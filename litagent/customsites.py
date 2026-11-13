"""Пользовательские сайты: описание в JSON-файле -> универсальный адаптер поиска + текст для контекста LLM.

Реестр (custom_sites.json) — единственный источник правды:
  * ручной режим и поиск читают из него параметры запроса (URL, имя параметра, операторы, пути в JSON);
  * авто-режим получает из него описание сайта и правила формирования запроса прямо в системный промпт.

Ограничения те же, что у остального агента: только официальные API, отдающие JSON (HTML выдачи не разбирается),
только GET, паузы по хосту, 401/403 = стоп, ключ не хранится в файле (только имя переменной окружения).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlparse

from .http import HttpError
from .models import Record, SearchSpec, strip_tags
from .policy import Policy
from .sources import SOURCE_CLASSES, Ctx, Source, _int, bool_expr, flat_terms, inv_abstract

DEFAULT_FILE = "custom_sites.json"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
FIELD_KEYS = ("title", "authors", "year", "doi", "abstract", "pdf_url", "url", "venue", "is_oa", "cited_by")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


# ---------------------------------------------------------------- пути в JSON
def get_path(obj, path: str):
    """'a.b' — вложенные ключи; 'authors[].name' — взять поле у каждого элемента списка.
    Без '[]' возвращает одно значение (или None), с '[]' — список."""
    if not path:
        return None
    cur = [obj]
    many = False
    for part in path.split("."):
        flat = part.endswith("[]")
        key = part[:-2] if flat else part
        many = many or flat
        nxt = []
        for node in cur:
            if key:
                node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                continue
            if flat:
                if isinstance(node, list):
                    nxt += node
            else:
                nxt.append(node)
        cur = nxt
        if not cur:
            return [] if many else None
    return cur if many else cur[0]


def _to_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        for x in v:
            s = _to_str(x)
            if s:
                return s
        return ""
    if isinstance(v, dict):
        for k in ("name", "display_name", "title", "full_name", "fullName", "value"):
            if v.get(k):
                return _to_str(v[k])
        fam, giv = v.get("family") or v.get("last_name") or v.get("lastName"), v.get("given") or v.get("first_name")
        return f"{fam or ''} {giv or ''}".strip()
    return str(v).strip()


def _str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (str, dict)):
        s = _to_str(v)
        return [s] if s else []
    if isinstance(v, list):
        return [s for s in (_to_str(x) for x in v) if s]
    return [str(v)]


def _to_year(v):
    while isinstance(v, list) and v:          # date-parts: [[2021, 5]]
        v = v[0]
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        y = int(v)
        return y if 1000 <= y <= 3000 else None
    m = re.search(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)", str(v))
    return int(m.group(1)) if m else None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return None


# ---------------------------------------------------------------- описание сайта
@dataclass
class Site:
    name: str = ""
    title: str = ""
    about: str = ""                       # для людей и для LLM: что это за база, для каких тем
    search_url: str = ""
    query_param: str = "q"
    static_params: dict = field(default_factory=dict)
    query_style: str = "boolean"          # boolean | plain
    op_and: str = " AND "
    op_or: str = " OR "
    op_not: str = " NOT "
    quote_phrases: bool = True
    query_template: str = "{query}"
    year_from_param: str = ""
    year_from_fmt: str = "{year}"
    year_to_param: str = ""
    year_to_fmt: str = "{year}"
    oa_param: str = ""
    oa_value: str = "true"
    limit_param: str = ""
    max_per_request: int = 100
    pagination: str = "none"              # page | offset | none
    page_param: str = ""
    page_start: int = 1
    key_env: str = ""                     # имя переменной окружения с ключом (сам ключ в файл не пишется)
    key_param: str = ""                   # ключ в параметре URL ...
    key_header: str = ""                  # ... или в заголовке
    key_prefix: str = ""                  # например 'Bearer '
    min_interval: float = 1.0
    results_path: str = ""
    total_path: str = ""
    fields: dict = field(default_factory=dict)
    notes: str = ""                       # синтаксис, лимиты, подводные камни — тоже уйдут в промпт LLM
    enabled: bool = True                  # включать ли автоматически во все запуски
    origin: str = ""                      # откуда импортирован (URL/файл), если не добавлен вручную

    # --- (де)сериализация
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Site":
        names = set(cls.__dataclass_fields__)
        s = cls(**{k: v for k, v in d.items() if k in names})
        s.name = (s.name or "").strip().lower()
        s.static_params = {str(k): str(v) for k, v in (s.static_params or {}).items()}
        s.fields = {k: str(v) for k, v in (s.fields or {}).items() if k in FIELD_KEYS and v}
        return s

    @property
    def host(self) -> str:
        return urlparse(self.search_url).netloc.lower()

    # --- проверка
    def validate(self, policy: Policy | None = None) -> list[str]:
        e = []
        if not NAME_RE.match(self.name or ""):
            e.append("имя: латиница в нижнем регистре, цифры и _, начинается с буквы (2–31 символ)")
        u = urlparse(self.search_url or "")
        if u.scheme not in ("http", "https") or not u.netloc:
            e.append("search_url: нужен полный адрес вида https://api.example.org/v1/search")
        elif u.scheme == "http" and (u.hostname or "") not in LOCAL_HOSTS:
            e.append("search_url: нужен https (http допустим только для localhost)")
        if not self.query_param:
            e.append("query_param: не задан параметр с поисковой строкой")
        if self.query_style not in ("boolean", "plain"):
            e.append("query_style: boolean или plain")
        if "{query}" not in (self.query_template or ""):
            e.append("query_template: должен содержать {query}")
        if self.pagination not in ("page", "offset", "none"):
            e.append("pagination: page, offset или none")
        if self.pagination in ("page", "offset") and not self.page_param:
            e.append("page_param: нужен для постраничной выдачи")
        if self.key_env and not (self.key_param or self.key_header):
            e.append("ключ: укажите, куда его подставлять (key_param или key_header)")
        if not self.fields.get("title"):
            e.append("fields.title: не указан путь к названию статьи")
        if not (0 < float(self.min_interval or 0) <= 600):
            e.append("min_interval: от 0 до 600 секунд (больше нуля)")
        if self.max_per_request < 1:
            e.append("max_per_request: минимум 1")
        if policy and u.netloc:
            ok, why = policy.url_allowed(self.search_url)
            if not ok:
                e.append(f"search_url: {why}")
        return e

    # --- текст для LLM
    def syntax_line(self) -> str:
        if self.query_style == "boolean":
            q = "multiword terms are put in double quotes" if self.quote_phrases else "multiword terms are NOT quoted"
            return (f"boolean query: AND = '{self.op_and.strip()}', OR = '{self.op_or.strip()}', "
                    f"NOT = '{self.op_not.strip()}'; {q}")
        return "plain keywords only, NO boolean operators (all terms are flattened into one keyword list)"

    def llm_details(self) -> str:
        L = [f"[{self.name}] {self.title or self.name} — USER-ADDED source"]
        if self.about:
            L.append(f"  About: {self.about}")
        fixed = f" plus fixed parameters {urlencode(self.static_params)}" if self.static_params else ""
        L.append(f"  Request: GET {self.search_url}, the query string goes into parameter `{self.query_param}`{fixed}")
        L.append(f"  Query syntax: {self.syntax_line()}")
        if self.query_template != "{query}":
            L.append(f"  The query is wrapped as: {self.query_template}")
        yrs = []
        if self.year_from_param:
            yrs.append(f"`{self.year_from_param}` = {self.year_from_fmt} (from)")
        if self.year_to_param:
            yrs.append(f"`{self.year_to_param}` = {self.year_to_fmt} (to)")
        L.append("  Year filter: " + (", ".join(yrs) if yrs else "none (years are filtered locally after the search)"))
        L.append("  Open-access filter: " + (f"`{self.oa_param}={self.oa_value}`" if self.oa_param else "none"))
        if self.pagination == "none":
            page = "single request, no paging"
        else:
            page = f"paging by {self.pagination} in `{self.page_param}`"
        if self.limit_param:
            page += f"; page size in `{self.limit_param}`, at most {self.max_per_request} per request"
        L.append(f"  Paging: {page}")
        if self.key_env:
            where = f"parameter `{self.key_param}`" if self.key_param else f"header `{self.key_header}`"
            L.append(f"  API key: required (env {self.key_env}, sent in {where}); the source is skipped without it")
        if self.notes:
            L.append(f"  User notes (syntax, limits, pitfalls): {self.notes}")
        return "\n".join(L)

    def human_line(self) -> str:
        key = "ключ" if self.key_env else "без ключа"
        flag = ("" if self.enabled else "  [выключен по умолчанию]") + (f"  <- {self.origin}" if self.origin else "")
        return (f"{self.name:<14} {self.title or self.name:<24} {self.host}  "
                f"({'булевы операторы' if self.query_style == 'boolean' else 'простые слова'}, {key}){flag}")


# ---------------------------------------------------------------- реестр (файл)
class SiteRegistry:
    def __init__(self, path: str | Path = DEFAULT_FILE):
        self.path = Path(path)
        self.sites: list[Site] = []
        self.error: str = ""                  # файл есть, но не читается — менять его нельзя, чтобы не затереть

    @classmethod
    def load(cls, path: str | Path | None = None, log: Callable | None = None) -> "SiteRegistry":
        reg = cls(path or DEFAULT_FILE)
        if not reg.path.exists():
            return reg
        try:
            data = json.loads(reg.path.read_text(encoding="utf-8"))
            items = data.get("sites", []) if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("ожидался список сайтов в ключе 'sites'")
            seen = set()
            for it in items:
                s = Site.from_dict(it)
                if s.name and s.name not in seen:
                    seen.add(s.name)
                    reg.sites.append(s)
        except (ValueError, OSError, TypeError, AttributeError) as e:
            reg.error = f"{reg.path} не читается ({e})"
            reg.sites = []
            if log:
                log(f"⚠ {reg.error}; пользовательские сайты не подключены. Исправьте файл или удалите его.")
        return reg

    def names(self) -> list[str]:
        return [s.name for s in self.sites]

    def enabled_names(self) -> list[str]:
        return [s.name for s in self.sites if s.enabled]

    def get(self, name: str) -> Site | None:
        name = (name or "").strip().lower()
        return next((s for s in self.sites if s.name == name), None)

    def put(self, site: Site) -> None:
        for i, s in enumerate(self.sites):
            if s.name == site.name:
                self.sites[i] = site
                return
        self.sites.append(site)

    def remove(self, name: str) -> bool:
        s = self.get(name)
        if s:
            self.sites.remove(s)
        return bool(s)

    def save(self) -> Path:
        if self.error:
            raise RuntimeError(self.error)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "sites": [s.to_dict() for s in self.sites]},
                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path


def name_error(name: str, reg: SiteRegistry, policy: Policy, editing: bool = False) -> str:
    if not NAME_RE.match(name):
        return "латиница в нижнем регистре, цифры и _, начинается с буквы, 2–31 символ"
    if name in SOURCE_CLASSES:
        return "это имя занято встроенным источником"
    if name in policy.forbidden_sources:
        return "этот источник запрещён политикой агента"
    if not editing and reg.get(name):
        return "сайт с таким именем уже есть (изменить: `sites edit`)"
    return ""


# ---------------------------------------------------------------- адаптер
class CustomSource(Source):
    """Универсальный адаптер по описанию Site. Только GET и JSON."""

    def __init__(self, ctx: Ctx, site: Site):
        super().__init__(ctx)
        self.site = site
        self.name = site.name
        self.boolean = site.query_style == "boolean"
        self.key_required = bool(site.key_env)

    # --- ключ
    def key(self):
        s = self.site
        if not s.key_env:
            return None
        return os.environ.get(s.key_env) or self.ctx.keys.get(s.name) or self.ctx.keys.get(s.key_env) or None

    def available(self) -> tuple[bool, str]:
        if self.site.key_env and not self.key():
            return False, f"не задан ключ: переменная окружения {self.site.key_env} (или [keys] {self.name} в litagent.toml)"
        return True, ""

    # --- запрос
    def build_query(self, spec: SearchSpec) -> str:
        s = self.site

        def fmt(t: str) -> str:
            t = t.replace('"', "").strip()
            return f'"{t}"' if (s.quote_phrases and re.search(r"\s", t)) else t
        expr = bool_expr(spec, fmt, s.op_and, s.op_or, s.op_not) if self.boolean else flat_terms(spec)
        return (s.query_template or "{query}").replace("{query}", expr)

    def _params(self, spec: SearchSpec) -> dict:
        s = self.site
        p = dict(s.static_params)
        p[s.query_param] = self.query_for(spec)
        if s.year_from_param and spec.year_from:
            p[s.year_from_param] = s.year_from_fmt.replace("{year}", str(spec.year_from))
        if s.year_to_param and spec.year_to:
            p[s.year_to_param] = s.year_to_fmt.replace("{year}", str(spec.year_to))
        if s.oa_param and spec.open_access:
            p[s.oa_param] = s.oa_value
        if self.key() and s.key_param:
            p[s.key_param] = self.key()
        return p

    def _headers(self):
        s = self.site
        if self.key() and s.key_header:
            return {s.key_header: f"{s.key_prefix}{self.key()}"}
        return None

    def fetch_json(self, spec: SearchSpec, n: int, extra: dict | None = None):
        s = self.site
        params = self._params(spec)
        if s.limit_param:
            params[s.limit_param] = n
        params.update(extra or {})
        r = self.ctx.client.get(s.search_url, params, self._headers())
        if r.status != 200:
            raise HttpError(f"{self.name}: HTTP {r.status}")
        try:
            return r.json()
        except ValueError as e:
            raise HttpError(f"{self.name}: ответ не JSON (HTML-страницы агент не разбирает; нужен API): {e}") from e

    # --- разбор
    def _items(self, data, first: bool):
        s = self.site
        items = get_path(data, s.results_path) if s.results_path else data
        if not isinstance(items, list):
            if first:
                top = ", ".join(list(data)[:10]) if isinstance(data, dict) else type(data).__name__
                raise HttpError(f"{self.name}: в ответе нет списка по пути '{s.results_path}' (верхний уровень: {top})")
            return []
        return [x for x in items if isinstance(x, dict)]

    def record_from(self, item: dict) -> Record | None:
        f = self.site.fields
        g = lambda k: get_path(item, f[k]) if f.get(k) else None      # noqa: E731
        title = _to_str(g("title"))
        if not title:
            return None
        ab = g("abstract")
        abstract = inv_abstract(ab) if isinstance(ab, dict) and all(isinstance(v, list) for v in ab.values()) \
            else strip_tags(_to_str(ab))
        pdfs = [u for u in _str_list(g("pdf_url")) if u.lower().startswith(("http://", "https://"))]
        doi = _to_str(g("doi")) or None
        url = _to_str(g("url")) or (f"https://doi.org/{doi}" if doi else None)
        oa = _to_bool(g("is_oa")) if f.get("is_oa") else None
        if oa is None and pdfs:
            oa = True
        return Record(title=title, authors=_str_list(g("authors")), year=_to_year(g("year")), doi=doi,
                      abstract=abstract, venue=_to_str(g("venue")), cited_by=_int(g("cited_by")),
                      is_oa=oa, pdf_urls=pdfs, url=url)

    def search(self, spec: SearchSpec, limit: int):
        s = self.site
        q = self.query_for(spec)
        per = max(1, s.max_per_request)
        recs, fetched, found, page, first = [], 0, None, s.page_start, True
        for _ in range(200):                                        # предохранитель от бесконечного цикла
            if len(recs) >= limit:
                break
            n = min(per, limit - len(recs)) if s.limit_param else per
            extra = {}
            if s.pagination == "page":
                extra[s.page_param] = page
            elif s.pagination == "offset":
                extra[s.page_param] = fetched
            data = self.fetch_json(spec, n, extra)
            items = self._items(data, first)
            if s.total_path:
                t = get_path(data, s.total_path)
                found = _int(t.get("value") if isinstance(t, dict) else t) if t is not None else found
            first = False
            if not items:
                break
            fetched += len(items)
            recs += [r for r in (self.record_from(it) for it in items) if r]
            page += 1
            if s.pagination == "none" or (s.limit_param and len(items) < n) or (found is not None and fetched >= found):
                break
        return recs[:limit], {"query": q, "found": found if found is not None else len(recs)}

    # --- описание для LLM
    def llm_info(self) -> dict:
        s = self.site
        return {"about": f"{s.title or s.name} (USER-ADDED). {s.about}".strip(),
                "hint": f"{s.syntax_line()}. {s.notes}".strip()[:600],
                "details": s.llm_details(), "custom": True}


# ---------------------------------------------------------------- подсказки по структуре JSON (для опросника)
_LIST_KEYS = ("results", "items", "data", "hits", "records", "docs", "entries", "papers", "works", "articles", "result")
_FIELD_CANDIDATES = {
    "title": ["title", "name", "display_name", "headline"],
    "authors": ["authors[].name", "authors[].display_name", "authorships[].author.display_name", "author[].name",
                "creators[].name", "authors", "creators", "author"],
    "year": ["year", "publication_year", "pubYear", "yearPublished", "published_year", "publication_date",
             "published", "date", "issued"],
    "doi": ["doi", "DOI", "externalIds.DOI", "metadata.doi"],
    "abstract": ["abstract", "summary", "description", "abstractText"],
    "pdf_url": ["pdf_url", "pdf", "downloadUrl", "download_url", "fulltext_url", "openAccessPdf.url",
                "best_oa_location.pdf_url", "links.pdf"],
    "url": ["url", "link", "landing_page_url", "html_url", "id"],
    "venue": ["venue", "journal.title", "journal", "container-title", "source"],
    "is_oa": ["is_oa", "isOpenAccess", "open_access.is_oa", "open_access"],
    "cited_by": ["cited_by_count", "citationCount", "citations", "cited_by"],
}
_TOTAL_CANDIDATES = ["total", "totalHits", "total_results", "totalResults", "count", "hitCount", "numFound",
                     "meta.count", "meta.total", "hits.total.value", "hits.total"]


def _find_list(data, depth: int = 0):
    """Все списки словарей в ответе: [(путь, список)]."""
    found = []
    if isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data[:3]):
            found.append(("", data))
    elif isinstance(data, dict) and depth < 3:
        for k, v in data.items():
            for sub, lst in _find_list(v, depth + 1):
                found.append((k + ("." + sub if sub else ""), lst))
    return found


def guess_mapping(data) -> dict:
    """По образцу ответа предлагает results_path, total_path и пути к полям. Это подсказка, а не истина."""
    cands = _find_list(data)
    out = {"results_path": "", "total_path": "", "fields": {}, "sample": None}
    if not cands:
        return out
    cands.sort(key=lambda c: (c[0].split(".")[-1].lower() not in _LIST_KEYS, -len(c[1]), c[0].count(".")))
    path, items = cands[0]
    out["results_path"] = path
    item = items[0]
    out["sample"] = item
    for key, options in _FIELD_CANDIDATES.items():
        for o in options:
            v = get_path(item, o)
            if v not in (None, "", [], {}):
                out["fields"][key] = o
                break
    if isinstance(data, dict):
        for o in _TOTAL_CANDIDATES:
            v = get_path(data, o)
            if isinstance(v, dict):
                v = v.get("value")
            if _int(v) is not None:
                out["total_path"] = o
                break
    return out
