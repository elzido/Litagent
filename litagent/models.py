"""Модели данных: Record (статья), SearchSpec (нейтральное описание запроса), дедупликация."""
from __future__ import annotations

import datetime
import html
import re
import unicodedata
from dataclasses import asdict, dataclass, field


def current_year() -> int:
    return datetime.date.today().year


# ---------------------------------------------------------------- нормализация
def norm_doi(d):
    if not d:
        return None
    d = str(d).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    d = d.strip().rstrip(".,;")
    return d or None


def norm_arxiv(a):
    if not a:
        return None
    a = str(a).strip()
    a = re.sub(r"^(https?://)?(www\.|export\.)?arxiv\.org/(abs|pdf)/", "", a, flags=re.I)
    a = re.sub(r"^arxiv:", "", a, flags=re.I)
    a = re.sub(r"\.pdf$", "", a)
    a = re.sub(r"v\d+$", "", a)
    return a or None


def norm_title(t: str) -> str:
    return re.sub(r"[\W_]+", " ", (t or "").casefold()).strip()


def strip_tags(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def safe_name(s: str, maxlen: int = 80) -> str:
    s = ascii_fold(s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:maxlen] or "paper"


# ---------------------------------------------------------------- Record
@dataclass
class Record:
    title: str = ""
    authors: list = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    abstract: str = ""
    venue: str = ""
    cited_by: int | None = None
    is_oa: bool | None = None
    pdf_urls: list = field(default_factory=list)
    supp_urls: list = field(default_factory=list)  # ссылки на доп. материалы/пакеты от источника
    xml_url: str | None = None
    url: str | None = None
    sources: list = field(default_factory=list)
    keywords: list = field(default_factory=list)   # ключевые слова источников (для авто-тегов)
    has_supp: bool | None = None                   # есть ли у статьи доп. материалы (известно из Europe PMC / проверки)
    score: float | None = None
    file: str | None = None
    supp: list = field(default_factory=list)   # доп. материалы: [{file, name, size, origin, added}]
    status: str = ""

    def __post_init__(self):
        self.doi = norm_doi(self.doi)
        self.arxiv_id = norm_arxiv(self.arxiv_id)
        if self.pmcid:
            self.pmcid = str(self.pmcid).upper()
            if not self.pmcid.startswith("PMC"):
                self.pmcid = "PMC" + self.pmcid

    def keys(self) -> list[str]:
        ks = []
        if self.doi:
            ks.append("doi:" + self.doi)
        if self.arxiv_id:
            ks.append("arxiv:" + self.arxiv_id)
        if self.pmid:
            ks.append("pmid:" + str(self.pmid))
        if self.pmcid:
            ks.append("pmcid:" + self.pmcid)
        nt = norm_title(self.title)
        if len(nt) >= 25:
            ks.append("title:" + nt)
        return ks

    def primary_key(self) -> str:
        ks = self.keys()
        return ks[0] if ks else "title:" + norm_title(self.title)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})

    def short(self) -> str:
        a = (self.authors[0] if self.authors else "?")
        return f"{a} ({self.year or '?'}) {self.title[:90]}"


def merge_into(a: Record, b: Record) -> None:
    for f in ("doi", "pmid", "pmcid", "arxiv_id", "openalex_id", "url", "xml_url", "venue", "year"):
        if not getattr(a, f) and getattr(b, f):
            setattr(a, f, getattr(b, f))
    if not a.title and b.title:
        a.title = b.title
    if len(b.abstract or "") > len(a.abstract or ""):
        a.abstract = b.abstract
    if not a.authors and b.authors:
        a.authors = b.authors
    if b.cited_by is not None:
        a.cited_by = max(a.cited_by or 0, b.cited_by)
    if a.is_oa is not True:
        a.is_oa = True if b.is_oa else (a.is_oa if a.is_oa is not None else b.is_oa)
    for u in b.pdf_urls:
        if u not in a.pdf_urls:
            a.pdf_urls.append(u)
    for u in b.supp_urls:
        if u not in a.supp_urls:
            a.supp_urls.append(u)
    for k in b.keywords:
        if k not in a.keywords:
            a.keywords.append(k)
    if a.has_supp is None:
        a.has_supp = b.has_supp
    elif b.has_supp:
        a.has_supp = True
    for s in b.sources:
        if s not in a.sources:
            a.sources.append(s)


def dedupe(records: list[Record]) -> list[Record]:
    out: list[Record] = []
    index: dict[str, int] = {}
    for r in records:
        hit = None
        for k in r.keys():
            if k in index:
                hit = index[k]
                break
        if hit is None:
            out.append(r)
            hit = len(out) - 1
        else:
            merge_into(out[hit], r)
        for k in out[hit].keys():
            index.setdefault(k, hit)
    return out


# ---------------------------------------------------------------- SearchSpec
def _term_re(term: str):
    core = term.strip().rstrip("*")
    return re.compile(r"(?<!\w)" + re.escape(core.casefold()) + r"\w*", re.U)


def _has(text: str, term: str) -> bool:
    if not term.strip():
        return False
    return bool(_term_re(term).search(text))


@dataclass
class SearchSpec:
    """Нейтральный запрос: groups = AND из OR-групп; exclude = исключаемые термины."""
    groups: list = field(default_factory=list)      # list[list[str]]
    exclude: list = field(default_factory=list)     # list[str]
    year_from: int | None = None
    year_to: int | None = None
    open_access: bool = False
    title_only: bool = False
    strict: bool = False                            # локально требовать каждую группу в тексте
    arxiv_cats: list = field(default_factory=list)  # например ["cs.CL"]
    raw: dict = field(default_factory=dict)         # {источник: готовая строка запроса}
    label: str = ""

    def describe(self) -> str:
        g = " AND ".join("(" + " | ".join(x) + ")" if len(x) > 1 else x[0] for x in self.groups)
        if self.exclude:
            g += " NOT (" + " | ".join(self.exclude) + ")"
        yr = f" [{self.year_from or ''}..{self.year_to or ''}]" if (self.year_from or self.year_to) else ""
        return g + yr + (" OA" if self.open_access else "")

    def local_ok(self, rec: Record, strict: bool = False) -> bool:
        text = f"{rec.title} {rec.abstract}".casefold()
        if any(_has(text, t) for t in self.exclude):
            return False
        if rec.year:
            if self.year_from and rec.year < self.year_from:
                return False
            if self.year_to and rec.year > self.year_to:
                return False
        if (strict or self.strict) and rec.abstract:
            for g in self.groups:
                if not any(_has(text, t) for t in g):
                    return False
        return True


def _clean_term(t: str) -> str:
    t = t.strip().strip("\"'«»").replace('"', "")
    return re.sub(r"\s+", " ", t)


def parse_topic(text: str, **kw) -> SearchSpec:
    """Мини-язык темы:  'CRISPR; sickle cell disease | SCD; -review'
       ';' или ',' — И;  '|' — ИЛИ внутри группы;  '-термин' — исключить."""
    groups, exclude = [], []
    for part in re.split(r"[;,\n]", text or ""):
        part = part.strip()
        if not part:
            continue
        if part.startswith("-") and len(part) > 1:
            exclude += [t for t in (_clean_term(x) for x in part[1:].split("|")) if t]
            continue
        g = [t for t in (_clean_term(x) for x in part.split("|")) if t]
        if g:
            groups.append(g)
    return SearchSpec(groups=groups, exclude=exclude, label=(text or "").strip()[:80], **kw)
