"""Опросник «вопрос-ответ»: собирает описание сайта (Site) и сохраняет его в реестр.
ask/say передаются снаружи — поэтому опросник тестируется без клавиатуры."""
from __future__ import annotations

import json
from typing import Callable
from urllib.parse import parse_qsl, urlparse

from .customsites import (FIELD_KEYS, LOCAL_HOSTS, Site, SiteRegistry, guess_mapping, name_error)
from .policy import Policy

INTRO = """\
Добавление сайта для поиска статей. Отвечайте на вопросы:
  Enter — принять значение в [скобках];  «-» — очистить необязательное поле;  «/q» — отмена (ничего не сохранится).
Подходят только официальные API, которые отдают JSON и принимают GET-запрос. HTML-страницы выдачи агент не разбирает."""

FIELD_LABELS = {
    "title": "Поле с названием статьи (обязательно)",
    "authors": "Поле с авторами (список или строка)",
    "year": "Поле с годом или датой публикации",
    "doi": "Поле с DOI",
    "abstract": "Поле с аннотацией",
    "pdf_url": "Поле со ссылкой на PDF (если API её отдаёт)",
    "url": "Поле со ссылкой на страницу статьи",
    "venue": "Поле с названием журнала",
    "is_oa": "Поле-признак открытого доступа (true/false)",
    "cited_by": "Поле с числом цитирований",
}


class Cancelled(Exception):
    pass


class Wizard:
    def __init__(self, ask: Callable[[str], str], say: Callable[[str], None]):
        self.ask, self.say = ask, say

    def text(self, prompt: str, default: str = "", required: bool = False, check=None) -> str:
        while True:
            try:
                raw = self.ask(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
            except (EOFError, KeyboardInterrupt):
                raise Cancelled from None
            if raw.lower() in ("/q", "/выход"):
                raise Cancelled
            val = "" if raw == "-" else (default if raw == "" else raw)
            if required and not val:
                self.say("  Это поле обязательно.")
                continue
            if val and check:
                err = check(val)
                if err:
                    self.say(f"  ✗ {err}")
                    continue
            return val

    def yesno(self, prompt: str, default: bool) -> bool:
        while True:
            try:
                raw = self.ask(f"{prompt} ({'Д/н' if default else 'д/Н'}): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raise Cancelled from None
            if raw in ("/q", "/выход"):
                raise Cancelled
            if raw == "":
                return default
            if raw in ("д", "да", "y", "yes"):
                return True
            if raw in ("н", "нет", "n", "no"):
                return False
            self.say("  Ответьте «да» или «нет».")

    def choice(self, prompt: str, options: list[str], default: int = 1) -> int:
        self.say(prompt)
        for i, o in enumerate(options, 1):
            self.say(f"  {i}) {o}")
        while True:
            val = self.text("  Номер", str(default))
            if val.isdigit() and 1 <= int(val) <= len(options):
                return int(val)
            self.say(f"  Введите число от 1 до {len(options)}.")

    def number(self, prompt: str, default, cast=int, lo=None, hi=None):
        def check(v):
            try:
                x = cast(v)
            except ValueError:
                return "нужно число"
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                return f"допустимо от {lo} до {hi}"
            return ""
        return cast(self.text(prompt, str(default), required=True, check=check))


def _url_error(policy: Policy):
    def check(url: str) -> str:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.netloc:
            return "нужен полный адрес, например https://api.example.org/v1/search"
        if u.scheme == "http" and (u.hostname or "") not in LOCAL_HOSTS:
            return "нужен https (http допустим только для localhost)"
        ok, why = policy.url_allowed(url)
        return "" if ok else why
    return check


def _params_error(v: str) -> str:
    for part in v.replace(",", "&").split("&"):
        if part.strip() and "=" not in part:
            return f"«{part.strip()}» — ожидается вид имя=значение"
    return ""


def _parse_params(v: str) -> dict:
    return {k.strip(): val.strip() for k, val in parse_qsl(v.replace(",", "&"), keep_blank_values=True) if k.strip()}


def _pad(op: str) -> str:
    return f" {op.strip()} "


def interview(reg: SiteRegistry, existing: Site | None = None, ask=input, say=print,
              probe: Callable[[Site], tuple] | None = None, policy: Policy | None = None) -> Site | None:
    """Возвращает заполненный Site или None (отмена / ошибка проверки). В реестр сам ничего не пишет."""
    policy = policy or Policy()
    w = Wizard(ask, say)
    d = Site.from_dict(existing.to_dict()) if existing else Site()
    try:
        say(INTRO)
        if existing:
            say(f"\nРедактируем «{existing.name}» (имя менять нельзя).")
        else:
            d.name = w.text("\n1. Короткое имя для --sources (латиница, например mysite)", required=True,
                            check=lambda v: name_error(v.strip().lower(), reg, policy)).strip().lower()
        d.title = w.text("2. Название для людей", d.title or d.name)
        d.about = w.text("3. Что это за база и для каких тем она подходит (это прочитает нейросеть)",
                         d.about, required=True)
        d.search_url = w.text("4. Адрес поискового API (https://…)", d.search_url, required=True,
                              check=_url_error(policy))
        d.query_param = w.text("5. Имя параметра, куда подставляется поисковая строка (q, query, search…)",
                               d.query_param or "q", required=True)
        cur = "&".join(f"{k}={v}" for k, v in d.static_params.items())
        d.static_params = _parse_params(w.text("6. Постоянные параметры, например format=json&lang=en (необязательно)",
                                               cur, check=_params_error))

        # --- как писать запрос
        style = w.choice("7. Как сайт понимает поисковую строку?",
                         ["булевы операторы (AND / OR / NOT, скобки)", "просто слова, без операторов (поиск по релевантности)"],
                         1 if d.query_style == "boolean" else 2)
        d.query_style = "boolean" if style == 1 else "plain"
        if d.query_style == "boolean":
            d.op_and = _pad(w.text("   Оператор «И»", d.op_and.strip() or "AND", required=True))
            d.op_or = _pad(w.text("   Оператор «ИЛИ»", d.op_or.strip() or "OR", required=True))
            d.op_not = _pad(w.text("   Оператор «НЕ»", d.op_not.strip() or "NOT", required=True))
            d.quote_phrases = w.yesno("   Брать многословные термины в двойные кавычки?", d.quote_phrases)
        d.query_template = w.text("   Обёртка запроса: шаблон с {query}, например title:({query}); Enter — без обёртки",
                                  d.query_template or "{query}", required=True,
                                  check=lambda v: "" if "{query}" in v else "в шаблоне должно быть {query}")

        # --- фильтры
        say("\n8. Фильтр по годам (если сайт не умеет — оставьте пустым, годы отфильтруются локально).")
        d.year_from_param = w.text("   Параметр «год от»", d.year_from_param)
        if d.year_from_param:
            d.year_from_fmt = w.text("   Формат значения ({year} — год, например {year}-01-01)", d.year_from_fmt or "{year}",
                                     required=True, check=lambda v: "" if "{year}" in v else "нужен {year}")
        d.year_to_param = w.text("   Параметр «год до»", d.year_to_param)
        if d.year_to_param:
            d.year_to_fmt = w.text("   Формат значения ({year} — год, например {year}-12-31)", d.year_to_fmt or "{year}",
                                   required=True, check=lambda v: "" if "{year}" in v else "нужен {year}")
        d.oa_param = w.text("9. Параметр «только открытый доступ» (необязательно)", d.oa_param)
        if d.oa_param:
            d.oa_value = w.text("   Его значение", d.oa_value or "true", required=True)

        # --- страницы
        pg = w.choice("10. Как устроена выдача по страницам?",
                      ["номер страницы (page=1, 2, 3…)", "смещение (offset=0, 20, 40…)", "один запрос, без страниц"],
                      {"page": 1, "offset": 2, "none": 3}.get(d.pagination, 3))
        d.pagination = ("page", "offset", "none")[pg - 1]
        if d.pagination != "none":
            d.page_param = w.text("    Имя параметра", d.page_param or ("page" if pg == 1 else "offset"), required=True)
        if d.pagination == "page":
            d.page_start = w.number("    С какого номера начинаются страницы", d.page_start, lo=0, hi=1)
        d.limit_param = w.text("11. Параметр «сколько записей вернуть» (limit, per_page, size…; необязательно)", d.limit_param)
        if d.limit_param:
            d.max_per_request = w.number("    Максимум записей за один запрос", d.max_per_request or 100, lo=1, hi=10000)

        # --- ключ
        if w.yesno("12. Нужен API-ключ?", bool(d.key_env)):
            d.key_env = w.text("    Имя переменной окружения с ключом (сам ключ в файл не записывается)",
                               d.key_env or f"{d.name.upper()}_API_KEY", required=True)
            where = w.choice("    Куда подставлять ключ?", ["параметр в адресе", "заголовок HTTP"],
                             2 if d.key_header else 1)
            if where == 1:
                d.key_param = w.text("    Имя параметра", d.key_param or "api_key", required=True)
                d.key_header = d.key_prefix = ""
            else:
                d.key_header = w.text("    Имя заголовка", d.key_header or "Authorization", required=True)
                d.key_prefix = w.text("    Префикс значения (например «Bearer »; необязательно)", d.key_prefix)
                d.key_param = ""
        else:
            d.key_env = d.key_param = d.key_header = d.key_prefix = ""
        d.min_interval = w.number("13. Минимальная пауза между запросами к сайту, сек (см. его правила)",
                                  d.min_interval or 1.0, cast=float, lo=0.05, hi=600)

        # --- структура ответа
        data = None
        if probe and w.yesno("14. Сделать пробный запрос («machine learning») и подсказать пути в ответе?", True):
            data, err = probe(d)
            if err:
                say(f"   Пробный запрос не удался: {err}\n   Укажите пути вручную.")
                data = None
            elif isinstance(data, dict):
                say("   Верхний уровень ответа: " + ", ".join(list(data)[:15]))
        g = guess_mapping(data) if data is not None else {"results_path": "", "total_path": "", "fields": {}, "sample": None}
        if data is not None and g["sample"] is not None:
            say("   Пример записи из ответа:\n" + json.dumps(g["sample"], ensure_ascii=False, indent=2)[:1200])
        elif data is not None:
            say("   Не нашёл в ответе список записей — укажите путь вручную.")
        say("   Пути пишутся через точку; для списка — [] (например authors[].name).")
        d.results_path = w.text("   Путь к списку статей в ответе (пусто — весь ответ и есть список)",
                                d.results_path or g["results_path"])
        d.total_path = w.text("   Путь к общему числу найденных (необязательно)", d.total_path or g["total_path"])
        for k in FIELD_KEYS:
            d.fields[k] = w.text(f"   {FIELD_LABELS[k]}", d.fields.get(k) or g["fields"].get(k, ""),
                                 required=(k == "title"))
        d.fields = {k: v for k, v in d.fields.items() if v}

        # --- пометки и режим
        d.notes = w.text("15. Особенности для нейросети: синтаксис, лимиты, подводные камни (необязательно)", d.notes)
        d.enabled = w.yesno("16. Использовать сайт автоматически во всех запусках (иначе — только через --sources)?", d.enabled)

        errs = d.validate(policy)
        if errs:
            say("\nНе сохранено, найдены ошибки:\n" + "\n".join(f"  ✗ {e}" for e in errs))
            return None
        say("\nИтог — именно это увидит нейросеть в авто-режиме:\n" + d.llm_details())
        if not w.yesno("\nСохранить?", True):
            say("Отменено, ничего не записано.")
            return None
        return d
    except Cancelled:
        say("\nОтменено, ничего не записано.")
        return None
