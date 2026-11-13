"""Командная строка: manual | auto | check | notes."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import tomllib
from pathlib import Path

from .agent import Agent
from .customsites import DEFAULT_FILE, CustomSource, SiteRegistry
from .helptext import render as render_help
from .http import PoliteClient
from .llm import LLM
from .models import SearchSpec, parse_topic
from .notes import Notes
from .remote import fetch_text, is_url, parse_sites, pick, sanitize
from .policy import Policy
from .planner import auto_search, heuristic_plan, llm_filter, request_plan
from .sources import DEFAULT_SOURCES, SOURCE_CLASSES, Ctx
from .store import Library
from .wizard import interview

DEFAULT_CFG = {
    "email": "", "tool": "litagent", "out": "library", "sources": DEFAULT_SOURCES,
    "limit_per_source": 50, "max_downloads": 20, "cache_ttl": 86400, "custom_sites": DEFAULT_FILE, "pubmed_pmc": True, "pdf_only": False,
    "download_workers": 4, "download_timeout": 25, "download_retries": 2,
    "keys": {"ncbi": "", "openalex": "", "s2": "", "core": ""},
    "llm": {"provider": "ollama", "base_url": "", "model": "gemma4", "temperature": 0.2, "timeout": 300, "api_key": ""},
    "auto": {"rounds": 2, "min_results": 10, "llm_filter": False, "min_score": 6, "notes": []},
}
ENV_KEYS = {"ncbi": "NCBI_API_KEY", "openalex": "OPENALEX_API_KEY", "s2": "S2_API_KEY", "core": "CORE_API_KEY"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def ask(prompt: str) -> str:
    return input(prompt)


def load_config(path: str | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    p = Path(path) if path else Path("litagent.toml")
    if p.exists():
        with open(p, "rb") as f:
            data = tomllib.load(f)
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    elif path:
        raise SystemExit(f"Конфиг не найден: {path}")
    if os.environ.get("LITAGENT_EMAIL"):
        cfg["email"] = os.environ["LITAGENT_EMAIL"]
    for k, env in ENV_KEYS.items():
        if os.environ.get(env):
            cfg["keys"][k] = os.environ[env]
    return cfg


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="путь к litagent.toml (по умолчанию ./litagent.toml)")
    p.add_argument("--out", help="папка библиотеки (pdf/, xml/, index.json, runs/)")
    p.add_argument("--sites-file", help="файл со своими сайтами (по умолчанию ./custom_sites.json)")
    p.add_argument("--email", help="контактный e-mail (обязателен для реальных запросов)")
    p.add_argument("--sources", help="источники через запятую: " + ",".join(DEFAULT_SOURCES))
    p.add_argument("--site", help="искать ТОЛЬКО на этом сайте (встроенном или своём из custom_sites.json); в авто-режиме "
                                  "нейросеть тоже увидит только его")
    p.add_argument("--pick-site", action="store_true", help="выбрать один сайт из списка (вопрос с номерами)")
    p.add_argument("--no-pmc", action="store_true", help="PubMed не ищет в PMC и не скачивает оттуда PDF")
    p.add_argument("--limit", type=int, help="сколько записей брать с каждого источника")
    p.add_argument("--year-from", type=int)
    p.add_argument("--year-to", type=int)
    p.add_argument("--oa-only", action="store_true", help="только открытый доступ")
    p.add_argument("--title-only", action="store_true", help="искать только в названиях")
    p.add_argument("--strict", action="store_true", help="локально требовать каждую группу терминов в тексте")
    p.add_argument("--no-download", action="store_true", help="только найти, не скачивать")
    p.add_argument("--workers", type=int, help="параллельных скачиваний (по умолчанию 4; 1 — по одному). "
                                              "Паузы и одно соединение на хост сохраняются")
    p.add_argument("--pdf-only", action="store_true", help="оставить только статьи с реально доступным полным PDF "
                                                           "(включает --oa-only; проверка = скачивание, XML не берётся)")
    p.add_argument("--max-downloads", type=int, help="максимум успешных скачиваний за запуск")
    p.add_argument("--dry-run", action="store_true", help="показать запросы и выйти (без обращений к источникам)")


def _resolve_single_site(cfg: dict, a) -> str | None:
    """--site NAME или --pick-site -> имя единственного сайта; иначе None. Ошибки — SystemExit с понятным текстом."""
    site = (getattr(a, "site", None) or "").strip().lower()
    if not site and not getattr(a, "pick_site", False):
        return None
    if getattr(a, "sources", None):
        raise SystemExit("Используйте что-то одно: --site/--pick-site или --sources.")
    reg = SiteRegistry.load(cfg.get("custom_sites") or DEFAULT_FILE)
    mine = [n for n in reg.names() if n not in SOURCE_CLASSES]
    names = list(SOURCE_CLASSES) + mine
    if not site:
        log("Выберите сайт, на котором искать:")
        for i, n in enumerate(names, 1):
            log(f"  {i:>2}) {n}" + ("  (свой)" if n in mine else ""))
        try:
            ans = ask("Номер или имя: ").strip().lower()
        except EOFError:
            raise SystemExit("Отменено.") from None
        site = names[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(names) else ans
    if site not in names:
        why = Policy().forbidden_sources.get(site)
        raise SystemExit(f"Сайт «{site}» недоступен агенту: {why}" if why else
                         f"Нет сайта «{site}». Доступны: {', '.join(names)}")
    return site


def apply_overrides(cfg: dict, a) -> dict:
    for k in ("out", "email"):
        if getattr(a, k, None):
            cfg[k] = getattr(a, k)
    if getattr(a, "sites_file", None):
        cfg["custom_sites"] = a.sites_file
    if getattr(a, "workers", None):
        cfg["download_workers"] = max(1, a.workers)
    if getattr(a, "no_pmc", False):
        cfg["pubmed_pmc"] = False
    if cfg.get("pdf_only") and hasattr(a, "pdf_only"):
        a.pdf_only = True
    if getattr(a, "pdf_only", False):
        if getattr(a, "no_download", False):
            raise SystemExit("--pdf-only проверяет наличие PDF скачиванием: уберите --no-download.")
        a.oa_only = True                                  # источники сразу сузят выдачу до открытого доступа
    one = _resolve_single_site(cfg, a)
    if one:
        cfg["sources"] = [one]
    elif getattr(a, "sources", None):
        cfg["sources"] = [s.strip() for s in a.sources.split(",") if s.strip()]
    else:                                   # свои включённые сайты подключаются сами, если источники не заданы явно
        reg = SiteRegistry.load(cfg.get("custom_sites") or DEFAULT_FILE)
        cfg["sources"] = list(cfg["sources"]) + [n for n in reg.enabled_names()
                                                 if n not in cfg["sources"] and n not in SOURCE_CLASSES]
    if getattr(a, "max_downloads", None) is not None:
        cfg["max_downloads"] = a.max_downloads
    return cfg


def table(records, n=15) -> None:
    for i, r in enumerate(records[:n], 1):
        oa = {True: "OA", False: "--", None: "? "}[r.is_oa]
        sc = f" s={r.score:.0f}" if r.score is not None else ""
        log(f"  {i:>2}. {oa} [{','.join(r.sources)}]{sc} {r.short()}")
    if len(records) > n:
        log(f"  … ещё {len(records) - n}")


def need_email(cfg: dict) -> bool:
    if cfg.get("email"):
        return True
    log("Нужен контактный e-mail (--email, litagent.toml или LITAGENT_EMAIL): "
        "его требуют Crossref/Unpaywall/NCBI, и он нужен для вежливого User-Agent.")
    return False


def finish_and_report(agent: Agent, mode: str, specs, records, args, cfg, extra=None) -> int:
    counts = None
    records = agent.rank(records)
    log(f"\nУникальных записей: {len(records)}")
    table(records)
    extra = dict(extra or {})
    pdf_only = getattr(args, "pdf_only", False)
    if records and not args.no_download:
        log(f"\nСкачивание (до {cfg['max_downloads']} " + ("статей с полным PDF" if pdf_only else "файлов")
            + ", только легальные OA-копии):")
        t0, st0 = time.monotonic(), dict(agent.client.stats)
        counts = agent.download(records, cfg["max_downloads"], args.oa_only, pdf_only)
        st1 = agent.client.stats
        log(f"\nИтог скачивания: {counts}")
        log(f"Скачивание заняло {time.monotonic() - t0:.0f} с (потоков: {agent.workers}; запросов: "
            f"{st1['requests'] - st0['requests']}, повторов: {st1['retries'] - st0['retries']})")
        if pdf_only:                                       # в результате остаются только статьи с PDF
            keep = [r for r in records if r.status in ("pdf", "exists")]
            ids = {id(r) for r in keep}
            dropped = [r for r in records if id(r) not in ids]
            extra["pdf_only"] = {"with_pdf": len(keep), "dropped": len(dropped),
                                 "no_pdf": counts["no_pdf"] + counts["failed"], "not_checked": sum(1 for r in dropped if not r.status),
                                 "dropped_titles": [f"{r.status or 'not_checked'}: {r.title[:100]}" for r in dropped][:200]}
            log(f"С доступным PDF: {len(keep)} из {len(records)} найденных (без PDF/ошибка: {extra['pdf_only']['no_pdf']}, "
                f"не проверялось: {extra['pdf_only']['not_checked']}; остальные отброшены как не-OA).")
            records = keep
    run_dir = agent.finish(mode, specs, records, {"downloads": counts, **extra})
    log(f"Библиотека: {agent.lib.root}  |  отчёт запуска: {run_dir}")
    return 0


# ------------------------------------------------------------------ manual
def cmd_manual(a) -> int:
    cfg = apply_overrides(load_config(a.config), a)
    common = dict(year_from=a.year_from, year_to=a.year_to, open_access=a.oa_only,
                  title_only=a.title_only, strict=a.strict, arxiv_cats=a.arxiv_cat or [])
    raw = {}
    for item in a.raw or []:
        k, _, v = item.partition("=")
        if not v:
            raise SystemExit(f"--raw ожидает ИСТОЧНИК=ЗАПРОС, получено: {item}")
        raw[k.strip()] = v.strip()
    topics = list(a.query or [])
    if a.topics_file:
        topics += [ln.strip() for ln in Path(a.topics_file).read_text(encoding="utf-8").splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
    extra_groups = [g for g in (parse_topic(x).groups[0] if parse_topic(x).groups else None for x in (a.group or [])) if g]
    extra_excl = [t for x in (a.exclude or []) for t in parse_topic("-" + x).exclude]
    specs: list[SearchSpec] = []
    for t in topics or ([""] if extra_groups else []):
        s = parse_topic(t, **common)
        s.groups += copy.deepcopy(extra_groups)
        s.exclude += extra_excl
        s.raw = dict(raw)
        if s.groups:
            specs.append(s)
    if not specs:
        raise SystemExit("Не задан запрос: используйте -q \"термин1; термин2|синоним; -исключить\", --group или --topics-file")
    agent = Agent(cfg, log)
    names = agent.check_sources(cfg["sources"])
    if a.dry_run:
        for q in agent.build_queries(specs, names):
            log(f"[{q['source']}] {q['spec']}\n    {q['query']}" + (f"\n    {q['extra']}" if q.get("extra") else ""))
        return 0
    if not need_email(cfg) or not names:
        return 2
    limit = a.limit or cfg["limit_per_source"]
    log(f"Ручной режим: {len(specs)} запрос(ов), источники: {', '.join(names)}, лимит {limit}/источник")
    records = agent.search(specs, names, limit)
    return finish_and_report(agent, "manual", specs, records, a, cfg)


# ------------------------------------------------------------------ auto
def make_llm(cfg: dict, a) -> LLM | None:
    if a.no_llm:
        return None
    c = cfg["llm"]
    llm = LLM(a.llm_provider or c["provider"], a.llm_url or c["base_url"] or None, a.model or c["model"],
              float(c["temperature"]), int(c["timeout"]), c.get("api_key") or None)
    if not llm.ping():
        log(f"⚠ Локальная LLM не отвечает ({llm.provider} @ {llm.base_url}). Запустите её (например `ollama serve`) "
            f"или используйте --no-llm. Продолжаю с эвристикой.")
        return None
    return llm


def cmd_auto(a) -> int:
    cfg = apply_overrides(load_config(a.config), a)
    goal = a.goal or (Path(a.goal_file).read_text(encoding="utf-8").strip() if a.goal_file else "")
    if not goal:
        raise SystemExit("Укажите цель: litagent auto \"описание темы\" или --goal-file")
    note_paths = list(a.notes or cfg["auto"].get("notes") or [])
    for p in note_paths:
        if not Path(p).exists():
            raise SystemExit(f"Файл заметок не найден: {p}")
    notes = Notes.load(*note_paths)
    log(f"Заметки: файлов {len(note_paths)}, систем в карточках {len(notes.systems)}, "
        f"общих пометок {len(notes.global_notes)}, AGENT-строк {len(notes.agent_lines)}")
    agent = Agent(cfg, log)
    names = agent.check_sources(cfg["sources"])
    defaults = {"year_from": a.year_from, "year_to": a.year_to, "open_access": a.oa_only,
                "title_only": a.title_only, "strict": a.strict, "limit": a.limit}
    llm = make_llm(cfg, a)
    if a.dry_run:
        try:
            plan = request_plan(llm, agent.policy, notes, goal, names, defaults) if llm else heuristic_plan(goal, names, defaults)
        except Exception as e:  # noqa: BLE001
            log(f"LLM не дала план ({e}); эвристика")
            plan = heuristic_plan(goal, names, defaults)
        log(f"План [{plan.source}]: {plan.rationale}\n{json.dumps(plan.raw_plan, ensure_ascii=False, indent=2)}")
        for n in agent.check_sources(plan.sources):
            log(f"[{n}] {agent.sources[n].query_for(plan.spec)}")
        return 0
    if not need_email(cfg) or not names:
        return 2
    au = cfg["auto"]
    records, specs, plan_log = auto_search(agent, llm, notes, goal, names, defaults,
                                           rounds=a.rounds or au["rounds"], min_results=a.min_results or au["min_results"], log=log)
    if llm and (a.llm_filter or au.get("llm_filter")) and records:
        thr = a.min_score if a.min_score is not None else au["min_score"]
        log(f"\nФильтр релевантности LLM (порог {thr}) для {len(records)} записей…")
        before = len(records)
        records = llm_filter(llm, goal, records, float(thr), log=log)
        log(f"  оставлено {len(records)} из {before}")
    return finish_and_report(agent, "auto", specs, records, a, cfg, {"goal": goal, "rounds": plan_log})


# ------------------------------------------------------------------ check / notes
def cmd_check(a) -> int:
    cfg = apply_overrides(load_config(a.config), a)
    if not need_email(cfg):
        return 2
    agent = Agent(cfg, log)
    spec = parse_topic("machine learning")
    bad = 0
    for n in agent.check_sources(cfg["sources"]):
        ok, why = agent.sources[n].available()
        if not ok:
            log(f"[{n:<16}] SKIP  {why}")
            continue
        try:
            recs, info = agent.sources[n].search(spec, 1)
            log(f"[{n:<16}] OK    найдено {info.get('found')}, первая: {recs[0].title[:60] if recs else '-'}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            log(f"[{n:<16}] FAIL  {type(e).__name__}: {str(e)[:140]}")
    llm = LLM(cfg["llm"]["provider"], cfg["llm"]["base_url"] or None, cfg["llm"]["model"])
    log(f"[LLM {llm.provider:<11}] {'OK' if llm.ping() else 'недоступна'} ({llm.base_url}, модель {llm.model})")
    return 1 if bad else 0


def cmd_notes(a) -> int:
    notes = Notes.load(*a.files)
    names = [a.source] if a.source else sorted(notes.systems)
    log(f"Систем в карточках: {len(notes.systems)} → {', '.join(sorted(notes.systems))}")
    for n in names:
        log(notes.for_sources([n], maxlen=300) or f"[{n}] нет данных")
        for w in notes.warnings(n):
            log(f"  ⚠ {w}")
    if notes.global_notes or notes.agent_lines:
        log("\nОбщие пометки:\n" + notes.global_text(300))
    return 0


# ------------------------------------------------------------------ свои сайты (sites)
def _probe_factory(cfg: dict):
    """Пробный запрос для опросника: (json | None, ошибка | '')."""
    def probe(site):
        if not cfg.get("email"):
            return None, "нужен e-mail (--email, litagent.toml или LITAGENT_EMAIL)"
        policy = Policy()
        policy.intervals[site.host] = max(policy.intervals.get(site.host, 0.0), float(site.min_interval))
        client = PoliteClient(policy, f"litagent/1.0 (mailto:{cfg['email']})", cache_dir=None, retries=2, log=log)
        src = CustomSource(Ctx(client, cfg["email"], cfg["keys"], "litagent", log), site)
        ok, why = src.available()
        if not ok:
            return None, why
        try:
            return src.fetch_json(parse_topic("machine learning"), 3), ""
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:200]}"
    return probe


def _pick_site(reg: SiteRegistry, name: str | None):
    if not name:
        if not reg.sites:
            log("Список пуст.")
            return None
        log("Сайты: " + ", ".join(reg.names()))
        name = ask("Имя сайта: ").strip()
    site = reg.get(name)
    if not site:
        log(f"Нет сайта «{name}». Список: python -m litagent sites list")
    return site


def cmd_sites(a) -> int:
    cfg = apply_overrides(load_config(a.config), a)
    path = a.sites_file or cfg.get("custom_sites") or DEFAULT_FILE
    reg = SiteRegistry.load(path, log)
    act = a.action
    if act == "list":
        if reg.error:
            return 2
        log(f"Файл: {reg.path}" + ("" if reg.path.exists() else "  (ещё не создан: появится при первом `sites add`)"))
        log(f"Встроенные: {', '.join(DEFAULT_SOURCES)}")
        if reg.sites:
            log("Свои:")
            for s in reg.sites:
                log("  " + s.human_line())
        else:
            log("Свои сайты не добавлены: python -m litagent sites add")
        return 0
    if act in ("add", "edit", "remove") and reg.error:
        log("Файл со своими сайтами повреждён — не буду его перезаписывать. Исправьте или удалите его вручную.")
        return 2
    if act == "add":
        site = interview(reg, None, ask, log, _probe_factory(cfg), Policy())
    elif act == "edit":
        old = _pick_site(reg, a.name)
        if not old:
            return 1
        site = interview(reg, old, ask, log, _probe_factory(cfg), Policy())
    elif act == "remove":
        old = _pick_site(reg, a.name)
        if not old:
            return 1
        if not a.yes and ask(f"Удалить «{old.name}» из списка? (да/нет): ").strip().lower() not in ("д", "да", "y", "yes"):
            log("Отменено.")
            return 0
        reg.remove(old.name)
        log(f"Удалено. Файл: {reg.save()}")
        return 0
    elif act == "import":
        return _import_sites(reg, a, cfg)
    elif act == "show":
        old = _pick_site(reg, a.name)
        if not old:
            return 1
        log(json.dumps(old.to_dict(), ensure_ascii=False, indent=2))
        log("\nТо, что видит нейросеть в авто-режиме:\n" + old.llm_details())
        return 0
    elif act == "test":
        old = _pick_site(reg, a.name)
        if not old:
            return 1
        return _test_site(old, cfg, a.query or "machine learning")
    else:
        return 2
    if site is None:
        return 1
    reg.put(site)
    log(f"\nСохранено: {reg.save()}")
    log(f"Проверить:  python -m litagent sites test {site.name}\n"
        f"Ручной режим: python -m litagent manual -q \"тема\" --sources {site.name}"
        + ("" if site.enabled else "   (сайт выключен по умолчанию)"))
    return 0


def _import_sites(reg: SiteRegistry, a, cfg: dict) -> int:
    if reg.error:
        log("Файл со своими сайтами повреждён — не буду его перезаписывать. Исправьте или удалите его вручную.")
        return 2
    if not a.name:
        log("Укажите адрес или файл: python -m litagent sites import https://example.org/sites.json   (help import)")
        return 2
    policy = Policy()
    ua = f"litagent/1.0 (mailto:{cfg['email']})" if cfg.get("email") else "litagent/1.0"
    client = PoliteClient(policy, ua, cache_dir=None, retries=1, log=log)       # один запрос, без кэша и лишних повторов
    try:
        items = parse_sites(fetch_text(a.name, client, policy))
    except Exception as e:  # noqa: BLE001
        log(f"Не удалось загрузить: {type(e).__name__}: {str(e)[:300]}")
        return 1
    origin = a.name if is_url(a.name) else Path(a.name).name
    todo, seen = [], set()
    log(f"Найдено сайтов в файле: {len(items)}")
    for raw in items:
        site, note = sanitize(raw, origin, reg, policy, enable=a.enable, overwrite=a.overwrite)
        if site is None or site.name in seen:
            log(f"  ✗ {raw.get('name', '?')}: {note or 'повтор имени в файле'}")
            continue
        seen.add(site.name)
        todo.append(site)
        log(f"  {len(todo):>2}) {site.name:<14} {site.host}  [{'обновит' if reg.get(site.name) else 'новый'}, "
            f"{'включён' if site.enabled else 'выключен'}]" + (f"  ({note})" if note else ""))
    if not todo:
        log("Добавлять нечего.")
        return 1
    yes = lambda t: t.strip().lower() in ("д", "да", "y", "yes")          # noqa: E731
    if a.only:                                                           # заранее заданный выбор по именам
        todo, bad = pick(a.only.replace(",", " "), todo)
        if bad:
            log(f"Нет в списке: {', '.join(bad)}")
        if not todo:
            log("Из заданных --only добавлять нечего.")
            return 1
        if not a.yes and not yes(ask(f"Добавить {', '.join(s.name for s in todo)}? (да/нет): ")):
            log("Отменено, ничего не записано.")
            return 0
    elif not a.yes:                                                      # интерактивный выбор: все или конкретные
        for _ in range(3):
            try:
                ans = ask("Что добавить? Enter — все; номера/имена через запятую (1,3 или 2-4); «нет» — отмена: ")
            except EOFError:
                ans = "нет"
            chosen, bad = pick(ans, todo)
            if bad:
                log(f"  Не понял: {', '.join(bad)}. Повторите.")
                continue
            todo = chosen
            break
        else:
            todo = []
        if not todo:
            log("Отменено, ничего не записано.")
            return 0
        log(f"Выбрано: {', '.join(s.name for s in todo)}")
    for s in todo:
        reg.put(s)
    log(f"Сохранено: {reg.save()}. Проверить: python -m litagent sites test <имя>; включить: sites edit <имя>.")
    return 0


def cmd_help(a) -> int:
    log(render_help(a.topic))
    return 0


def _test_site(site, cfg: dict, query: str) -> int:
    if not need_email(cfg):
        return 2
    errs = site.validate(Policy())
    if errs:
        log("Описание сайта некорректно:\n" + "\n".join(f"  ✗ {e}" for e in errs))
        return 1
    policy = Policy()
    policy.intervals[site.host] = max(policy.intervals.get(site.host, 0.0), float(site.min_interval))
    client = PoliteClient(policy, f"litagent/1.0 (mailto:{cfg['email']})", cache_dir=None, log=log)
    src = CustomSource(Ctx(client, cfg["email"], cfg["keys"], "litagent", log), site)
    ok, why = src.available()
    if not ok:
        log(f"[{site.name}] пропуск: {why}")
        return 2
    spec = parse_topic(query)
    log(f"[{site.name}] запрос: {src.query_for(spec)}")
    try:
        recs, info = src.search(spec, 3)
    except Exception as e:  # noqa: BLE001
        log(f"[{site.name}] FAIL {type(e).__name__}: {str(e)[:300]}")
        return 1
    log(f"[{site.name}] OK найдено {info.get('found')}, взято {len(recs)}")
    for i, r in enumerate(recs, 1):
        log(f"  {i}. {r.short()} | doi={r.doi or '-'} | pdf={'да' if r.pdf_urls else 'нет'}")
    for k in ("authors", "year", "doi", "abstract", "pdf_urls"):
        if recs and not any(getattr(r, k) for r in recs):
            log(f"  ⚠ поле «{k}» пустое у всех записей — проверьте путь (`sites edit {site.name}`)")
    return 0 if recs else 1


# ------------------------------------------------------------------ очистка папки со статьями
def _mb(n: int) -> str:
    return f"{n / 1048576:.1f} МБ"


def cmd_clean(a) -> int:
    cfg = load_config(a.config)
    if a.out:
        cfg["out"] = a.out
    root = Path(cfg["out"])
    if not root.is_dir() or not any((root / x).exists() for x in ("pdf", "xml", "index.json")):
        log(f"В {root.resolve()} нет библиотеки litagent (нет pdf/, xml/, index.json) — ничего не делаю.")
        return 0
    lib = Library(root)
    st = lib.stats()
    runs, cache = a.runs or a.all, a.cache or a.all
    log(f"Библиотека: {root.resolve()}")
    log(f"  статьи: {st['pdf']} PDF + {st['xml']} XML ({_mb(st['articles_bytes'])}), в индексе записей: {st['indexed']}")
    log(f"  отчёты запусков: {st['runs']} ({_mb(st['runs_bytes'])}){' — будут удалены' if runs else ' — останутся'}")
    log(f"  HTTP-кэш: {st['cache']} файлов ({_mb(st['cache_bytes'])}){' — будет удалён' if cache else ' — останется'}")
    todo = st["pdf"] + st["xml"] + (st["runs"] if runs else 0) + (st["cache"] if cache else 0)
    if not todo:
        log("Удалять нечего.")
        return 0
    if a.dry_run:
        log("--dry-run: ничего не удалено.")
        return 0
    if not a.yes:
        try:
            ok = ask("Удалить? Это необратимо. Введите «да»: ").strip().lower() in ("д", "да", "y", "yes")
        except EOFError:
            ok = False
        if not ok:
            log("Отменено.")
            return 0
    res = lib.clean(runs=runs, cache=cache)
    log(f"Удалено: {res['pdf']} PDF, {res['xml']} XML" + (f", {res['runs']} отчётов" if runs else "")
        + (f", {res['cache']} файлов кэша" if cache else "") + f" ({_mb(res['bytes'])}). Индекс обновлён.")
    for e in res["errors"][:5]:
        log(f"  не удалось: {e}")
    return 1 if res["errors"] else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="litagent", description="Агент поиска и легального скачивания научных статей")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manual", help="ручной режим: запросы задаёте вы")
    add_common(m)
    m.add_argument("-q", "--query", action="append", help="тема: 'термин1; термин2|синоним; -исключить' (можно несколько -q)")
    m.add_argument("--topics-file", help="файл: одна тема на строку, # — комментарий")
    m.add_argument("--group", action="append", help="доп. OR-группа 'a|b' (добавляется ко всем темам)")
    m.add_argument("--exclude", action="append", help="доп. исключаемые термины 'a|b'")
    m.add_argument("--raw", action="append", metavar="ИСТОЧНИК=ЗАПРОС", help="готовый запрос в родном синтаксисе источника")
    m.add_argument("--arxiv-cat", action="append", help="категория arXiv, например cs.CL")
    m.set_defaults(fn=cmd_manual)
    au = sub.add_parser("auto", help="авто-режим: запрос формирует локальная LLM по цели и заметкам")
    add_common(au)
    au.add_argument("goal", nargs="?", help="цель на естественном языке")
    au.add_argument("--goal-file")
    au.add_argument("--notes", action="append", help=".tex с аналитикой (или .md/.txt со своими пометками); можно несколько")
    au.add_argument("--no-llm", action="store_true", help="не использовать LLM (эвристика)")
    au.add_argument("--llm-provider", choices=["ollama", "openai"])
    au.add_argument("--llm-url")
    au.add_argument("--model")
    au.add_argument("--rounds", type=int, help="максимум раундов уточнения плана")
    au.add_argument("--min-results", type=int)
    au.add_argument("--llm-filter", action="store_true", help="отфильтровать результаты оценкой релевантности LLM")
    au.add_argument("--min-score", type=float)
    au.set_defaults(fn=cmd_auto)
    c = sub.add_parser("check", help="проверить доступность источников и LLM из вашей сети")
    add_common(c)
    c.set_defaults(fn=cmd_check)
    n = sub.add_parser("notes", help="показать, что агент извлёк из .tex/.md (то, что видит LLM)")
    n.add_argument("files", nargs="+")
    n.add_argument("--source")
    n.set_defaults(fn=cmd_notes)
    st = sub.add_parser("sites", help="свои сайты для поиска: list | add | edit | remove | show | test | import (через вопросы-ответы)")
    st.add_argument("action", choices=["list", "add", "edit", "remove", "show", "test", "import"])
    st.add_argument("name", nargs="?", help="имя сайта (edit/remove/show/test) или адрес/файл списка (import)")
    st.add_argument("--config")
    st.add_argument("--sites-file", help="файл со своими сайтами (по умолчанию ./custom_sites.json)")
    st.add_argument("--email", help="e-mail для пробных запросов (опросник и test)")
    st.add_argument("--query", help="для test: тема в мини-языке, по умолчанию 'machine learning'")
    st.add_argument("-y", "--yes", action="store_true", help="remove/import: без подтверждения")
    st.add_argument("--overwrite", action="store_true", help="import: перезаписывать сайты с тем же именем")
    st.add_argument("--only", help="import: добавить только эти сайты (имена или номера через запятую); без --only спросит интерактивно")
    st.add_argument("--enable", action="store_true", help="import: сразу включить сайты (по умолчанию приходят выключенными)")
    st.set_defaults(fn=cmd_sites)
    hp = sub.add_parser("help", help="справка по темам: start, manual, auto, sites, import, pmc, pdf, speed, clean, llm, files, limits")
    hp.add_argument("topic", nargs="?")
    hp.set_defaults(fn=cmd_help)
    cl = sub.add_parser("clean", help="очистить папку со скачанными статьями (pdf/, xml/) и привести индекс в порядок")
    cl.add_argument("--config")
    cl.add_argument("--out", help="папка библиотеки")
    cl.add_argument("--runs", action="store_true", help="удалить ещё и отчёты запусков (runs/)")
    cl.add_argument("--cache", action="store_true", help="удалить ещё и HTTP-кэш")
    cl.add_argument("--all", action="store_true", help="статьи + отчёты + кэш")
    cl.add_argument("--dry-run", action="store_true", help="только показать, что будет удалено")
    cl.add_argument("-y", "--yes", action="store_true", help="без вопроса о подтверждении")
    cl.set_defaults(fn=cmd_clean)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        log("\nПрервано. Скачанное сохранено; повторный запуск пропустит уже скачанное.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
