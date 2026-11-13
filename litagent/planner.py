"""Авто-режим: цель на естественном языке -> JSON-план (локальная LLM, с учётом заметок из файла)
-> валидация по политикам -> поиск -> (опционально) уточнение плана по результатам -> фильтр релевантности."""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field

from .llm import LLM, LLMError
from .models import Record, SearchSpec, _clean_term, current_year, dedupe, parse_topic
from .notes import Notes
from .policy import Policy

SCHEMA = """{
  "rationale": "one short sentence",
  "sources": ["openalex", "pubmed"],
  "groups": [["term", "synonym or abbreviation"], ["second concept"]],
  "exclude": ["term to exclude"],
  "year_from": 2020,
  "year_to": null,
  "open_access_only": false,
  "title_only": false,
  "arxiv_categories": [],
  "raw": {},
  "max_per_source": 50
}"""


@dataclass
class Plan:
    spec: SearchSpec
    sources: list
    limit: int
    rationale: str = ""
    source: str = "llm"          # llm | heuristic
    raw_plan: dict = field(default_factory=dict)


# ------------------------------------------------------------------ промпты
def build_system(policy: Policy, notes: Notes, available: list[str]) -> str:
    about = "\n".join(f"- {n}: {policy.sources_info.get(n, {}).get('about', '')}" for n in available)
    hints = "\n".join(f"- {n}: {policy.sources_info.get(n, {}).get('hint', '')}" for n in available)
    file_notes = notes.for_sources(available)
    parts = [
        "You are the query-planning component of a scientific-literature download agent.",
        "Reply with ONE JSON object and nothing else. No markdown, no comments.",
        "",
        "JSON schema (all keys optional except groups):",
        SCHEMA,
        "",
        "Planning rules:",
        "- Translate the goal to English keywords (databases index English). Keep proper names, gene names, acronyms.",
        "- 2-4 groups. Groups are combined with AND; terms inside a group are OR (synonyms, abbreviations).",
        "- Do NOT add quotes, field tags or boolean operators inside terms; multiword terms are treated as phrases.",
        "- Avoid generic words (study, analysis, method, review) unless the user asked for them.",
        "- Pick only sources that fit the topic: biomedicine -> pubmed, europepmc; CS/physics/math preprints -> arxiv; any field -> openalex, semanticscholar, crossref; open-access full text -> core, doaj.",
        "- Sources marked USER-ADDED were configured by the user: choose them by the same logic, when their description fits the topic, and follow their request format exactly.",
        "- Fill 'raw' only if you are certain of a source's native syntax; otherwise leave it empty.",
        "- Use year_from/year_to only if the goal implies a period.",
        "",
        "HARD RULES (never violate):",
        *[f"- {r}" for r in policy.rules],
        "",
        "AVAILABLE SOURCES:", about,
        "",
        "SYNTAX HINTS:", hints,
    ]
    if len(available) == 1:
        parts += ["", f"NOTE: the user restricted this search to ONE source ({available[0]}). "
                      "Use only it and write every query strictly in its syntax."]
    custom = [policy.sources_info[n]["details"] for n in available if policy.sources_info.get(n, {}).get("custom")]
    if custom:
        parts += ["", "USER-ADDED SOURCES (full request format as configured by the user):", *custom]
    if file_notes:
        parts += ["", "NOTES FROM THE USER'S ANALYTICAL FILE (syntax, pitfalls, limits, blocking behaviour):", file_notes]
    g = notes.global_text()
    if g:
        parts += ["", "GLOBAL NOTES FROM THE FILE:", g]
    u = notes.user_notes()
    if u:
        parts += ["", "USER NOTES (highest priority):", u]
    return "\n".join(parts)


# ------------------------------------------------------------------ валидация
def _strs(x, max_items: int, max_len: int = 80) -> list[str]:
    out = []
    for t in (x if isinstance(x, list) else []):
        if isinstance(t, str):
            t = _clean_term(t)
            if t and len(t) <= max_len and t.lower() not in {o.lower() for o in out}:
                out.append(t)
        if len(out) >= max_items:
            break
    return out


def _year(v):
    try:
        y = int(v)
    except (TypeError, ValueError):
        return None
    return y if 1900 <= y <= current_year() + 1 else None


def plan_to_spec(plan: dict, available: list[str], defaults: dict, source: str = "llm") -> Plan:
    groups = []
    for g in (plan.get("groups") or [])[:6]:
        terms = _strs([g] if isinstance(g, str) else g, 8)
        if terms:
            groups.append(terms)
    if not groups:
        raise ValueError("в плане нет непустого 'groups'")
    srcs = [s for s in (plan.get("sources") or []) if isinstance(s, str) and s in available] or list(available)
    raw = {}
    if isinstance(plan.get("raw"), dict):
        raw = {k: v.strip()[:600] for k, v in plan["raw"].items()
               if k in available and isinstance(v, str) and v.strip()}
    d = defaults
    yf, yt = _year(plan.get("year_from")), _year(plan.get("year_to"))
    if yf and yt and yf > yt:
        yf, yt = yt, yf
    try:
        limit = max(1, min(int(plan.get("max_per_source") or d.get("limit") or 50), 500))
    except (TypeError, ValueError):
        limit = d.get("limit") or 50
    if d.get("limit"):
        limit = d["limit"]                         # явный выбор пользователя приоритетнее
    spec = SearchSpec(
        groups=groups, exclude=_strs(plan.get("exclude"), 6),
        year_from=d.get("year_from") or yf, year_to=d.get("year_to") or yt,
        open_access=bool(d.get("open_access") or plan.get("open_access_only")),
        title_only=bool(d.get("title_only") or plan.get("title_only")),
        strict=bool(d.get("strict")), arxiv_cats=_strs(plan.get("arxiv_categories"), 5, 20), raw=raw,
        label=str(plan.get("rationale") or "")[:80])
    return Plan(spec, srcs, limit, str(plan.get("rationale") or ""), source, plan)


# ------------------------------------------------------------------ эвристика без LLM
_STOP = set("""и в во на по о об от для при про с со к из за как что это только найти найди найдите скачать скачай
статьи статья статей научные научных работы работа работ последние последних лет год года годы про при the a an of in on
for and or to with about from by using use based study studies paper papers article articles find download recent latest
only""".split())


def heuristic_plan(goal: str, available: list[str], defaults: dict) -> Plan:
    """Без LLM: понимает мини-язык 'a; b|c; -x' и простые фразы ('последние 5 лет', 'открытый доступ')."""
    yf = yt = None
    oa = False
    period = re.compile(r"(?:за\s+)?(?:последн\w*|last|past)\s+(\d{1,2})\s+(?:лет|года|год|years?)", re.I)
    m = period.search(goal)
    if m:
        yf = current_year() - int(m.group(1))
        goal = period.sub(" ", goal)
    oa_re = re.compile(r"(?:только\s+)?открыт\w*\s+доступ\w*|open[- ]access", re.I)
    if oa_re.search(goal):
        oa = True
        goal = oa_re.sub(" ", goal)
    years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", goal)]
    if years and not yf:
        yf = min(years)
        yt = max(years) if len(years) > 1 else None
    # мини-язык включается только явными ';' или '|' (запятая в обычной фразе — не разделитель)
    spec_obj = parse_topic(goal) if re.search(r"[;|]", goal) else None
    groups = spec_obj.groups if spec_obj else []
    exclude = spec_obj.exclude if spec_obj else []
    note = "эвристика без LLM (без перевода и синонимов)"
    if not groups:
        toks = [w for w in re.findall(r"[\w\-]{3,}", goal)
                if w.lower() not in _STOP and not re.fullmatch(r"(?:19|20)\d{2}", w)]
        groups = [[w] for w in toks[:4]]
        if any(re.search("[А-Яа-яЁё]", w) for w in toks):
            note += "; кириллица не переводится — для англоязычных баз запустите LLM или задайте термины по-английски"
    if not groups:
        raise ValueError("не удалось выделить ключевые слова из цели")
    plan = {"groups": groups, "exclude": exclude, "year_from": yf, "year_to": yt, "open_access_only": oa,
            "sources": available, "rationale": note}
    return plan_to_spec(plan, available, defaults, source="heuristic")


# ------------------------------------------------------------------ запросы к LLM
def request_plan(llm: LLM, policy: Policy, notes: Notes, goal: str, available: list[str], defaults: dict) -> Plan:
    system = build_system(policy, notes, available)
    user = f"Today: {datetime.date.today().isoformat()}\nResearch goal: {goal}"
    return plan_to_spec(llm.chat_json(system, user), available, defaults)


def summarize_results(records: list[Record], runlog: list[dict], n_titles: int = 12) -> str:
    lines = []
    for e in runlog:
        if "error" in e or "skipped" in e:
            lines.append(f"- {e['source']}: {e.get('error') or 'skipped: ' + e.get('skipped', '')}"[:160])
        else:
            tr = f"; translation: {e['translation']}" if e.get("translation") else ""
            lines.append(f"- {e['source']}: total hits {e.get('found')}, fetched {e.get('fetched')}, kept {e.get('kept')}{tr}")
    lines.append(f"Unique records: {len(records)}")
    lines.append("Sample titles:")
    for r in records[:n_titles]:
        lines.append(f"  * ({r.year or '?'}) {r.title[:110]}")
    return "\n".join(lines)


def review_plan(llm: LLM, policy: Policy, notes: Notes, goal: str, plan: Plan, records: list[Record],
                runlog: list[dict], available: list[str], defaults: dict, min_results: int):
    """Возвращает (verdict, reason, новый Plan | None)."""
    system = (build_system(policy, notes, available) +
              "\n\nNOW YOU REVIEW SEARCH RESULTS. Reply with JSON: "
              '{"verdict": "ok" | "too_few" | "too_broad" | "off_topic", "reason": "...", "plan": {…schema above…}}. '
              "Include 'plan' only when verdict is not ok: a revised plan (more synonyms / fewer groups for too_few; "
              "more specific groups or exclude terms for too_broad / off_topic). "
              f"At least {min_results} on-topic unique records are wanted.")
    user = (f"Goal: {goal}\nCurrent plan: {json.dumps(plan.raw_plan or {'groups': plan.spec.groups, 'exclude': plan.spec.exclude}, ensure_ascii=False)}\n\n"
            f"Results:\n{summarize_results(records, runlog)}")
    d = llm.chat_json(system, user)
    verdict = str(d.get("verdict", "ok")).lower()
    new = None
    if verdict != "ok" and isinstance(d.get("plan"), dict):
        try:
            new = plan_to_spec(d["plan"], available, defaults)
        except ValueError:
            new = None
    return verdict, str(d.get("reason", ""))[:300], new


def llm_filter(llm: LLM, goal: str, records: list[Record], min_score: float, batch: int = 6, log=print) -> list[Record]:
    system = ("You grade how relevant scientific papers are to a research goal. "
              'Reply with JSON: {"scores": [{"n": 1, "score": 0-10}, ...]}. 10 = exactly on topic, 0 = unrelated. '
              "Judge by title and abstract only.")
    for i in range(0, len(records), batch):
        chunk = records[i:i + batch]
        items = "\n".join(f"{k + 1}. ({r.year or '?'}) {r.title}\n   {(r.abstract or '')[:500]}" for k, r in enumerate(chunk))
        try:
            d = llm.chat_json(system, f"Goal: {goal}\n\nPapers:\n{items}")
        except LLMError as e:
            log(f"  фильтр: пакет {i // batch + 1} пропущен ({str(e)[:80]})")
            continue
        for s in d.get("scores", []) if isinstance(d, dict) else []:
            try:
                n, sc = int(s["n"]), float(s["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= n <= len(chunk):
                chunk[n - 1].score = max(0.0, min(10.0, sc))
    return [r for r in records if r.score is None or r.score >= min_score]


# ------------------------------------------------------------------ главный цикл авто-режима
def auto_search(agent, llm: LLM | None, notes: Notes, goal: str, available: list[str], defaults: dict,
                rounds: int = 2, min_results: int = 10, log=print):
    """Возвращает (records, specs, plan_log)."""
    plan_log, specs, allrecs = [], [], []
    plan = None
    if llm is not None:
        try:
            plan = request_plan(llm, agent.policy, notes, goal, available, defaults)
        except (LLMError, ValueError) as e:
            log(f"LLM не дала план ({str(e)[:120]}); использую эвристику")
            llm = None
    if plan is None:
        plan = heuristic_plan(goal, available, defaults)
    for rnd in range(1, max(1, rounds) + 1):
        log(f"\nРаунд {rnd}: [{plan.source}] {plan.rationale}\n  запрос: {plan.spec.describe()}\n  источники: {', '.join(plan.sources)}")
        names = agent.check_sources(plan.sources)
        for n in names:
            for w in notes.warnings(n):
                log(f"  ⚠ [{n}] {w}")
        start = len(agent.runlog)
        recs = agent.search([plan.spec], names, plan.limit)
        round_log = agent.runlog[start:]
        for e in round_log:
            e["round"] = rnd
        specs.append(plan.spec)
        allrecs = dedupe(allrecs + recs)
        plan_log.append({"round": rnd, "source": plan.source, "plan": plan.raw_plan,
                         "spec": plan.spec.describe(), "unique_after": len(allrecs)})
        if rnd >= rounds or llm is None:
            break
        try:
            verdict, reason, new = review_plan(llm, agent.policy, notes, goal, plan, allrecs, round_log,
                                               available, defaults, min_results)
        except LLMError as e:
            log(f"  ревью пропущено: {str(e)[:100]}")
            break
        plan_log[-1].update(verdict=verdict, reason=reason)
        log(f"  ревью LLM: {verdict} — {reason}")
        if verdict == "ok" and len(allrecs) >= min_results:
            break
        if new is None:
            break
        plan = new
    return allrecs, specs, plan_log
