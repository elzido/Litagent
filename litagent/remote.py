"""Импорт списка сайтов из интернета (или локального файла) — экономно и безопасно.

* ОДИН GET, не более MAX_BYTES, без повторов на ошибках парсинга, паузы/блокировки — через PoliteClient;
* файл из сети — недоверенный: ключи из него вырезаются (иначе чужой файл мог бы заставить отправить ваш секрет
  на чужой хост), тексты чистятся и режутся (они попадают в промпт LLM), сайты приходят выключенными;
* в реестр ничего не пишется без вашего подтверждения (это делает cli).
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from .customsites import LOCAL_HOSTS, Site, SiteRegistry, name_error
from .http import HttpError, PoliteClient
from .policy import Policy

MAX_BYTES = 256 * 1024
MAX_SITES = 50
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")
_LIMITS = {"title": 80, "about": 500, "notes": 1000}


def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def fetch_text(source: str, client: PoliteClient, policy: Policy, max_bytes: int = MAX_BYTES) -> str:
    if not is_url(source):
        p = Path(source)
        if not p.is_file():
            raise HttpError(f"нет такого файла: {source}")
        if p.stat().st_size > max_bytes:
            raise HttpError(f"файл больше {max_bytes // 1024} КБ")
        return p.read_text(encoding="utf-8")
    u = urlparse(source)
    if u.scheme == "http" and (u.hostname or "") not in LOCAL_HOSTS:
        raise HttpError("нужен https (http допустим только для localhost)")
    ok, why = policy.url_allowed(source)
    if not ok:
        raise HttpError(why)
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d, "sites.json")
        r = client.download(source, dest, max_bytes=max_bytes)
        if r.status != 200:
            raise HttpError(f"HTTP {r.status}")
        return dest.read_text(encoding="utf-8", errors="replace")


def parse_sites(text: str) -> list[dict]:
    try:
        data = json.loads(text)
    except ValueError as e:
        hint = " Это HTML-страница? Для GitHub берите ссылку «Raw»." if text.lstrip()[:1] == "<" else ""
        raise HttpError(f"это не JSON ({e}).{hint}") from e
    items = data.get("sites") if isinstance(data, dict) else data
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        raise HttpError("ожидался JSON вида {\"sites\": [ {...}, ... ]}")
    return items[:MAX_SITES]


def sanitize(raw: dict, origin: str, reg: SiteRegistry, policy: Policy, enable: bool = False,
             overwrite: bool = False) -> tuple[Site | None, str]:
    """-> (сайт | None, комментарий для пользователя)."""
    site = Site.from_dict(raw)
    notes = []
    for k, n in _LIMITS.items():
        v = _CTRL.sub(" ", getattr(site, k) or "").strip()
        setattr(site, k, v[:n])
    if site.key_env or site.key_param or site.key_header:
        site.key_env = site.key_param = site.key_header = site.key_prefix = ""
        notes.append("ключ вырезан — если нужен, настройте `sites edit`")
    site.min_interval = max(float(site.min_interval or 1.0), 1.0)
    site.max_per_request = min(int(site.max_per_request or 100), 200)
    site.enabled = bool(enable)
    site.origin = origin
    err = name_error(site.name, reg, policy, editing=overwrite)
    if err:
        return None, f"пропущен: имя «{site.name}» — {err}"
    errs = site.validate(policy)
    if errs:
        return None, f"пропущен: {errs[0]}"
    return site, "; ".join(notes)


_ALL = {"", "все", "всё", "all", "*", "да", "д", "y", "yes"}
_NONE = {"нет", "н", "n", "no", "0"}


def pick(answer: str, sites: list[Site]) -> tuple[list[Site], list[str]]:
    """Разбор выбора: пусто/«все» — все; «нет»/0 — ничего; иначе номера (1,3), диапазоны (2-4) и/или имена.
    -> (выбранные в порядке списка, нераспознанные токены)."""
    a = answer.strip().lower()
    if a in _ALL:
        return list(sites), []
    if a in _NONE:
        return [], []
    by_name = {s.name: i for i, s in enumerate(sites)}
    idx, bad = set(), []
    for tok in re.split(r"[,\s;]+", a):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        nums = range(int(m.group(1)), int(m.group(2)) + 1) if m else ([int(tok)] if tok.isdigit() else None)
        if nums is not None:
            for n in nums:
                (idx.add(n - 1) if 1 <= n <= len(sites) else bad.append(str(n)))
        elif tok in by_name:
            idx.add(by_name[tok])
        else:
            bad.append(tok)
    return [sites[i] for i in sorted(idx)], bad
