"""Пресеты ручного режима: именованные наборы параметров поиска в presets.json.

Сохраняется всё, что описывает сам поиск (темы, группы, исключения, источники, годы, лимиты, режимы скачивания);
e-mail, ключи и пути (--out, --config, --sites-file) в пресет не попадают.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shlex
from pathlib import Path

DEFAULT_FILE = "presets.json"
NAME_RE = re.compile(r"^[\w.-]{1,40}$", re.U)
LIST_KEYS = ("query", "group", "exclude", "raw", "arxiv_cat")
FLAG_KEYS = ("oa_only", "title_only", "strict", "no_download", "pdf_only", "no_pmc", "flat", "supplements")
VALUE_KEYS = ("topics_file", "sources", "site", "limit", "year_from", "year_to", "workers",
              "max_downloads", "count", "deepen_rounds")
INT_KEYS = {"limit", "year_from", "year_to", "workers", "max_downloads", "count", "deepen_rounds"}
FLAGS_CLI = {"oa_only": "--oa-only", "title_only": "--title-only", "strict": "--strict", "no_download": "--no-download",
             "pdf_only": "--pdf-only", "no_pmc": "--no-pmc", "flat": "--flat", "supplements": "--supplements"}
LISTS_CLI = {"query": "-q", "group": "--group", "exclude": "--exclude", "raw": "--raw", "arxiv_cat": "--arxiv-cat"}
VALUES_CLI = {"topics_file": "--topics-file", "sources": "--sources", "site": "--site", "limit": "--limit",
              "year_from": "--year-from", "year_to": "--year-to", "workers": "--workers",
              "max_downloads": "--max-downloads", "count": "--count", "deepen_rounds": "--deepen-rounds"}


def check_name(name: str) -> str:
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise SystemExit(f"Недопустимое имя пресета «{name}»: буквы, цифры, «_», «-», «.», до 40 символов.")
    return name


def load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise SystemExit(f"Файл пресетов {p} не читается: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("presets", {}), dict):
        raise SystemExit(f"Файл пресетов {p}: ожидается объект {{\"presets\": {{имя: {{...}}}}}}.")
    return data.get("presets", {})


def _write(path: str | Path, presets: dict) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"presets": presets}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def clean_args(args: dict) -> dict:
    """Оставляет только известные ключи с правильными типами (файл можно править руками)."""
    out = {}
    for k in LIST_KEYS:
        v = args.get(k)
        if isinstance(v, list) and v:
            out[k] = [str(x) for x in v if str(x).strip()]
    for k in FLAG_KEYS:
        if args.get(k) is True:
            out[k] = True
    for k in VALUE_KEYS:
        v = args.get(k)
        if v in (None, ""):
            continue
        if k in INT_KEYS:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                raise SystemExit(f"В пресете ключ «{k}» должен быть числом, а не {v!r}.") from None
        else:
            out[k] = str(v)
    return out


def get(path: str | Path, name: str) -> dict:
    presets = load(path)
    if name not in presets:
        have = ", ".join(sorted(presets)) or "—"
        raise SystemExit(f"Пресет «{name}» не найден в {path}. Есть: {have}. Сохранить: manual ... --save-preset {name}")
    entry = presets[name]
    return clean_args(entry.get("args", {}) if isinstance(entry, dict) else {})


def snapshot(a) -> dict:
    """Параметры текущего вызова manual, которые стоит сохранить."""
    args = {k: list(getattr(a, k)) for k in LIST_KEYS if getattr(a, k, None)}
    args.update({k: True for k in FLAG_KEYS if getattr(a, k, False)})
    args.update({k: getattr(a, k) for k in VALUE_KEYS if getattr(a, k, None) not in (None, "")})
    return clean_args(args)


def save(path: str | Path, name: str, args: dict, desc: str = "") -> bool:
    """Записывает пресет; True, если перезаписан существующий."""
    name = check_name(name)
    if not args.get("query") and not args.get("topics_file") and not args.get("group"):
        raise SystemExit("В пресете нет темы: добавьте -q, --topics-file или --group.")
    presets = load(path)
    existed = name in presets
    presets[name] = {"description": desc.strip(), "saved": datetime.datetime.now().isoformat(timespec="seconds"),
                     "args": clean_args(args)}
    _write(path, presets)
    return existed


def remove(path: str | Path, name: str) -> None:
    presets = load(path)
    if name not in presets:
        raise SystemExit(f"Пресет «{name}» не найден.")
    del presets[name]
    _write(path, presets)


def apply(a, args: dict):
    """Подмешивает пресет в аргументы CLI. Явные значения в командной строке главнее:
    списки (-q, --group, …) дополняются (пресет, затем CLI), флаги включаются, значения берутся из пресета, только если не заданы в CLI.
    Выбор источников и числа статей из CLI заменяет соответствующие настройки пресета целиком."""
    own_sources = bool(getattr(a, "sources", None) or getattr(a, "site", None) or getattr(a, "pick_site", False))
    own_count = getattr(a, "count", None) is not None or getattr(a, "max_downloads", None) is not None
    for k in LIST_KEYS:
        if args.get(k):
            setattr(a, k, list(args[k]) + list(getattr(a, k, None) or []))
    for k in FLAG_KEYS:
        if args.get(k):
            setattr(a, k, True)
    for k in VALUE_KEYS:
        if k not in args or getattr(a, k, None) is not None:
            continue
        if (k in ("sources", "site") and own_sources) or (k in ("count", "max_downloads") and own_count):
            continue
        setattr(a, k, args[k])
    return a


def to_cli(args: dict) -> str:
    parts = []
    for k, flag in LISTS_CLI.items():
        parts += [f"{flag} {shlex.quote(v)}" for v in args.get(k, [])]
    for k, flag in VALUES_CLI.items():
        if k in args:
            parts.append(f"{flag} {shlex.quote(str(args[k]))}")
    parts += [flag for k, flag in FLAGS_CLI.items() if args.get(k)]
    return "python -m litagent manual " + " ".join(parts)
