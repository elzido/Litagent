"""Веб-интерфейс: поиск, библиотека, запуски, пресеты, сайты и редактор конфига в браузере.

Только стандартная библиотека. Поиск и проверка запускаются как фоновые задачи через тот же CLI
(python -m litagent ...), поэтому веб-морда и консоль всегда ведут себя одинаково.
Сервер по умолчанию слушает 127.0.0.1: конфиг и ключи не должны уходить в сеть.
"""
from __future__ import annotations

import copy
import csv
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import tomllib
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import cli, presets
from .customsites import (DEFAULT_FILE as SITES_DEFAULT, CustomSource, Site, SiteRegistry,
                          guess_mapping, name_error)
from .http import PoliteClient
from .llm import GIGACHAT_SCOPES
from .models import parse_topic
from .policy import Policy
from .sources import DEFAULT_SOURCES, SOURCE_CLASSES, Ctx
from .store import Library

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}$")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_BODY = 2 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


# ---------------------------------------------------------------- конфиг: чтение и запись TOML
def _merge_file(path: Path) -> dict:
    """Значения из файла поверх дефолтов, БЕЗ переменных окружения (для редактора)."""
    cfg = copy.deepcopy(cli.DEFAULT_CFG)
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def _toml_str(s: str) -> str:
    out = []
    for ch in str(s):
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 32:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    return _toml_str(v)


COMMENTS = {
    "email": "обязателен: идёт в User-Agent, Crossref mailto, Unpaywall, NCBI",
    "out": "куда класть pdf/, xml/, index.json, runs/",
    "limit_per_source": "сколько записей брать с каждого источника",
    "download_workers": "параллельных скачиваний (1 — по одному); к одному хосту всё равно по одному соединению",
    "download_timeout": "сек на файл (простой сокета)",
    "download_retries": "попыток на файл (обычные запросы API: 5)",
    "pdf_only": "true: оставлять только статьи с реально скачанным PDF (как --pdf-only)",
    "pubmed_pmc": "PubMed ищет ещё и в PMC и качает оттуда OA-PDF (false — выключить)",
    "custom_sites": "свои сайты (создаётся командой `sites add`)",
    "max_downloads": "потолок успешных скачиваний за запуск (--max-downloads)",
    "folders_by_source": "раскладывать файлы по папкам источников: pdf/<источник>/",
    "presets_file": "пресеты ручного режима (manual --save-preset / --preset)",
    "deepen_rounds": "при --count N: сколько раз углублять поиск, если N PDF ещё не набрано",
    "deepen_limit_cap": "при углублении лимит на источник растёт ×2, но не выше этого значения",
    "cache_ttl": "сколько секунд живут ответы в HTTP-кэше",
    "keys.ncbi": "PubMed: 10 запр./с вместо 3",
    "keys.openalex": "ОБЯЗАТЕЛЕН для OpenAlex (бесплатный)",
    "keys.s2": "Semantic Scholar: без ключа часто 429",
    "keys.core": "ОБЯЗАТЕЛЕН для CORE (бесплатный)",
    "keys.gigachat_client_id": "GigaChat: Client ID из кабинета Sber Studio",
    "keys.gigachat_client_secret": "GigaChat: Client Secret оттуда же; Base64-ключ агент соберёт сам",
    "keys.gigachat": "запасной вариант: готовый ключ авторизации (лучше переменная GIGACHAT_CREDENTIALS)",
    "llm.provider": "ollama, openai (LM Studio, llama.cpp server, vLLM) или gigachat (облако Сбера)",
    "llm.base_url": "пусто = адрес по умолчанию для провайдера",
    "llm.model": "для gigachat: GigaChat, GigaChat-Pro, GigaChat-Max, GigaChat-2, GigaChat-2-Pro, GigaChat-2-Max, GigaChat-3-Lightning",
    "llm.num_ctx": "окно контекста Ollama; без него Ollama режет длинный промпт (заметки из .tex)",
    "llm.api_key": "ключ для openai-совместимого сервера; для gigachat лучше пара client_id + client_secret",
    "llm.scope": "версия ключа GigaChat: GIGACHAT_API_PERS (физлица) | GIGACHAT_API_CORP | GIGACHAT_API_B2B (юрлица)",
    "llm.ca_bundle": "PEM сертификата Минцифры для GigaChat, если он не установлен в системе",
    "llm.oauth_url": "пусто = https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
    "auto.rounds": "1 = без уточнения плана",
    "auto.llm_filter": "оценка релевантности каждой найденной статьи моделью",
    "auto.notes": "файл(ы) с пометками: .tex, .md, .txt",
    "web.host": "127.0.0.1 — только этот компьютер; другой адрес откроет конфиг и ключи сети",
    "web.port": "порт веб-интерфейса",
    "web.open_browser": "открывать браузер при запуске `litagent web`",
}
SECTION_TITLES = {"keys": "ключи API (необязательно; лучше через переменные окружения)",
                  "llm": "модель для авто-режима: локальная или облачная",
                  "auto": "авто-режим", "web": "веб-интерфейс"}


def dump_toml(cfg: dict) -> str:
    """Сериализует конфиг в TOML с комментариями-подсказками (стандартная библиотека не умеет писать TOML)."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [f"# litagent — конфигурация. Сохранено веб-интерфейсом {now}.",
             "# Ключи также можно задавать переменными окружения (NCBI_API_KEY, OPENALEX_API_KEY, S2_API_KEY,",
             "# CORE_API_KEY, GIGACHAT_CLIENT_ID, GIGACHAT_CLIENT_SECRET, GIGACHAT_CREDENTIALS, LITAGENT_EMAIL).", ""]

    def emit(key: str, val, full: str):
        c = COMMENTS.get(full, "")
        lines.append(f"{key} = {_toml_val(val)}" + (f"    # {c}" if c else ""))

    for k, v in cfg.items():
        if not isinstance(v, dict):
            emit(k, v, k)
    for sec in ("keys", "llm", "auto", "web"):
        if not isinstance(cfg.get(sec), dict):
            continue
        lines += ["", f"[{sec}]" + (f"    # {SECTION_TITLES[sec]}" if sec in SECTION_TITLES else "")]
        for k, v in cfg[sec].items():
            emit(k, v, f"{sec}.{k}")
    return "\n".join(lines) + "\n"


def validate_config(data: dict, known_sources: list[str], base: dict | None = None) -> tuple[dict, dict]:
    """Проверяет присланные поля, приводит типы. Неприсланные поля берутся из base (текущий файл) или дефолтов.
    Возвращает (итоговый конфиг, ошибки по полям)."""
    cfg = copy.deepcopy(base if base is not None else cli.DEFAULT_CFG)
    errors: dict[str, str] = {}

    def num(field: str, v, lo, hi, integer=True):
        try:
            n = int(v) if integer else float(v)
        except (TypeError, ValueError):
            errors[field] = "должно быть числом"
            return None
        if not lo <= n <= hi:
            errors[field] = f"допустимо от {lo} до {hi}"
            return None
        return n

    def take(field: str, sec: str | None, key: str, kind):
        """kind — фабричный дефолт (задаёт ожидаемый тип); неприсланное поле сохраняет значение из base."""
        base_val = cfg[sec].get(key, kind) if sec else cfg.get(key, kind)
        src = data.get(sec) if sec else data
        if not isinstance(src, dict) or key not in src or src[key] is None:
            return base_val
        v = src[key]
        if isinstance(kind, bool):
            if isinstance(v, bool):
                return v
            errors[field] = "ожидается true/false"
            return base_val
        if isinstance(kind, (int, float)):
            if isinstance(v, bool):
                errors[field] = "ожидается число"
                return base_val
            n = num(field, v, -(10**9), 10**9, integer=isinstance(kind, int))
            return base_val if n is None else n
        if isinstance(kind, list):
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return [x.strip() for x in v if x.strip()]
            errors[field] = "ожидается список строк"
            return base_val
        return str(v)

    for k, kind in cli.DEFAULT_CFG.items():
        if isinstance(kind, dict):
            for sk, skind in kind.items():
                cfg[k][sk] = take(f"{k}.{sk}", k, sk, skind)
        else:
            cfg[k] = take(k, None, k, kind)

    if not isinstance(cfg["email"], str):
        errors["email"], cfg["email"] = "ожидается строка", ""
    elif cfg["email"] and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", cfg["email"]):
        errors["email"] = "не похож на e-mail"
    if not isinstance(cfg["out"], str) or not cfg["out"].strip():
        errors["out"] = "папка библиотеки не может быть пустой"
    if not isinstance(cfg["sources"], list):
        errors["sources"], cfg["sources"] = "ожидается список строк", list(cli.DEFAULT_CFG["sources"])
    if not isinstance(cfg["web"]["host"], str) or not cfg["web"]["host"].strip():
        errors["web.host"] = "адрес не может быть пустым"
    for f, lo, hi in (("limit_per_source", 1, 10000), ("max_downloads", 0, 10000), ("download_workers", 1, 64),
                      ("download_timeout", 1, 600), ("download_retries", 0, 10), ("cache_ttl", 0, 10**7),
                      ("deepen_rounds", 0, 100), ("deepen_limit_cap", 1, 100000)):
        if f in errors:
            continue
        if not isinstance(cfg[f], (int, float)) or isinstance(cfg[f], bool):
            errors[f] = "должно быть числом"
        elif not lo <= cfg[f] <= hi:
            errors[f] = f"допустимо от {lo} до {hi}"
    bad = [s for s in cfg["sources"] if s not in known_sources]
    if bad:
        errors["sources"] = "неизвестные источники: " + ", ".join(bad)
    if not cfg["sources"]:
        errors["sources"] = "выберите хотя бы один источник"
    if cfg["llm"]["provider"] not in ("ollama", "openai", "gigachat"):
        errors["llm.provider"] = "ollama, openai или gigachat"
    if cfg["llm"]["scope"] not in GIGACHAT_SCOPES:
        errors["llm.scope"] = " или ".join(GIGACHAT_SCOPES)
    for f, lo, hi in (("llm.temperature", 0, 2), ("llm.timeout", 1, 3600), ("llm.num_ctx", 512, 10**6),
                      ("auto.rounds", 1, 20), ("auto.min_results", 0, 10000), ("auto.min_score", 0, 10),
                      ("web.port", 1, 65535)):
        sec, key = f.split(".")
        if f in errors:
            continue
        if not isinstance(cfg[sec][key], (int, float)) or isinstance(cfg[sec][key], bool):
            errors[f] = "должно быть числом"
        elif not lo <= cfg[sec][key] <= hi:
            errors[f] = f"допустимо от {lo} до {hi}"
    return cfg, errors


# ---------------------------------------------------------------- фоновые задачи (запуски CLI)
class Job:
    def __init__(self, jid: str, kind: str, argv: list[str], title: str):
        self.id, self.kind, self.argv, self.title = jid, kind, argv, title
        self.started = datetime.datetime.now().isoformat(timespec="seconds")
        self.finished: str | None = None
        self.state = "running"                     # running | done | error | cancelled
        self.exit_code: int | None = None
        self.lines: list[str] = []
        self.run_dir: str | None = None
        self.cancel_requested = False
        self.proc: subprocess.Popen | None = None

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state,
                "started": self.started, "finished": self.finished, "exit_code": self.exit_code,
                "run_id": Path(self.run_dir).name if self.run_dir else None}


class JobManager:
    """Одна активная задача за раз: библиотека и index.json не рассчитаны на параллельные запуски."""

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.lock = threading.Lock()
        self.current: Job | None = None
        self.history: list[Job] = []

    def start(self, kind: str, argv: list[str], title: str) -> Job:
        with self.lock:
            if self.current and self.current.state == "running":
                raise ApiError(409, f"Уже идёт задача «{self.current.title}» — дождитесь окончания или отмените её.")
            job = Job(datetime.datetime.now().strftime("%H%M%S-") + str(len(self.history)), kind, argv, title)
            self.current = job
            self.history.append(job)
            self.history = self.history[-30:]
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"job-{job.id}").start()
        return job

    def _run(self, job: Job) -> None:
        try:
            proc = subprocess.Popen(job.argv, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
        except OSError as e:
            with self.lock:
                job.lines.append(f"Не удалось запустить: {e}")
                job.state, job.exit_code = "error", -1
                job.finished = datetime.datetime.now().isoformat(timespec="seconds")
            return
        job.proc = proc
        if job.cancel_requested:                     # отмена пришла раньше, чем процесс успел стартовать
            self._interrupt(proc)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            with self.lock:
                job.lines.append(line)
                if "отчёт запуска: " in line:
                    job.run_dir = line.split("отчёт запуска: ", 1)[1].strip()
        proc.stdout.close()
        proc.wait()
        with self.lock:
            job.exit_code = proc.returncode
            job.state = "cancelled" if job.cancel_requested else ("done" if proc.returncode == 0 else "error")
            job.finished = datetime.datetime.now().isoformat(timespec="seconds")

    def get(self, jid: str) -> Job:
        with self.lock:
            for j in reversed(self.history):
                if j.id == jid:
                    return j
        raise ApiError(404, f"Нет задачи {jid}")

    def cancel(self, jid: str) -> Job:
        job = self.get(jid)
        with self.lock:
            if job.state != "running":
                raise ApiError(409, "Задача уже завершилась.")
            job.cancel_requested = True
            proc = job.proc
        if proc:
            self._interrupt(proc)                    # если процесса ещё нет — _run прервёт его сам после старта
        return job

    @staticmethod
    def _interrupt(proc: subprocess.Popen) -> None:
        try:
            proc.send_signal(signal.SIGINT)          # CLI корректно обрабатывает KeyboardInterrupt
        except OSError:
            pass

        def hard_stop():
            if proc.poll() is None:
                proc.terminate()
        threading.Timer(6.0, hard_stop).start()

    def snapshot(self, job: Job, since: int = 0) -> dict:
        with self.lock:
            return {**job.summary(), "total": len(job.lines), "lines": job.lines[since:],
                    "cancel_requested": job.cancel_requested}


# ---------------------------------------------------------------- аргументы CLI из формы поиска
FLAGS = {"oa_only": "--oa-only", "title_only": "--title-only", "strict": "--strict", "pdf_only": "--pdf-only",
         "no_download": "--no-download", "no_pmc": "--no-pmc", "flat": "--flat", "dry_run": "--dry-run"}
NUMS = {"year_from": "--year-from", "year_to": "--year-to", "limit": "--limit", "count": "--count",
        "max_downloads": "--max-downloads", "workers": "--workers", "deepen_rounds": "--deepen-rounds"}


def _int_opt(p: dict, key: str) -> int | None:
    v = p.get(key)
    if v in (None, ""):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ApiError(400, f"«{key}» должно быть целым числом.") from None


def _common_argv(p: dict, known: list[str], forbidden: dict) -> list[str]:
    argv: list[str] = []
    srcs = p.get("sources") or []
    if isinstance(srcs, str):
        srcs = [s.strip() for s in srcs.split(",") if s.strip()]
    if srcs:
        bad = [s for s in srcs if s in forbidden]
        if bad:
            raise ApiError(400, "; ".join(f"{s}: {forbidden[s]}" for s in bad))
        unknown = [s for s in srcs if s not in known]
        if unknown:
            raise ApiError(400, "Неизвестные источники: " + ", ".join(unknown))
        argv += ["--sources", ",".join(srcs)]
    for key, flag in NUMS.items():
        n = _int_opt(p, key)
        if n is not None:
            if key == "count" and n < 1:
                raise ApiError(400, "--count должен быть положительным числом.")
            if key in ("limit", "workers", "max_downloads", "deepen_rounds") and n < 0:
                raise ApiError(400, f"«{key}» не может быть отрицательным.")
            argv += [flag, str(n)]
    for key, flag in FLAGS.items():
        if p.get(key):
            argv.append(flag)
    if p.get("pdf_only") and p.get("no_download"):
        raise ApiError(400, "--pdf-only проверяет наличие PDF скачиванием: уберите «только найти».")
    return argv


def build_search_argv(p: dict, cfg: dict, known: list[str], forbidden: dict, config_path: Path) -> tuple[list[str], str, str]:
    """(argv без python -m litagent, kind, заголовок для UI)."""
    mode = p.get("mode") or "manual"
    common = _common_argv(p, known, forbidden)
    if config_path.exists():
        common += ["--config", str(config_path)]
    if mode == "manual":
        argv = ["manual"]
        topics = [t.strip() for t in (p.get("topics") or []) if isinstance(t, str) and t.strip()]
        preset = (p.get("preset") or "").strip()
        groups = [g.strip() for g in (p.get("group") or []) if isinstance(g, str) and g.strip()]
        if not topics and not preset and not groups:
            raise ApiError(400, "Задайте тему или выберите пресет.")
        if preset:
            argv += ["--preset", preset]
        for t in topics:
            argv += ["-q", t]
        for g in groups:
            argv += ["--group", g]
        for x in p.get("exclude") or []:
            if isinstance(x, str) and x.strip():
                argv += ["--exclude", x.strip()]
        for c in p.get("arxiv_cats") or []:
            if isinstance(c, str) and c.strip():
                argv += ["--arxiv-cat", c.strip()]
        for src, q in (p.get("raw") or {}).items():
            if isinstance(q, str) and q.strip():
                argv += ["--raw", f"{src}={q.strip()}"]
        save = (p.get("save_preset") or "").strip()
        if save:
            argv += ["--save-preset", save]
            if p.get("preset_desc"):
                argv += ["--preset-desc", str(p["preset_desc"])]
        title = preset and f"Пресет «{preset}»" or "; ".join(topics)[:90] or "Ручной поиск"
        return argv + common, "manual", title
    if mode == "auto":
        goal = (p.get("goal") or "").strip()
        if not goal:
            raise ApiError(400, "Опишите цель поиска.")
        argv = ["auto", goal]
        notes = p.get("notes")
        if notes is None:
            notes = cfg.get("auto", {}).get("notes") or []
        for n in notes:
            if isinstance(n, str) and n.strip():
                argv += ["--notes", n.strip()]
        if p.get("no_llm"):
            argv.append("--no-llm")
        if p.get("llm_filter"):
            argv.append("--llm-filter")
        if p.get("min_score") not in (None, ""):
            argv += ["--min-score", str(p["min_score"])]
        for key, flag in (("rounds", "--rounds"), ("min_results", "--min-results")):
            n = _int_opt(p, key)
            if n is not None:
                argv += [flag, str(n)]
        return argv + common, "auto", goal[:90]
    raise ApiError(400, f"Неизвестный режим: {mode}")


# ---------------------------------------------------------------- приложение (логика API)
class WebApp:
    def __init__(self, config_path: str | Path | None = None, cwd: str | Path | None = None):
        self.cwd = Path(cwd).resolve() if cwd else Path.cwd()
        self.config_path = Path(config_path) if config_path else self.cwd / "litagent.toml"
        if not self.config_path.is_absolute():
            self.config_path = self.cwd / self.config_path
        self.jobs = JobManager(self.cwd)

    # --- помощники
    def cfg(self) -> dict:
        return cli.load_config(str(self.config_path)) if self.config_path.exists() else _env_defaults()

    def known_sources(self) -> list[str]:
        reg = self._sites_registry()
        return list(SOURCE_CLASSES) + [n for n in reg.names() if n not in SOURCE_CLASSES]

    def library_root(self) -> Path:
        return (self.cwd / (self.cfg().get("out") or "library")).resolve()

    def _presets_file(self) -> Path:
        return self.cwd / (self.cfg().get("presets_file") or presets.DEFAULT_FILE)

    def _sites_registry(self) -> SiteRegistry:
        return SiteRegistry.load(self.cwd / (self.cfg().get("custom_sites") or SITES_DEFAULT))

    # --- состояние и конфиг
    def get_state(self) -> dict:
        cfg = self.cfg()
        reg = self._sites_registry()
        pol = Policy()
        try:
            allp = presets.load(self._presets_file())
        except SystemExit:
            allp = {}
        lib = Library(self.library_root(), by_source=bool(cfg.get("folders_by_source", True)))
        return {
            "config_path": str(self.config_path), "config_exists": self.config_path.exists(),
            "email": cfg.get("email") or "", "out": str(self.library_root()),
            "llm": {k: cfg["llm"].get(k) for k in ("provider", "model", "base_url")},
            "sources": {"builtin": DEFAULT_SOURCES,
                        "custom": [{"name": s.name, "title": s.title, "enabled": s.enabled, "host": s.host} for s in reg.sites],
                        "forbidden": pol.forbidden_sources, "selected": cfg.get("sources") or []},
            "presets": [{"name": n, "description": (e.get("description", "") if isinstance(e, dict) else "")}
                        for n, e in sorted(allp.items())],
            "library": lib.stats(), "web": cfg.get("web", {}),
            "job": self.jobs.current.summary() if self.jobs.current else None,
        }

    def get_config(self) -> dict:
        env = []
        if os.environ.get("LITAGENT_EMAIL"):
            env.append({"field": "email", "env": "LITAGENT_EMAIL"})
        for k, name in cli.ENV_KEYS.items():
            if os.environ.get(name):
                env.append({"field": f"keys.{k}", "env": name})
        return {"path": str(self.config_path), "exists": self.config_path.exists(),
                "cfg": _merge_file(self.config_path), "env_overrides": env}

    def post_config(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ApiError(400, "Ожидается объект с полями конфига.")
        cfg, errors = validate_config(data, self.known_sources(), base=_merge_file(self.config_path))
        if errors:
            raise ApiError(400, json.dumps(errors, ensure_ascii=False))
        text = dump_toml(cfg)
        try:
            tomllib.loads(text)                      # страховка: файл обязан читаться тем же tomllib
        except ValueError as e:
            raise ApiError(500, f"Внутренняя ошибка сериализации TOML: {e}") from None
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.config_path)
        return {"saved": str(self.config_path), "cfg": cfg}

    # --- поиск и проверка
    def post_search(self, p: dict) -> dict:
        cfg = self.cfg()
        pol = Policy()
        argv, kind, title = build_search_argv(p, cfg, self.known_sources(), pol.forbidden_sources, self.config_path)
        job = self.jobs.start(kind, [sys.executable, "-m", "litagent", *argv], title)
        return {"job": job.summary()}

    def post_check(self) -> dict:
        argv = ["check"] + (["--config", str(self.config_path)] if self.config_path.exists() else [])
        job = self.jobs.start("check", [sys.executable, "-m", "litagent", *argv], "Проверка источников и LLM")
        return {"job": job.summary()}

    def get_jobs(self) -> dict:
        with self.jobs.lock:
            return {"jobs": [j.summary() for j in reversed(self.jobs.history)]}

    def get_job(self, jid: str, since: int = 0) -> dict:
        return self.jobs.snapshot(self.jobs.get(jid), max(0, since))

    def post_cancel(self, jid: str) -> dict:
        return {"job": self.jobs.cancel(jid).summary()}

    # --- библиотека
    def get_library(self) -> dict:
        root = self.library_root()
        lib = Library(root)
        records = []
        for pk, e in lib.records.items():
            f = e.get("file")
            records.append({**e, "pk": pk, "tags": lib.tags.get(pk, []),
                            "has_file": bool(f and (root / f).is_file())})
        records.sort(key=lambda r: (-(r.get("year") or 0), (r.get("title") or "").casefold()))
        return {"root": str(root), "stats": lib.stats(), "records": records, "all_tags": lib.all_tags()}

    def post_tags(self, data: dict) -> dict:
        pk = data.get("pk")
        tags = data.get("tags")
        if not isinstance(pk, str) or not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ApiError(400, "Ожидается {pk: строка, tags: список строк}.")
        lib = Library(self.library_root())
        try:
            clean = lib.set_tags(pk, tags)
        except KeyError:
            raise ApiError(404, f"Нет записи {pk} в библиотеке.") from None
        lib.save()
        return {"pk": pk, "tags": clean, "all_tags": lib.all_tags()}

    def post_library_delete(self, data: dict) -> dict:
        keys = data.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            raise ApiError(400, "Ожидается {keys: непустой список идентификаторов записей}.")
        return Library(self.library_root()).delete(keys)

    def get_file(self, rel: str):
        """(content_type, bytes, имя файла) — только файлы внутри библиотеки."""
        root = self.library_root()
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ApiError(404, "Файл не найден в библиотеке.")
        if target.suffix.lower() not in (".pdf", ".xml", ".tgz"):
            raise ApiError(403, "Отдаю только PDF/XML из библиотеки.")
        ctype = "application/pdf" if target.suffix.lower() == ".pdf" else "application/xml"
        return ctype, target.read_bytes(), target.name

    # --- запуски
    def get_runs(self) -> dict:
        d = self.library_root() / "runs"
        out = []
        if d.is_dir():
            for p in sorted(d.iterdir(), reverse=True):
                if not p.is_dir() or not RUN_ID_RE.match(p.name):
                    continue
                meta = _read_json(p / "run.json") or {}
                out.append({"id": p.name, "mode": meta.get("mode", "?"), "started": meta.get("started", ""),
                            "total_unique": meta.get("total_unique"), "downloads": meta.get("downloads"),
                            "goal": meta.get("goal", ""), "target": meta.get("target"),
                            "topics": [s.get("label", "") for s in meta.get("specs", [])][:5]})
                if len(out) >= 200:
                    break
        return {"runs": out}

    def get_run(self, run_id: str) -> dict:
        if not RUN_ID_RE.match(run_id):
            raise ApiError(400, "Неверный идентификатор запуска.")
        d = self.library_root() / "runs" / run_id
        meta = _read_json(d / "run.json")
        if meta is None:
            raise ApiError(404, f"Нет запуска {run_id}.")
        results = []
        csv_path = d / "results.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                results = list(csv.DictReader(f))
        return {"id": run_id, "meta": meta, "results": results}

    def post_runs_delete(self, data: dict) -> dict:
        ids = data.get("ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise ApiError(400, "Ожидается {ids: непустой список идентификаторов запусков}.")
        base = (self.library_root() / "runs").resolve()
        res = {"deleted": 0, "errors": []}
        for rid in ids:
            if not RUN_ID_RE.match(rid):
                res["errors"].append(f"{rid}: неверный идентификатор")
                continue
            d = (base / rid).resolve()
            if not d.is_relative_to(base) or not d.is_dir():
                continue
            try:
                shutil.rmtree(d)
                res["deleted"] += 1
            except OSError as e:
                res["errors"].append(f"{rid}: {e}")
        return res

    def post_cache_clear(self) -> dict:
        return Library(self.library_root()).clear_cache()

    # --- пресеты
    def get_presets(self) -> dict:
        try:
            allp = presets.load(self._presets_file())
        except SystemExit as e:
            raise ApiError(409, str(e)) from None
        out = []
        for name in sorted(allp):
            e = allp[name] if isinstance(allp[name], dict) else {}
            args = presets.clean_args(e.get("args", {}))
            out.append({"name": name, "description": e.get("description", ""), "saved": e.get("saved", ""),
                        "args": args, "cli": presets.to_cli(args)})
        return {"file": str(self._presets_file()), "presets": out}

    def post_preset(self, data: dict) -> dict:
        name = (data.get("name") or "").strip()
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ApiError(400, "args должен быть объектом.")
        try:
            presets.save(self._presets_file(), name, args, str(data.get("description") or ""))
        except SystemExit as e:
            raise ApiError(400, str(e)) from None
        return {"saved": name}

    def delete_preset(self, name: str) -> dict:
        try:
            presets.remove(self._presets_file(), name)
        except SystemExit as e:
            raise ApiError(404, str(e)) from None
        return {"removed": name}

    # --- свои сайты
    def get_sites(self) -> dict:
        reg = self._sites_registry()
        pol = Policy()
        return {"file": str(reg.path), "error": reg.error, "builtin": DEFAULT_SOURCES,
                "forbidden": pol.forbidden_sources,
                "custom": [{**s.to_dict(), "host": s.host, "valid": not s.validate(pol)} for s in reg.sites]}

    def post_site(self, name: str, data: dict) -> dict:
        reg = self._sites_registry()
        if reg.error:
            raise ApiError(409, reg.error)
        site = reg.get(name)
        if not site:
            raise ApiError(404, f"Нет сайта «{name}».")
        if "enabled" in data:
            site.enabled = bool(data["enabled"])
        reg.put(site)
        reg.save()
        return {"name": site.name, "enabled": site.enabled}

    def delete_site(self, name: str) -> dict:
        reg = self._sites_registry()
        if reg.error:
            raise ApiError(409, reg.error)
        if not reg.remove(name):
            raise ApiError(404, f"Нет сайта «{name}».")
        reg.save()
        return {"removed": name}

    def post_site_create(self, data: dict) -> dict:
        reg = self._sites_registry()
        if reg.error:
            raise ApiError(409, reg.error)
        pol = Policy()
        site = Site.from_dict(data)
        errs = []
        ne = name_error(site.name, reg, pol)
        if ne:
            errs.append(f"имя: {ne}")
        errs += site.validate(pol)
        if errs:
            raise ApiError(400, json.dumps(errs, ensure_ascii=False))
        reg.put(site)
        reg.save()
        return {"site": {**site.to_dict(), "host": site.host}, "llm_details": site.llm_details()}

    def _site_ctx(self, site: Site) -> Ctx:
        cfg = self.cfg()
        pol = Policy()
        pol.intervals[site.host] = max(pol.intervals.get(site.host, 0.0), float(site.min_interval or 1.0))
        client = PoliteClient(pol, f"litagent/1.0 (mailto:{cfg.get('email') or 'unknown'})", cache_dir=None, retries=2)
        return Ctx(client, cfg.get("email", ""), cfg.get("keys", {}), "litagent", lambda m: None)

    def post_site_probe(self, data: dict) -> dict:
        """Пробный запрос «machine learning» и подсказки путей в JSON — как в опроснике sites add."""
        if not self.cfg().get("email"):
            raise ApiError(400, "Нужен e-mail в конфиге: он уходит в User-Agent пробного запроса.")
        site = Site.from_dict(data)
        pol = Policy()
        if not site.search_url:
            raise ApiError(400, "Укажите адрес поискового API (search_url).")
        ok, why = pol.url_allowed(site.search_url)
        if not ok:
            raise ApiError(400, why)
        src = CustomSource(self._site_ctx(site), site)
        ok, why = src.available()
        if not ok:
            raise ApiError(400, why)
        try:
            raw = src.fetch_json(parse_topic("machine learning"), 3)
        except Exception as e:  # noqa: BLE001 — сетевая ошибка должна вернуться текстом, а не 500
            raise ApiError(502, f"Пробный запрос не удался: {type(e).__name__}: {str(e)[:200]}") from None
        g = guess_mapping(raw)
        return {"top": list(raw)[:15] if isinstance(raw, dict) else [],
                "results_path": g["results_path"], "total_path": g["total_path"],
                "fields": g["fields"], "sample": g["sample"]}

    def post_site_test(self, data: dict) -> dict:
        """Проверка описания сайта (можно несохранённого): реальный запрос и разбор полей."""
        if not self.cfg().get("email"):
            raise ApiError(400, "Нужен e-mail в конфиге: он уходит в User-Agent пробного запроса.")
        site = Site.from_dict(data)
        pol = Policy()
        errs = site.validate(pol)
        if errs:
            raise ApiError(400, json.dumps(errs, ensure_ascii=False))
        src = CustomSource(self._site_ctx(site), site)
        ok, why = src.available()
        if not ok:
            raise ApiError(400, why)
        spec = parse_topic(data.get("query") or "machine learning")
        try:
            recs, info = src.search(spec, 3)
        except Exception as e:  # noqa: BLE001
            raise ApiError(502, f"{type(e).__name__}: {str(e)[:300]}") from None
        warnings = [f"поле «{k}» пустое у всех записей — проверьте путь"
                    for k in ("authors", "year", "doi", "abstract", "pdf_urls")
                    if recs and not any(getattr(r, k) for r in recs)]
        return {"query": info.get("query", ""), "found": info.get("found"),
                "records": [{**r.to_dict(), "abstract": (r.abstract or "")[:300]} for r in recs],
                "warnings": warnings}


def _env_defaults() -> dict:
    cfg = copy.deepcopy(cli.DEFAULT_CFG)
    if os.environ.get("LITAGENT_EMAIL"):
        cfg["email"] = os.environ["LITAGENT_EMAIL"]
    for k, env in cli.ENV_KEYS.items():
        if os.environ.get(env):
            cfg["keys"][k] = os.environ[env]
    return cfg


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    app: WebApp
    protocol_version = "HTTP/1.1"
    server_version = "litagent-web"

    def log_message(self, fmt, *args):      # тихий сервер: только ошибки видны в консоли
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return {}
        if n > MAX_BODY:
            raise ApiError(413, "Слишком большой запрос.")
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "Тело запроса — не JSON.") from None
        if not isinstance(data, dict):
            raise ApiError(400, "Ожидается JSON-объект.")
        return data

    def _dispatch(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        app = self.app
        try:
            if self.command == "GET":
                if path in ("/", "/index.html"):
                    html = Path(__file__).with_name("webui.html").read_bytes()
                    self._send(200, html, "text/html; charset=utf-8")
                elif path.startswith("/files/"):
                    ctype, data, name = app.get_file(path[len("/files/"):])
                    self._send(200, data, ctype, {"Content-Disposition": f'inline; filename="{name}"'})
                elif path == "/api/state":
                    self._json(200, app.get_state())
                elif path == "/api/config":
                    self._json(200, app.get_config())
                elif path == "/api/library":
                    self._json(200, app.get_library())
                elif path == "/api/runs":
                    self._json(200, app.get_runs())
                elif path.startswith("/api/runs/"):
                    self._json(200, app.get_run(path[len("/api/runs/"):]))
                elif path == "/api/presets":
                    self._json(200, app.get_presets())
                elif path == "/api/sites":
                    self._json(200, app.get_sites())
                elif path == "/api/jobs":
                    self._json(200, app.get_jobs())
                elif path.startswith("/api/jobs/"):
                    try:
                        since = int(q.get("since") or 0)
                    except ValueError:
                        since = 0
                    self._json(200, app.get_job(path[len("/api/jobs/"):], since))
                else:
                    raise ApiError(404, "Нет такого адреса.")
            elif self.command == "POST":
                data = self._body()
                if path == "/api/config":
                    self._json(200, app.post_config(data))
                elif path == "/api/search":
                    self._json(200, app.post_search(data))
                elif path == "/api/check":
                    self._json(200, app.post_check())
                elif path == "/api/presets":
                    self._json(200, app.post_preset(data))
                elif path == "/api/library/tags":
                    self._json(200, app.post_tags(data))
                elif path == "/api/library/delete":
                    self._json(200, app.post_library_delete(data))
                elif path == "/api/runs/delete":
                    self._json(200, app.post_runs_delete(data))
                elif path == "/api/cache/clear":
                    self._json(200, app.post_cache_clear())
                elif path == "/api/sites":
                    self._json(200, app.post_site_create(data))
                elif path == "/api/sites/probe":
                    self._json(200, app.post_site_probe(data))
                elif path == "/api/sites/test":
                    self._json(200, app.post_site_test(data))
                elif re.match(r"^/api/sites/[^/]+$", path):
                    self._json(200, app.post_site(path[len("/api/sites/"):], data))
                elif re.match(r"^/api/jobs/[^/]+/cancel$", path):
                    self._json(200, app.post_cancel(path.split("/")[3]))
                else:
                    raise ApiError(404, "Нет такого адреса.")
            elif self.command == "DELETE":
                if path.startswith("/api/presets/"):
                    self._json(200, app.delete_preset(unquote(path[len("/api/presets/"):])))
                elif path.startswith("/api/sites/"):
                    self._json(200, app.delete_site(unquote(path[len("/api/sites/"):])))
                else:
                    raise ApiError(404, "Нет такого адреса.")
            else:
                raise ApiError(405, "Метод не поддерживается.")
        except ApiError as e:
            detail = e.message
            try:                                   # ошибки валидации конфига приходят JSON-строкой
                detail = json.loads(e.message)
            except (ValueError, TypeError):
                pass
            self._json(e.code, {"error": detail})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 — одна ошибка не должна ронять сервер
            import traceback
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    do_GET = do_POST = do_DELETE = _dispatch


def serve(config_path: str | None = None, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    app = WebApp(config_path)
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    url = f"http://{'localhost' if host in LOOPBACK else host}:{port}"
    print(f"litagent web: {url}", flush=True)
    print(f"  конфиг: {app.config_path}" + ("" if app.config_path.exists() else "  (не найден — будет создан при сохранении)"), flush=True)
    print(f"  библиотека: {app.library_root()}", flush=True)
    if host not in LOOPBACK:
        print("  ⚠ сервер слушает не только localhost: конфиг и ключи будут доступны всем, кто может подключиться.", flush=True)
    print("  Остановить: Ctrl+C", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.", flush=True)
    finally:
        server.server_close()
