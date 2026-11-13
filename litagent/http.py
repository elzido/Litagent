"""Вежливый HTTP-клиент на stdlib: паузы по хостам, Retry-After, стоп на 401/403, дисковый кэш."""
from __future__ import annotations

import email.utils
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urlparse

from .policy import Policy


class Blocked(RuntimeError):
    """401/403 или запрет политикой: остановиться и разобраться, не повторять."""


class HttpError(RuntimeError):
    pass


class Response:
    def __init__(self, status: int, headers: dict, body: bytes, url: str = ""):
        self.status, self.headers, self.body, self.url = status, headers, body or b"", url

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body.decode("utf-8", errors="replace"))


def redact(url: str) -> str:
    return re.sub(r"(api_key|apikey|key|token)=[^&]+", r"\1=***", url, flags=re.I)


class PoliteClient:
    def __init__(self, policy: Policy, user_agent: str, cache_dir: str | Path | None = None,
                 cache_ttl: int = 86400, timeout: int = 60, retries: int = 5, log=None,
                 download_timeout: int = 25, download_retries: int = 2, dead_after: int = 3):
        self.policy = policy
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl, self.timeout, self.retries = cache_ttl, timeout, retries
        self.log = log or (lambda m: None)
        # скачивание файлов: быстрый отказ вместо долгих ожиданий (до 5 попыток × минуты на каждую статью)
        self.download_timeout, self.download_retries, self.dead_after = download_timeout, download_retries, dead_after
        self.blocked: dict[str, str] = {}
        self._last: dict[str, float] = {}
        self._fails: dict[str, int] = {}                 # подряд идущие сбои скачивания по хосту
        self._lock = threading.Lock()                    # защищает словари/счётчики (клиент общий для потоков)
        self._host_locks: dict[str, threading.Lock] = {}
        self._tl = threading.local()                     # таймаут текущего запроса, не меняя сигнатуру _raw
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0}

    # --- низкий уровень (переопределяется в тестах)
    def _sleep(self, s: float) -> None:
        time.sleep(s)

    def _raw(self, url: str, headers: dict, dest: Path | None = None, max_bytes: int | None = None):
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=getattr(self._tl, "timeout", None) or self.timeout)
            status = resp.status
        except urllib.error.HTTPError as e:  # тело ошибки читаем как обычный ответ
            resp, status = e, e.code
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        try:
            if dest is not None and status == 200:
                written = 0
                want = headers.get("Accept", "")
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        if written == 0:                  # первый кусок: не качаем мегабайты заглушки до проверки
                            if want.startswith("application/pdf") and b"%PDF" not in chunk[:1024]:
                                raise HttpError("не PDF (лендинг или paywall): начало ответа не похоже на PDF")
                            if want.startswith("application/gzip") and chunk[:2] != b"\x1f\x8b":
                                raise HttpError("не gzip-пакет (лендинг или paywall)")
                        written += len(chunk)
                        if max_bytes and written > max_bytes:
                            raise HttpError(f"файл больше {max_bytes} байт, прервано")
                        f.write(chunk)
                return status, hdrs, None
            return status, hdrs, resp.read()
        finally:
            resp.close()

    # --- логика
    @staticmethod
    def build_url(url: str, params: dict | None) -> str:
        if not params:
            return url
        clean = {k: v for k, v in params.items() if v is not None}
        return url + ("&" if "?" in url else "?") + urlencode(clean, doseq=True)

    def _throttle(self, host: str) -> None:
        """Резервирует ближайший свободный слот хоста: запросы к одному хосту идут не чаще policy.interval, даже из
        нескольких потоков; запросы к разным хостам друг друга не ждут."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._last.get(host, 0.0) + self.policy.interval(host))
            self._last[host] = slot
        if slot > now:
            self._sleep(slot - now)

    def _host_lock(self, host: str) -> threading.Lock:
        with self._lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] += n

    def _note(self, host: str, ok: bool) -> None:
        with self._lock:
            self._fails[host] = 0 if ok else self._fails.get(host, 0) + 1

    @staticmethod
    def _retry_after(headers: dict, attempt: int) -> float:
        ra = headers.get("retry-after")
        if ra:
            try:
                return min(float(ra), 300.0)
            except ValueError:
                try:
                    dt = email.utils.parsedate_to_datetime(ra)
                    return max(0.0, min((dt.timestamp() - time.time()), 300.0))
                except Exception:
                    pass
        return min(2 ** attempt * 2, 120)

    def request(self, url: str, params: dict | None = None, headers: dict | None = None,
                dest: Path | None = None, max_bytes: int | None = None, use_cache: bool = True,
                retries: int | None = None, max_wait: float | None = None, timeout: float | None = None) -> Response:
        """max_wait: если сервер просит ждать дольше (Retry-After) — не ждём, а отдаём HttpError (вызывающий пойдёт
        к следующему кандидату). Для скачивания файлов хост, давший dead_after сбоев подряд, пропускается до конца запуска."""
        full = self.build_url(url, params)
        host = urlparse(full).netloc.lower()
        if host in self.blocked:
            raise Blocked(f"{host} заблокировал нас ранее в этом запуске ({self.blocked[host]}); повтор не делается")
        ok, why = self.policy.url_allowed(full)
        if not ok:
            raise Blocked(why)
        hdrs = {"User-Agent": self.user_agent, "Accept": "*/*", **(headers or {})}
        is_dl = dest is not None
        if is_dl and self._fails.get(host, 0) >= self.dead_after:
            raise HttpError(f"{host} пропущен: {self._fails[host]} сбоев подряд в этом запуске")
        retries = retries or self.retries
        wait_cap = max_wait if max_wait is not None else 60.0

        cache_file = None
        if dest is None and use_cache and self.cache_dir:
            key = hashlib.sha256((full + json.dumps(sorted(hdrs.items()))).encode()).hexdigest()
            cache_file = self.cache_dir / key
            if cache_file.exists() and time.time() - cache_file.stat().st_mtime < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return Response(200, {}, cache_file.read_bytes(), full)

        last_err = None
        for attempt in range(retries):
            self._throttle(host)
            self._bump("requests")
            try:
                with self._host_lock(host):               # к одному хосту — одно соединение за раз, как раньше
                    self._tl.timeout = timeout
                    status, rh, body = self._raw(full, hdrs, dest, max_bytes)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = e
                self._bump("retries")
                self._sleep(min(2 ** attempt * 2, wait_cap if max_wait is not None else 60))
                continue
            if 200 <= status < 300:
                if is_dl:
                    self._note(host, True)
                if cache_file is not None:
                    tmp = cache_file.with_name(f"{cache_file.name}.{threading.get_ident()}.tmp")
                    tmp.write_bytes(body or b"")
                    tmp.replace(cache_file)               # атомарно: параллельные потоки не увидят половину файла
                return Response(status, rh, body, full)
            if status == 404:
                return Response(status, rh, body, full)
            if status in (429, 500, 502, 503, 504):
                self._bump("retries")
                delay = self._retry_after(rh, attempt)
                if max_wait is not None and delay > max_wait:
                    if is_dl:
                        self._note(host, False)
                    raise HttpError(f"HTTP {status} от {host}: просит ждать {delay:.0f} с, кандидат пропущен")
                self.log(f"  HTTP {status} от {host}, пауза {delay:.0f} с")
                self._sleep(delay)
                last_err = HttpError(f"HTTP {status}")
                continue
            if status in (401, 403):
                self.blocked[host] = f"HTTP {status}"
                raise Blocked(f"HTTP {status} от {host} ({redact(full)[:120]}): проверьте ключ, условия и robots.txt; повтор не делается")
            snippet = (body or b"")[:200].decode("utf-8", "replace")
            raise HttpError(f"HTTP {status} {redact(full)[:150]} {snippet}")
        if is_dl:
            self._note(host, False)
        raise HttpError(f"не удалось получить {redact(full)[:150]}: {last_err}")

    def get(self, url: str, params: dict | None = None, headers: dict | None = None, use_cache: bool = True) -> Response:
        return self.request(url, params, headers, use_cache=use_cache)

    def download(self, url: str, dest: Path, max_bytes: int = 120 * 1024 * 1024, headers: dict | None = None) -> Response:
        dest.parent.mkdir(parents=True, exist_ok=True)
        return self.request(url, None, headers, dest=dest, max_bytes=max_bytes, use_cache=False,
                            retries=self.download_retries, max_wait=20.0, timeout=self.download_timeout)
