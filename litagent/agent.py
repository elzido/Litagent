"""Оркестратор: поиск по источникам -> дедупликация -> ранжирование -> скачивание -> отчёты."""
from __future__ import annotations

import dataclasses
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime
from pathlib import Path

from .customsites import DEFAULT_FILE, CustomSource, SiteRegistry
from .http import Blocked, HttpError, PoliteClient
from .models import Record, SearchSpec, dedupe
from .policy import Policy
from .resolve import Downloader
from .sources import Ctx, make_sources
from .store import Library


class Agent:
    def __init__(self, cfg: dict, log=print, policy: Policy | None = None):
        self.cfg, self.log = cfg, log
        self.policy = policy or Policy()
        keys = cfg.get("keys", {})
        if keys.get("ncbi"):
            self.policy.intervals["eutils.ncbi.nlm.nih.gov"] = 0.11     # 10 запр./с с ключом
        self.lib = Library(cfg.get("out", "library"))
        ua = f"{cfg.get('tool', 'litagent')}/1.0 (mailto:{cfg.get('email') or 'unknown'})"
        self.client = PoliteClient(self.policy, ua, cache_dir=self.lib.cache_dir,
                                   cache_ttl=int(cfg.get("cache_ttl", 86400)), log=log,
                                   download_timeout=int(cfg.get("download_timeout", 25)),
                                   download_retries=int(cfg.get("download_retries", 2)))
        self.workers = max(1, int(cfg.get("download_workers", 4)))
        self.ctx = Ctx(self.client, cfg.get("email", ""), keys, cfg.get("tool", "litagent"), log,
                       pmc=bool(cfg.get("pubmed_pmc", True)))
        self.sources = make_sources(self.ctx)
        self.registry = SiteRegistry.load(cfg.get("custom_sites") or DEFAULT_FILE, log)
        self._attach_custom_sites()
        self.downloader = Downloader(self.ctx, self.lib, self.policy)
        self.started = datetime.datetime.now()
        self.run_id = self.started.strftime("%Y%m%d-%H%M%S")
        self.runlog: list[dict] = []

    # ------------------------------------------------------------ пользовательские сайты
    def _attach_custom_sites(self) -> None:
        """Подключает сайты из custom_sites.json: адаптер, пауза по хосту и описание для контекста LLM."""
        for site in self.registry.sites:
            if site.name in self.sources or site.name in self.policy.forbidden_sources:
                self.log(f"  [{site.name}] пользовательский сайт пропущен: имя занято встроенным или запрещённым источником")
                continue
            errs = site.validate(self.policy)
            if errs:
                self.log(f"  [{site.name}] пользовательский сайт пропущен: {errs[0]}")
                continue
            src = CustomSource(self.ctx, site)
            self.sources[site.name] = src
            self.policy.intervals[site.host] = max(self.policy.intervals.get(site.host, 0.0), float(site.min_interval))
            self.policy.sources_info[site.name] = src.llm_info()

    # ------------------------------------------------------------ поиск
    def check_sources(self, names: list[str]) -> list[str]:
        ok = []
        for n in names:
            if n in self.policy.forbidden_sources:
                self.log(f"  [{n}] недоступен: {self.policy.forbidden_sources[n]}")
            elif n not in self.sources:
                self.log(f"  [{n}] неизвестный источник; доступны: {', '.join(self.sources)}")
            else:
                ok.append(n)
        return ok

    def build_queries(self, specs: list[SearchSpec], names: list[str]) -> list[dict]:
        out = []
        for s in specs:
            for n in names:
                src = self.sources[n]
                q = {"spec": s.describe(), "source": n, "query": src.query_for(s)}
                if n == "pubmed" and self.ctx.pmc:
                    q["extra"] = "PMC (db=pmc): " + src.build_pmc_query(s)
                out.append(q)
        return out

    def search(self, specs: list[SearchSpec], names: list[str], limit: int) -> list[Record]:
        allrecs: list[Record] = []
        disabled: dict[str, str] = {}
        for spec in specs:
            for name in names:
                if name in disabled:
                    continue
                src = self.sources[name]
                ok, why = src.available()
                if not ok:
                    disabled[name] = why
                    self.runlog.append({"source": name, "skipped": why})
                    self.log(f"  [{name}] пропуск: {why}")
                    continue
                entry = {"source": name, "spec": spec.describe()}
                try:
                    try:
                        recs, info = src.search(spec, limit)
                    except HttpError as e:
                        if not spec.raw.get(name):
                            raise
                        self.log(f"  [{name}] сырой запрос не сработал ({str(e)[:80]}), повтор со стандартным")
                        spec2 = dataclasses.replace(spec, raw={k: v for k, v in spec.raw.items() if k != name})
                        recs, info = src.search(spec2, limit)
                        info["raw_failed"] = str(e)[:200]
                except Blocked as e:
                    disabled[name] = str(e)
                    entry["error"] = str(e)
                    self.log(f"  [{name}] СТОП: {e}")
                except Exception as e:  # noqa: BLE001 — один источник не должен валить весь запуск
                    entry["error"] = f"{type(e).__name__}: {e}"
                    self.log(f"  [{name}] ошибка: {entry['error'][:160]}")
                else:
                    for r in recs:
                        r.sources = [name]
                    kept = [r for r in recs if spec.local_ok(r, strict=not src.boolean)]
                    entry.update(info, fetched=len(recs), kept=len(kept))
                    allrecs += kept
                    tr = f" | translation: {info['translation']}" if info.get("translation") else ""
                    self.log(f"  [{name}] найдено {info.get('found')}, взято {len(recs)}, после фильтра {len(kept)}{tr}")
                self.runlog.append(entry)
        return dedupe(allrecs)

    @staticmethod
    def rank(records: list[Record]) -> list[Record]:
        return sorted(records, key=lambda r: (-(r.score if r.score is not None else 5.0),
                                              -len(r.sources), -(r.cited_by or 0)))

    # ------------------------------------------------------------ скачивание
    def download(self, records: list[Record], max_downloads: int, oa_only: bool = False, pdf_only: bool = False) -> dict:
        """pdf_only: цель — max_downloads статей с ДЕЙСТВИТЕЛЬНО скачанным PDF. Записи без PDF получают status 'no_pdf'
        (или 'failed…'), XML не берётся. Сначала проверяются записи, у которых PDF вероятен (ссылка из источника, PMCID, is_oa)."""
        counts = {"pdf": 0, "xml": 0, "exists": 0, "no_oa": 0, "no_pdf": 0, "failed": 0, "skipped": 0}
        oa_only = oa_only or pdf_only
        if pdf_only:                                      # стабильно: внутри групп порядок по релевантности сохраняется
            likely = lambda r: 0 if (r.pdf_urls or r.pmcid or r.is_oa) else 1          # noqa: E731
            records = sorted(records, key=likely)
        cap = max_downloads * (10 if pdf_only else 4)     # без PDF отсеивается много — попыток нужно больше
        queue = iter(records)
        got = attempts = done_n = 0
        pending: dict = {}

        def next_record():
            for r in queue:
                if oa_only and r.is_oa is False:
                    r.status = "skip:not_oa"
                    counts["skipped"] += 1
                    continue
                return r
            return None

        # Скользящее окно: в работе не больше self.workers записей, и «успешные + в работе» не превышают max_downloads —
        # поэтому лишнего не скачивается. Паузы по хостам и одно соединение на хост обеспечивает PoliteClient.
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="dl") as ex:
            def fill():
                nonlocal attempts
                while len(pending) < self.workers and got + len(pending) < max_downloads and attempts < cap:
                    r = next_record()
                    if r is None:
                        return
                    attempts += 1
                    pending[ex.submit(self.downloader.fetch, r, pdf_only)] = r
            fill()
            while pending:
                finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in finished:                       # учёт и запись в индекс — только в главном потоке
                    r = pending.pop(fut)
                    try:
                        st = fut.result()
                    except Exception as e:  # noqa: BLE001  — сбой одной статьи не должен ронять весь запуск
                        st = f"failed: {type(e).__name__}: {str(e)[:60]}"
                    r.status = st
                    counts[st if st in counts else "failed"] += 1
                    if st in ("pdf", "xml", "exists"):
                        got += 1
                        self.lib.register(r)
                    done_n += 1
                    self.log(f"  [{done_n}] {st[:60]:<12} {r.short()}")
                fill()
        return counts

    # ------------------------------------------------------------ итог
    def finish(self, mode: str, specs: list[SearchSpec], records: list[Record], extra: dict | None = None) -> Path:
        self.lib.save()
        meta = {"run_id": self.run_id, "mode": mode, "started": self.started.isoformat(timespec="seconds"),
                "queries": self.runlog, "specs": [dataclasses.asdict(s) for s in specs],
                "http": self.client.stats, "total_unique": len(records), **(extra or {})}
        return self.lib.write_run(self.run_id, meta, records)
