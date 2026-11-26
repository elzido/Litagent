"""Легальное скачивание: только открытые копии (PDF от источников, Unpaywall, Europe PMC XML).
Не PDF (лендинг, paywall) отбрасывается; HTML-страницы не разбираются; 403 — стоп для хоста."""
from __future__ import annotations

import datetime
import os
import re
import shutil
import threading
import tarfile
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from urllib.parse import quote, urlparse

from .http import Blocked, HttpError, PoliteClient
from .models import Record
from .policy import Policy
from .sources import Ctx
from .store import Library


PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"       # официальный OA Web Service PMC
PMC_FTP_HTTPS = "https://ftp.ncbi.nlm.nih.gov/"
PMC_TGZ_MAX = 80 * 1024 * 1024                                       # пакет статьи больше — не качаем
EUPMC_SUPP = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/supplementaryFiles"
EUPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SUPP_MAX_FILE = 40 * 1024 * 1024                                     # один файл доп. материалов
SUPP_MAX_TOTAL = 200 * 1024 * 1024                                   # все доп. материалы статьи
SUPP_MAX_FILES = 40
_SUPP = re.compile(r"(supp|media|appendix|additional|mmc\d|moesm|dataset)", re.I)


def _pick_main_pdf(tf, max_pdf: int, max_members: int, max_total: int):
    """Элемент tarfile с основным PDF пакета (не приложение, не разметка)."""
    best, total = None, 0
    for i, m in enumerate(tf):
        total += max(m.size, 0)
        if i >= max_members or total > max_total:
            break
        base = Path(m.name).name
        if m.isfile() and base.lower().endswith(".pdf") and not _SUPP.search(base) and 1000 < m.size <= max_pdf:
            if best is None or m.size > best.size:
                best = m
    return best


def extract_main_pdf(tgz: Path, dest: Path, max_pdf: int = 80 * 1024 * 1024, max_members: int = 500,
                     max_total: int = 1 << 30, name_out: list | None = None) -> bool:
    """Достаёт из пакета PMC основной PDF статьи (не приложения). Ничего не распаковывается на диск по именам
    из архива: читается один выбранный элемент и пишется в dest, поэтому path traversal невозможен.
    name_out: если передан список, в него добавляется имя выбранного элемента (чтобы не класть его в доп. материалы)."""
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            best = _pick_main_pdf(tf, max_pdf, max_members, max_total)
            src = tf.extractfile(best) if best is not None else None
            if src is None:
                return False
            if name_out is not None:
                name_out.append(Path(best.name).name)
            with src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
        return True
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        return False


def main_pdf_name(tgz: Path, max_pdf: int = 80 * 1024 * 1024) -> str | None:
    """Имя основного PDF в пакете, без распаковки (чтобы исключить его из доп. материалов)."""
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            best = _pick_main_pdf(tf, max_pdf, 500, 1 << 30)
            return Path(best.name).name if best is not None else None
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        return None


def _safe_member_name(name: str) -> str | None:
    """Имя файла из архива -> безопасное плоское имя (без путей, .., абсолютных)."""
    n = Path(str(name).replace("\\", "/")).name.strip()
    n = re.sub(r"[^\w.\- ]+", "_", n).strip("._ ")
    return n[:120] or None


def _supp_candidate_name(base: str) -> bool:
    """Основной PDF статьи и служебные файлы (.nxml, .xml, .txt-разметка) в доп. материалы не берём."""
    low = base.lower()
    if low.endswith(".nxml") or low.endswith(".sgm") or low.endswith(".xml"):
        return False
    return True


def _reserve(dest_dir: Path, base: str) -> Path:
    """Уникальный путь в папке: name.pdf, name_2.pdf, … (не перезаписываем существующее)."""
    p = dest_dir / base
    if not p.exists():
        return p
    stem, suf = Path(base).stem, Path(base).suffix
    for i in range(2, 1000):
        p = dest_dir / f"{stem}_{i}{suf}"
        if not p.exists():
            return p
    return dest_dir / f"{stem}_{datetime.datetime.now().strftime('%H%M%S%f')}{suf}"


def extract_supp_from_tgz(tgz: Path, dest_dir: Path, main_pdf_name: str | None = None,
                          max_file: int = SUPP_MAX_FILE, max_total: int = SUPP_MAX_TOTAL,
                          max_files: int = SUPP_MAX_FILES) -> list[dict]:
    """Доп. материалы из пакета PMC: всё, кроме основного PDF и разметки.
    Пишем только по безопасным плоским именам (path traversal невозможен)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    main_low = (main_pdf_name or "").lower()
    out, total = [], 0
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            for i, m in enumerate(tf):
                if i >= max_files * 4 or len(out) >= max_files or total > max_total:
                    break
                if not m.isfile() or m.size <= 0 or m.size > max_file:
                    continue
                base = _safe_member_name(m.name)
                if not base or not _supp_candidate_name(base) or base.lower() == main_low:
                    continue
                src = tf.extractfile(m)
                if src is None:
                    continue
                target = _reserve(dest_dir, base)
                with src, target.open("wb") as f:
                    shutil.copyfileobj(src, f, 256 * 1024)
                total += m.size
                out.append({"file": target.name, "name": target.name, "size": m.size, "origin": "pmc-package"})
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        return out
    return out


def extract_supp_from_zip(zp: Path, dest_dir: Path,
                          max_file: int = SUPP_MAX_FILE, max_total: int = SUPP_MAX_TOTAL,
                          max_files: int = SUPP_MAX_FILES) -> list[dict]:
    """Доп. материалы из zip (Europe PMC supplementaryFiles)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out, total = [], 0
    try:
        with zipfile.ZipFile(zp) as zf:
            for i, info in enumerate(zf.infolist()):
                if i >= max_files * 4 or len(out) >= max_files or total > max_total:
                    break
                if info.is_dir() or info.file_size <= 0 or info.file_size > max_file:
                    continue
                base = _safe_member_name(info.filename)
                if not base or not _supp_candidate_name(base):
                    continue
                target = _reserve(dest_dir, base)
                with zf.open(info) as src, target.open("wb") as f:
                    shutil.copyfileobj(src, f, 256 * 1024)
                total += info.file_size
                out.append({"file": target.name, "name": target.name, "size": info.file_size, "origin": "europepmc"})
    except (zipfile.BadZipFile, OSError, EOFError):
        return out
    return out


class Downloader:
    def __init__(self, ctx: Ctx, lib: Library, policy: Policy):
        self.ctx, self.lib, self.policy = ctx, lib, policy
        self.client: PoliteClient = ctx.client
        self._pmc_cache: dict[str, list[tuple[str, str]]] = {}   # PMCID -> [('pdf'|'tgz', url)]: один пакетный oa.fcgi на всех
        self._pmc_lock = threading.Lock()

    def _pmc_oa_parse(self, ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """PMC OA Web Service принимает список id через запятую — получаем ссылки для всех разом."""
        out: dict[str, list[tuple[str, str]]] = {i: [] for i in ids}
        try:
            r = self.client.get(PMC_OA, {"id": ",".join(ids)})
            if r.status != 200:
                return out
            root = ET.fromstring(r.text())
        except (HttpError, Blocked, ET.ParseError, ValueError):
            return out
        if root.find(".//error") is not None:                    # idIsNotOpenAccess и т.п.
            return out
        for rec in root.iter("record"):
            rid = (rec.get("id") or "").strip().upper()
            if rid and not rid.startswith("PMC"):
                rid = "PMC" + rid
            if rid not in out or rec.get("retracted") == "yes":  # отозванная — как без ссылок
                continue
            links = {}
            for ln in rec.findall("link"):
                href = (ln.get("href") or "").replace("ftp://ftp.ncbi.nlm.nih.gov/", PMC_FTP_HTTPS, 1)
                if ln.get("format") in ("pdf", "tgz") and href.startswith(PMC_FTP_HTTPS):
                    links.setdefault(ln.get("format"), href)
            out[rid] = [(k, links[k]) for k in ("pdf", "tgz") if k in links]
        return out

    def prefetch_pmc_oa(self, pmcids: list[str]) -> int:
        """Пакетно (по 60 id) заполнить кэш PMC OA для списка записей — вместо запроса oa.fcgi на каждую статью."""
        ids = [p for p in dict.fromkeys(pmcids) if p]
        with self._pmc_lock:
            todo = [i for i in ids if i not in self._pmc_cache]
        n = 0
        for j in range(0, len(todo), 60):
            chunk = todo[j:j + 60]
            res = self._pmc_oa_parse(chunk)
            with self._pmc_lock:
                self._pmc_cache.update(res)
            n += len(chunk)
        return n

    # --- кандидаты в порядке приоритета (лениво: Unpaywall только если прямые ссылки не сработали)
    def _unpaywall(self, doi: str) -> list[str]:
        if not self.ctx.email or not doi:
            return []
        try:
            r = self.client.get(f"https://api.unpaywall.org/v2/{quote(doi, safe='/')}", {"email": self.ctx.email})
            if r.status != 200:
                return []
            d = r.json()
        except (HttpError, Blocked, ValueError):
            return []
        urls = []
        locs = ([d["best_oa_location"]] if d.get("best_oa_location") else []) + (d.get("oa_locations") or [])
        for loc in locs:
            u = loc.get("url_for_pdf")
            if u and u not in urls:
                urls.append(u)
        return urls

    def _pmc_oa(self, pmcid: str) -> list[tuple[str, str]]:
        """PMC OA Web Service: [('pdf', url), ('tgz', url)] только для статей из Open Access Subset.
        Ссылки приходят как ftp://ftp.ncbi.nlm.nih.gov/..., берём их же по https. Сайт PMC не затрагивается.
        Ответ кэшируется на весь запуск; prefetch_pmc_oa заполняет его пакетно по списку id."""
        with self._pmc_lock:
            if pmcid in self._pmc_cache:
                return list(self._pmc_cache[pmcid])
        res = self._pmc_oa_parse([pmcid]).get(pmcid, [])
        with self._pmc_lock:
            self._pmc_cache[pmcid] = res
        return list(res)

    def prefetch_supp_flags(self, pmcids: list[str]) -> dict[str, bool]:
        """Один пакетный lite-запрос Europe PMC (PMCID:a OR PMCID:b …) на признак hasSuppl для всех статей."""
        out: dict[str, bool] = {}
        ids = [p for p in dict.fromkeys(pmcids) if p]
        for j in range(0, len(ids), 50):
            chunk = ids[j:j + 50]
            try:
                r = self.client.get(EUPMC_SEARCH, {"query": " OR ".join(f"PMCID:{p}" for p in chunk),
                                                   "format": "json", "resultType": "lite",
                                                   "pageSize": str(len(chunk))})
                if r.status != 200:
                    continue
                res = (r.json().get("resultList") or {}).get("result") or []
            except (HttpError, Blocked, ValueError):
                continue
            for item in res:
                pid = (item.get("pmcid") or "").upper()
                if pid:
                    out[pid] = item.get("hasSuppl") == "Y"
        return out

    def candidates(self, rec: Record):
        seen = set()

        def fresh(u):
            if u and u not in seen:
                seen.add(u)
                return True
            return False
        for u in rec.pdf_urls:
            if fresh(u):
                yield "pdf", u, "source"
        if rec.pmcid and getattr(self.ctx, "pmc", True):
            for kind, u in self._pmc_oa(rec.pmcid):                # официальный канал PMC (PDF или пакет)
                if fresh(u):
                    yield kind, u, "pmc-oa"
        if rec.doi and rec.is_oa is not False:
            for u in self._unpaywall(rec.doi):
                if fresh(u):
                    yield "pdf", u, "unpaywall"
        xml = rec.xml_url or (f"https://www.ebi.ac.uk/europepmc/webservices/rest/{rec.pmcid}/fullTextXML"
                              if rec.pmcid and rec.is_oa is not False else None)
        if fresh(xml):
            yield "xml", xml, "europepmc"

    @staticmethod
    def _valid(path: Path, kind: str) -> bool:
        try:
            if path.stat().st_size < 1000:
                return False
            with path.open("rb") as f:
                head = f.read(1024)
        except OSError:
            return False
        if kind == "pdf":
            return b"%PDF" in head
        h = head.lstrip()
        return h.startswith((b"<?xml", b"<article", b"<pmc-articleset"))

    def fetch(self, rec: Record, pdf_only: bool = False, supplements: bool = False) -> str:
        """pdf_only: нужен именно PDF — XML-копии (Europe PMC) не запрашиваются, а ранее сохранённый XML за «есть файл» не считается.
        supplements: из пакетов PMC дополнительно доставать доп. материалы (supp/<статья>/)."""
        existing = self.lib.find(rec)
        if existing and (not pdf_only or str(existing["file"]).lower().endswith(".pdf")):
            rec.file = existing["file"]
            rec.supp = list(existing.get("supp") or [])
            if existing.get("has_supp") is not None:
                rec.has_supp = bool(existing["has_supp"])
            if supplements and rec.pmcid and not rec.supp:
                try:
                    self.fetch_supplements(rec)
                except Exception:  # noqa: BLE001 — доп. материалы не должны ронять скачивание
                    pass
            return "exists"
        tried, errs = 0, []
        for kind, url, origin in self.candidates(rec):
            if pdf_only and kind == "xml":
                continue
            url = self.policy.rewrite(url)
            ok, why = self.policy.url_allowed(url)
            if not ok:
                errs.append(why)
                continue
            is_tgz = kind == "tgz"
            dest = self.lib.path_for(rec, ".xml" if kind == "xml" else ".pdf")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + (".tgz.part" if is_tgz else ".part"))
            tried += 1
            try:
                resp = self.client.download(url, tmp, max_bytes=PMC_TGZ_MAX if is_tgz else 120 * 1024 * 1024, headers={
                    "Accept": "application/pdf,*/*;q=0.5" if kind == "pdf" else
                              "application/gzip,*/*;q=0.5" if is_tgz else "application/xml,*/*;q=0.5"})
            except Blocked as e:
                errs.append(f"blocked {self.policy.host(url)}")
                tmp.unlink(missing_ok=True)
                continue
            except HttpError as e:
                errs.append(str(e)[:80])
                tmp.unlink(missing_ok=True)
                continue
            if resp.status != 200:
                errs.append(f"HTTP {resp.status} ({origin})")
                tmp.unlink(missing_ok=True)
                continue
            tgz = None
            pkg_main: list[str] = []
            if is_tgz:                                            # пакет PMC -> основной PDF
                pdf_tmp = dest.with_name(dest.name + ".part")
                ok = extract_main_pdf(tmp, pdf_tmp, name_out=pkg_main)
                if not ok:
                    tmp.unlink(missing_ok=True)
                    errs.append("в пакете PMC нет основного PDF")
                    pdf_tmp.unlink(missing_ok=True)
                    continue
                tgz, tmp, kind = tmp, pdf_tmp, "pdf"
            if not self._valid(tmp, kind):
                errs.append(f"не {kind.upper()} ({origin}: лендинг или paywall)")
                tmp.unlink(missing_ok=True)
                if tgz:
                    tgz.unlink(missing_ok=True)
                continue
            os.replace(tmp, dest)
            rec.file = str(dest.relative_to(self.lib.root))
            if tgz:
                if supplements:                                    # доп. материалы из того же пакета
                    if self._supp_from_tgz(rec, tgz, pkg_main[0] if pkg_main else None):
                        rec.has_supp = True
                tgz.unlink(missing_ok=True)
            elif supplements and (rec.pmcid or rec.supp_urls):
                try:
                    if self.fetch_supplements(rec):
                        rec.has_supp = True
                except Exception:  # noqa: BLE001 — доп. материалы не должны ронять скачивание
                    pass
            return kind
        if tried == 0 and not errs:
            return "no_pdf" if pdf_only else "no_oa"
        return "failed: " + "; ".join(errs[:3])

    # ------------------------------------------------------------ дополнительные материалы
    def _register_supp(self, rec: Record, p: Path, origin: str, have: set | None = None) -> dict | None:
        """Добавляет файл доп. материалов (уже на диске) в rec.supp; None — если уже учтён."""
        if have is not None:
            if p.resolve() in have or not p.is_file():
                return None
            have.add(p.resolve())
        entry = {"file": str(p.relative_to(self.lib.root)), "name": p.name, "size": p.stat().st_size, "origin": origin,
                 "added": datetime.datetime.now().isoformat(timespec="seconds")}
        rec.supp = [s for s in rec.supp if s.get("file") != entry["file"]] + [entry]
        return entry

    def check_supplements(self, rec: Record) -> bool | None:
        """Есть ли у статьи доп. материалы: сначала ссылки, которые дал источник (в т.ч. свой сайт),
        иначе — дешёвый lite-запрос к Europe PMC (по PMCID, а если его нет — по DOI; ответ в HTTP-кэше)
        + признак OA-пакета PMC. None — проверить нельзя."""
        if rec.supp_urls:
            rec.has_supp = True
            return True
        if rec.has_supp is not None:
            return bool(rec.has_supp)
        query = f"PMCID:{rec.pmcid}" if rec.pmcid else (f'DOI:"{rec.doi}"' if rec.doi else "")
        if not query:
            return None
        try:
            r = self.client.get(EUPMC_SEARCH, {"query": query, "format": "json",
                                               "resultType": "lite", "pageSize": 1})
            if r.status == 200:
                res = (r.json().get("resultList") or {}).get("result") or []
                if res:
                    pid = (res[0].get("pmcid") or "").upper()
                    if pid.startswith("PMC") and not rec.pmcid:
                        rec.pmcid = pid                    # Europe PMC дал PMCID — им и будем качать пакет
                    rec.has_supp = res[0].get("hasSuppl") == "Y"
                    return bool(rec.has_supp)
        except (HttpError, Blocked, ValueError):
            pass
        if rec.pmcid:
            for kind, _ in self._pmc_oa(rec.pmcid):
                if kind == "tgz":
                    rec.has_supp = True
                    return True
        return rec.has_supp

    def _supp_from_tgz(self, rec: Record, tgz: Path, main_name: str | None) -> int:
        """Индекс не сохраняем: вызывающий поток (agent.download) запишет rec.supp сам."""
        dest_dir = self.lib.supp_dir_for(rec.file)
        have = {(self.lib.root / s["file"]).resolve() for s in rec.supp}
        added = 0
        for item in extract_supp_from_tgz(tgz, dest_dir, main_pdf_name=main_name):
            if self._register_supp(rec, dest_dir / item["file"], "pmc-package", have):
                added += 1
        return added

    def fetch_supplements(self, rec: Record) -> int:
        """Докачать доп. материалы для уже скачанной статьи: ссылки самого источника (rec.supp_urls),
        пакет PMC (если есть PMCID) и zip Europe PMC. Возвращает число новых файлов."""
        if not rec.file:
            return 0
        dest_dir = self.lib.supp_dir_for(rec.file)
        have = {(self.lib.root / s["file"]).resolve() for s in rec.supp}
        added = 0

        for url in (rec.supp_urls or [])[:SUPP_MAX_FILES]:            # сначала то, что дал источник (в т.ч. свой сайт)
            url = self.policy.rewrite(url)
            ok, why = self.policy.url_allowed(url)
            if not ok:
                continue                                              # запрещённый политикой адрес источника пропускается
            dest_dir.mkdir(parents=True, exist_ok=True)
            name = self.lib.supp_name(Path(urlparse(url).path).name or "supplement")
            tmp = dest_dir / (name + ".part")
            try:
                resp = self.client.download(url, tmp, max_bytes=SUPP_MAX_TOTAL,
                                           headers={"Accept": "application/zip,application/gzip,*/*;q=0.5"})
            except (Blocked, HttpError):
                tmp.unlink(missing_ok=True)
                continue
            if resp.status != 200 or tmp.stat().st_size <= 0:
                tmp.unlink(missing_ok=True)
                continue
            if zipfile.is_zipfile(tmp) or tarfile.is_tarfile(tmp):
                items = (extract_supp_from_zip(tmp, dest_dir) if zipfile.is_zipfile(tmp)
                         else extract_supp_from_tgz(tmp, dest_dir))
                prefix = "source-zip" if zipfile.is_zipfile(tmp) else "source-tgz"
                got = sum(1 for item in items
                          if self._register_supp(rec, dest_dir / item["file"], prefix + ":" + self.policy.host(url), have))
                if got:                                                 # архив распакован — служебный файл не нужен
                    tmp.unlink(missing_ok=True)
                    added += got
                    continue
            final = _reserve(dest_dir, name)                           # одиночный файл (xlsx, csv, pdf-приложение…)
            os.replace(tmp, final)
            if self._register_supp(rec, final, "source:" + self.policy.host(url), have):
                added += 1
        if not rec.pmcid:
            if added:
                self.lib.register(rec)
                self.lib.save()
            return added

        if rec.pmcid and getattr(self.ctx, "pmc", True):
            for kind, url in self._pmc_oa(rec.pmcid):
                if kind != "tgz":
                    continue
                dest_dir.mkdir(parents=True, exist_ok=True)
                tmp = dest_dir.parent / (dest_dir.name + ".tgz.part")
                try:
                    resp = self.client.download(url, tmp, max_bytes=PMC_TGZ_MAX,
                                                headers={"Accept": "application/gzip,*/*;q=0.5"})
                except (Blocked, HttpError):
                    tmp.unlink(missing_ok=True)
                    break
                if resp.status != 200:
                    tmp.unlink(missing_ok=True)
                    break
                main_name = main_pdf_name(tmp)                         # основной PDF пакета в доп. материалы не идёт
                for item in extract_supp_from_tgz(tmp, dest_dir, main_pdf_name=main_name):
                    if self._register_supp(rec, dest_dir / item["file"], "pmc-package", have):
                        added += 1
                tmp.unlink(missing_ok=True)
                break
        if rec.pmcid:
            dest_dir.mkdir(parents=True, exist_ok=True)
            tmp = dest_dir / "_eupmc.zip.part"
            try:
                resp = self.client.download(EUPMC_SUPP.format(rec.pmcid), tmp, max_bytes=SUPP_MAX_TOTAL,
                                            headers={"Accept": "application/zip,*/*;q=0.5"})
            except (Blocked, HttpError):
                resp = None
            if resp is not None and resp.status == 200 and zipfile.is_zipfile(tmp):
                for item in extract_supp_from_zip(tmp, dest_dir):
                    if self._register_supp(rec, dest_dir / item["file"], "europepmc", have):
                        added += 1
            tmp.unlink(missing_ok=True)
        return added

    def fetch_supp_url(self, rec: Record, url: str) -> dict:
        """Ручная докачка одного файла доп. материалов по URL (только разрешённые политикой адреса)."""
        if not rec.file:
            raise HttpError("у статьи ещё нет файла в библиотеке")
        url = self.policy.rewrite(url)
        ok, why = self.policy.url_allowed(url)
        if not ok:
            raise HttpError(why)
        dest_dir = self.lib.supp_dir_for(rec.file)
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = Path(urlparse(url).path).name or "supplement"
        tmp = _reserve(dest_dir, self.lib.supp_name(name))
        part = tmp.with_name(tmp.name + ".part")
        try:
            resp = self.client.download(url, part, max_bytes=SUPP_MAX_FILE)
        except (Blocked, HttpError):
            part.unlink(missing_ok=True)
            raise
        if resp.status != 200 or part.stat().st_size <= 0:
            part.unlink(missing_ok=True)
            raise HttpError(f"HTTP {resp.status}" if resp.status != 200 else "пустой файл")
        os.replace(part, tmp)
        entry = self._register_supp(rec, tmp, "manual:" + self.policy.host(url))
        rec.has_supp = True
        return entry
