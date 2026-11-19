"""Легальное скачивание: только открытые копии (PDF от источников, Unpaywall, Europe PMC XML).
Не PDF (лендинг, paywall) отбрасывается; HTML-страницы не разбираются; 403 — стоп для хоста."""
from __future__ import annotations

import os
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from urllib.parse import quote

from .http import Blocked, HttpError, PoliteClient
from .models import Record
from .policy import Policy
from .sources import Ctx
from .store import Library


PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"       # официальный OA Web Service PMC
PMC_FTP_HTTPS = "https://ftp.ncbi.nlm.nih.gov/"
PMC_TGZ_MAX = 80 * 1024 * 1024                                       # пакет статьи больше — не качаем
_SUPP = re.compile(r"(supp|media|appendix|additional|mmc\d|moesm|dataset)", re.I)


def extract_main_pdf(tgz: Path, dest: Path, max_pdf: int = 80 * 1024 * 1024, max_members: int = 500,
                     max_total: int = 1 << 30) -> bool:
    """Достаёт из пакета PMC основной PDF статьи (не приложения). Ничего не распаковывается на диск по именам
    из архива: читается один выбранный элемент и пишется в dest, поэтому path traversal невозможен."""
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            best, total = None, 0
            for i, m in enumerate(tf):
                total += max(m.size, 0)
                if i >= max_members or total > max_total:
                    break
                base = Path(m.name).name
                if m.isfile() and base.lower().endswith(".pdf") and not _SUPP.search(base) and 1000 < m.size <= max_pdf:
                    if best is None or m.size > best.size:
                        best = m
            src = tf.extractfile(best) if best is not None else None
            if src is None:
                return False
            with src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
        return True
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        return False


class Downloader:
    def __init__(self, ctx: Ctx, lib: Library, policy: Policy):
        self.ctx, self.lib, self.policy = ctx, lib, policy
        self.client: PoliteClient = ctx.client

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
        Ссылки приходят как ftp://ftp.ncbi.nlm.nih.gov/..., берём их же по https. Сайт PMC не затрагивается."""
        try:
            r = self.client.get(PMC_OA, {"id": pmcid})
            if r.status != 200:
                return []
            root = ET.fromstring(r.text())
        except (HttpError, Blocked, ET.ParseError, ValueError):
            return []
        if root.find(".//error") is not None:                    # idIsNotOpenAccess и т.п.
            return []
        out = {}
        for rec in root.iter("record"):
            if rec.get("retracted") == "yes":
                return []
            for ln in rec.findall("link"):
                href = (ln.get("href") or "").replace("ftp://ftp.ncbi.nlm.nih.gov/", PMC_FTP_HTTPS, 1)
                if ln.get("format") in ("pdf", "tgz") and href.startswith(PMC_FTP_HTTPS):
                    out.setdefault(ln.get("format"), href)
        return [(k, out[k]) for k in ("pdf", "tgz") if k in out]

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

    def fetch(self, rec: Record, pdf_only: bool = False) -> str:
        """pdf_only: нужен именно PDF — XML-копии (Europe PMC) не запрашиваются, а ранее сохранённый XML за «есть файл» не считается."""
        existing = self.lib.find(rec)
        if existing and (not pdf_only or str(existing["file"]).lower().endswith(".pdf")):
            rec.file = existing["file"]
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
            if is_tgz:                                            # пакет PMC -> основной PDF
                pdf_tmp = dest.with_name(dest.name + ".part")
                ok = extract_main_pdf(tmp, pdf_tmp)
                tmp.unlink(missing_ok=True)
                if not ok:
                    errs.append("в пакете PMC нет основного PDF")
                    pdf_tmp.unlink(missing_ok=True)
                    continue
                tmp, kind = pdf_tmp, "pdf"
            if not self._valid(tmp, kind):
                errs.append(f"не {kind.upper()} ({origin}: лендинг или paywall)")
                tmp.unlink(missing_ok=True)
                continue
            os.replace(tmp, dest)
            rec.file = str(dest.relative_to(self.lib.root))
            return kind
        if tried == 0 and not errs:
            return "no_pdf" if pdf_only else "no_oa"
        return "failed: " + "; ".join(errs[:3])
