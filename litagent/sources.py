"""Адаптеры источников. Каждый: build_query(spec) -> строка, search(spec, limit) -> (records, info).
Пишутся по документации сервисов и обзору; проверить вживую можно командой `check`."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote

from .http import HttpError, PoliteClient
from .models import (Record, SearchSpec, current_year, norm_arxiv, norm_doi, strip_tags)


@dataclass
class Ctx:
    client: PoliteClient
    email: str = ""
    keys: dict = None
    tool: str = "litagent"
    log: Callable = print
    pmc: bool = True              # PubMed дополнительно ищет в PMC (db=pmc) и скачивает OA-PDF оттуда

    def __post_init__(self):
        self.keys = self.keys or {}


# ---------------------------------------------------------------- помощники запроса
def _q(t: str) -> str:
    t = t.replace('"', "").strip()
    star = t.endswith("*")
    core = t.rstrip("*")
    if re.search(r"[^\w*]", core, re.U):
        return f'"{core}"'
    return core + ("*" if star else "")


def bool_expr(spec: SearchSpec, fmt, op_and=" AND ", op_or=" OR ", op_not=" NOT ") -> str:
    parts = []
    for g in spec.groups:
        items = [fmt(t) for t in g]
        parts.append(items[0] if len(items) == 1 else "(" + op_or.join(items) + ")")
    s = op_and.join(parts)
    if spec.exclude:
        ex = [fmt(t) for t in spec.exclude]
        s = f"({s}){op_not}" + (ex[0] if len(ex) == 1 else "(" + op_or.join(ex) + ")")
    return s


def flat_terms(spec: SearchSpec) -> str:
    return " ".join(t for g in spec.groups for t in g)


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _year_from_text(s: str | None):
    m = re.search(r"(19|20)\d\d", s or "")
    return int(m.group(0)) if m else None


class Source:
    name = ""
    boolean = True          # понимает ли источник булевы операторы (иначе локальная доп. фильтрация)
    key_name: str | None = None
    key_required = False

    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    def key(self):
        return self.ctx.keys.get(self.key_name) if self.key_name else None

    def available(self) -> tuple[bool, str]:
        if self.key_required and not self.key():
            return False, f"не задан ключ ({self.key_name}); см. README"
        return True, ""

    def query_for(self, spec: SearchSpec) -> str:
        return spec.raw.get(self.name) or self.build_query(spec)

    def build_query(self, spec: SearchSpec) -> str:
        raise NotImplementedError

    def search(self, spec: SearchSpec, limit: int):
        raise NotImplementedError


# ---------------------------------------------------------------- PubMed
class PubMed(Source):
    name = "pubmed"
    key_name = "ncbi"
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def build_query(self, spec):
        tag = "ti" if spec.title_only else "tiab"
        expr = bool_expr(spec, lambda t: f"{_q(t)}[{tag}]")
        extra = []
        if spec.year_from or spec.year_to:
            extra.append(f"{spec.year_from or 1800}:{spec.year_to or 3000}[dp]")
        if spec.open_access:
            extra.append("free full text[sb]")
        if extra:
            expr = f"({expr}) AND " + " AND ".join(extra)
        return expr

    def _common(self):
        p = {"tool": self.ctx.tool, "email": self.ctx.email}
        if self.key():
            p["api_key"] = self.key()
        return p

    # --- PMC (db=pmc): тот же E-utilities, поиск по полнотекстовому архиву
    def build_pmc_query(self, spec, tagged: bool = True) -> str:
        """Сырые запросы (spec.raw) пишутся в синтаксисе PubMed и в PMC не отправляются.
        tagged=False — запасной вариант без тегов полей: годы тогда отфильтрует локальный фильтр."""
        if tagged:
            def fmt(t):
                q = _q(t)
                return f"{q}[Title]" if spec.title_only else f"({q}[Title] OR {q}[Abstract])"
        else:
            fmt = _q
        expr = bool_expr(spec, fmt)
        if tagged and (spec.year_from or spec.year_to):
            expr = f"({expr}) AND {spec.year_from or 1800}:{spec.year_to or 3000}[pdat]"
        return expr

    def _search_pmc(self, spec, limit):
        c = self.ctx.client
        for tagged in (True, False):
            term = self.build_pmc_query(spec, tagged)
            res = c.get(self.BASE + "esearch.fcgi", {**self._common(), "db": "pmc", "term": term,
                        "retmax": min(limit, 10000), "retmode": "json"}).json().get("esearchresult", {})
            if res.get("ERROR"):
                raise HttpError("PMC: " + str(res["ERROR"]))
            bad = (res.get("errorlist") or {}).get("fieldsnotfound")
            if bad and tagged:                      # PMC не знает какой-то тег поля -> повтор без тегов
                self.ctx.log(f"  [pubmed] PMC не знает поля {bad}; повтор запроса без тегов")
                continue
            break
        ids = res.get("idlist", [])[:limit]
        info = {"query": term, "found": _int(res.get("count")), "translation": res.get("querytranslation"),
                "tags": tagged}
        recs = []
        for i in range(0, len(ids), 100):
            r = c.get(self.BASE + "esummary.fcgi", {**self._common(), "db": "pmc",
                      "id": ",".join(ids[i:i + 100]), "retmode": "json"})
            recs += parse_pmc_summary(r.json())
        return recs, info

    def search(self, spec, limit):
        recs, info = self._search_pubmed(spec, limit)
        if self.ctx.pmc:
            try:
                pmc_recs, pinfo = self._search_pmc(spec, limit)
                recs += pmc_recs
                info["pmc"] = pinfo
                self.ctx.log(f"  [pubmed] PMC: найдено {pinfo['found']}, взято {len(pmc_recs)}"
                             + (f" | translation: {pinfo['translation']}" if pinfo.get("translation") else ""))
            except (HttpError, ValueError, KeyError, TypeError, AttributeError) as e:   # сбой PMC-части не роняет PubMed-часть
                info["pmc"] = {"error": str(e)[:200]}
                self.ctx.log(f"  [pubmed] PMC-часть не сработала: {str(e)[:120]}")
        return recs, info

    def _search_pubmed(self, spec, limit):
        term = self.query_for(spec)
        c = self.ctx.client
        r = c.get(self.BASE + "esearch.fcgi", {**self._common(), "db": "pubmed", "term": term,
                  "retmax": min(limit, 10000), "retmode": "json", "sort": "relevance"})
        res = r.json().get("esearchresult", {})
        if res.get("ERROR"):
            raise HttpError("PubMed: " + str(res["ERROR"]))
        ids = res.get("idlist", [])[:limit]
        info = {"query": term, "found": _int(res.get("count")), "translation": res.get("querytranslation")}
        recs = []
        for i in range(0, len(ids), 100):
            batch = ids[i:i + 100]
            x = c.get(self.BASE + "efetch.fcgi", {**self._common(), "db": "pubmed", "id": ",".join(batch),
                      "retmode": "xml", "rettype": "abstract"})
            recs += parse_pubmed(x.text())
        return recs, info


def parse_pmc_summary(data: dict) -> list[Record]:
    """esummary db=pmc (JSON). uid — числовая часть PMCID; DOI и PMID берутся из articleids, если есть."""
    res = (data or {}).get("result", {})
    out = []
    for uid in res.get("uids", []):
        d = res.get(str(uid)) or {}
        title = re.sub(r"<[^>]+>", "", d.get("title") or "").strip()
        if not title:
            continue
        ids = {a.get("idtype"): str(a.get("value") or "") for a in d.get("articleids", []) if isinstance(a, dict)}
        doi = ids.get("doi") or None
        pmid = re.sub(r"\D", "", ids.get("pmid", "")) or None
        authors = [a["name"] for a in d.get("authors", []) if isinstance(a, dict) and a.get("name")]
        out.append(Record(title=title, authors=authors, year=_year_from_text(d.get("pubdate") or d.get("epubdate")),
                          doi=doi, pmid=pmid, pmcid=f"PMC{uid}", venue=d.get("fulljournalname") or d.get("source") or "",
                          is_oa=True, url=f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{uid}/"))
    return out


def parse_pubmed(xml_text: str) -> list[Record]:
    out = []
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        mc = art.find("MedlineCitation")
        if mc is None:
            continue
        a = mc.find("Article")
        if a is None:
            continue
        title = "".join(a.find("ArticleTitle").itertext()).strip() if a.find("ArticleTitle") is not None else ""
        abstract = " ".join("".join(x.itertext()).strip() for x in a.findall("Abstract/AbstractText"))
        authors = []
        for au in a.findall("AuthorList/Author"):
            ln, ini = au.findtext("LastName"), au.findtext("Initials")
            if ln:
                authors.append(f"{ln} {ini}".strip())
            elif au.findtext("CollectiveName"):
                authors.append(au.findtext("CollectiveName"))
        year = _int(a.findtext("Journal/JournalIssue/PubDate/Year")) \
            or _year_from_text(a.findtext("Journal/JournalIssue/PubDate/MedlineDate")) \
            or _int(a.findtext("ArticleDate/Year"))
        doi = pmcid = None
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            t = aid.get("IdType")
            if t == "doi":
                doi = aid.text
            elif t == "pmc":
                pmcid = aid.text
        if not doi:
            for e in a.findall("ELocationID"):
                if e.get("EIdType") == "doi":
                    doi = e.text
        pmid = mc.findtext("PMID")
        out.append(Record(title=title, authors=authors, year=year, doi=doi, pmid=pmid, pmcid=pmcid,
                          abstract=abstract, venue=a.findtext("Journal/Title") or "",
                          url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None))
    return out


# ---------------------------------------------------------------- Europe PMC
class EuropePMC(Source):
    name = "europepmc"
    URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def build_query(self, spec):
        def fmt(t):
            q = _q(t)
            if spec.title_only:
                return f"TITLE:{q}"
            return f"(TITLE:{q} OR ABSTRACT:{q})"
        expr = bool_expr(spec, fmt)
        if spec.year_from or spec.year_to:
            expr = f"({expr}) AND PUB_YEAR:[{spec.year_from or 1800} TO {spec.year_to or current_year() + 1}]"
        if spec.open_access:
            expr += " AND OPEN_ACCESS:y"
        return expr

    def search(self, spec, limit):
        q = self.query_for(spec)
        params = {"query": q, "format": "json", "resultType": "core", "cursorMark": "*",
                  "pageSize": min(100, limit)}
        recs, found = [], None
        while len(recs) < limit:
            params["pageSize"] = min(100, limit - len(recs))
            d = self.ctx.client.get(self.URL, params).json()
            found = d.get("hitCount", found)
            res = d.get("resultList", {}).get("result", [])
            if not res:
                break
            for r in res:
                recs.append(self._rec(r))
            nxt = d.get("nextCursorMark")
            if not nxt or nxt == params["cursorMark"]:
                break
            params["cursorMark"] = nxt
        return recs[:limit], {"query": q, "found": found}

    @staticmethod
    def _rec(r) -> Record:
        pdfs = []
        for u in (r.get("fullTextUrlList") or {}).get("fullTextUrl", []):
            if u.get("documentStyle") == "pdf" and u.get("availability") in ("Open access", "Free"):
                pdfs.append(u.get("url"))
        oa = r.get("isOpenAccess") == "Y"
        pmcid = r.get("pmcid")
        return Record(
            title=(r.get("title") or "").rstrip("."), year=_int(r.get("pubYear")),
            authors=[a.strip() for a in (r.get("authorString") or "").rstrip(".").split(",") if a.strip()],
            doi=r.get("doi"), pmid=r.get("pmid"),
            pmcid=pmcid, abstract=strip_tags(r.get("abstractText") or ""),
            venue=((r.get("journalInfo") or {}).get("journal") or {}).get("title", ""),
            cited_by=_int(r.get("citedByCount")), is_oa=oa, pdf_urls=[u for u in pdfs if u],
            xml_url=(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML" if (oa and pmcid) else None),
            url=f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}" if r.get("id") else None)


# ---------------------------------------------------------------- arXiv
ATOM = "{http://www.w3.org/2005/Atom}"
OS = "{http://a9.com/-/spec/opensearch/1.1/}"
AX = "{http://arxiv.org/schemas/atom}"


class Arxiv(Source):
    name = "arxiv"
    URL = "https://export.arxiv.org/api/query"

    def build_query(self, spec):
        field = "ti" if spec.title_only else "all"

        def fmt(t):
            t = t.replace('"', "").strip().rstrip("*")
            return f'{field}:"{t}"' if re.search(r"\W", t) else f"{field}:{t}"
        expr = bool_expr(spec, fmt, op_not=" ANDNOT ")
        if spec.arxiv_cats:
            expr += " AND (" + " OR ".join(f"cat:{c}" for c in spec.arxiv_cats) + ")"
        if spec.year_from or spec.year_to:
            a = f"{spec.year_from or 1991}01010000"
            b = f"{spec.year_to or current_year()}12312359"
            expr += f" AND submittedDate:[{a} TO {b}]"
        return expr

    def search(self, spec, limit):
        q = self.query_for(spec)
        recs, start, total = [], 0, None
        while len(recs) < limit:
            n = min(100, limit - len(recs))
            r = self.ctx.client.get(self.URL, {"search_query": q, "start": start, "max_results": n,
                                               "sortBy": "relevance", "sortOrder": "descending"})
            root = ET.fromstring(r.text())
            total = _int((root.findtext(OS + "totalResults") or "0"))
            entries = root.findall(ATOM + "entry")
            if not entries:
                break
            for e in entries:
                eid = e.findtext(ATOM + "id") or ""
                if "/api/errors" in eid:
                    raise HttpError("arXiv: " + (e.findtext(ATOM + "summary") or "ошибка запроса").strip())
                aid = norm_arxiv(eid)
                recs.append(Record(
                    title=" ".join((e.findtext(ATOM + "title") or "").split()),
                    abstract=" ".join((e.findtext(ATOM + "summary") or "").split()),
                    authors=[(a.findtext(ATOM + "name") or "").strip() for a in e.findall(ATOM + "author")],
                    year=_year_from_text(e.findtext(ATOM + "published")), arxiv_id=aid,
                    doi=e.findtext(AX + "doi"), is_oa=True, venue="arXiv",
                    pdf_urls=[f"https://export.arxiv.org/pdf/{aid}"], url=f"https://arxiv.org/abs/{aid}"))
            start += len(entries)
            if total is not None and start >= total:
                break
        return recs[:limit], {"query": q, "found": total}


# ---------------------------------------------------------------- OpenAlex
def inv_abstract(ii) -> str:
    if not ii:
        return ""
    pos = {}
    for w, idxs in ii.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


class OpenAlex(Source):
    name = "openalex"
    key_name = "openalex"
    key_required = True          # с 13.02.2026 нужен бесплатный ключ
    URL = "https://api.openalex.org/works"
    SELECT = ("id,doi,title,publication_year,cited_by_count,open_access,best_oa_location,"
              "primary_location,authorships,abstract_inverted_index,ids")

    def build_query(self, spec):
        return bool_expr(spec, _q)

    def _filter(self, spec):
        f = []
        if spec.year_from:
            f.append(f"from_publication_date:{spec.year_from}-01-01")
        if spec.year_to:
            f.append(f"to_publication_date:{spec.year_to}-12-31")
        if spec.open_access:
            f.append("is_oa:true")
        return ",".join(f)

    def search(self, spec, limit):
        q = self.query_for(spec)
        params = {"search": q, "filter": self._filter(spec) or None, "select": self.SELECT,
                  "per_page": min(100, limit), "cursor": "*", "api_key": self.key(),
                  "mailto": self.ctx.email or None}
        recs, found = [], None
        while len(recs) < limit and params["cursor"]:
            params["per_page"] = min(100, limit - len(recs))
            d = self.ctx.client.get(self.URL, params).json()
            found = (d.get("meta") or {}).get("count", found)
            res = d.get("results", [])
            if not res:
                break
            recs += [self._rec(w) for w in res]
            params["cursor"] = (d.get("meta") or {}).get("next_cursor")
        return recs[:limit], {"query": q, "found": found, "filter": self._filter(spec)}

    @staticmethod
    def _rec(w) -> Record:
        ids = w.get("ids") or {}
        pdfs = []
        for loc in (w.get("best_oa_location"), w.get("primary_location")):
            if loc and loc.get("pdf_url") and loc["pdf_url"] not in pdfs:
                pdfs.append(loc["pdf_url"])
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        pm = (ids.get("pmid") or "").rstrip("/").split("/")[-1] or None
        pmc = (ids.get("pmcid") or "").rstrip("/").split("/")[-1] or None
        return Record(
            title=w.get("title") or "", year=_int(w.get("publication_year")), doi=w.get("doi"),
            authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships", [])],
            abstract=inv_abstract(w.get("abstract_inverted_index")), venue=src, cited_by=_int(w.get("cited_by_count")),
            is_oa=(w.get("open_access") or {}).get("is_oa"), pdf_urls=pdfs, pmid=pm, pmcid=pmc,
            openalex_id=(w.get("id") or "").rsplit("/", 1)[-1] or None, url=w.get("doi") or w.get("id"))


# ---------------------------------------------------------------- Crossref
class Crossref(Source):
    name = "crossref"
    boolean = False
    URL = "https://api.crossref.org/works"

    def build_query(self, spec):
        return flat_terms(spec)

    def search(self, spec, limit):
        q = self.query_for(spec)
        flt = ["type:journal-article"]
        if spec.year_from:
            flt.append(f"from-pub-date:{spec.year_from}-01-01")
        if spec.year_to:
            flt.append(f"until-pub-date:{spec.year_to}-12-31")
        params = {"query": q, "filter": ",".join(flt), "rows": min(100, limit), "cursor": "*",
                  "select": "DOI,title,issued,author,abstract,container-title,is-referenced-by-count",
                  "mailto": self.ctx.email or None}
        recs, found = [], None
        while len(recs) < limit:
            params["rows"] = min(100, limit - len(recs))
            m = self.ctx.client.get(self.URL, params).json().get("message", {})
            found = m.get("total-results", found)
            items = m.get("items", [])
            if not items:
                break
            for it in items:
                issued = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
                recs.append(Record(
                    title=" ".join(it.get("title") or []), doi=it.get("DOI"), year=_int(issued[0] if issued else None),
                    authors=[f"{a.get('family', '')} {a.get('given', '')}".strip() for a in it.get("author", [])],
                    abstract=strip_tags(it.get("abstract") or ""), venue=(it.get("container-title") or [""])[0],
                    cited_by=_int(it.get("is-referenced-by-count")), url=f"https://doi.org/{it.get('DOI')}"))
            nxt = m.get("next-cursor")
            if not nxt:
                break
            params["cursor"] = nxt
        return recs[:limit], {"query": q, "found": found}


# ---------------------------------------------------------------- Semantic Scholar
class SemanticScholar(Source):
    name = "semanticscholar"
    boolean = False
    key_name = "s2"
    URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,year,externalIds,abstract,openAccessPdf,citationCount,venue,authors,isOpenAccess"

    def build_query(self, spec):
        return flat_terms(spec)

    def search(self, spec, limit):
        q = self.query_for(spec)
        params = {"query": q, "fields": self.FIELDS, "limit": min(100, limit), "offset": 0}
        if spec.year_from or spec.year_to:
            params["year"] = f"{spec.year_from or ''}-{spec.year_to or ''}"
        if spec.open_access:
            params["openAccessPdf"] = ""
        headers = {"x-api-key": self.key()} if self.key() else None
        recs, found = [], None
        limit = min(limit, 1000)     # глубина релевантного поиска ограничена
        while len(recs) < limit:
            params["limit"] = min(100, limit - len(recs))
            d = self.ctx.client.get(self.URL, params, headers).json()
            found = d.get("total", found)
            data = d.get("data", [])
            if not data:
                break
            for p in data:
                ex = p.get("externalIds") or {}
                oa = (p.get("openAccessPdf") or {}).get("url")
                recs.append(Record(
                    title=p.get("title") or "", year=_int(p.get("year")), doi=ex.get("DOI"), pmid=ex.get("PubMed"),
                    pmcid=ex.get("PubMedCentral"), arxiv_id=ex.get("ArXiv"), abstract=p.get("abstract") or "",
                    authors=[a.get("name", "") for a in p.get("authors", [])], venue=p.get("venue") or "",
                    cited_by=_int(p.get("citationCount")), is_oa=p.get("isOpenAccess"), pdf_urls=[oa] if oa else [],
                    url=f"https://www.semanticscholar.org/paper/{p.get('paperId')}" if p.get("paperId") else None))
            params["offset"] += len(data)
            if "next" not in d:
                break
        return recs[:limit], {"query": q, "found": found}


# ---------------------------------------------------------------- CORE
class Core(Source):
    name = "core"
    key_name = "core"
    key_required = True
    URL = "https://api.core.ac.uk/v3/search/works/"

    def build_query(self, spec):
        expr = bool_expr(spec, _q)
        if spec.year_from:
            expr = f"({expr}) AND yearPublished>={spec.year_from}"
        if spec.year_to:
            expr = f"({expr}) AND yearPublished<={spec.year_to}"
        return expr

    def search(self, spec, limit):
        q = self.query_for(spec)
        headers = {"Authorization": f"Bearer {self.key()}"}
        recs, offset, found = [], 0, None
        while len(recs) < limit:
            d = self.ctx.client.get(self.URL, {"q": q, "limit": min(100, limit - len(recs)), "offset": offset}, headers).json()
            found = d.get("totalHits", found)
            res = d.get("results", [])
            if not res:
                break
            for w in res:
                dl = w.get("downloadUrl")
                recs.append(Record(
                    title=w.get("title") or "", abstract=strip_tags(w.get("abstract") or ""), year=_int(w.get("yearPublished")),
                    authors=[a.get("name", "") for a in w.get("authors", [])], doi=w.get("doi"),
                    pdf_urls=[dl] if dl else [], is_oa=True if dl else None, cited_by=_int(w.get("citationCount")),
                    venue=((w.get("journals") or [{}])[0] or {}).get("title", ""),
                    url=f"https://core.ac.uk/works/{w.get('id')}" if w.get("id") else None))
            offset += len(res)
        return recs[:limit], {"query": q, "found": found}


# ---------------------------------------------------------------- DOAJ
class DOAJ(Source):
    name = "doaj"
    URL = "https://doaj.org/api/search/articles/"

    def build_query(self, spec):
        def fmt(t):
            q = _q(t)
            if spec.title_only:
                return f"bibjson.title:{q}"
            return f"(bibjson.title:{q} OR bibjson.abstract:{q})"
        expr = bool_expr(spec, fmt, op_not=" AND NOT ")
        if spec.year_from or spec.year_to:
            expr = f"({expr}) AND bibjson.year:[{spec.year_from or 1800} TO {spec.year_to or current_year() + 1}]"
        return expr

    def search(self, spec, limit):
        q = self.query_for(spec)
        recs, page, found = [], 1, None
        while len(recs) < limit:
            size = min(50, limit - len(recs))
            d = self.ctx.client.get(self.URL + quote(q, safe=""), {"page": page, "pageSize": size}).json()
            found = d.get("total", found)
            res = d.get("results", [])
            if not res:
                break
            for it in res:
                b = it.get("bibjson", {})
                doi = next((i.get("id") for i in b.get("identifier", []) if i.get("type") == "doi"), None)
                link = next((l.get("url") for l in b.get("link", []) if l.get("type") == "fulltext"), None)
                recs.append(Record(
                    title=b.get("title") or "", abstract=strip_tags(b.get("abstract") or ""), year=_int(b.get("year")),
                    authors=[a.get("name", "") for a in b.get("author", [])], doi=doi, is_oa=True,
                    venue=(b.get("journal") or {}).get("title", ""), url=link))
            page += 1
        return recs[:limit], {"query": q, "found": found}


SOURCE_CLASSES = {c.name: c for c in (PubMed, EuropePMC, Arxiv, OpenAlex, Crossref, SemanticScholar, Core, DOAJ)}
DEFAULT_SOURCES = ["openalex", "pubmed", "europepmc", "arxiv", "semanticscholar", "crossref", "core", "doaj"]


def make_sources(ctx: Ctx) -> dict[str, Source]:
    return {n: c(ctx) for n, c in SOURCE_CLASSES.items()}
