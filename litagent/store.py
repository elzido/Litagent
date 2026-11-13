"""Библиотека на диске: pdf/, xml/, index.json (без повторных скачиваний), runs/<id>/ (запросы и отчёты)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

from .models import Record, safe_name

ARTICLE_EXT = {".pdf", ".xml", ".part"}     # что считается скачанной статьёй при очистке


class Library:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for d in ("pdf", "xml", "runs"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.records: dict[str, dict] = {}
        self.aliases: dict[str, str] = {}
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.records, self.aliases = data.get("records", {}), data.get("aliases", {})
            except (ValueError, OSError):
                pass

    @property
    def cache_dir(self) -> Path:
        return self.root / ".http_cache"

    def find(self, rec: Record) -> dict | None:
        for k in rec.keys():
            pk = self.aliases.get(k)
            if pk and pk in self.records:
                e = self.records[pk]
                if e.get("file") and (self.root / e["file"]).exists():
                    return e
        return None

    def path_for(self, rec: Record, ext: str) -> Path:
        author = safe_name(rec.authors[0], 20) if rec.authors else "anon"
        base = safe_name(f"{author}_{rec.year or 'nd'}_{rec.title}", 90)
        h = hashlib.sha1(rec.primary_key().encode()).hexdigest()[:8]
        return self.root / ("xml" if ext == ".xml" else "pdf") / f"{base}_{h}{ext}"

    def register(self, rec: Record) -> None:
        pk = next((self.aliases[k] for k in rec.keys() if k in self.aliases), None) or rec.primary_key()
        self.records[pk] = rec.to_dict()
        for k in rec.keys():
            self.aliases.setdefault(k, pk)

    def save(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"records": self.records, "aliases": self.aliases},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------ очистка
    def _article_files(self, sub: str) -> list[Path]:
        d = self.root / sub
        if not d.is_dir():
            return []
        return [p for p in d.iterdir() if (p.is_file() or p.is_symlink()) and p.suffix.lower() in ARTICLE_EXT]

    def _cache_files(self) -> list[Path]:
        return [p for p in self.cache_dir.iterdir() if p.is_file()] if self.cache_dir.is_dir() else []

    def _run_dirs(self) -> list[Path]:
        d = self.root / "runs"
        return [p for p in d.iterdir() if p.is_dir() and not p.is_symlink()] if d.is_dir() else []

    @staticmethod
    def _size(paths) -> int:
        total = 0
        for p in paths:
            try:
                total += p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            except OSError:
                pass
        return total

    def stats(self) -> dict:
        pdf, xml, cache, runs = (self._article_files("pdf"), self._article_files("xml"),
                                 self._cache_files(), self._run_dirs())
        return {"pdf": len(pdf), "xml": len(xml), "articles_bytes": self._size(pdf + xml),
                "cache": len(cache), "cache_bytes": self._size(cache),
                "runs": len(runs), "runs_bytes": self._size(runs), "indexed": len(self.records)}

    def clean(self, runs: bool = False, cache: bool = False) -> dict:
        """Удаляет скачанные .pdf/.xml (и недокачанные .part) из pdf/ и xml/; чужие файлы и подпапки не трогает.
        runs=True — ещё и отчёты запусков, cache=True — ещё и HTTP-кэш. Индекс приводится в соответствие с диском."""
        res = {"pdf": 0, "xml": 0, "runs": 0, "cache": 0, "bytes": 0, "errors": []}
        victims = [(k, p) for k in ("pdf", "xml") for p in self._article_files(k)]
        if cache:
            victims += [("cache", p) for p in self._cache_files()]
        for kind, p in victims:
            try:
                size = p.lstat().st_size
                p.unlink()
                res[kind] += 1
                res["bytes"] += size
            except OSError as e:
                res["errors"].append(f"{p.name}: {e}")
        if runs:
            for d in self._run_dirs():
                try:
                    res["bytes"] += self._size([d])
                    shutil.rmtree(d)
                    res["runs"] += 1
                except OSError as e:
                    res["errors"].append(f"{d.name}: {e}")
        self.records = {k: e for k, e in self.records.items() if e.get("file") and (self.root / e["file"]).exists()}
        self.aliases = {a: k for a, k in self.aliases.items() if k in self.records}
        self.save()
        return res

    def write_run(self, run_id: str, meta: dict, records: list[Record]) -> Path:
        d = self.root / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(d / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["n", "status", "score", "year", "title", "authors", "doi", "pmid", "pmcid",
                        "arxiv_id", "sources", "cited_by", "is_oa", "file", "url"])
            for i, r in enumerate(records, 1):
                w.writerow([i, r.status, r.score if r.score is not None else "", r.year or "", r.title,
                            "; ".join(r.authors[:3]), r.doi or "", r.pmid or "", r.pmcid or "",
                            r.arxiv_id or "", ",".join(r.sources), r.cited_by if r.cited_by is not None else "",
                            "" if r.is_oa is None else int(r.is_oa), r.file or "", r.url or ""])
        return d
