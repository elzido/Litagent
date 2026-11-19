"""Библиотека на диске: pdf/, xml/, index.json (без повторных скачиваний), runs/<id>/ (запросы и отчёты)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from .models import Record, safe_name

ARTICLE_EXT = {".pdf", ".xml", ".part"}     # что считается скачанной статьёй при очистке
# Файл, созданный агентом: Автор_Год_Название_<8 hex>.pdf|xml[.tgz][.part]. В подпапках источников очистка трогает только такие файлы.
OURS = re.compile(r"_[0-9a-f]{8}\.(?:pdf|xml)(?:\.tgz)?(?:\.part)?$", re.I)


class Library:
    def __init__(self, root: str | Path, by_source: bool = True):
        self.root = Path(root)
        self.by_source = by_source                # pdf/<источник>/файл.pdf, xml/<источник>/файл.xml
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

    @staticmethod
    def source_dir(rec: Record) -> str:
        """Имя папки источника: первый источник, нашедший статью (для своего сайта — его имя из custom_sites.json)."""
        name = next((x for x in rec.sources if x), "unknown")
        d = safe_name(name, 40).lower()
        return d if d != "paper" or name.lower() == "paper" else "unknown"

    def path_for(self, rec: Record, ext: str) -> Path:
        author = safe_name(rec.authors[0], 20) if rec.authors else "anon"
        base = safe_name(f"{author}_{rec.year or 'nd'}_{rec.title}", 90)
        h = hashlib.sha1(rec.primary_key().encode()).hexdigest()[:8]
        folder = self.root / ("xml" if ext == ".xml" else "pdf")
        if self.by_source:
            folder = folder / self.source_dir(rec)
        return folder / f"{base}_{h}{ext}"

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
        out = [p for p in d.iterdir() if (p.is_file() or p.is_symlink()) and p.suffix.lower() in ARTICLE_EXT]
        for sd in d.iterdir():                                  # папки источников: только файлы, созданные агентом
            if sd.is_dir() and not sd.is_symlink():
                out += [p for p in sd.iterdir() if (p.is_file() or p.is_symlink()) and OURS.search(p.name)]
        return out

    def prune_empty(self) -> None:
        """Убирает пустые папки источников (остаются после неудачных скачиваний)."""
        for sd in self._source_dirs():
            try:
                sd.rmdir()
            except OSError:
                pass

    def _source_dirs(self) -> list[Path]:
        return [sd for k in ("pdf", "xml") if (self.root / k).is_dir()
                for sd in (self.root / k).iterdir() if sd.is_dir() and not sd.is_symlink()]

    def by_source_counts(self) -> dict[str, int]:
        """{источник: число файлов} по папкам pdf/ (для отчёта)."""
        d = self.root / "pdf"
        return {sd.name: n for sd in sorted(d.iterdir()) if sd.is_dir() and (n := len(self._article_files_in(sd)))} if d.is_dir() else {}

    @staticmethod
    def _article_files_in(sd: Path) -> list[Path]:
        return [p for p in sd.iterdir() if p.is_file() and OURS.search(p.name)]

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

    def clean(self, runs: bool = False, cache: bool = False, articles: bool = True) -> dict:
        """articles — удалить скачанные .pdf/.xml (и недокачанные .part) из pdf/ и xml/ (чужие файлы не трогает),
        runs — отчёты запусков (логи: runs/), cache — HTTP-кэш. Индекс приводится в соответствие с диском."""
        res = {"pdf": 0, "xml": 0, "runs": 0, "cache": 0, "bytes": 0, "errors": []}
        victims = [(k, p) for k in ("pdf", "xml") for p in self._article_files(k)] if articles else []
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
        self.prune_empty()
        self.records = {k: e for k, e in self.records.items() if e.get("file") and (self.root / e["file"]).exists()}
        self.aliases = {a: k for a, k in self.aliases.items() if k in self.records}
        self.save()
        return res

    # ------------------------------------------------------------ разбивка уже скачанного по папкам источников
    def organize(self, dry_run: bool = False) -> dict:
        """Переносит файлы из плоских pdf/ и xml/ в pdf/<источник>/ и xml/<источник>/ по данным index.json и обновляет индекс."""
        res = {"moved": 0, "skipped": 0, "by_source": {}, "errors": []}
        for e in self.records.values():
            f = e.get("file")
            if not f:
                continue
            src = self.root / f
            kind = Path(f).parts[0] if Path(f).parts else ""
            if kind not in ("pdf", "xml") or len(Path(f).parts) != 2 or not src.is_file():
                res["skipped"] += 1                              # уже в подпапке, нет файла или нестандартный путь
                continue
            name = self.source_dir(Record.from_dict(e))
            dest = self.root / kind / name / src.name
            res["by_source"][name] = res["by_source"].get(name, 0) + 1
            if dry_run:
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    res["skipped"] += 1
                    continue
                os.replace(src, dest)
                e["file"] = str(dest.relative_to(self.root))
                res["moved"] += 1
            except OSError as ex:
                res["errors"].append(f"{src.name}: {ex}")
        if not dry_run:
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
