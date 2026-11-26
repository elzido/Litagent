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
        self.tags: dict[str, list[str]] = {}        # пользовательские теги по primary key (не трогаются при register)
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.records, self.aliases = data.get("records", {}), data.get("aliases", {})
                self.tags = data.get("tags", {})
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

    def supp_dir_for(self, file_rel: str) -> Path:
        """Папка доп. материалов статьи: pdf/<src>/A_b1c2.pdf -> supp/<src>/A_b1c2/."""
        p = Path(file_rel)
        parts = list(p.parts)
        if parts and parts[0] in ("pdf", "xml"):
            parts[0] = "supp"
        else:
            parts = ["supp"] + parts
        parts[-1] = p.stem
        return self.root.joinpath(*parts)

    def register(self, rec: Record) -> None:
        pk = next((self.aliases[k] for k in rec.keys() if k in self.aliases), None) or rec.primary_key()
        self.records[pk] = rec.to_dict()
        for k in rec.keys():
            self.aliases.setdefault(k, pk)

    def save(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"records": self.records, "aliases": self.aliases, "tags": self.tags},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------ теги и удаление записей
    @staticmethod
    def _norm_tags(tags: list) -> list[str]:
        clean: list[str] = []
        for t in tags:
            t = re.sub(r"\s+", " ", str(t).strip())[:80]
            if t and t.casefold() not in {x.casefold() for x in clean}:
                clean.append(t)
        return clean[:30]

    def set_tags(self, pk: str, tags: list) -> list[str]:
        if pk not in self.records:
            raise KeyError(pk)
        self.tags[pk] = self._norm_tags(tags)
        return self.tags[pk]

    def all_tags(self) -> list[str]:
        return sorted({t for ts in self.tags.values() for t in ts}, key=str.casefold)

    def delete(self, pks: list) -> dict:
        """Удаляет записи вместе с файлами (только внутри библиотеки и только если файл не нужен другой записи)."""
        gone = set(pks)
        keep_files = {e.get("file") for k, e in self.records.items() if k not in gone}
        res = {"deleted": 0, "files": 0, "bytes": 0, "errors": []}
        for pk in pks:
            e = self.records.get(pk)
            if e is None:
                continue
            f = e.get("file")
            if f and f not in keep_files:
                p = (self.root / f).resolve()
                if p.is_relative_to(self.root.resolve()) and p.is_file() and p.suffix.lower() in ARTICLE_EXT:
                    try:
                        res["bytes"] += p.stat().st_size
                        p.unlink()
                        res["files"] += 1
                    except OSError as ex:
                        res["errors"].append(f"{p.name}: {ex}")
                sd = self.supp_dir_for(f)
                if sd.is_dir() and sd.resolve().is_relative_to(self.root.resolve()):
                    try:
                        res["bytes"] += self._size([sd])
                        shutil.rmtree(sd)
                    except OSError as ex:
                        res["errors"].append(f"{sd.name}: {ex}")
            del self.records[pk]
            res["deleted"] += 1
        self.aliases = {a: k for a, k in self.aliases.items() if k in self.records}
        self.tags = {k: t for k, t in self.tags.items() if k in self.records}
        self.prune_empty()
        self.save()
        return res

    # ------------------------------------------------------------ дополнительные материалы
    def _supp_files(self) -> list[Path]:
        d = self.root / "supp"
        return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() != ".part"] if d.is_dir() else []

    def supp_list(self, pk: str) -> list[dict]:
        e = self.records.get(pk)
        if e is None:
            raise KeyError(pk)
        out = []
        for s in e.get("supp") or []:
            p = self.root / s.get("file", "")
            if p.is_file():
                out.append({**s, "size": p.stat().st_size})
        return out

    @staticmethod
    def supp_name(name: str) -> str:
        """Безопасное имя файла доп. материалов (плоское, без путей; расширение сохраняется)."""
        n = Path(str(name)).name.replace("\\", "/")
        n = re.sub(r"[^\w.\- ]+", "_", n).strip("._ ")
        return n[:120] or "file"

    def delete_supp(self, pk: str, rel_file: str) -> dict:
        e = self.records.get(pk)
        if e is None:
            raise KeyError(pk)
        before = e.get("supp") or []
        e["supp"] = [s for s in before if s.get("file") != rel_file]
        if len(e["supp"]) == len(before):
            raise KeyError(rel_file)
        p = (self.root / rel_file).resolve()
        removed = False
        if p.is_relative_to(self.root.resolve()) and p.is_file():
            try:
                p.unlink()
                removed = True
            except OSError:
                pass
        self.prune_empty()
        self.save()
        return {"removed": removed, "supp": e["supp"]}

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

    def _part_files(self) -> list[Path]:
        """Недокачанные временные файлы (*.part) внутри библиотеки."""
        out: list[Path] = []
        for sub in ("pdf", "xml", "supp"):
            d = self.root / sub
            if d.is_dir():
                out += [p for p in d.rglob("*.part") if p.is_file()]
        return out

    def purge_parts(self) -> dict:
        """Удаляет оставшиеся .part (обрыв скачивания не должен выглядеть как статья)."""
        res = {"files": 0, "bytes": 0, "errors": []}
        for p in self._part_files():
            try:
                res["bytes"] += p.stat().st_size
                p.unlink()
                res["files"] += 1
            except OSError as e:
                res["errors"].append(f"{p.name}: {e}")
        self.prune_empty()
        return res

    def prune_empty(self) -> None:
        """Убирает пустые папки источников (остаются после неудачных скачиваний)."""
        for sd in self._source_dirs():
            try:
                sd.rmdir()
            except OSError:
                pass
        supp = self.root / "supp"
        if supp.is_dir():
            for sd in sorted(supp.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if sd.is_dir() and not sd.is_symlink():
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
        return [p for p in sd.iterdir() if p.is_file() and OURS.search(p.name) and p.suffix.lower() != ".part"]

    def _cache_files(self) -> list[Path]:
        return [p for p in self.cache_dir.iterdir() if p.is_file()] if self.cache_dir.is_dir() else []

    def clear_cache(self) -> dict:
        """Удаляет только HTTP-кэш ответов; статьи, индекс и запуски не трогает."""
        res = {"files": 0, "bytes": 0, "errors": []}
        for p in self._cache_files():
            try:
                res["bytes"] += p.stat().st_size
                p.unlink()
                res["files"] += 1
            except OSError as e:
                res["errors"].append(f"{p.name}: {e}")
        return res

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
        pdf = [p for p in pdf if p.suffix.lower() != ".part"]      # недокачанные .part статьями не считаются
        xml = [p for p in xml if p.suffix.lower() != ".part"]
        supp = self._supp_files()
        return {"pdf": len(pdf), "xml": len(xml), "articles_bytes": self._size(pdf + xml),
                "supp": len(supp), "supp_bytes": self._size(supp),
                "cache": len(cache), "cache_bytes": self._size(cache),
                "runs": len(runs), "runs_bytes": self._size(runs), "indexed": len(self.records)}

    def clean(self, runs: bool = False, cache: bool = False, articles: bool = True) -> dict:
        """articles — удалить скачанные .pdf/.xml, их доп. материалы (supp/) и недокачанные .part из pdf/ и xml/
        (чужие файлы не трогает), runs — отчёты запусков (логи: runs/), cache — HTTP-кэш. Индекс приводится в соответствие с диском."""
        res = {"pdf": 0, "xml": 0, "supp": 0, "parts": 0, "runs": 0, "cache": 0, "bytes": 0, "errors": []}
        victims = [(k, p) for k in ("pdf", "xml") for p in self._article_files(k)] if articles else []
        if cache:
            victims += [("cache", p) for p in self._cache_files()]
        for kind, p in victims:
            try:
                size = p.lstat().st_size
                p.unlink()
                if p.suffix.lower() == ".part":
                    res["parts"] += 1
                else:
                    res[kind] += 1
                res["bytes"] += size
            except OSError as e:
                res["errors"].append(f"{p.name}: {e}")
        if articles:
            supp_root = self.root / "supp"
            if supp_root.is_dir():
                try:
                    res["bytes"] += self._size([supp_root])
                    shutil.rmtree(supp_root)
                    res["supp"] += 1
                except OSError as e:
                    res["errors"].append(f"supp: {e}")
            for e in self.records.values():
                e.pop("supp", None)
        pr = self.purge_parts()
        res["parts"] += pr["files"]
        res["bytes"] += pr["bytes"]
        res["errors"] += pr["errors"]
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
        self.tags = {k: t for k, t in self.tags.items() if k in self.records}
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
                old_supp = self.supp_dir_for(f)
                new_supp = self.supp_dir_for(e["file"])
                if old_supp.is_dir() and not new_supp.exists():          # доп. материалы переезжают со статьёй
                    new_supp.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(old_supp, new_supp)
                    e["supp"] = [{**s, "file": (new_supp / Path(s["file"]).name).relative_to(self.root).as_posix()}
                                 for s in (e.get("supp") or [])]
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
