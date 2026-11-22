"""Офлайн-тесты веб-интерфейса (сеть не нужна). Запуск: python -m unittest discover -s tests -v"""
import copy
import json
import os
import sys
import tempfile
import threading
import time
import tomllib
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent import cli                                            # noqa: E402
from litagent.http import PoliteClient                              # noqa: E402
from litagent.models import Record                                  # noqa: E402
from litagent.policy import Policy                                 # noqa: E402
from litagent.store import Library                                  # noqa: E402
from litagent.web import (ApiError, Handler, JobManager, WebApp,   # noqa: E402
                          build_search_argv, dump_toml, validate_config)

PDF = b"%PDF-1.4\n" + b"x" * 500


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.app = WebApp(cwd=self.cwd)

    def tearDown(self):
        self.tmp.cleanup()

    def write_index(self):
        lib = self.cwd / "library"
        (lib / "pdf" / "pubmed").mkdir(parents=True, exist_ok=True)
        (lib / "pdf" / "pubmed" / "Smith_2022_Test_1a2b3c4d.pdf").write_bytes(PDF)
        rec = {"title": "CRISPR therapy", "authors": ["Smith J"], "year": 2022, "doi": "10.1/x",
               "sources": ["pubmed"], "file": "pdf/pubmed/Smith_2022_Test_1a2b3c4d.pdf", "status": "pdf"}
        (lib / "index.json").write_text(json.dumps({"records": {"doi:10.1/x": rec}, "aliases": {"doi:10.1/x": "doi:10.1/x"}}),
                                        encoding="utf-8")
        return lib

    def write_run(self, rid="20261122-120000"):
        d = self.cwd / "library" / "runs" / rid
        d.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": rid, "mode": "manual", "started": "2026-11-22T12:00:00",
                "total_unique": 1, "downloads": {"pdf": 1, "xml": 0, "exists": 0},
                "queries": [{"source": "pubmed", "spec": "cancer", "found": 10, "fetched": 10, "kept": 1}],
                "specs": [{"label": "cancer"}], "http": {"requests": 3, "retries": 0}}
        (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
        (d / "results.csv").write_text(
            "n,status,score,year,title,authors,doi,pmid,pmcid,arxiv_id,sources,cited_by,is_oa,file,url\n"
            '1,pdf,,2022,CRISPR therapy,Smith J,10.1/x,,,,pubmed,,1,pdf/pubmed/Smith_2022_Test_1a2b3c4d.pdf,\n',
            encoding="utf-8-sig")
        return d


class TestToml(unittest.TestCase):
    def test_roundtrip(self):
        cfg = copy.deepcopy(cli.DEFAULT_CFG)
        cfg["email"] = 'a"b\\c@d.org'
        cfg["auto"]["notes"] = ["файл.tex", "путь с пробелом.md"]
        cfg["auto"]["min_score"] = 6.5
        cfg["llm"]["temperature"] = 0.35
        cfg["keys"]["gigachat_client_id"] = "01a0fd8c-0033"
        parsed = tomllib.loads(dump_toml(cfg))
        self.assertEqual(parsed, cfg)

    def test_validate_ok(self):
        data = {"email": "t@e.org", "sources": ["pubmed", "arxiv"],
                "llm": {"provider": "gigachat", "temperature": 0.3},
                "auto": {"min_score": 6.5}, "web": {"port": 9000}}
        cfg, errors = validate_config(data, ["pubmed", "arxiv", "openalex"])
        self.assertEqual(errors, {})
        self.assertEqual(cfg["llm"]["provider"], "gigachat")
        self.assertEqual(cfg["auto"]["min_score"], 6.5)
        self.assertEqual(cfg["web"]["port"], 9000)
        self.assertEqual(cfg["limit_per_source"], cli.DEFAULT_CFG["limit_per_source"])   # пропущенное поле — дефолт

    def test_validate_errors(self):
        data = {"email": "не почта", "limit_per_source": -3, "sources": ["google_scholar"],
                "llm": {"provider": "bad"}, "web": {"port": 99999}}
        _, errors = validate_config(data, ["pubmed"])
        for k in ("email", "limit_per_source", "sources", "llm.provider", "web.port"):
            self.assertIn(k, errors)

    def test_validate_null_keeps_default(self):
        cfg, errors = validate_config({"email": None, "llm": {"model": None}}, list(cli.DEFAULT_CFG["sources"]))
        self.assertEqual(errors, {})
        self.assertEqual(cfg["email"], "")
        self.assertEqual(cfg["llm"]["model"], cli.DEFAULT_CFG["llm"]["model"])


class TestArgv(Tmp):
    known = ["pubmed", "arxiv", "openalex"]
    forbidden = Policy().forbidden_sources

    def build(self, p):
        return build_search_argv(p, self.app.cfg(), self.known, self.forbidden, self.app.config_path)

    def test_manual(self):
        argv, kind, title = self.build({"mode": "manual", "topics": ["cancer; -mouse"], "sources": ["pubmed"],
                                        "year_from": 2020, "oa_only": True, "count": 5, "dry_run": True})
        self.assertEqual(kind, "manual")
        self.assertEqual(argv[:3], ["manual", "-q", "cancer; -mouse"])
        for piece in (["--sources", "pubmed"], ["--year-from", "2020"], ["--count", "5"], ["--oa-only"], ["--dry-run"]):
            self.assertIn(piece[0], argv)
            if len(piece) > 1:
                self.assertEqual(argv[argv.index(piece[0]) + 1], piece[1])
        self.assertIn("cancer", title)

    def test_manual_preset_and_save(self):
        argv, _, title = self.build({"mode": "manual", "preset": "gnn", "save_preset": "gnn2", "preset_desc": "описание"})
        self.assertIn("--preset", argv)
        self.assertIn("--save-preset", argv)
        self.assertIn("описание", argv)
        self.assertIn("gnn", title)

    def test_auto(self):
        argv, kind, _ = self.build({"mode": "auto", "goal": "найди статьи про CRISPR", "llm_filter": True,
                                    "min_score": 7, "rounds": 3, "no_llm": False})
        self.assertEqual(kind, "auto")
        self.assertEqual(argv[:2], ["auto", "найди статьи про CRISPR"])
        self.assertIn("--llm-filter", argv)
        self.assertEqual(argv[argv.index("--min-score") + 1], "7")
        self.assertEqual(argv[argv.index("--rounds") + 1], "3")
        self.assertNotIn("--no-llm", argv)

    def test_errors(self):
        with self.assertRaises(ApiError):
            self.build({"mode": "manual"})                                   # нет темы
        with self.assertRaises(ApiError):
            self.build({"mode": "auto", "goal": "  "})                       # нет цели
        with self.assertRaises(ApiError):
            self.build({"mode": "manual", "topics": ["x"], "pdf_only": True, "no_download": True})
        with self.assertRaises(ApiError):
            self.build({"mode": "manual", "topics": ["x"], "sources": ["google_scholar"]})
        with self.assertRaises(ApiError):
            self.build({"mode": "manual", "topics": ["x"], "count": 0})
        with self.assertRaises(ApiError):
            self.build({"mode": "manual", "topics": ["x"], "sources": ["nosuch"]})


class TestConfigApi(Tmp):
    def test_create_and_read(self):
        self.assertFalse(self.app.config_path.exists())
        r = self.app.post_config({"email": "t@e.org", "sources": ["pubmed"], "llm": {"provider": "gigachat"}})
        self.assertTrue(self.app.config_path.exists())
        self.assertEqual(r["cfg"]["email"], "t@e.org")
        got = self.app.get_config()
        self.assertTrue(got["exists"])
        self.assertEqual(got["cfg"]["llm"]["provider"], "gigachat")
        self.assertEqual(got["cfg"]["sources"], ["pubmed"])

    def test_env_not_baked_into_file(self):
        self.app.post_config({"email": "file@e.org", "keys": {"ncbi": "filekey"}})
        with mock.patch.dict(os.environ, {"LITAGENT_EMAIL": "env@e.org", "NCBI_API_KEY": "envkey"}):
            got = self.app.get_config()
            self.assertEqual(got["cfg"]["email"], "file@e.org")              # редактор показывает значения файла
            self.assertEqual(got["cfg"]["keys"]["ncbi"], "filekey")
            self.assertIn({"field": "email", "env": "LITAGENT_EMAIL"}, got["env_overrides"])
            self.assertEqual(self.app.cfg()["email"], "env@e.org")           # запуски используют окружение
        self.app.post_config({"email": "file@e.org"})
        self.assertNotIn("envkey", self.app.config_path.read_text(encoding="utf-8"))

    def test_invalid_returns_field_errors(self):
        with self.assertRaises(ApiError) as cm:
            self.app.post_config({"llm": {"provider": "bad"}})
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("llm.provider", json.loads(cm.exception.message))
        self.assertFalse(self.app.config_path.exists())                      # ничего не записано

    def test_existing_file_values_preserved(self):
        self.app.config_path.write_text('email = "old@e.org"\n[keys]\nncbi = "secret"\n', encoding="utf-8")
        self.app.post_config({"email": "new@e.org"})
        data = tomllib.loads(self.app.config_path.read_text(encoding="utf-8"))
        self.assertEqual(data["email"], "new@e.org")
        self.assertEqual(data["keys"]["ncbi"], "secret")                     # не присланные поля не теряются


class TestLibraryApi(Tmp):
    def test_library_and_files(self):
        self.write_index()
        d = self.app.get_library()
        self.assertEqual(len(d["records"]), 1)
        self.assertTrue(d["records"][0]["has_file"])
        self.assertEqual(d["stats"]["pdf"], 1)
        ctype, body, name = self.app.get_file("pdf/pubmed/Smith_2022_Test_1a2b3c4d.pdf")
        self.assertEqual(ctype, "application/pdf")
        self.assertEqual(body, PDF)
        self.assertEqual(name, "Smith_2022_Test_1a2b3c4d.pdf")

    def test_file_guards(self):
        self.write_index()
        (self.cwd / "secret.txt").write_text("секрет", encoding="utf-8")
        with self.assertRaises(ApiError) as cm:
            self.app.get_file("../secret.txt")
        self.assertEqual(cm.exception.code, 404)
        with self.assertRaises(ApiError) as cm:
            self.app.get_file("index.json")
        self.assertEqual(cm.exception.code, 403)
        with self.assertRaises(ApiError):
            self.app.get_file("pdf/nope.pdf")

    def test_tags_set_normalize_persist(self):
        self.write_index()
        r = self.app.post_tags({"pk": "doi:10.1/x", "tags": ["CRISPR", " gene therapy ", "crispr", ""]})
        self.assertEqual(r["tags"], ["CRISPR", "gene therapy"])          # дедуп без учёта регистра, пробелы схлопнуты
        self.assertEqual(r["all_tags"], ["CRISPR", "gene therapy"])
        saved = json.loads((self.cwd / "library" / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["tags"]["doi:10.1/x"], ["CRISPR", "gene therapy"])
        rec = self.app.get_library()["records"][0]
        self.assertEqual(rec["tags"], ["CRISPR", "gene therapy"])
        self.assertEqual(rec["pk"], "doi:10.1/x")
        with self.assertRaises(ApiError) as cm:
            self.app.post_tags({"pk": "doi:нет", "tags": ["x"]})
        self.assertEqual(cm.exception.code, 404)
        with self.assertRaises(ApiError):
            self.app.post_tags({"pk": "doi:10.1/x", "tags": "не список"})

    def test_tags_survive_reregister_and_pruned_on_clean(self):
        self.write_index()
        self.app.post_tags({"pk": "doi:10.1/x", "tags": ["важное"]})
        lib = Library(self.cwd / "library")
        lib.register(Record.from_dict(lib.records["doi:10.1/x"]))         # повторная регистрация не затирает теги
        lib.save()
        self.assertEqual(Library(self.cwd / "library").tags["doi:10.1/x"], ["важное"])
        lib.clean(articles=True)
        self.assertEqual(lib.tags, {})                                    # удалённые записи уносят теги с собой

    def test_delete_records_with_files(self):
        lib = self.write_index()
        self.app.post_tags({"pk": "doi:10.1/x", "tags": ["тест"]})
        f = lib / "pdf" / "pubmed" / "Smith_2022_Test_1a2b3c4d.pdf"
        self.assertTrue(f.exists())
        r = self.app.post_library_delete({"keys": ["doi:10.1/x", "doi:нет-такой"]})
        self.assertEqual((r["deleted"], r["files"]), (1, 1))
        self.assertGreater(r["bytes"], 0)
        self.assertFalse(f.exists())
        data = json.loads((lib / "index.json").read_text(encoding="utf-8"))
        self.assertEqual((data["records"], data["aliases"], data["tags"]), ({}, {}, {}))
        self.assertEqual(self.app.get_library()["records"], [])

    def test_delete_keeps_shared_file_and_guards_path(self):
        lib = self.write_index()
        idx = json.loads((lib / "index.json").read_text(encoding="utf-8"))
        idx["records"]["doi:10.1/y"] = {**idx["records"]["doi:10.1/x"], "title": "Duplicate entry"}
        idx["records"]["doi:10.1/evil"] = {"title": "Evil", "file": "../secret.txt"}
        (lib / "index.json").write_text(json.dumps(idx), encoding="utf-8")
        (self.cwd / "secret.txt").write_text("секрет", encoding="utf-8")
        r = self.app.post_library_delete({"keys": ["doi:10.1/x", "doi:10.1/evil"]})
        self.assertEqual(r["deleted"], 2)
        self.assertEqual(r["files"], 0)                                   # общий файл нужен другой записи; чужой путь вне библиотеки
        self.assertTrue((lib / "pdf" / "pubmed" / "Smith_2022_Test_1a2b3c4d.pdf").exists())
        self.assertTrue((self.cwd / "secret.txt").exists())
        with self.assertRaises(ApiError):
            self.app.post_library_delete({"keys": []})
        with self.assertRaises(ApiError):
            self.app.post_library_delete({"keys": "не список"})

    def test_runs(self):
        self.write_run()
        runs = self.app.get_runs()["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["mode"], "manual")
        self.assertEqual(runs[0]["downloads"]["pdf"], 1)
        run = self.app.get_run("20261122-120000")
        self.assertEqual(len(run["results"]), 1)
        self.assertEqual(run["results"][0]["title"], "CRISPR therapy")
        with self.assertRaises(ApiError) as cm:
            self.app.get_run("../../etc")
        self.assertEqual(cm.exception.code, 400)
        with self.assertRaises(ApiError) as cm:
            self.app.get_run("20990101-000000")
        self.assertEqual(cm.exception.code, 404)

    def test_runs_delete(self):
        self.write_run()
        self.write_run("20261122-130000")
        (self.cwd / "library" / "runs" / "keep.txt").write_text("не запуск", encoding="utf-8")
        r = self.app.post_runs_delete({"ids": ["20261122-120000", "../runs", "20990101-000000"]})
        self.assertEqual(r["deleted"], 1)
        self.assertTrue(any("неверный идентификатор" in e for e in r["errors"]))
        self.assertFalse((self.cwd / "library" / "runs" / "20261122-120000").exists())
        self.assertTrue((self.cwd / "library" / "runs" / "20261122-130000").exists())
        self.assertTrue((self.cwd / "library" / "runs" / "keep.txt").exists())
        self.assertEqual(len(self.app.get_runs()["runs"]), 1)
        with self.assertRaises(ApiError):
            self.app.post_runs_delete({"ids": []})
        with self.assertRaises(ApiError):
            self.app.post_runs_delete({"ids": "не список"})

    def test_clear_cache(self):
        lib = self.write_index()
        cache = lib / ".http_cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "abc123").write_bytes(b"cached response" * 100)
        (cache / "def456").write_bytes(b"another")
        r = self.app.post_cache_clear()
        self.assertEqual((r["files"], r["errors"]), (2, []))
        self.assertGreater(r["bytes"], 0)
        self.assertEqual(list(cache.iterdir()), [])
        self.assertTrue((lib / "pdf" / "pubmed" / "Smith_2022_Test_1a2b3c4d.pdf").exists())   # статьи целы
        self.assertEqual(len(self.app.get_library()["records"]), 1)                          # индекс цел
        self.assertEqual(self.app.post_cache_clear()["files"], 0)                            # повтор — пусто


class TestPresetsSites(Tmp):
    def test_presets_crud(self):
        self.app.post_preset({"name": "gnn", "description": "графы", "args": {"query": ["graph; neural|gnn"], "oa_only": True}})
        got = self.app.get_presets()["presets"]
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["args"]["query"], ["graph; neural|gnn"])
        self.assertIn("--oa-only", got[0]["cli"])
        self.app.delete_preset("gnn")
        self.assertEqual(self.app.get_presets()["presets"], [])
        with self.assertRaises(ApiError):
            self.app.delete_preset("gnn")
        with self.assertRaises(ApiError):
            self.app.post_preset({"name": "плохое имя!", "args": {"query": ["x"]}})

    def write_sites(self):
        site = {"name": "mysite", "title": "My Base", "search_url": "https://api.example.org/search",
                "query_param": "q", "results_path": "results", "fields": {"title": "title"}, "enabled": True}
        (self.cwd / "custom_sites.json").write_text(json.dumps({"version": 1, "sites": [site]}), encoding="utf-8")

    def test_sites_toggle_and_remove(self):
        self.write_sites()
        d = self.app.get_sites()
        self.assertEqual(len(d["custom"]), 1)
        self.assertTrue(d["custom"][0]["valid"])
        self.app.post_site("mysite", {"enabled": False})
        saved = json.loads((self.cwd / "custom_sites.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["sites"][0]["enabled"])
        self.assertIn("mysite", self.app.known_sources())
        self.app.delete_site("mysite")
        self.assertEqual(self.app.get_sites()["custom"], [])
        with self.assertRaises(ApiError):
            self.app.post_site("mysite", {"enabled": True})

    def test_sites_broken_file_guard(self):
        (self.cwd / "custom_sites.json").write_text("{не json", encoding="utf-8")
        self.assertTrue(self.app.get_sites()["error"])
        with self.assertRaises(ApiError) as cm:
            self.app.post_site("mysite", {"enabled": True})
        self.assertEqual(cm.exception.code, 409)


SITE_DATA = {"name": "mysite", "title": "My Base", "about": "Тестовая база",
             "search_url": "https://api.example.org/search", "query_param": "q",
             "results_path": "results", "total_path": "meta.count", "min_interval": 1,
             "fields": {"title": "title", "authors": "authors[].name", "year": "publication_year",
                        "doi": "doi", "abstract": "abstract", "pdf_url": "pdf_url"}}
SAMPLE = {"meta": {"count": 42}, "results": [
    {"title": "Graph neural networks", "authors": [{"name": "Smith J"}], "publication_year": 2023,
     "doi": "10.1/gnn", "abstract": "About graphs.", "pdf_url": "https://example.org/p.pdf"}]}


def sample_raw(self, url, headers, dest=None, max_bytes=None):
    return 200, {}, json.dumps(SAMPLE).encode()


class TestSiteForm(Tmp):
    def test_create_site(self):
        r = self.app.post_site_create(SITE_DATA)
        self.assertEqual(r["site"]["name"], "mysite")
        self.assertIn("USER-ADDED", r["llm_details"])
        self.assertIn("mysite", self.app.known_sources())
        saved = json.loads((self.cwd / "custom_sites.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["sites"][0]["fields"]["title"], "title")
        with self.assertRaises(ApiError) as cm:                            # повтор имени
            self.app.post_site_create(SITE_DATA)
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("имя", json.loads(cm.exception.message)[0])

    def test_create_validation(self):
        bad = {"name": "Bad Name", "search_url": "http://evil.com/x", "fields": {}}
        with self.assertRaises(ApiError) as cm:
            self.app.post_site_create(bad)
        errs = json.loads(cm.exception.message)
        self.assertTrue(any("имя" in e for e in errs))
        self.assertTrue(any("https" in e for e in errs))
        self.assertTrue(any("fields.title" in e for e in errs))
        with self.assertRaises(ApiError) as cm:                            # запрещённый политикой хост
            self.app.post_site_create({**SITE_DATA, "name": "shadow", "search_url": "https://sci-hub.se/api"})
        self.assertTrue(any("политик" in e.lower() for e in json.loads(cm.exception.message)))
        self.assertFalse((self.cwd / "custom_sites.json").exists())        # ничего не записано

    def test_probe_suggests_paths(self):
        self.app.post_config({"email": "t@e.org"})
        with mock.patch.object(PoliteClient, "_raw", sample_raw), \
             mock.patch.object(PoliteClient, "_sleep", lambda s, x: None):
            d = self.app.post_site_probe({"search_url": "https://api.example.org/search", "query_param": "q"})
        self.assertEqual(d["results_path"], "results")
        self.assertEqual(d["total_path"], "meta.count")
        self.assertEqual(d["fields"]["title"], "title")
        self.assertEqual(d["fields"]["authors"], "authors[].name")
        self.assertEqual(d["sample"]["doi"], "10.1/gnn")
        self.assertIn("meta", d["top"])

    def test_probe_requires_email_and_url(self):
        with self.assertRaises(ApiError) as cm:
            self.app.post_site_probe({"search_url": "https://api.example.org/search"})
        self.assertIn("e-mail", cm.exception.message)
        self.app.post_config({"email": "t@e.org"})
        with self.assertRaises(ApiError):
            self.app.post_site_probe({"search_url": ""})

    def test_site_test_parses_records(self):
        self.app.post_config({"email": "t@e.org"})
        with mock.patch.object(PoliteClient, "_raw", sample_raw), \
             mock.patch.object(PoliteClient, "_sleep", lambda s, x: None):
            d = self.app.post_site_test(SITE_DATA)
        self.assertEqual(d["found"], 42)
        self.assertEqual(len(d["records"]), 1)
        self.assertEqual(d["records"][0]["title"], "Graph neural networks")
        self.assertEqual(d["records"][0]["year"], 2023)
        self.assertEqual(d["warnings"], [])
        with self.assertRaises(ApiError) as cm:                            # невалидное описание — список ошибок
            self.app.post_site_test({**SITE_DATA, "fields": {}})
        self.assertTrue(any("fields.title" in e for e in json.loads(cm.exception.message)))


class TestJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jm = JobManager(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def wait(self, job, timeout=15):
        t0 = time.monotonic()
        while job.state == "running" and time.monotonic() - t0 < timeout:
            time.sleep(0.05)
        return job

    def test_run_and_capture(self):
        job = self.jm.start("manual", [sys.executable, "-c", "print('строка раз'); print('строка два')"], "тест")
        self.wait(job)
        self.assertEqual(job.state, "done")
        self.assertEqual(job.exit_code, 0)
        self.assertEqual(job.lines, ["строка раз", "строка два"])
        snap = self.jm.snapshot(job, since=1)
        self.assertEqual(snap["lines"], ["строка два"])
        self.assertEqual(snap["total"], 2)

    def test_run_dir_marker(self):
        code = "print('Библиотека: library  |  отчёт запуска: library/runs/20261122-120000')"
        job = self.jm.start("manual", [sys.executable, "-c", code], "тест")
        self.wait(job)
        self.assertEqual(job.run_dir, "library/runs/20261122-120000")
        self.assertEqual(job.summary()["run_id"], "20261122-120000")

    def test_error_exit(self):
        job = self.jm.start("manual", [sys.executable, "-c", "import sys; print('ой'); sys.exit(2)"], "тест")
        self.wait(job)
        self.assertEqual(job.state, "error")
        self.assertEqual(job.exit_code, 2)

    def test_busy_and_cancel(self):
        job = self.jm.start("manual", [sys.executable, "-c", "import time; time.sleep(60)"], "долгая")
        with self.assertRaises(ApiError) as cm:
            self.jm.start("manual", [sys.executable, "-c", "pass"], "вторая")
        self.assertEqual(cm.exception.code, 409)
        self.jm.cancel(job.id)
        self.wait(job)
        self.assertEqual(job.state, "cancelled")
        self.jm.start("manual", [sys.executable, "-c", "pass"], "третья")     # после отмены слот свободен

    def test_unknown_job(self):
        with self.assertRaises(ApiError) as cm:
            self.jm.get("нет-такой")
        self.assertEqual(cm.exception.code, 404)


class TestHttp(Tmp):
    """Дымовой тест настоящего HTTP-сервера на случайном порту."""

    def setUp(self):
        super().setUp()
        self.write_index()
        self.write_run()
        handler = type("BoundHandler", (Handler,), {"app": self.app})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        super().tearDown()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path):
        with urllib.request.urlopen(self.url(path), timeout=10) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    def test_index_html(self):
        status, ctype, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("litagent", body.decode("utf-8"))

    def test_state_and_config(self):
        status, _, body = self.get("/api/state")
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertIn("pubmed", state["sources"]["builtin"])
        self.assertEqual(state["library"]["pdf"], 1)
        req = urllib.request.Request(self.url("/api/config"), method="POST",
                                     data=json.dumps({"email": "t@e.org", "sources": ["pubmed"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(r.status, 200)
        self.assertTrue(self.app.config_path.exists())

    def test_files_and_traversal(self):
        status, ctype, body = self.get("/files/pdf/pubmed/Smith_2022_Test_1a2b3c4d.pdf")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "application/pdf")
        self.assertEqual(body, PDF)
        for bad in ("/files/../secret.txt", "/files/%2e%2e%2fsecret.txt", "/files/index.json"):
            (self.cwd / "secret.txt").write_text("секрет", encoding="utf-8")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.get(bad)
            self.assertIn(cm.exception.code, (403, 404))
            self.assertNotIn("секрет", cm.exception.read().decode("utf-8", "replace"))

    def test_run_detail_and_404(self):
        status, _, body = self.get("/api/runs/20261122-120000")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["results"]), 1)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/api/" + urllib.parse.quote("несуществует"))
        self.assertEqual(cm.exception.code, 404)

    def test_bad_json_body(self):
        req = urllib.request.Request(self.url("/api/config"), method="POST", data="{не json".encode(),
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 400)

    def post(self, path, obj):
        req = urllib.request.Request(self.url(path), method="POST", data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())

    def test_tags_and_delete_via_http(self):
        status, r = self.post("/api/library/tags", {"pk": "doi:10.1/x", "tags": ["CRISPR"]})
        self.assertEqual(status, 200)
        self.assertEqual(r["tags"], ["CRISPR"])
        _, _, lib = self.get("/api/library")
        self.assertEqual(json.loads(lib)["records"][0]["tags"], ["CRISPR"])
        status, r = self.post("/api/library/delete", {"keys": ["doi:10.1/x"]})
        self.assertEqual((status, r["deleted"], r["files"]), (200, 1, 1))
        _, _, lib = self.get("/api/library")
        self.assertEqual(json.loads(lib)["records"], [])

    def test_site_create_via_http(self):
        status, r = self.post("/api/sites", SITE_DATA)
        self.assertEqual(status, 200)
        self.assertEqual(r["site"]["name"], "mysite")
        _, _, sites = self.get("/api/sites")
        self.assertEqual(len(json.loads(sites)["custom"]), 1)

    def test_runs_delete_and_cache_clear_via_http(self):
        cache = self.cwd / "library" / ".http_cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "x").write_bytes(b"data")
        status, r = self.post("/api/cache/clear", {})
        self.assertEqual((status, r["files"]), (200, 1))
        status, r = self.post("/api/runs/delete", {"ids": ["20261122-120000"]})
        self.assertEqual((status, r["deleted"]), (200, 1))
        _, _, runs = self.get("/api/runs")
        self.assertEqual(json.loads(runs)["runs"], [])


if __name__ == "__main__":
    unittest.main()
