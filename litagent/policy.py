"""Политики доступа: интервалы по хостам, запрещённые URL, запрещённые источники."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

_DEFAULT = Path(__file__).with_name("policies.json")
_ARXIV_RE = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:pdf|abs)/(.+?)(?:\.pdf)?(?:[?#].*)?$", re.I)


class Policy:
    def __init__(self, path: str | Path | None = None):
        data = json.loads(Path(path or _DEFAULT).read_text(encoding="utf-8"))
        self.data = data
        self.default_interval = float(data.get("default_min_interval", 1.0))
        self.intervals: dict[str, float] = dict(data.get("intervals", {}))
        self._blocked = [re.compile(p, re.I) for p in data.get("blocked_url_patterns", [])]
        self.forbidden_sources: dict[str, str] = data.get("forbidden_sources", {})
        self.rules: list[str] = data.get("rules", [])
        self.sources_info: dict[str, dict] = data.get("sources", {})

    def interval(self, host: str) -> float:
        return self.intervals.get(host.lower(), self.default_interval)

    def url_allowed(self, url: str) -> tuple[bool, str]:
        for rx in self._blocked:
            if rx.search(url):
                return False, f"URL запрещён политикой ({rx.pattern})"
        return True, ""

    def rewrite(self, url: str) -> str:
        """arXiv: автоматическая загрузка только с export.arxiv.org."""
        m = _ARXIV_RE.match(url)
        if m:
            return f"https://export.arxiv.org/pdf/{m.group(1)}"
        return url

    @staticmethod
    def host(url: str) -> str:
        return urlparse(url).netloc.lower()
