"""Парсер «важных пометок» из аналитического .tex (карточки syscard, warnbox/notebox, комментарии % AGENT:)
и из обычных .md/.txt заметок пользователя."""
from __future__ import annotations

import re
from pathlib import Path

# заголовок карточки -> имя источника в litagent
_TITLE_MAP = [
    (r"pubmed central", "pmc"), (r"pubmed", "pubmed"), (r"europe pmc", "europepmc"), (r"arxiv", "arxiv"),
    (r"openalex", "openalex"), (r"crossref", "crossref"), (r"semantic scholar", "semanticscholar"),
    (r"\bcore\b", "core"), (r"unpaywall", "unpaywall"), (r"doaj", "doaj"), (r"dblp", "dblp"),
    (r"google scholar", "google_scholar"),
]
ROW_SYNTAX, ROW_PITFALLS, ROW_LIMITS, ROW_BLOCK, ROW_STATUS = (
    "Синтаксис запросов", "Подводные камни", "Лимиты", "Блокировка веб-агентов", "Статус данных")


def read_group(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{'. Возвращает (содержимое, индекс после закрывающей скобки)."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def clean_tex(s: str) -> str:
    for a, b in (("\\%", "%"), ("\\_", "_"), ("\\&", "&"), ("\\#", "#"), ("\\$", "$"), ("\\{", "{"), ("\\}", "}")):
        s = s.replace(a, "\x00" + b)
    s = s.replace("---", "—").replace("--", "–").replace("~", " ").replace("\\,", " ").replace("\\ldots", "...")
    s = s.replace("$\\rightarrow$", "→").replace("$\\leftrightarrow$", "↔")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s).replace("\\\\", " ")
    s = s.replace("{", "").replace("}", "").replace("\x00", "")
    return re.sub(r"\s+", " ", s).strip()


def map_title(title: str) -> str | None:
    t = clean_tex(title).lower()
    for pat, name in _TITLE_MAP:
        if re.search(pat, t):
            return name
    return None


class Notes:
    def __init__(self):
        self.systems: dict[str, dict[str, str]] = {}     # имя источника -> {метка строки: текст}
        self.global_notes: list[tuple[str, str]] = []    # (заголовок, текст) из warnbox/notebox
        self.agent_lines: list[str] = []                 # из комментариев "% AGENT: ..."
        self.user_text: str = ""                         # из .md/.txt

    # -------- загрузка
    @classmethod
    def load(cls, *paths: str | Path) -> "Notes":
        n = cls()
        for p in paths:
            if not p:
                continue
            p = Path(p)
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix.lower() == ".tex":
                n._parse_tex(text)
            else:
                n.user_text += ("\n" if n.user_text else "") + text.strip()
        return n

    def _parse_tex(self, tex: str) -> None:
        self.agent_lines += [m.group(1).strip() for m in re.finditer(r"^\s*%\s*AGENT:\s*(.+)$", tex, re.M)]
        tex = re.sub(r"(?<!\\)%.*", "", tex)          # убрать комментарии
        for env in ("warnbox", "notebox"):
            for m in re.finditer(r"\\begin\{%s\}(?:\[([^\]]*)\])?(.*?)\\end\{%s\}" % (env, env), tex, re.S):
                self.global_notes.append((clean_tex(m.group(1) or env), clean_tex(m.group(2))))
        for m in re.finditer(r"\\begin\{syscard\}\s*", tex):
            title, j = read_group(tex, m.end())
            end = tex.find("\\end{syscard}", j)
            body = tex[j:end if end != -1 else len(tex)]
            name = map_title(title)
            if not name:
                continue
            rows = self.systems.setdefault(name, {})
            for rm in re.finditer(r"\\row\s*\{", body):
                label, k = read_group(body, rm.end() - 1)
                while k < len(body) and body[k] in " \n\t":
                    k += 1
                if k < len(body) and body[k] == "{":
                    text, _ = read_group(body, k)
                    rows[clean_tex(label)] = clean_tex(text)

    # -------- использование
    def warnings(self, source: str) -> list[str]:
        st = self.systems.get(source, {}).get(ROW_STATUS, "").lower()
        out = []
        if "противореч" in st:
            out.append("данные о лимитах противоречивы — сверьте с сайтом сервиса")
        elif "вторичн" in st and "официальн" not in st:
            out.append("сведения только из вторичных источников")
        return out

    def for_sources(self, names: list[str], rows=(ROW_SYNTAX, ROW_PITFALLS, ROW_LIMITS, ROW_BLOCK),
                    maxlen: int = 700) -> str:
        blocks = []
        for n in names:
            data = self.systems.get(n)
            if not data:
                continue
            lines = []
            for r in rows:
                if r in data:
                    t = data[r]
                    lines.append(f"  {r}: {t[:maxlen]}{'…' if len(t) > maxlen else ''}")
            if lines:
                blocks.append(f"[{n}]\n" + "\n".join(lines))
        return "\n".join(blocks)

    def global_text(self, maxlen: int = 900) -> str:
        parts = [f"{t}: {x[:maxlen]}" for t, x in self.global_notes]
        parts += [f"AGENT note: {a}" for a in self.agent_lines]
        return "\n".join(parts)

    def user_notes(self, maxlen: int = 3000) -> str:
        return self.user_text[:maxlen]
