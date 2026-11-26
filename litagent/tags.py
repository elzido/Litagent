"""Авто-теги по ключевым словам: чистка keyword-полей источников + разбор заголовка/аннотации.

Ключевые слова приходят мусорными строками («Crispr/cas; Gene Editing, survey 2023», один текст на все темы),
поэтому каждая фраза нормализуется, разбивается по разделителям, отсеивается стоп-лист и шум — получается множество тегов.
"""
from __future__ import annotations

import re

MAX_TAG_WORDS = 4
MAX_TAG_LEN = 32
DEFAULT_LIMIT = 6

STOP = {
    # служебные (EN)
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "between", "both", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "in", "into", "is", "it", "its", "may",
    "not", "of", "on", "or", "our", "per", "than", "that", "the", "their", "them", "this", "those", "to",
    "under", "up", "upon", "us", "using", "was", "we", "were", "what", "when", "which", "while", "who",
    "why", "with", "without", "within", "via",
    # научный шум (EN)
    "survey", "study", "studies", "analysis", "analyses", "approach", "article", "based", "data", "effect", "effects",
    "method", "methods", "model", "models", "new", "novel", "paper", "potential", "proposed", "result",
    "results", "review", "role", "systematic", "used",
    # биомедицинские MeSH-демографические маркеры
    "adult", "aged", "animals", "child", "female", "humans", "infant", "male", "middle aged", "adolescent",
    "aged 80 and over", "young adult",
    # служебные (RU)
    "анализ", "исследование", "исследования", "метод", "методы", "новой", "новый", "обзор", "получены",
    "работа", "результаты", "статья", "целью", "данные",
}
_NUM_ONLY = re.compile(r"^[\d\W_]+$")
_WORD = re.compile(r"[\w'\-/À-ſЁё]+", re.U)   # «/» внутри термина (Crispr/cas) сохраняем
_SPLIT = re.compile(r"\s*[;|,\n]\s*")       # «/» не разделяем: Crispr/cas — один термин


def _ok_phrase(p: str, keep_case: bool = False) -> str | None:
    """Нормализовать фразу-кандидат в тег или вернуть None. keep_case — сохранить регистр источника
    (keyword-поля уже размечены людьми: GNN, CRISPR-Cas9); обычные фразы заголовка — в нижний регистр."""
    p = re.sub(r"\s+", " ", p).strip(" .,:;()[]{}\"'“”‘’–—-")
    if not p or _NUM_ONLY.match(p):
        return None
    words = _WORD.findall(p)
    if not words or len(words) > MAX_TAG_WORDS:
        return None
    low = " ".join(words).casefold()
    if low in STOP or len(low) < 3 or len(low) > MAX_TAG_LEN:
        return None
    if all(w.casefold() in STOP for w in words):
        return None
    if keep_case:
        return " ".join(words)
    # капитализированные фразы (названия методов/белков) сохраняем как есть, прочие — в нижний регистр
    return " ".join(words) if any(w[:1].isupper() for w in words[1:]) else " ".join(w.lower() for w in words)


def split_keywords(raw) -> list[str]:
    """Значение keyword-поля (список/строка/вложенные словари) -> чистые фразы-кандидаты."""
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        for part in _SPLIT.split(str(s)):
            tag = _ok_phrase(part, keep_case=True)
            if tag and tag.casefold() not in seen:
                seen.add(tag.casefold())
                out.append(tag)

    def walk(v) -> None:
        if isinstance(v, str):
            add(v)
        elif isinstance(v, dict):
            for t in v.get("keyword") or v.get("terms") or v.get("value") or []:
                walk(t)
            if not any(k in v for k in ("keyword", "terms", "value")):
                for t in v.values():
                    if isinstance(t, str):
                        add(t)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(raw)
    return out


def title_phrases(title: str) -> list[str]:
    """Смысловые фразы из заголовка: капитализированные последовательности и содержательные n-граммы;
    короткая фраза, входящая в уже взятую длинную (или наоборот), не добавляется."""
    out: list[str] = []
    picked: list[list[str]] = []

    def subseq(small: list[str], big: list[str]) -> bool:
        return any(big[i:i + len(small)] == small for i in range(len(big) - len(small) + 1))

    def add(p: str) -> None:
        tag = _ok_phrase(p)
        if not tag:
            return
        words = [w.casefold() for w in _WORD.findall(tag)]
        if any(subseq(words, p_) or subseq(p_, words) for p_ in picked):
            return
        picked.append(words)
        out.append(tag)

    t = re.sub(r"\s+", " ", str(title or "")).strip()
    if not t:
        return out
    # капитализированные последовательности в середине заголовка — термины, а не начало предложения
    for m in re.finditer(r"(?<!^)(?:[A-Z][A-Za-z'*\-/]+)(?:\s+[A-Z][A-Za-z'*\-/]+)+", t):
        add(m.group(0))
    toks = _WORD.findall(t)
    grams = [toks[i:i + n] for n in (3, 2) for i in range(len(toks) - n + 1)]
    for g in grams:
        if all(w.casefold() not in STOP for w in g):
            add(" ".join(g))
    for w in toks:
        add(w)
    return out


def text_terms(text: str, limit: int) -> list[str]:
    """Термин из тела текста, встречающийся >= 2 раз (частотная подсказка при пустых keyword-полях)."""
    words = [w for w in _WORD.findall(str(text or "")) if len(w) >= 5 and w.casefold() not in STOP]
    freq: dict[str, int] = {}
    disp: dict[str, str] = {}
    for w in words:
        k = w.casefold().rstrip("s")
        freq[k] = freq.get(k, 0) + 1
        disp.setdefault(k, w)
    best = sorted(((n, k) for k, n in freq.items() if n >= 2), reverse=True)[:limit]
    return [disp[k] for _, k in best]


def suggest_tags(rec, limit: int = DEFAULT_LIMIT) -> list[str]:
    """Теги для одной записи: сначала keywords/arXiv-категории, затем фразы заголовка, затем частотные слова аннотации."""
    out: list[str] = []
    seen: set[str] = set()

    def add(tag: str) -> None:
        if tag and tag.casefold() not in seen:
            seen.add(tag.casefold())
            out.append(tag)

    for tag in split_keywords(getattr(rec, "keywords", None)):
        add(tag)
    if len(out) < min(3, limit):
        for tag in title_phrases(rec.title)[:limit]:
            add(tag)
    if len(out) < min(2, limit):
        for tag in text_terms(rec.abstract, limit):
            add(tag)
    return out[:limit]
