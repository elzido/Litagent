"""Клиент локальной LLM: Ollama (/api/chat) или OpenAI-совместимый сервер (LM Studio, llama.cpp, vLLM)."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def extract_json(text: str):
    """Достаёт первый JSON-объект из ответа модели (с ```json-заборами и мусором вокруг)."""
    if not text:
        raise ValueError("пустой ответ")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:json)?", "", text, flags=re.I)
    start = text.find("{")
    if start == -1:
        raise ValueError("нет JSON-объекта")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON не закрыт")


class LLM:
    def __init__(self, provider: str = "ollama", base_url: str | None = None, model: str = "gemma4",
                 temperature: float = 0.2, timeout: int = 300, api_key: str | None = None,
                 num_ctx: int = 8192):
        self.provider = provider
        self.num_ctx = int(num_ctx or 8192)
        self.last_info: dict = {}            # диагностика последнего ответа (токены, причина остановки)
        self.base_url = (base_url or ("http://127.0.0.1:11434" if provider == "ollama" else "http://localhost:1234/v1")).rstrip("/")
        self.model, self.temperature, self.timeout, self.api_key = model, temperature, timeout, api_key

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"LLM HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"LLM недоступна по {self.base_url}: {e}") from e

    def _ctx_for(self, msgs: list) -> int:
        """Контекст Ollama по умолчанию (2-4k токенов) молча обрезает длинный промпт. Берём num_ctx из настроек,
        а если промпт (русский текст ~2.5 симв./токен) в него не влезает — увеличиваем, но не выше 32768."""
        need = int(sum(len(m["content"]) for m in msgs) / 2.5) + 1200
        ctx = self.num_ctx
        while ctx < need and ctx < 32768:
            ctx *= 2
        return min(ctx, 32768)

    def chat(self, system: str, user: str, json_mode: bool = True) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.provider == "ollama":
            payload = {"model": self.model, "messages": msgs, "stream": False,
                       "think": False,       # think-модели иначе тратят токены на рассуждения и отдают пустой content
                       "options": {"temperature": self.temperature, "num_ctx": self._ctx_for(msgs)}}
            if json_mode:
                payload["format"] = "json"
            try:
                d = self._post("/api/chat", payload)
            except LLMError as e:
                if "think" not in str(e).lower():
                    raise
                payload.pop("think")          # старый Ollama / модель без поддержки think
                d = self._post("/api/chat", payload)
            self.last_info = {"prompt_tokens": d.get("prompt_eval_count"), "num_ctx": payload["options"]["num_ctx"],
                              "done_reason": d.get("done_reason")}
            msg = d.get("message") or {}
            text = msg.get("content") or ""
            if not text.strip():
                why = f"done_reason={d.get('done_reason')}"
                if msg.get("thinking"):
                    why += ", модель потратила ответ на рассуждения (think)"
                raise ValueError(f"пустой ответ модели ({why})")
            return text
        payload = {"model": self.model, "messages": msgs, "temperature": self.temperature}
        d = self._post("/chat/completions", payload)
        try:
            return d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"неожиданный ответ LLM: {str(d)[:200]}") from e

    def chat_json(self, system: str, user: str, retries: int = 2) -> dict:
        last = None
        for i in range(retries + 1):
            hint = "" if i == 0 else "\n\nYour previous reply was not valid JSON. Reply with ONE valid JSON object and nothing else."
            try:
                return extract_json(self.chat(system, user + hint))
            except ValueError as e:
                last = e
        raise LLMError(f"модель не вернула валидный JSON: {last}")

    def ping(self) -> bool:
        try:
            if self.provider == "ollama":
                urllib.request.urlopen(self.base_url + "/api/tags", timeout=5).read()
            else:
                urllib.request.urlopen(self.base_url + "/models", timeout=5).read()
            return True
        except Exception:  # noqa: BLE001
            return False
