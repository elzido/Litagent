"""Клиент LLM: Ollama (/api/chat), OpenAI-совместимый сервер (LM Studio, llama.cpp, vLLM) или GigaChat (облако Сбера)."""
from __future__ import annotations

import base64
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

GIGACHAT_URL = "https://api.giga.chat/v1"
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_SCOPES = ("GIGACHAT_API_PERS", "GIGACHAT_API_CORP", "GIGACHAT_API_B2B")
GIGACHAT_CA_URL = "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt"


def gigachat_key(client_id: str, client_secret: str) -> str:
    """Ключ авторизации GigaChat: Base64 от «ClientID:ClientSecret». Пробелы и переносы по краям убираются."""
    return base64.b64encode(f"{client_id.strip()}:{client_secret.strip()}".encode()).decode()


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
                 num_ctx: int = 8192, scope: str = "GIGACHAT_API_PERS", ca_bundle: str | None = None,
                 oauth_url: str | None = None):
        self.provider = provider
        self.num_ctx = int(num_ctx or 8192)
        self.last_info: dict = {}            # диагностика последнего ответа (токены, причина остановки)
        if provider == "gigachat":
            if model in ("", "gemma4"):      # gemma4 — дефолт Ollama из конфига, для GigaChat не существует
                model = "GigaChat"
            base_url = base_url or GIGACHAT_URL
        else:
            base_url = base_url or ("http://127.0.0.1:11434" if provider == "ollama" else "http://localhost:1234/v1")
        self.base_url = base_url.rstrip("/")
        self.model, self.temperature, self.timeout, self.api_key = model, temperature, timeout, api_key
        self.scope = scope or "GIGACHAT_API_PERS"
        if self.scope not in GIGACHAT_SCOPES:
            raise ValueError(f"неизвестный scope GigaChat: {self.scope} (допустимы: {', '.join(GIGACHAT_SCOPES)})")
        self.ca_bundle = ca_bundle or None
        self.oauth_url = (oauth_url or GIGACHAT_OAUTH_URL).rstrip("/")
        self._gc_token: str | None = None
        self._gc_expires = 0.0               # мс, как в ответе OAuth

    def _ssl_ctx(self):
        """GigaChat ходит по TLS с сертификатом Минцифры: либо ca_bundle из настроек, либо системное хранилище."""
        if self.provider == "gigachat" and self.ca_bundle:
            return ssl.create_default_context(cafile=self.ca_bundle)
        return None

    def _net_error_text(self, e: Exception, where: str) -> str:
        if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            return (f"GigaChat ({where}): не прошёл проверку сертификата — сервер использует сертификат Минцифры РФ. "
                    f"Установите корневой сертификат Russian Trusted Root CA ({GIGACHAT_CA_URL}) в систему, "
                    f"либо укажите PEM-файл: [llm] ca_bundle / --gigachat-ca-bundle / переменная SSL_CERT_FILE.")
        return f"LLM недоступна по {where}: {e}"

    def _gigachat_token(self, force: bool = False, timeout: int | None = None) -> str:
        """Access token GigaChat (живёт 30 минут): OAuth по ключу авторизации, кэшируется до истечения."""
        now = time.time() * 1000
        if not force and self._gc_token and now < self._gc_expires - 60_000:
            return self._gc_token
        if not self.api_key:
            raise LLMError("GigaChat: нужны Client ID и Client Secret — [keys] gigachat_client_id и gigachat_client_secret "
                           "в litagent.toml (или переменные GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET); "
                           "значения выдаются в личном кабинете Sber Studio")
        req = urllib.request.Request(
            self.oauth_url, data=urllib.parse.urlencode({"scope": self.scope}).encode(),
            headers={"Authorization": f"Basic {self.api_key}", "RqUID": str(uuid.uuid4()),
                     "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout, context=self._ssl_ctx()) as r:
                d = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"GigaChat OAuth HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')} "
                           f"(проверьте ключ авторизации и scope {self.scope})") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(self._net_error_text(e, self.oauth_url)) from e
        token = d.get("access_token")
        if not token:
            raise LLMError(f"GigaChat OAuth не вернул токен: {str(d)[:200]}")
        self._gc_token = token
        self._gc_expires = float(d.get("expires_at") or now + 30 * 60 * 1000)
        return token

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.provider == "gigachat":
            headers["Authorization"] = f"Bearer {self._gigachat_token()}"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in (1, 2):
            req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_ctx()) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if self.provider == "gigachat" and e.code == 401 and attempt == 1:
                    e.close()
                    headers["Authorization"] = f"Bearer {self._gigachat_token(force=True)}"   # токен протух раньше срока
                    continue
                raise LLMError(f"LLM HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise LLMError(self._net_error_text(e, self.base_url)) from e

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
        if self.provider == "gigachat":
            payload["stream"] = False        # JSON-режима у GigaChat нет: chat_json сам вытаскивает JSON и переспрашивает
        d = self._post("/chat/completions", payload)
        if self.provider == "gigachat":
            self.last_info = {"prompt_tokens": (d.get("usage") or {}).get("prompt_tokens"),
                              "done_reason": (d.get("choices") or [{}])[0].get("finish_reason")}
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
            elif self.provider == "gigachat":
                token = self._gigachat_token(timeout=10)
                req = urllib.request.Request(self.base_url + "/models",
                                             headers={"Authorization": f"Bearer {token}"})
                urllib.request.urlopen(req, timeout=10, context=self._ssl_ctx()).read()
            else:
                urllib.request.urlopen(self.base_url + "/models", timeout=5).read()
            return True
        except Exception:  # noqa: BLE001
            return False
