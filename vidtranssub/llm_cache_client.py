"""Stage 6:透過外部 OpenAI-compatible LLM cache API 翻譯 cue。

穩定請求原則(規格 §6):
- 每個唯一 cue 一個穩定 request,確保跨影片也能命中相同翻譯。
- messages 不含 timestamp、bbox、sample index、檔名、request id 等每次執行會變的內容。
- 固定 temperature=0、stream=false,要求回傳等長 JSON 字串陣列。
- 回傳數量不符或無法解析時重試一次,再降級成每行一個穩定 request。
- 翻譯失敗預設保留原文;429 遵守 Retry-After;401/403 不重試。
- API key 只從環境變數或上游自身設定取得,不寫入 manifest/log。
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Protocol

import httpx

from .config import TRANSLATION_PROMPT_VERSION

CallChat = Callable[[list[dict]], str]  # messages -> assistant text


class TranslationClient(Protocol):
    def translate_cue(
        self, texts: list[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        ...


class TranslationError(RuntimeError):
    """無法從 LLM cache API 取得回應(連線失敗、認證失敗或多次重試後仍失敗)。"""


def build_messages(
    texts: list[str], source_lang: str | None, target_lang: str
) -> list[dict]:
    """建立穩定 messages;source/target language 與 prompt version 都在穩定內容中。"""
    src = source_lang or "auto"
    system = (
        f"{TRANSLATION_PROMPT_VERSION}. Translate {src} into {target_lang}."
        " Return only a JSON array with exactly the same number of strings,"
        " one translated string per input string, in the same order."
        " Do not merge, split, add, or omit any element. Do not add commentary."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
    ]


def parse_json_array(text: str) -> list[str] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    return [str(x) for x in arr]


def translate_texts(
    texts: list[str],
    source_lang: str | None,
    target_lang: str,
    call_chat: CallChat,
) -> tuple[list[str], list[int]]:
    """翻譯一個 cue 的多行原文,回傳 (譯文, 保留原文的行索引)。

    整批(JSON 陣列)先試一次 + 重試一次;仍失敗才降級成每行一個穩定 request。
    connection/HTTP 層錯誤由 call_chat 直接拋出(不在此吞掉),讓上層保留斷點後退出。
    """
    if not texts:
        return [], []

    translated: list[str] = list(texts)
    failed: list[int] = []

    result: list[str] | None = None
    for _ in range(2):  # 第一次 + 重試一次
        out = parse_json_array(call_chat(build_messages(texts, source_lang, target_lang)))
        if out is not None and len(out) == len(texts):
            result = out
            break

    if result is not None:
        return result, []

    # 降級:每行一個穩定 request(仍是穩定內容,可跨影片命中)。
    for i, line in enumerate(texts):
        out = parse_json_array(
            call_chat(build_messages([line], source_lang, target_lang))
        )
        if out is not None and len(out) == 1:
            translated[i] = out[0]
        else:
            failed.append(i)  # 保留原文
    return translated, failed


def resolve_model(
    base_url: str,
    llm_model: str | None,
    timeout: float = 10.0,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """未指定 --llm-model 時,取上游第一個模型;連線失敗時有限重試(§10)。"""
    if llm_model:
        return llm_model
    url = f"{base_url.rstrip('/')}/v1/models"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = httpx.get(url, timeout=timeout)
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
            if models:
                return models[0]
            raise TranslationError(
                f"LLM cache API 未回傳任何模型({base_url});請以 --llm-model 指定"
            )
        except httpx.HTTPError as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep(min(2.0 * (attempt + 1), 10.0))
                continue
    raise TranslationError(
        f"無法從 LLM cache API 取得模型清單({base_url}):{last_err};請以 --llm-model 指定"
    )


class LLMCacheClient:
    """OpenAI-compatible LLM cache API 的翻譯 client。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 600.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))
        self.cache_hits = 0
        self.cache_misses = 0
        self.upstream_calls = 0
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMCacheClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _chat(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": False,
        }
        url = f"{self.base_url}/v1/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(url, json=body)
            except httpx.HTTPError as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    self._sleep(min(2.0 * (attempt + 1), 10.0))
                    continue
                raise TranslationError(f"LLM cache API 連線失敗:{e}") from e

            if r.status_code == 200:
                self.request_count += 1
                hit = r.headers.get("x-vtf-cache", "").lower() == "hit"
                if hit:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                    self.upstream_calls += 1
                return r.json()["choices"][0]["message"]["content"]

            if r.status_code in (401, 403):
                raise TranslationError(
                    f"LLM cache API 認證失敗({r.status_code});API key 由環境變數或上游設定,不重試"
                )

            if r.status_code == 429:
                retry_after = _parse_retry_after(r.headers.get("Retry-After"))
                if attempt < self.max_retries - 1:
                    self._sleep(retry_after if retry_after is not None else 2.0 * (attempt + 1))
                    continue
                raise TranslationError("LLM cache API 持續回傳 429(rate limit)")

            # 其他 5xx/4xx:有限重試
            last_err = TranslationError(f"LLM cache API 回傳 {r.status_code}:{r.text[:500]}")
            if attempt < self.max_retries - 1:
                self._sleep(2.0 * (attempt + 1))
                continue
            raise last_err

        raise TranslationError(f"LLM cache API 重試耗盡:{last_err}")

    def translate_cue(
        self, texts: list[str], source_lang: str | None, target_lang: str
    ) -> list[str]:
        translated, _failed = translate_texts(texts, source_lang, target_lang, self._chat)
        return translated

    def translate_cue_detailed(
        self, texts: list[str], source_lang: str | None, target_lang: str
    ) -> tuple[list[str], list[int]]:
        return translate_texts(texts, source_lang, target_lang, self._chat)


def _parse_retry_after(value: str | None) -> float | None:
    """支援 RFC-7231 的兩種形式:delta-seconds 或 HTTP-date。"""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None
