"""OpenAI-compatible LLM 翻譯快取 proxy(真正的「LLM cache API」服務端)。

VideoTransSub 本身只實作呼叫端([llm_cache_client.py]);此模組補上與其對齊的
server:對唯一 request 快取翻譯結果,cache miss 時轉發到一個 OpenAI-compatible
上游(例如 Ollama、llama.cpp,或任何相容端點),把回應存檔後回傳。

對齊 client 與規格 §6 的介面:
- POST /v1/chat/completions:以「正規化後的完整 request body」SHA-256 為 cache key;
  命中回應帶 header ``x-vtf-cache: hit``,未命中轉發上游成功後存檔並帶 ``miss``。
- GET  /v1/models:轉發上游模型清單(client 的 resolve_model 用)。
- GET  /vtf/stats:回 ``{hits, misses, entries}``。

設計要點:
- cache key 正規化與 tests/mock_llm.py 一致(丟掉 stream/request_id 等每次會變的欄位),
  因此每個唯一 cue 的穩定 request 可跨影片命中。
- 上游 API key 只從環境變數讀取,不寫入 log/DB;client → 本 proxy 之間不需金鑰。
- 只支援非串流(client 固定 ``stream=false``);轉發上游時一律以非串流取得完整 JSON 以供快取。
- 錯誤回應不快取;429 的 ``Retry-After`` 會原樣轉回,讓 client 遵守。

FastAPI/uvicorn 屬 ``[server]`` extra,於函式內延遲匯入,未安裝時給明確提示。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol

import httpx

# fastapi/uvicorn 屬 [server] extra。於模組頂端匯入(而非函式內),路由的 `Request`
# 型別註解在 `from __future__ import annotations` 下才能由模組命名空間正確解析;
# 缺相依時設為 None,由 main() 給出明確安裝提示。
try:  # pragma: no cover - 視是否安裝 [server] 而定
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = Request = JSONResponse = None  # type: ignore[assignment,misc]

# 計算 cache key 時忽略的欄位(每次執行會變、或與翻譯結果無關)。與 mock_llm 對齊。
DROP_KEYS = {"stream", "stream_options", "request_id", "id", "metadata", "user"}


def normalize_key(body: dict) -> str:
    """以正規化後的 request body 計算穩定 cache key(規格 §6)。"""
    norm = {k: v for k, v in body.items() if k not in DROP_KEYS}
    payload = json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheStore:
    """持久化 SQLite 快取:cache_key -> 完整 response JSON。

    允許跨執行緒共用單一連線(uvicorn 以執行緒池跑同步路由),以 lock 序列化存取。
    """

    def __init__(self, db_path: str | os.PathLike):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            " cache_key TEXT PRIMARY KEY,"
            " response TEXT NOT NULL,"
            " created_at REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, response: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, response, created_at)"
                " VALUES (?, ?, ?)",
                (key, json.dumps(response, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Upstream(Protocol):
    """上游 OpenAI-compatible 端點介面(以此隔離,方便測試注入假上游)。"""

    def chat(self, body: dict) -> tuple[int, dict, dict]:
        """回傳 (status_code, json_body, extra_headers)。"""
        ...

    def models(self) -> tuple[int, dict]:
        """回傳 (status_code, json_body)。"""
        ...


class HttpUpstream:
    """轉發到 OpenAI-compatible 上游(Ollama / llama.cpp / 其他相容端點)。"""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat(self, body: dict) -> tuple[int, dict, dict]:
        # 強制非串流,確保取得完整 JSON 以供快取。
        forwarded = {**body, "stream": False}
        r = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=forwarded,
            headers=self._headers(),
        )
        extra: dict = {}
        if "Retry-After" in r.headers:
            extra["Retry-After"] = r.headers["Retry-After"]
        try:
            data = r.json()
        except json.JSONDecodeError:
            data = {"error": {"message": r.text[:500]}}
        return r.status_code, data, extra

    def models(self) -> tuple[int, dict]:
        r = self._client.get(f"{self.base_url}/v1/models", headers=self._headers())
        try:
            return r.status_code, r.json()
        except json.JSONDecodeError:
            return r.status_code, {"object": "list", "data": []}

    def close(self) -> None:
        self._client.close()


def _models_fallback(default_model: str | None) -> dict:
    data = [{"id": default_model, "object": "model"}] if default_model else []
    return {"object": "list", "data": data}


def create_app(upstream: Upstream, store: CacheStore, default_model: str | None = None):
    """建立 FastAPI app;hits/misses 為記憶體計數(client 以前後差值算命中率)。"""
    app = FastAPI(title="VideoTransSub LLM cache proxy")
    app.state.hits = 0
    app.state.misses = 0

    @app.get("/v1/models")
    async def models():
        try:
            status, data = upstream.models()
        except httpx.HTTPError:
            return JSONResponse(content=_models_fallback(default_model))
        if status != 200:
            if default_model:
                return JSONResponse(content=_models_fallback(default_model))
            return JSONResponse(status_code=status, content=data)
        return JSONResponse(content=data)

    @app.get("/vtf/stats")
    async def stats():
        return {
            "hits": app.state.hits,
            "misses": app.state.misses,
            "entries": store.count(),
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        key = normalize_key(body)

        cached = store.get(key)
        if cached is not None:
            app.state.hits += 1
            return JSONResponse(content=cached, headers={"x-vtf-cache": "hit"})

        try:
            status, data, extra = upstream.chat(body)
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"上游連線失敗:{e}"}},
                headers={"x-vtf-cache": "miss"},
            )

        if status != 200:
            # 錯誤不快取;保留 Retry-After 讓 client 遵守 429。
            headers = {"x-vtf-cache": "miss"}
            if "Retry-After" in extra:
                headers["Retry-After"] = extra["Retry-After"]
            return JSONResponse(status_code=status, content=data, headers=headers)

        app.state.misses += 1
        store.put(key, data)
        return JSONResponse(content=data, headers={"x-vtf-cache": "miss"})

    return app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidtranssub-llm-cache",
        description="OpenAI-compatible LLM 翻譯快取 proxy(cache miss 轉發到上游)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument(
        "--upstream-url",
        required=True,
        help="OpenAI-compatible 上游 base URL,例如 http://127.0.0.1:11434(Ollama)"
        " 或 http://127.0.0.1:8080(llama.cpp)",
    )
    p.add_argument(
        "--api-key-env",
        default="LLM_API_KEY",
        help="存放上游 API key 的環境變數名稱(上游免金鑰時可忽略)",
    )
    p.add_argument("--db", default="./videosub_llm_cache.db", help="SQLite 快取檔路徑")
    p.add_argument(
        "--default-model",
        default=None,
        help="上游 /v1/models 無法取得時,回報給 client 的模型名稱",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if FastAPI is None:
        raise SystemExit(
            "需要 server 相依(fastapi + uvicorn)。請執行:"
            ' pip install "vidtranssub[server]"'
        )
    import uvicorn

    api_key = os.environ.get(args.api_key_env) or None
    upstream = HttpUpstream(args.upstream_url, api_key=api_key)
    store = CacheStore(args.db)
    app = create_app(upstream, store, default_model=args.default_model)

    print(
        f"[llm-cache] 監聽 http://{args.host}:{args.port}"
        f"  上游={args.upstream_url}"
        f"  金鑰={'有' if api_key else '無'}"
        f"  DB={args.db}"
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        upstream.close()
        store.close()


if __name__ == "__main__":
    main()
