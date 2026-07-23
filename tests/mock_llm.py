"""測試用的假 LLM cache API(OpenAI-compatible)。

行為對齊真實 LLM cache proxy:
- POST /v1/chat/completions:讀 user 訊息中的 JSON 陣列,回等長 "[譯] "+x 陣列。
  以正規化後的 request body 計算 cache key,回應帶 x-vtf-cache: hit|miss。
- GET /vtf/stats:回 hits/misses/entries。
- GET /v1/models:回一個模型。

獨立執行:python tests/mock_llm.py --port 8790
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DROP_KEYS = {"stream", "stream_options", "request_id", "id", "metadata", "user"}


def _normalize_key(body: dict) -> str:
    norm = {k: v for k, v in body.items() if k not in DROP_KEYS}
    payload = json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _find_json_array(text: str) -> list[str] | None:
    start = text.find("[")
    while start != -1:
        end = text.find("]", start)
        while end != -1:
            try:
                arr = json.loads(text[start : end + 1])
                if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                    return arr
            except json.JSONDecodeError:
                pass
            end = text.find("]", end + 1)
        start = text.find("[", start + 1)
    return None


def create_mock_llm() -> FastAPI:
    app = FastAPI(title="Mock LLM cache API")
    app.state.cache = {}
    app.state.hits = 0
    app.state.misses = 0

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "mock-translator", "object": "model"}]}

    @app.get("/vtf/stats")
    async def stats():
        return {
            "hits": app.state.hits,
            "misses": app.state.misses,
            "entries": len(app.state.cache),
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        key = _normalize_key(body)
        cached = app.state.cache.get(key)
        if cached is not None:
            app.state.hits += 1
            return JSONResponse(content=cached, headers={"x-vtf-cache": "hit"})

        app.state.misses += 1
        user = next(
            (m["content"] for m in reversed(body.get("messages", []))
             if m.get("role") == "user"),
            "",
        )
        arr = _find_json_array(user)
        if arr is not None:
            content = json.dumps([f"[譯] {x}" for x in arr], ensure_ascii=False)
        else:
            content = f"[譯] {user[:200]}"
        response = {
            "id": "mock-1",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-translator"),
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        app.state.cache[key] = response
        return JSONResponse(content=response, headers={"x-vtf-cache": "miss"})

    return app


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    uvicorn.run(create_mock_llm(), host="127.0.0.1", port=args.port, log_level="warning")
