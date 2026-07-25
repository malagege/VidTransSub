"""LLM cache proxy(llm_cache_server)測試。

以 FastAPI TestClient + 假上游驗證:cache miss/hit、x-vtf-cache header、
/vtf/stats、/v1/models 轉發,以及跨執行(SQLite 持久化)命中。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")  # server 屬 [server]/[dev] extra
from fastapi.testclient import TestClient  # noqa: E402

from vidtranssub.llm_cache_server import CacheStore, create_app, normalize_key  # noqa: E402


class FakeUpstream:
    """記錄呼叫次數的假上游;chat 回等長「TR:」前綴譯文。"""

    def __init__(self, models_status: int = 200):
        self.chat_calls = 0
        self.models_calls = 0
        self.models_status = models_status

    def chat(self, body: dict) -> tuple[int, dict, dict]:
        self.chat_calls += 1
        user = next(
            (m["content"] for m in reversed(body.get("messages", []))
             if m.get("role") == "user"),
            "",
        )
        content = f"TR:{user}"
        return (
            200,
            {
                "id": "fake-1",
                "object": "chat.completion",
                "model": body.get("model", "fake-model"),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}
                ],
            },
            {},
        )

    def models(self) -> tuple[int, dict]:
        self.models_calls += 1
        return self.models_status, {
            "object": "list",
            "data": [{"id": "fake-model", "object": "model"}],
        }


def _body(text: str, model: str = "fake-model") -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "translate"},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "stream": False,
    }


@pytest.fixture
def client_and_upstream(tmp_path):
    upstream = FakeUpstream()
    store = CacheStore(str(tmp_path / "cache.db"))
    app = create_app(upstream, store, default_model="fallback-model")
    with TestClient(app) as client:
        yield client, upstream, store
    store.close()


def test_first_call_is_miss_and_hits_upstream(client_and_upstream):
    client, upstream, _ = client_and_upstream
    r = client.post("/v1/chat/completions", json=_body("hello"))
    assert r.status_code == 200
    assert r.headers["x-vtf-cache"] == "miss"
    assert r.json()["choices"][0]["message"]["content"] == "TR:hello"
    assert upstream.chat_calls == 1


def test_identical_request_is_cache_hit_without_upstream(client_and_upstream):
    client, upstream, _ = client_and_upstream
    client.post("/v1/chat/completions", json=_body("hello"))
    r2 = client.post("/v1/chat/completions", json=_body("hello"))
    assert r2.headers["x-vtf-cache"] == "hit"
    assert r2.json()["choices"][0]["message"]["content"] == "TR:hello"
    # 第二次未再打上游
    assert upstream.chat_calls == 1


def test_stream_flag_does_not_affect_cache_key(client_and_upstream):
    client, upstream, _ = client_and_upstream
    client.post("/v1/chat/completions", json={**_body("hi"), "stream": False})
    r2 = client.post("/v1/chat/completions", json={**_body("hi"), "stream": True})
    assert r2.headers["x-vtf-cache"] == "hit"
    assert upstream.chat_calls == 1


def test_different_text_is_a_new_miss(client_and_upstream):
    client, upstream, _ = client_and_upstream
    client.post("/v1/chat/completions", json=_body("a"))
    r2 = client.post("/v1/chat/completions", json=_body("b"))
    assert r2.headers["x-vtf-cache"] == "miss"
    assert upstream.chat_calls == 2


def test_stats_reflect_hits_misses_entries(client_and_upstream):
    client, _, _ = client_and_upstream
    client.post("/v1/chat/completions", json=_body("a"))   # miss
    client.post("/v1/chat/completions", json=_body("a"))   # hit
    client.post("/v1/chat/completions", json=_body("b"))   # miss
    stats = client.get("/vtf/stats").json()
    assert stats == {"hits": 1, "misses": 2, "entries": 2}


def test_models_proxied_from_upstream(client_and_upstream):
    client, upstream, _ = client_and_upstream
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == ["fake-model"]
    assert upstream.models_calls == 1


def test_models_fallback_when_upstream_errors(tmp_path):
    upstream = FakeUpstream(models_status=500)
    store = CacheStore(str(tmp_path / "c.db"))
    app = create_app(upstream, store, default_model="fallback-model")
    with TestClient(app) as client:
        data = client.get("/v1/models").json()
    store.close()
    assert [m["id"] for m in data["data"]] == ["fallback-model"]


def test_cache_persists_across_restart(tmp_path):
    db = str(tmp_path / "persist.db")

    up1 = FakeUpstream()
    store1 = CacheStore(db)
    app1 = create_app(up1, store1)
    with TestClient(app1) as c1:
        c1.post("/v1/chat/completions", json=_body("keep"))
    store1.close()
    assert up1.chat_calls == 1

    # 新的 store/app 指向同一個 DB → 同一 request 應命中,不再打上游。
    up2 = FakeUpstream()
    store2 = CacheStore(db)
    app2 = create_app(up2, store2)
    with TestClient(app2) as c2:
        r = c2.post("/v1/chat/completions", json=_body("keep"))
    store2.close()
    assert r.headers["x-vtf-cache"] == "hit"
    assert up2.chat_calls == 0


def test_request_log_callback_reports_miss_then_hit(tmp_path):
    upstream = FakeUpstream()
    store = CacheStore(str(tmp_path / "c.db"))
    lines: list[str] = []
    app = create_app(upstream, store, log=lines.append)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_body("hello"))  # miss
        client.post("/v1/chat/completions", json=_body("hello"))  # hit
    store.close()
    assert any("MISS" in ln for ln in lines)
    assert any("HIT" in ln for ln in lines)
    # 預覽文字出現在日誌中
    assert any("hello" in ln for ln in lines)


def test_no_log_callback_is_silent(tmp_path, capsys):
    upstream = FakeUpstream()
    store = CacheStore(str(tmp_path / "c.db"))
    app = create_app(upstream, store)  # log=None
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_body("x"))
    store.close()
    assert capsys.readouterr().out == ""


def test_normalize_key_ignores_volatile_fields():
    a = normalize_key({"model": "m", "messages": [], "stream": False, "request_id": "x"})
    b = normalize_key({"model": "m", "messages": [], "stream": True, "request_id": "y"})
    assert a == b
    c = normalize_key({"model": "m2", "messages": []})
    assert a != c
