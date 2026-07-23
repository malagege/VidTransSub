import json

from vidtranssub.config import TRANSLATION_PROMPT_VERSION, Config
from vidtranssub.llm_cache_client import (
    build_messages,
    parse_json_array,
    translate_texts,
)


def test_parse_json_array_variants():
    assert parse_json_array('["a", "b"]') == ["a", "b"]
    assert parse_json_array('```json\n["a"]\n```') == ["a"]
    assert parse_json_array('Here: ["x", "y"] done') == ["x", "y"]
    assert parse_json_array("not json") is None
    assert parse_json_array('{"a": 1}') is None


def test_build_messages_is_stable_and_clean():
    texts = ["今日はいい天気です"]
    m1 = build_messages(texts, "ja", "zh-TW")
    m2 = build_messages(texts, "ja", "zh-TW")
    assert m1 == m2  # 穩定
    system = m1[0]["content"]
    user = m1[1]["content"]
    assert TRANSLATION_PROMPT_VERSION in system
    assert "ja" in system and "zh-TW" in system
    assert user == json.dumps(texts, ensure_ascii=False)
    # 不得含 timestamp/bbox/sample index/檔名等每次會變的內容
    blob = system + user
    for forbidden in ("timestamp", "bbox", "sample", "index", ".jpg", "request_id"):
        assert forbidden not in blob


def test_build_messages_changes_with_lang():
    a = build_messages(["x"], "ja", "zh-TW")[0]["content"]
    b = build_messages(["x"], "en", "zh-TW")[0]["content"]
    c = build_messages(["x"], "ja", "en")[0]["content"]
    assert a != b and a != c


def test_translate_texts_happy_path():
    calls = []

    def chat(messages):
        arr = json.loads(messages[-1]["content"])
        calls.append(arr)
        return json.dumps([f"tr:{x}" for x in arr], ensure_ascii=False)

    out, failed = translate_texts(["A", "B"], "ja", "zh-TW", chat)
    assert out == ["tr:A", "tr:B"]
    assert failed == []
    assert len(calls) == 1


def test_translate_texts_retries_once_then_succeeds():
    n = {"i": 0}

    def chat(messages):
        n["i"] += 1
        arr = json.loads(messages[-1]["content"])
        if n["i"] == 1:
            return json.dumps(["only-one"])  # 數量不符 -> 重試
        return json.dumps([f"tr:{x}" for x in arr], ensure_ascii=False)

    out, failed = translate_texts(["A", "B"], "ja", "zh-TW", chat)
    assert n["i"] == 2
    assert out == ["tr:A", "tr:B"] and failed == []


def test_translate_texts_degrades_to_per_line():
    def chat(messages):
        arr = json.loads(messages[-1]["content"])
        if len(arr) > 1:
            return "garbage"  # 整批總是失敗 -> 降級逐行
        return json.dumps([f"one:{arr[0]}"], ensure_ascii=False)

    out, failed = translate_texts(["A", "B"], "ja", "zh-TW", chat)
    assert out == ["one:A", "one:B"] and failed == []


def test_translate_texts_total_failure_keeps_original():
    out, failed = translate_texts(["A", "B"], "ja", "zh-TW", lambda m: "garbage")
    assert out == ["A", "B"]
    assert failed == [0, 1]


def test_parse_retry_after_forms():
    from vidtranssub.llm_cache_client import _parse_retry_after

    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("garbage") is None
    # HTTP-date 形式:過去的時間 -> 0(不為負、不拋例外)
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_translate_params_invalidation():
    cfg = Config(target_lang="zh-TW", source_lang="ja")
    base = cfg.translate_params("m1")
    assert base != Config(target_lang="zh-TW", source_lang="ja").translate_params("m2")
    assert base != Config(target_lang="en", source_lang="ja").translate_params("m1")
    assert base != Config(target_lang="zh-TW", source_lang="en").translate_params("m1")
    assert base["prompt_version"] == TRANSLATION_PROMPT_VERSION
