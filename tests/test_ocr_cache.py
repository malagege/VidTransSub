from vidtranssub.config import Config, stable_hash
from vidtranssub.ocr_cache import OCRCache, ocr_cache_key, sha256_bytes

PARAMS = {"provider": "paddleocr-vl", "model": "PaddleOCR-VL", "confidence": None}


def test_key_depends_on_image_and_params():
    sha_a = sha256_bytes(b"image-a")
    sha_b = sha256_bytes(b"image-b")
    assert ocr_cache_key(sha_a, PARAMS) == ocr_cache_key(sha_a, PARAMS)
    # 不同圖片 -> 不同 key
    assert ocr_cache_key(sha_a, PARAMS) != ocr_cache_key(sha_b, PARAMS)
    # 不同 OCR 參數 -> 不同 key(即使圖片相同)
    changed = {**PARAMS, "model": "other"}
    assert ocr_cache_key(sha_a, PARAMS) != ocr_cache_key(sha_a, changed)


def test_roundtrip(tmp_path):
    cache = OCRCache(tmp_path / "ocr.db")
    key = ocr_cache_key(sha256_bytes(b"img"), PARAMS)
    assert cache.get(key) is None
    normalized = {"sample_index": 1, "timestamp": 0.0, "status": "ok", "blocks": []}
    raw = {"parsing_res_list": [{"block_content": "x"}]}
    cache.put(key, normalized, raw)
    got = cache.get(key)
    assert got is not None
    n, r = got
    assert n["status"] == "ok"
    assert r["parsing_res_list"][0]["block_content"] == "x"
    assert cache.hits == 1 and cache.misses == 1
    cache.close()


def test_only_exact_match_hits(tmp_path):
    cache = OCRCache(tmp_path / "ocr.db")
    key = ocr_cache_key(sha256_bytes(b"img"), PARAMS)
    cache.put(key, {"status": "ok", "blocks": []}, None)
    # 只有畫面相似(不同 bytes)不得命中
    other = ocr_cache_key(sha256_bytes(b"img2"), PARAMS)
    assert cache.get(other) is None
    # 參數不同不得命中
    diff_params = ocr_cache_key(sha256_bytes(b"img"), {**PARAMS, "confidence": 0.5})
    assert cache.get(diff_params) is None
    cache.close()


def test_batch_size_not_in_ocr_params():
    """batch_size 只影響一次送幾張,不影響辨識結果,不得參與 hash/cache key。"""
    cfg = Config()
    assert "batch_size" not in cfg.ocr_params({"paddleocr_version": "3.0"})
    a = stable_hash(Config(ocr_batch_size=8).ocr_params())
    b = stable_hash(Config(ocr_batch_size=1).ocr_params())
    assert a == b
    # 但真正影響結果的參數仍須改變 hash
    assert stable_hash(Config(ocr_confidence=0.5).ocr_params()) != a
    # v1 相容用的參數組仍含 batch_size(搬遷比對用)
    legacy = Config(ocr_batch_size=1).legacy_ocr_params_v1()
    assert legacy["batch_size"] == 1
    assert stable_hash(legacy) != a


def test_migrate_param_hash(tmp_path):
    cache = OCRCache(tmp_path / "ocr.db")
    old_hash = stable_hash(PARAMS)
    new_hash = stable_hash({**PARAMS, "extra": 1})
    sha = sha256_bytes(b"img")
    cache.put(f"{sha}:{old_hash}", {"status": "ok", "blocks": []}, {"raw": 1})

    assert cache.migrate_param_hash(old_hash, new_hash) == 1
    assert cache.get(f"{sha}:{old_hash}") is None
    got = cache.get(f"{sha}:{new_hash}")
    assert got is not None and got[0]["status"] == "ok" and got[1] == {"raw": 1}
    # 重複搬遷與 old == new 都是 no-op
    assert cache.migrate_param_hash(old_hash, new_hash) == 0
    assert cache.migrate_param_hash(new_hash, new_hash) == 0
    assert cache.stats()["entries"] == 1
    cache.close()


def test_migrate_param_hash_keeps_existing_target(tmp_path):
    """新 key 已存在時保留既有那筆,舊 key 清掉,不留重複。"""
    cache = OCRCache(tmp_path / "ocr.db")
    old_hash, new_hash = stable_hash({"v": 1}), stable_hash({"v": 2})
    sha = sha256_bytes(b"img")
    cache.put(f"{sha}:{old_hash}", {"status": "ok", "blocks": ["old"]}, None)
    cache.put(f"{sha}:{new_hash}", {"status": "ok", "blocks": ["new"]}, None)

    cache.migrate_param_hash(old_hash, new_hash)
    assert cache.stats()["entries"] == 1
    got = cache.get(f"{sha}:{new_hash}")
    assert got is not None and got[0]["blocks"] == ["new"]
    cache.close()
