from vidtranssub.ocr_cache import OCRCache, ocr_cache_key, sha256_bytes

PARAMS = {"provider": "paddleocr-vl", "model": "PaddleOCR-VL", "batch_size": 8}


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
    diff_params = ocr_cache_key(sha256_bytes(b"img"), {**PARAMS, "batch_size": 4})
    assert cache.get(diff_params) is None
    cache.close()
