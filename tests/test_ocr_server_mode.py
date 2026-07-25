"""OCR VLM server 橋接模式的單元測試(不需 GPU / 不載入 paddleocr)。

只驗證 provider 有把正確的 vl_rec_* kwargs 組出來、fingerprint 有把 server 後端
納入(而不含 url / api key),以及 in-process 預設行為維持不變。
"""

import pytest

from vidtranssub.config import Config
from vidtranssub.paddleocr_provider import PaddleOCRInitError, PaddleOCRProvider


def test_in_process_default_has_no_server_kwargs():
    p = PaddleOCRProvider(model="PaddleOCR-VL")
    kwargs = p._pipeline_kwargs()
    assert not any(k.startswith("vl_rec_") for k in kwargs)
    # baseline 前處理選項維持不變
    assert kwargs["use_doc_orientation_classify"] is False
    fp = p._static_fingerprint()
    assert "vl_rec_backend" not in fp


def test_server_mode_forwards_vl_rec_kwargs():
    p = PaddleOCRProvider(
        model="PaddleOCR-VL",
        server_url="http://gpu-host:8118/v1",
        server_backend="vllm-server",
        server_model="PaddleOCR-VL-1.6-0.9B",
        server_api_key="secret",
    )
    kwargs = p._pipeline_kwargs()
    assert kwargs["vl_rec_backend"] == "vllm-server"
    assert kwargs["vl_rec_server_url"] == "http://gpu-host:8118/v1"
    assert kwargs["vl_rec_api_model_name"] == "PaddleOCR-VL-1.6-0.9B"
    assert kwargs["vl_rec_api_key"] == "secret"


def test_server_model_defaults_to_model():
    p = PaddleOCRProvider(model="PaddleOCR-VL", server_url="http://h:8118/v1")
    assert p._pipeline_kwargs()["vl_rec_api_model_name"] == "PaddleOCR-VL"


def test_api_key_omitted_when_absent():
    p = PaddleOCRProvider(model="m", server_url="http://h:8118/v1")
    assert "vl_rec_api_key" not in p._pipeline_kwargs()


def test_fingerprint_includes_backend_and_model_but_not_url_or_key():
    p = PaddleOCRProvider(
        model="PaddleOCR-VL",
        server_url="http://gpu-host:8118/v1",
        server_backend="sglang-server",
        server_model="PaddleOCR-VL-1.6-0.9B",
        server_api_key="secret",
    )
    fp = p._static_fingerprint()
    assert fp["vl_rec_backend"] == "sglang-server"
    assert fp["vl_rec_model"] == "PaddleOCR-VL-1.6-0.9B"
    # url / key 不得進 fingerprint(換機器不失效 cache;金鑰不落地)
    dumped = repr(fp)
    assert "gpu-host" not in dumped
    assert "secret" not in dumped
    assert "vl_rec_server_url" not in fp
    assert "vl_rec_api_key" not in fp


def test_fingerprint_differs_between_inprocess_and_server():
    local = PaddleOCRProvider(model="PaddleOCR-VL")
    server = PaddleOCRProvider(
        model="PaddleOCR-VL",
        server_url="http://h:8118/v1",
        server_model="PaddleOCR-VL-1.6-0.9B",
    )
    # 服務端可能是不同版本的 VLM;cache 不該互相污染
    assert local._static_fingerprint() != server._static_fingerprint()


def test_unknown_backend_rejected():
    with pytest.raises(PaddleOCRInitError):
        PaddleOCRProvider(model="m", server_url="http://h:8118/v1", server_backend="bogus")


def test_backend_not_validated_without_server_url():
    # 未啟用 server 時 backend 值不生效,也不該擋下(維持 in-process 預設)
    p = PaddleOCRProvider(model="m", server_backend="bogus")
    assert "vl_rec_backend" not in p._pipeline_kwargs()


def test_build_provider_reads_api_key_from_env(monkeypatch):
    from vidtranssub.pipeline import build_provider

    monkeypatch.setenv("MY_OCR_KEY", "env-secret")
    cfg = Config(
        ocr_server_url="http://h:8118/v1",
        ocr_server_backend="vllm-server",
        ocr_api_key_env="MY_OCR_KEY",
    )
    provider = build_provider(cfg)
    assert provider.server_api_key == "env-secret"
    assert provider._pipeline_kwargs()["vl_rec_api_key"] == "env-secret"


def test_build_provider_no_key_lookup_when_inprocess(monkeypatch):
    from vidtranssub.pipeline import build_provider

    monkeypatch.setenv("PADDLEOCR_VL_API_KEY", "should-be-ignored")
    provider = build_provider(Config())  # 預設 in-process
    assert provider.server_api_key is None
