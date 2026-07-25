"""VideoTransSub 執行設定。

每個階段有自己的 params hash;參數改變時,manifest 會讓該階段與所有下游階段失效
(見 §8 失效範圍表)。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# 翻譯 system prompt 的固定版本;改版即讓 translate 之後全部失效,也改變 LLM cache key。
TRANSLATION_PROMPT_VERSION = "VideoTransSub translation prompt v1"


def stable_hash(obj) -> str:
    """物件 canonical JSON 形式的 SHA-256。"""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Config:
    # --- 取樣 ---
    interval: float = 1.0
    max_width: int = 1920
    image_quality: int = 3  # ffmpeg mjpeg -q:v(2 最佳 .. 31 最差)

    # --- 路徑與語言 ---
    work_dir: str = "./work"
    target_lang: str = "zh-TW"
    source_lang: str | None = None  # None = auto

    # --- OCR ---
    ocr_provider: str = "paddleocr-vl"
    paddleocr_model: str = "PaddleOCR-VL"
    paddleocr_engine: str | None = None
    ocr_device: str = "auto"
    ocr_batch_size: int = 8
    ocr_confidence: float | None = None  # None = 停用門檻
    no_ocr_cache: bool = False

    # --- OCR VLM server 橋接(可選;None = 維持 in-process VLM) ---
    # 指定後,layout 仍在本地跑,VLM 辨識改走已啟動的 PaddleOCR genai_server
    # (OpenAI-compatible)。適合把吃 VRAM 的 VLM 推到另一台 Linux GPU server。
    ocr_server_url: str | None = None  # 例如 http://GPU_HOST:8118/v1
    ocr_server_backend: str = "vllm-server"  # vllm/sglang/fastdeploy/mlx-vlm/llama-cpp -server
    ocr_server_model: str | None = None  # 服務端模型名;None = 沿用 paddleocr_model
    ocr_api_key_env: str = "PADDLEOCR_VL_API_KEY"  # 存放 server API key 的環境變數名稱

    # --- 翻譯 ---
    llm_model: str | None = None  # None = 取上游第一個
    llm_cache_url: str = "http://127.0.0.1:8790"

    # --- 追蹤/合併 ---
    text_similarity: float = 0.85
    gap_tolerance: int = 1
    iou_threshold: float = 0.3
    center_dist_ratio: float = 0.08  # 佔畫面對角線比例
    min_event_duration: float = 0.4
    reading_order: str = "auto"  # auto/ltr/rtl/ttb

    # --- 輸出 ---
    bilingual: bool = False
    subtitle_position: str = "bottom"  # bottom/top
    max_lines: int = 2

    # --- 行為 ---
    strict: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- 各階段 params hash 來源(見 manifest 失效判斷) ----

    def sample_params(self) -> dict:
        return {
            "interval": self.interval,
            "max_width": self.max_width,
            "image_quality": self.image_quality,
        }

    def ocr_params(self, provider_fingerprint: dict | None = None) -> dict:
        """OCR 參數;provider_fingerprint 是 provider 啟動後解析出的實際版本/模型/選項。

        exact-image cache key 也用這份參數,確保 PaddleOCR 版本或選項改變時 cache 自然失效。
        """
        params = {
            "provider": self.ocr_provider,
            "model": self.paddleocr_model,
            "engine": self.paddleocr_engine,
            "device": self.ocr_device,
            "batch_size": self.ocr_batch_size,
            "confidence": self.ocr_confidence,
        }
        if provider_fingerprint:
            params["provider_fingerprint"] = provider_fingerprint
        return params

    def track_params(self) -> dict:
        return {
            "text_similarity": self.text_similarity,
            "gap_tolerance": self.gap_tolerance,
            "iou_threshold": self.iou_threshold,
            "center_dist_ratio": self.center_dist_ratio,
            "min_event_duration": self.min_event_duration,
            "reading_order": self.reading_order,
        }

    def translate_params(self, resolved_model: str) -> dict:
        return {
            "model": resolved_model,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "prompt_version": TRANSLATION_PROMPT_VERSION,
        }

    def cleanup_params(self) -> dict:
        return {"max_lines": self.max_lines}

    def emit_params(self) -> dict:
        return {
            "subtitle_position": self.subtitle_position,
            "bilingual": self.bilingual,
            "target_lang": self.target_lang,
        }
