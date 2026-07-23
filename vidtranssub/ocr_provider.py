"""OCR provider 介面與資料模型。

v1 的 provider 固定為 PaddleOCR-VL(見 paddleocr_provider.py),但以此 adapter 隔離:
未來更換 OCR 模型時,取樣、cache、追蹤與字幕輸出邏輯都不需重寫。

座標一律正規化到 0~1,避免依賴影片解析度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .normalize import has_meaningful_text


@dataclass
class OCRBlock:
    id: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) 正規化 0~1
    source_text: str
    label: str | None = None
    reading_order: int | None = None
    confidence: float | None = None
    language: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "source_text": self.source_text,
            "label": self.label,
            "reading_order": self.reading_order,
            "confidence": self.confidence,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OCRBlock":
        b = d["bbox"]
        return cls(
            id=d["id"],
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            source_text=d["source_text"],
            label=d.get("label"),
            reading_order=d.get("reading_order"),
            confidence=d.get("confidence"),
            language=d.get("language"),
        )


@dataclass
class OCRResult:
    sample_index: int
    timestamp: float
    blocks: list[OCRBlock] = field(default_factory=list)
    status: str = "ok"  # ok | failed
    raw: dict | None = None  # PaddleOCR-VL 原始 JSON,另外存 *.raw.json

    def to_dict(self) -> dict:
        """VideoTransSub 正規化 JSON(不含原始 raw)。"""
        return {
            "sample_index": self.sample_index,
            "timestamp": self.timestamp,
            "status": self.status,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OCRResult":
        return cls(
            sample_index=d["sample_index"],
            timestamp=d["timestamp"],
            status=d.get("status", "ok"),
            blocks=[OCRBlock.from_dict(b) for b in d.get("blocks", [])],
        )

    def reindex(self, sample_index: int, timestamp: float) -> "OCRResult":
        """設定樣本序號與時間,並把 block id 重新編為 <sample_index>-<n>。"""
        self.sample_index = sample_index
        self.timestamp = timestamp
        for n, block in enumerate(self.blocks, start=1):
            block.id = f"{sample_index}-{n}"
        return self


@runtime_checkable
class OCRProvider(Protocol):
    def recognize(self, images: list[Path]) -> list[OCRResult]:
        """對一批圖片辨識,回傳等長的 OCRResult 清單。

        provider 只負責 blocks 與 raw;sample_index/timestamp 由 pipeline 設定。
        """
        ...


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def blocks_from_parsing_list(
    parsing_res_list: list[dict],
    width: int,
    height: int,
    confidence_threshold: float | None = None,
) -> list[OCRBlock]:
    """把 PaddleOCR-VL 的 parsing_res_list 轉成 VideoTransSub 正規化 blocks。

    對照(規格 §4 結果轉換表):
      block_content -> source_text
      block_bbox    -> bbox(除以 sample 寬高後正規化到 0~1)
      block_label   -> label
      block_order   -> reading_order

    - confidence 為 optional:provider 有提供且 confidence_threshold 有設定時才過濾;
      沒有分數時寫 None,不捏造分數也不因缺分數而丟棄 block。
    - 空白、純標點、單一裝飾符號不建立 block。
    """
    if not width or not height:
        raise ValueError("blocks_from_parsing_list 需要有效的 sample 寬高")

    blocks: list[OCRBlock] = []
    for n, item in enumerate(parsing_res_list, start=1):
        content = _first(item, "block_content", "content", default="")
        content = ("" if content is None else str(content)).strip()
        if not has_meaningful_text(content):
            continue

        bbox_px = _first(item, "block_bbox", "bbox", default=None)
        if not bbox_px or len(bbox_px) < 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox_px[:4])
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        bbox = (
            _clamp01(x0 / width), _clamp01(y0 / height),
            _clamp01(x1 / width), _clamp01(y1 / height),
        )

        conf_raw = _first(item, "confidence", "score", default=None)
        confidence = None if conf_raw is None else float(conf_raw)
        if (
            confidence_threshold is not None
            and confidence is not None
            and confidence < confidence_threshold
        ):
            continue

        order = _first(item, "block_order", "order", "reading_order", default=None)
        blocks.append(OCRBlock(
            id=str(n),
            bbox=bbox,
            source_text=content,
            label=_first(item, "block_label", "label", default=None),
            reading_order=None if order is None else int(order),
            confidence=confidence,
            language=_first(item, "language", "lang", default=None),
        ))
    return blocks
