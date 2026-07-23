"""測試用的假 OCR provider。

以 sample 序號(從檔名解析)決定回傳哪些 block,不看實際像素,
方便重現規格 §12 的整合測試情境。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from vidtranssub.ocr_provider import OCRBlock, OCRResult


class FakeOCRProvider:
    def __init__(
        self,
        scene_fn: Callable[[int], list[OCRBlock]],
        fail_indices: set[int] | None = None,
    ):
        self.scene_fn = scene_fn
        self.fail_indices = fail_indices or set()
        self.calls = 0  # recognize 被呼叫的圖片數(用來驗證只送 miss)

    def fingerprint(self) -> dict:
        return {"provider": "fake", "version": "test-1"}

    def recognize(self, images: list[Path]) -> list[OCRResult]:
        out: list[OCRResult] = []
        for path in images:
            self.calls += 1
            index = int(Path(path).stem)
            if index in self.fail_indices:
                out.append(OCRResult(0, 0.0, [], status="failed", raw=None))
                continue
            blocks = self.scene_fn(index)
            out.append(OCRResult(0, 0.0, list(blocks), status="ok",
                                 raw={"parsing_res_list": []}))
        return out


def block(text: str, bbox: tuple, order: int | None = None,
          confidence: float | None = None) -> OCRBlock:
    return OCRBlock(
        id="0", bbox=bbox, source_text=text, label="text",
        reading_order=order, confidence=confidence, language=None,
    )
