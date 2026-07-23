"""產生整合測試用的靜音測試影片。

依「每秒一格」的場景描述輸出影片:相同 (rgb, text) 的秒數會產生 byte 相同的影格,
用來驗證完全相同圖片的 OCR cache 命中。

可獨立執行:python testdata/make_test_video.py out.mp4
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from PIL import ImageFont

    _FONT = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 40)
except Exception:  # pragma: no cover
    _FONT = None


# 預設 20 秒場景(規格 §12 情境的近似):
#   (rgb, 上方文字, 下方文字, 秒數)
DEFAULT_SCENES = [
    ((20, 20, 40), "", "こんにちは世界", 4),      # 固定 4 秒日文字幕
    ((20, 40, 20), "", "", 2),                      # 完全沒有文字
    ((20, 20, 40), "タイトル", "本文A", 3),         # 上下兩個文字區塊
    ((20, 20, 40), "タイトル", "本文B", 3),         # 同背景下方文字改變
    ((60, 20, 20), "タイトル", "本文B", 3),         # 畫面切換但文字不變
    ((40, 40, 40), "", "", 5),                      # 沒有文字
]


def _frame(size, rgb, top, bottom):
    img = Image.new("RGB", size, rgb)
    draw = ImageDraw.Draw(img)
    w, h = size
    if top:
        draw.text((w * 0.1, h * 0.08), top, fill="white", font=_FONT)
    if bottom:
        draw.text((w * 0.1, h * 0.75), bottom, fill="white", font=_FONT)
    return img


def build_video(out_path, scenes=DEFAULT_SCENES, fps=10, size=(640, 360)) -> Path:
    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        n = 0
        for rgb, top, bottom, seconds in scenes:
            img = _frame(size, rgb, top, bottom)
            for _ in range(int(fps * seconds)):
                n += 1
                img.save(frames_dir / f"{n:05d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", str(frames_dir / "%05d.png"),
                # 無損編碼(-qp 0):相同來源影格解碼後仍完全相同,
                # 讓完全相同的取樣圖片能命中 exact-image OCR cache。
                "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            check=True, capture_output=True,
        )
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    path = build_video(out)
    print(f"OK: {path}")
