"""Stage 1/2:用 ffprobe 讀影片資訊,用 ffmpeg 依固定 interval 取樣。

- 完全不需要音軌;無音軌影片一樣能處理。
- 取樣命令一律以參數陣列呼叫,不拼 shell 字串。
- 樣本時間用序號與 interval 計算,不用輸出 JPEG 的檔案時間。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

INTERVAL_MIN = 0.1
INTERVAL_MAX = 60.0


class FFmpegError(RuntimeError):
    pass


def check_tools() -> None:
    """啟動時檢查 ffmpeg/ffprobe 是否在 PATH 上,缺少即立即失敗。"""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise FFmpegError(
            f"找不到必要執行檔:{', '.join(missing)}。請安裝 ffmpeg 並確認在 PATH 上。"
        )


def validate_interval(interval: float) -> None:
    if not math.isfinite(interval) or interval <= 0 or interval > INTERVAL_MAX:
        raise ValueError(
            f"--interval 必須是 {INTERVAL_MIN}~{INTERVAL_MAX} 之間的有限正數,得到:{interval}"
        )
    if interval < INTERVAL_MIN:
        raise ValueError(
            f"--interval 不得小於 {INTERVAL_MIN} 秒,得到:{interval}"
        )


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise FFmpegError(
            f"指令失敗({proc.returncode}):{' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


def _eval_fraction(text: str, default: float = 0.0) -> float:
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError, TypeError):
        return default


def _extract_rotation(stream: dict) -> int:
    """從 side_data_list(Display Matrix)或 tags.rotate 取得旋轉角度,正規化到 0/90/180/270。"""
    rotation = 0
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
            break
    else:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag is not None:
            try:
                rotation = int(round(float(tag)))
            except ValueError:
                rotation = 0
    return rotation % 360


def probe_video(path: Path) -> dict:
    """讀取時長、起始時間、寬高、平均幀率、旋轉資訊與是否有音軌。"""
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams:
        raise FFmpegError(f"找不到視訊串流:{path}")
    stream = streams[0]

    duration = _eval_fraction(data.get("format", {}).get("duration"), 0.0)
    if duration <= 0:
        # 有些容器只在 stream 上有 duration
        duration = _eval_fraction(stream.get("duration"), 0.0)

    start_time = _eval_fraction(data.get("format", {}).get("start_time"), 0.0)
    if start_time == 0.0:
        start_time = _eval_fraction(stream.get("start_time"), 0.0)

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    r_frame_rate = stream.get("r_frame_rate", "0/0")
    avg_frame_rate = stream.get("avg_frame_rate", r_frame_rate)
    rotation = _extract_rotation(stream)

    # 旋轉 ±90/270 時,顯示寬高互換(ffmpeg 解碼時會自動套用旋轉)。
    if rotation in (90, 270):
        display_width, display_height = height, width
    else:
        display_width, display_height = width, height

    audio_out = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "json", str(path),
    ])
    has_audio = bool(json.loads(audio_out).get("streams", []))

    return {
        "duration": duration,
        "start_time": start_time,
        "coded_width": width,
        "coded_height": height,
        "width": display_width,
        "height": display_height,
        "rotation": rotation,
        "r_frame_rate": r_frame_rate,
        "avg_frame_rate": avg_frame_rate,
        "avg_fps": _eval_fraction(avg_frame_rate, 0.0),
        "has_audio": has_audio,
    }


def expected_sample_count(duration: float, interval: float) -> int:
    """規格 §4.1:sample_count = ceil(D / I)。"""
    if duration <= 0 or interval <= 0:
        return 0
    return math.ceil(duration / interval)


def _fmt(x: float) -> str:
    """把浮點數轉成不帶科學記號的字串,供 ffmpeg 濾鏡使用。"""
    return format(x, "f").rstrip("0").rstrip(".") or "0"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_frames(
    work: Path, input_path: Path, video_info: dict, interval: float,
    max_width: int, image_quality: int,
) -> list[dict]:
    """依 interval 取樣為 JPEG,回傳 samples 清單(含 index/timestamp/path/sha256/status)。"""
    validate_interval(interval)
    samples_dir = work / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for old in samples_dir.glob("*.jpg"):
        old.unlink()

    vf = f"fps=1/{_fmt(interval)}"
    if max_width and max_width > 0:
        # scale 只縮不放大;-2 保持長寬比且高度為偶數。
        vf += f",scale='min({int(max_width)},iw)':-2"

    pattern = str(samples_dir / "%08d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", vf,
        "-q:v", str(int(image_quality)),
        "-fps_mode", "vfr",
        pattern,
    ]
    run_cmd(cmd)

    files = sorted(samples_dir.glob("*.jpg"))
    duration = float(video_info.get("duration") or 0.0)
    samples: list[dict] = []
    for i, path in enumerate(files, start=1):
        ts = (i - 1) * interval
        if duration > 0 and ts > duration:
            ts = duration
        samples.append({
            "index": i,
            "timestamp": round(ts, 3),
            "path": f"samples/{path.name}",
            "sha256": file_sha256(path),
            "status": "pending",
        })
    return samples


def write_samples_json(work: Path, samples: list[dict], meta: dict) -> None:
    payload = {**meta, "samples": samples}
    tmp = work / "samples.json.tmp"
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, work / "samples.json")
