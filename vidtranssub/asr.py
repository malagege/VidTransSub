"""Stage: 語音辨識(ASR;預設關閉)。

用 ffmpeg 抽 16kHz 單聲道 wav,交給 faster-whisper 產生逐段時間軸,
輸出 work/audio_segments.json:{"language","model","segments":[{start,end,text}]}。

預設 cfg.audio_transcribe=False 時完全不辨識(由 pipeline 寫入空片段),
因此對現有 OCR/翻譯流程零影響。faster-whisper 為選用相依([asr] extra),
只有實際啟用時才 import,未安裝也不影響其他功能。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import ffmpeg

SEGMENTS_FILE = "audio_segments.json"


class ASRError(RuntimeError):
    pass


class ASRInitError(ASRError):
    pass


def clean_text(text: str) -> str:
    """壓縮空白並去頭尾,供翻譯 key 與輸出共用同一份正規化文字。"""
    return " ".join((text or "").split()).strip()


def read_segments(work: Path | str) -> list[dict]:
    """讀取 asr 階段產物;不存在時回傳空清單(視為無語音字幕)。"""
    path = Path(work) / SEGMENTS_FILE
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments", [])


def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_segments(work: Path | str, result: dict) -> None:
    _atomic_write(Path(work) / SEGMENTS_FILE, result)


def empty_result() -> dict:
    return {"language": None, "model": None, "segments": []}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2  # faster-whisper 的底層相依

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def transcribe(input_path: Path | str, work: Path | str, cfg, log=print) -> dict:
    """抽取音訊並以 faster-whisper 轉錄,回傳 {language, model, segments}。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - 取決於是否安裝 extra
        raise ASRInitError(
            "未安裝 faster-whisper。請安裝語音辨識相依:\n"
            '  uv pip install "vidtranssub[asr]"\n'
            "或  pip install faster-whisper"
        ) from e

    work = Path(work)
    wav = work / "audio.wav"
    log("[asr] 以 ffmpeg 抽取 16kHz 單聲道音訊…")
    ffmpeg.run_cmd([
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav),
    ])

    device = _resolve_device(cfg.asr_device)
    compute_type = _resolve_compute_type(cfg.asr_compute_type, device)
    log(f"[asr] 載入 faster-whisper:model={cfg.asr_model}、device={device}、compute={compute_type}")
    try:
        model = WhisperModel(cfg.asr_model, device=device, compute_type=compute_type)
    except Exception as e:  # pragma: no cover - 取決於環境/模型
        raise ASRInitError(f"faster-whisper 載入失敗:{e}") from e

    segments_iter, info = model.transcribe(
        str(wav), language=cfg.asr_language, vad_filter=True
    )
    segments: list[dict] = []
    for seg in segments_iter:
        text = clean_text(seg.text)
        if not text:
            continue
        segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": text,
        })

    # wav 可由原片重新抽取,轉錄後即刪除以節省空間。
    try:
        wav.unlink()
    except OSError:
        pass

    return {
        "language": getattr(info, "language", None),
        "model": cfg.asr_model,
        "segments": segments,
    }
