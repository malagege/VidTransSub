"""Stage 7:時間軸整理。

- 依 start 排序,移除完全重複事件。
- 不允許 end <= start。
- 所有時間裁切到 [0, duration]。
- 多行 cue 依閱讀順序保留;最大行數的拆分交給 emit(SRT 串接、ASS 分成同時 events)。
"""

from __future__ import annotations

from .config import Config
from .tracking import canonical_key


def finalize_cues(
    cues: list[dict], translations: dict[str, list[str]], cfg: Config, duration: float
) -> list[dict]:
    """把組好的 cue 套上譯文並整理時間軸,回傳最終字幕事件。"""
    out: list[dict] = []
    for cue in cues:
        source_lines = list(cue["source_texts"])
        translated = translations.get(canonical_key(source_lines))
        if not translated or len(translated) != len(source_lines):
            # 沒有(或不等長)譯文時保留原文,確保仍能輸出。
            translated = list(source_lines)

        start = max(0.0, float(cue["start"]))
        end = float(cue["end"])
        if duration and duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        if end <= start:
            continue

        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "source_lines": source_lines,
            "translated_lines": [t if t else s for t, s in zip(translated, source_lines)],
            "bboxes": list(cue["bboxes"]),
        })

    out.sort(key=lambda c: (c["start"], c["end"]))

    # 移除完全重複事件。
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for c in out:
        key = (
            c["start"], c["end"],
            tuple(c["source_lines"]), tuple(c["translated_lines"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped
