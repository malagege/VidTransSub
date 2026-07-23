"""Stage 5 後半:跨樣本追蹤,把重複出現的文字合併為有開始/結束時間的事件。

判定同一事件(規格 §5):
- bbox IoU > iou_threshold,或中心點距離 < 畫面對角線的 center_dist_ratio。
- 文字正規化相似度 > text_similarity。
- 中間消失不超過 gap_tolerance 個樣本。

重要:OCR 失敗的 sample 不視為文字消失,完全略過,不參與 gap 計算與事件關閉。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Config
from .normalize import normalize_display, similarity
from .ocr_provider import OCRResult

_DIAGONAL = math.sqrt(2.0)  # 正規化座標下的畫面對角線長度


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(b: tuple) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _center_distance(a: tuple, b: tuple) -> float:
    (ax, ay), (bx, by) = _center(a), _center(b)
    return math.hypot(ax - bx, ay - by)


def _region_match(a: tuple, b: tuple, cfg: Config) -> bool:
    if _iou(a, b) > cfg.iou_threshold:
        return True
    return _center_distance(a, b) < cfg.center_dist_ratio * _DIAGONAL


@dataclass
class _Active:
    start: float
    last_seen_ts: float
    last_seen_index: int
    source_text: str
    bbox: tuple
    reading_order: int | None
    label: str | None
    confidence: float | None
    sample_indices: list[int] = field(default_factory=list)
    observation_count: int = 0
    missed: int = 0


def track_events(results: list[OCRResult], cfg: Config, duration: float) -> list[dict]:
    """把 OCR 結果合併成事件清單。results 需可依 sample_index 排序。"""
    interval = cfg.interval
    active: list[_Active] = []
    closed: list[dict] = []

    def close(ev: _Active, end_ts: float) -> None:
        end = end_ts
        if duration and duration > 0:
            end = min(end, duration)
        start = max(0.0, ev.start)
        closed.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "source_text": ev.source_text,
            "bbox": [round(v, 5) for v in ev.bbox],
            "reading_order": ev.reading_order,
            "label": ev.label,
            "confidence": ev.confidence,
            "sample_indices": list(ev.sample_indices),
            "observation_count": ev.observation_count,
        })

    for result in sorted(results, key=lambda r: r.sample_index):
        if result.status == "failed":
            # OCR 失敗:不當成文字消失,完全略過。
            continue

        ts = result.timestamp
        idx = result.sample_index
        seen: set[int] = set()  # 本 sample 已配對/新建的 active 索引

        for block in result.blocks:
            disp = normalize_display(block.source_text)
            if not disp:
                continue

            # 1) 找「同區域且文字相似」的 active 事件 -> 延伸
            best_i = -1
            best_sim = -1.0
            for i, ev in enumerate(active):
                if i in seen:
                    continue
                if _region_match(block.bbox, ev.bbox, cfg):
                    sim = similarity(disp, ev.source_text)
                    if sim >= cfg.text_similarity and sim > best_sim:
                        best_sim, best_i = sim, i

            if best_i >= 0:
                ev = active[best_i]
                ev.last_seen_ts = ts
                ev.last_seen_index = idx
                ev.sample_indices.append(idx)
                ev.observation_count += 1
                ev.missed = 0
                ev.bbox = block.bbox
                if ev.reading_order is None and block.reading_order is not None:
                    ev.reading_order = block.reading_order
                if block.confidence is not None:
                    ev.confidence = (
                        block.confidence if ev.confidence is None
                        else max(ev.confidence, block.confidence)
                    )
                seen.add(best_i)
                continue

            # 2) 同區域但文字明確改變 -> 舊事件在此刻結束,新事件同時開始
            region_i = -1
            for i, ev in enumerate(active):
                if i in seen:
                    continue
                if _region_match(block.bbox, ev.bbox, cfg):
                    region_i = i
                    break
            if region_i >= 0:
                close(active[region_i], ts)
                active.pop(region_i)
                # pop 後,seen 內大於 region_i 的索引需左移;重建 seen。
                seen = {i - 1 if i > region_i else i for i in seen}

            # 3) 建立新事件
            active.append(_Active(
                start=ts,
                last_seen_ts=ts,
                last_seen_index=idx,
                source_text=disp,
                bbox=block.bbox,
                reading_order=block.reading_order,
                label=block.label,
                confidence=block.confidence,
                sample_indices=[idx],
                observation_count=1,
                missed=0,
            ))
            seen.add(len(active) - 1)

        # 本 sample 沒看到的 active 事件:累計 missed,超過容忍值就關閉。
        survivors: list[_Active] = []
        for i, ev in enumerate(active):
            if i in seen:
                survivors.append(ev)
                continue
            ev.missed += 1
            if ev.missed > cfg.gap_tolerance:
                close(ev, ev.last_seen_ts + interval)
            else:
                survivors.append(ev)
        active = survivors

    # 影片結束:關閉所有仍開啟的事件。
    for ev in active:
        close(ev, ev.last_seen_ts + interval)

    # 最短事件過濾 + 排序。
    events = [
        e for e in closed
        if (e["end"] - e["start"]) >= cfg.min_event_duration - 1e-9
    ]
    events.sort(key=lambda e: (e["start"], e["bbox"][1], e["bbox"][0]))
    return events


# ---------- Cue 組成(供 translate 與 cleanup 共用) ----------

def _order_key(line: dict, mode: str):
    bbox = line["bbox"]
    x0, y0 = bbox[0], bbox[1]
    if mode == "rtl":
        return (round(y0, 3), -x0)
    if mode == "ttb":
        return (round(-x0, 3), y0)
    # ltr / auto
    return (round(y0, 3), x0)


def compose_cues(events: list[dict], cfg: Config) -> list[dict]:
    """把時間視窗完全相同的事件組成一條多行 cue,依閱讀順序排列各行。

    同一 cue 內:所有行都有 reading_order 時優先照它排;否則依 auto/ltr/rtl/ttb 幾何規則。
    """
    groups: dict[tuple, list[dict]] = {}
    for ev in events:
        key = (round(ev["start"], 3), round(ev["end"], 3))
        groups.setdefault(key, []).append(ev)

    cues: list[dict] = []
    for (start, end), lines in sorted(groups.items()):
        if all(ln.get("reading_order") is not None for ln in lines):
            ordered = sorted(lines, key=lambda ln: ln["reading_order"])
        else:
            ordered = sorted(lines, key=lambda ln: _order_key(ln, cfg.reading_order))
        cues.append({
            "start": start,
            "end": end,
            "source_texts": [ln["source_text"] for ln in ordered],
            "bboxes": [ln["bbox"] for ln in ordered],
            "lines": ordered,
        })
    cues.sort(key=lambda c: (c["start"], c["end"]))
    return cues


def canonical_key(source_texts: list[str]) -> str:
    """cue 的去重/翻譯 key:只由按閱讀順序排列的原文決定。"""
    import json

    return json.dumps(source_texts, ensure_ascii=False)
