from vidtranssub.config import Config
from vidtranssub.ocr_provider import OCRBlock, OCRResult
from vidtranssub.tracking import track_events

TOP = (0.1, 0.05, 0.9, 0.2)
BOTTOM = (0.1, 0.7, 0.9, 0.9)
BIGDUR = 1000.0


def blk(text, bbox, order=None):
    return OCRBlock("x", bbox, text, "text", order, None, None)


def res(idx, ts, blocks, status="ok"):
    return OCRResult(idx, ts, list(blocks), status=status)


def cfg(**kw):
    return Config(**kw)


def test_same_text_same_place_one_event():
    results = [res(i, float(i - 1), [blk("こんにちは", BOTTOM)]) for i in (1, 2, 3)]
    events = track_events(results, cfg(), BIGDUR)
    assert len(events) == 1
    e = events[0]
    assert e["source_text"] == "こんにちは"
    assert e["start"] == 0.0
    assert e["end"] == 3.0  # last_seen(2) + interval(1)
    assert e["observation_count"] == 3
    assert e["sample_indices"] == [1, 2, 3]


def test_minor_ocr_diff_merges():
    results = [
        res(1, 0.0, [blk("今日はいい天気です", BOTTOM)]),
        res(2, 1.0, [blk("今日はいい天気でず", BOTTOM)]),  # 1 字差,相似度 > 0.85
    ]
    events = track_events(results, cfg(), BIGDUR)
    assert len(events) == 1
    assert events[0]["observation_count"] == 2


def test_same_text_different_position_two_events():
    results = [
        res(1, 0.0, [blk("X", TOP)]),
        res(2, 1.0, [blk("X", BOTTOM)]),
    ]
    events = track_events(results, cfg(), BIGDUR)
    assert len(events) == 2


def test_same_position_text_changed_splits():
    results = [
        res(1, 0.0, [blk("本文A", BOTTOM)]),
        res(2, 1.0, [blk("本文A", BOTTOM)]),
        res(3, 2.0, [blk("本文B", BOTTOM)]),
    ]
    events = track_events(results, cfg(), BIGDUR)
    assert len(events) == 2
    a = [e for e in events if e["source_text"] == "本文A"][0]
    b = [e for e in events if e["source_text"] == "本文B"][0]
    assert a["start"] == 0.0 and a["end"] == 2.0   # 舊事件在改變時刻結束
    assert b["start"] == 2.0 and b["end"] == 3.0   # 新事件同時開始


def test_gap_tolerance_merges_one_missing_sample():
    results = [
        res(1, 0.0, [blk("X", BOTTOM)]),
        res(2, 1.0, []),                 # 短暫消失一個 sample
        res(3, 2.0, [blk("X", BOTTOM)]),
    ]
    events = track_events(results, cfg(gap_tolerance=1), BIGDUR)
    assert len(events) == 1
    assert events[0]["sample_indices"] == [1, 3]


def test_gap_zero_does_not_merge():
    results = [
        res(1, 0.0, [blk("X", BOTTOM)]),
        res(2, 1.0, []),
        res(3, 2.0, [blk("X", BOTTOM)]),
    ]
    events = track_events(results, cfg(gap_tolerance=0), BIGDUR)
    assert len(events) == 2


def test_ocr_failure_not_treated_as_disappearance():
    # 即使 gap_tolerance=0,中間的失敗 sample 也不能當成文字消失
    results = [
        res(1, 0.0, [blk("X", BOTTOM)]),
        res(2, 1.0, [], status="failed"),
        res(3, 2.0, [blk("X", BOTTOM)]),
    ]
    events = track_events(results, cfg(gap_tolerance=0), BIGDUR)
    assert len(events) == 1
    assert events[0]["sample_indices"] == [1, 3]


def test_end_clamped_to_duration():
    results = [res(1, 9.5, [blk("X", BOTTOM)])]
    events = track_events(results, cfg(), duration=10.0)
    assert events[0]["end"] == 10.0  # 9.5 + 1.0 -> 夾到 duration


def test_min_duration_filter():
    # interval 0.2:單次出現 -> 0.2s < 0.4s -> 過濾
    results = [res(1, 0.0, [blk("X", BOTTOM)])]
    assert track_events(results, cfg(interval=0.2), BIGDUR) == []
    # interval 1.0:單次出現 -> 1.0s >= 0.4s -> 保留
    assert len(track_events(results, cfg(interval=1.0), BIGDUR)) == 1
