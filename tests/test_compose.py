from vidtranssub.config import Config
from vidtranssub.tracking import canonical_key, compose_cues

TOP = [0.1, 0.05, 0.9, 0.2]
BOTTOM = [0.1, 0.7, 0.9, 0.9]


def ev(start, end, text, bbox, order=None):
    return {
        "start": start, "end": end, "source_text": text, "bbox": bbox,
        "reading_order": order, "label": "text", "confidence": None,
        "sample_indices": [1], "observation_count": 1,
    }


def test_simultaneous_blocks_merge_by_reading_order():
    events = [
        ev(0.0, 3.0, "下", BOTTOM, order=2),
        ev(0.0, 3.0, "上", TOP, order=1),
    ]
    cues = compose_cues(events, Config())
    assert len(cues) == 1
    assert cues[0]["source_texts"] == ["上", "下"]  # 依 reading_order


def test_geometry_fallback_when_no_reading_order():
    events = [
        ev(0.0, 3.0, "下", BOTTOM),
        ev(0.0, 3.0, "上", TOP),
    ]
    cues = compose_cues(events, Config(reading_order="ltr"))
    assert cues[0]["source_texts"] == ["上", "下"]  # 上(y 小)在前


def test_different_windows_stay_separate():
    events = [
        ev(0.0, 3.0, "A", TOP, order=1),
        ev(0.0, 5.0, "B", BOTTOM, order=2),  # 不同 end
    ]
    cues = compose_cues(events, Config())
    assert len(cues) == 2


def test_canonical_key_stable():
    assert canonical_key(["上", "下"]) == canonical_key(["上", "下"])
    assert canonical_key(["上", "下"]) != canonical_key(["下", "上"])
