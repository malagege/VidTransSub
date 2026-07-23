from vidtranssub.cleanup import finalize_cues
from vidtranssub.config import Config
from vidtranssub.tracking import canonical_key


def make_cue(start, end, texts):
    return {
        "start": start, "end": end,
        "source_texts": list(texts),
        "bboxes": [[0.1, 0.7, 0.9, 0.9]] * len(texts),
        "lines": [],
    }


def test_applies_translation_and_sorts():
    cues = [
        make_cue(2.0, 4.0, ["B"]),
        make_cue(0.0, 1.0, ["A"]),
    ]
    translations = {
        canonical_key(["A"]): ["譯A"],
        canonical_key(["B"]): ["譯B"],
    }
    out = finalize_cues(cues, translations, Config(), 100.0)
    assert [c["start"] for c in out] == [0.0, 2.0]  # 依 start 排序
    assert out[0]["translated_lines"] == ["譯A"]


def test_missing_translation_falls_back_to_source():
    cues = [make_cue(0.0, 1.0, ["A"])]
    out = finalize_cues(cues, {}, Config(), 100.0)
    assert out[0]["translated_lines"] == ["A"]


def test_clamps_and_drops_invalid():
    cues = [
        make_cue(-1.0, 5.0, ["A"]),   # start 夾到 0,end 夾到 duration
        make_cue(9.0, 8.0, ["B"]),    # end <= start -> 丟棄
    ]
    out = finalize_cues(cues, {}, Config(), 3.0)
    assert len(out) == 1
    assert out[0]["start"] == 0.0 and out[0]["end"] == 3.0


def test_dedup_exact_duplicates():
    cues = [make_cue(0.0, 1.0, ["A"]), make_cue(0.0, 1.0, ["A"])]
    out = finalize_cues(cues, {canonical_key(["A"]): ["譯A"]}, Config(), 100.0)
    assert len(out) == 1
