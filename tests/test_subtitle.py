from vidtranssub.subtitle import _ass_time, _srt_time, to_ass, to_srt


def cue(start, end, translated, source=None, bboxes=None):
    source = source if source is not None else translated
    return {
        "start": start, "end": end,
        "translated_lines": translated, "source_lines": source,
        "bboxes": bboxes or [[0.1, 0.7, 0.9, 0.9]] * len(translated),
    }


def test_srt_timecodes():
    assert _srt_time(0) == "00:00:00,000"
    assert _srt_time(61.25) == "00:01:01,250"
    assert _srt_time(3661.999) == "01:01:01,999"
    assert _srt_time(-1) == "00:00:00,000"


def test_ass_timecodes():
    assert _ass_time(0) == "0:00:00.00"
    assert _ass_time(61.25) == "0:01:01.25"
    assert _ass_time(3661.99) == "1:01:01.99"


def test_srt_structure_multiline():
    cues = [
        cue(0.0, 2.5, ["第一句"]),
        cue(3.0, 4.0, ["上", "下"]),
    ]
    srt = to_srt(cues)
    blocks = srt.strip().split("\n\n")
    assert blocks[0].splitlines() == ["1", "00:00:00,000 --> 00:00:02,500", "第一句"]
    # 多行 cue 依閱讀順序串接
    assert blocks[1].splitlines()[2:] == ["上", "下"]


def test_srt_bilingual():
    srt = to_srt([cue(0, 1, ["譯文"], ["原文"])], bilingual=True)
    lines = srt.strip().splitlines()
    assert lines[2] == "譯文" and lines[3] == "原文"


def test_empty_output():
    assert to_srt([]) == ""
    assert to_ass([], 1920, 1080).count("Dialogue:") == 0


def _style_alignment(ass: str) -> str:
    line = [l for l in ass.splitlines() if l.startswith("Style:")][0]
    fields = line[len("Style: "):].split(",")
    return fields[18]  # Alignment 欄位


def test_ass_bottom_vs_top_alignment():
    bottom = to_ass([cue(0, 1, ["x"])], 1920, 1080, position="bottom")
    top = to_ass([cue(0, 1, ["x"])], 1920, 1080, position="top")
    assert _style_alignment(bottom) == "2"  # 底部置中
    assert _style_alignment(top) == "8"     # 頂部置中


def test_ass_resolution_from_video():
    ass = to_ass([cue(0, 1, ["x"])], 1280, 720)
    assert "PlayResX: 1280" in ass
    assert "PlayResY: 720" in ass


def test_ass_maxlines_split():
    c = cue(0, 3, ["一", "二", "三"])
    # max_lines=2 -> 拆成兩個同時存在的 Dialogue
    ass = to_ass([c], 1920, 1080, max_lines=2)
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogues) == 2
    assert dialogues[0].endswith("一\\N二")
    assert dialogues[1].endswith("三")


def test_ass_brace_neutralized():
    ass = to_ass([cue(0, 1, ["{\\b1}bold?"])], 1920, 1080)
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")][0]
    assert "{" not in dialogue.split(",,")[-1]
