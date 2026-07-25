"""語音字幕:ASS 依 source 上色、音訊事件組成,以及「無語音時輸出不變」的保證。"""

from vidtranssub.cleanup import finalize_audio_events
from vidtranssub.config import Config
from vidtranssub.subtitle import _ass_color, to_ass
from vidtranssub.tracking import canonical_key


def ocr_cue(start, end, text):
    return {
        "start": start, "end": end,
        "source_lines": [text], "translated_lines": [text],
        "bboxes": [[0.1, 0.7, 0.9, 0.9]], "source": "ocr",
    }


def audio_cue(start, end, text):
    return {
        "start": start, "end": end,
        "source_lines": [text], "translated_lines": [text],
        "bboxes": [], "source": "audio",
    }


def _style_names(ass: str) -> list[str]:
    return [l[len("Style: "):].split(",")[0] for l in ass.splitlines() if l.startswith("Style:")]


def test_ass_color_parsing():
    assert _ass_color("yellow") == "&H0000FFFF"
    assert _ass_color("white") == "&H00FFFFFF"
    assert _ass_color("#FF0000") == "&H000000FF"  # 紅:#RRGGBB -> &H00BBGGRR
    assert _ass_color("&h0000ffff") == "&H0000FFFF"
    assert _ass_color("nonsense") == "&H00FFFFFF"  # 退回白色


def test_no_audio_keeps_single_default_style():
    """純 OCR(無 source=audio)時不得出現 Audio 樣式,維持與舊版相同的單一 Default。"""
    ass = to_ass([ocr_cue(0, 1, "x")], 1920, 1080)
    assert _style_names(ass) == ["Default"]
    assert "Audio" not in ass


def test_audio_adds_colored_style_and_assigns_per_source():
    cues = [ocr_cue(0, 2, "畫面字"), audio_cue(0, 2, "語音字")]
    ass = to_ass(cues, 1920, 1080, audio_position="top", audio_color="yellow")
    assert _style_names(ass) == ["Default", "Audio"]
    # Audio 樣式帶黃色 PrimaryColour
    audio_style = [l for l in ass.splitlines() if l.startswith("Style: Audio,")][0]
    assert "&H0000FFFF" in audio_style
    # Dialogue 依 source 指派樣式
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    ocr_line = [d for d in dialogues if d.endswith("畫面字")][0]
    audio_line = [d for d in dialogues if d.endswith("語音字")][0]
    assert ",Default,," in ocr_line
    assert ",Audio,," in audio_line


def test_finalize_audio_events_applies_translation_and_tag():
    cfg = Config()
    segments = [
        {"start": 1.0, "end": 2.5, "text": "hello world"},
        {"start": 3.0, "end": 3.0, "text": "zero-length"},  # end<=start 應丟棄
        {"start": 4.0, "end": 5.0, "text": "  "},            # 空白應丟棄
    ]
    translations = {canonical_key(["hello world"]): ["你好世界"]}
    events = finalize_audio_events(segments, translations, cfg, duration=10.0)
    assert len(events) == 1
    ev = events[0]
    assert ev["source"] == "audio"
    assert ev["source_lines"] == ["hello world"]
    assert ev["translated_lines"] == ["你好世界"]
    assert ev["bboxes"] == []


def test_finalize_audio_events_keeps_source_when_no_translation():
    cfg = Config()
    segments = [{"start": 0.0, "end": 1.0, "text": "untranslated"}]
    events = finalize_audio_events(segments, {}, cfg, duration=10.0)
    assert events[0]["translated_lines"] == ["untranslated"]
