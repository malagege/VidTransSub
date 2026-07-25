"""Stage 8:輸出 SRT / ASS 字幕。

- SRT 使用 HH:MM:SS,mmm;ASS 使用 H:MM:SS.cc。
- 檔案 UTF-8,不加 BOM(由呼叫端寫檔時保證)。
- --subtitle-position bottom|top 只影響 ASS(底部/頂部置中);SRT 位置由播放器決定。
- 一條 cue 可為多行:SRT 依閱讀順序串接;ASS 超過 max_lines 時分成同時存在的 events。
- --bilingual:先譯文,再原文。
"""

from __future__ import annotations

# ASS Alignment(numpad 方向):2 = 底部置中,8 = 頂部置中。
_ALIGNMENT = {"bottom": 2, "top": 8}

# ASS PrimaryColour 為 &HAABBGGRR(alpha+BGR)。OCR 維持白色。
_NAMED_COLORS = {
    "white": "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "cyan": "&H00FFFF00",
    "green": "&H0000FF00",
    "lime": "&H0000FF00",
    "red": "&H000000FF",
    "orange": "&H000080FF",
    "magenta": "&H00FF00FF",
    "pink": "&H00CBC0FF",
    "black": "&H00000000",
}


def _ass_color(value: str) -> str:
    """把顏色名 / #RRGGBB / &H.. 轉成 ASS 的 &HAABBGGRR;無法解析時退回白色。"""
    if not value:
        return "&H00FFFFFF"
    v = value.strip()
    low = v.lower()
    if low in _NAMED_COLORS:
        return _NAMED_COLORS[low]
    if low.startswith("&h"):
        return "&H" + v[2:].upper()
    if v.startswith("#") and len(v) == 7:
        r, g, b = v[1:3], v[3:5], v[5:7]
        return f"&H00{b}{g}{r}".upper()
    return "&H00FFFFFF"


def _srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = round(seconds * 100)
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def display_lines(cue: dict, bilingual: bool) -> list[str]:
    """一條 cue 要顯示的文字行(已依閱讀順序)。譯文缺漏時以原文遞補。"""
    source = cue.get("source_lines", [])
    translated = cue.get("translated_lines", []) or []
    shown = [t if t else (source[i] if i < len(source) else "")
             for i, t in enumerate(translated)]
    if not shown:
        shown = list(source)
    if bilingual:
        return shown + list(source)
    return shown


def to_srt(cues: list[dict], bilingual: bool = False) -> str:
    """SRT:每條 cue 一個區塊,多行以換行串接。空輸入回傳空字串。"""
    blocks = []
    n = 0
    for cue in cues:
        lines = [ln for ln in display_lines(cue, bilingual) if ln != ""]
        if not lines:
            continue
        n += 1
        text = "\n".join(lines)
        blocks.append(
            f"{n}\n{_srt_time(cue['start'])} --> {_srt_time(cue['end'])}\n{text}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def _style_line(
    name: str, fontsize: int, outline: int, alignment: int,
    margin_h: int, margin_v: int, primary: str,
) -> str:
    return (
        f"Style: {name},Arial,{fontsize},{primary},&H000000FF,&H00000000,&H80000000,"
        f"0,0,0,0,100,100,0,0,1,{outline},1,{alignment},{margin_h},{margin_h},{margin_v},1"
    )


def _ass_header(width: int, height: int, styles: list[tuple[str, str, str]]) -> str:
    """styles:list of (name, position, primary_colour_ass)。第一個為 Default。"""
    width = width or 1920
    height = height or 1080
    # 依解析度縮放字級、外框與安全邊距。
    fontsize = max(20, round(height * 0.05))
    outline = max(1, round(height * 0.002))
    margin_v = max(10, round(height * 0.04))
    margin_h = max(10, round(width * 0.03))
    style_lines = "\n".join(
        _style_line(
            name, fontsize, outline, _ALIGNMENT.get(position, 2),
            margin_h, margin_v, primary,
        )
        for (name, position, primary) in styles
    )
    return (
        "[Script Info]\n"
        "Title: VideoTransSub\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: None\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{style_lines}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _ass_escape(text: str) -> str:
    # 大括號會開啟 override 區塊,先中和;換行轉成 \N。
    text = text.replace("{", "(").replace("}", ")")
    return text.replace("\r\n", "\\N").replace("\n", "\\N").replace("\r", "\\N")


def _chunk(lines: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [lines] if lines else []
    return [lines[i : i + size] for i in range(0, len(lines), size)]


def to_ass(
    cues: list[dict],
    width: int,
    height: int,
    position: str = "bottom",
    bilingual: bool = False,
    max_lines: int = 2,
    audio_position: str = "top",
    audio_color: str = "yellow",
) -> str:
    """ASS:多行 cue 超過 max_lines 時,拆成多個同時存在的 Dialogue。

    以 cue["source"] 區分樣式:OCR(或未標記)用白色 Default,語音用 Audio 樣式著色。
    只有實際存在語音事件時才加入 Audio 樣式,無語音時輸出與純 OCR 版本完全相同。
    """
    has_audio = any(c.get("source") == "audio" for c in cues)
    styles: list[tuple[str, str, str]] = [("Default", position, "&H00FFFFFF")]
    if has_audio:
        styles.append(("Audio", audio_position, _ass_color(audio_color)))

    parts = [_ass_header(width, height, styles)]
    dialogues: list[str] = []
    for cue in cues:
        lines = [ln for ln in display_lines(cue, bilingual) if ln != ""]
        if not lines:
            continue
        style_name = "Audio" if cue.get("source") == "audio" else "Default"
        start = _ass_time(cue["start"])
        end = _ass_time(cue["end"])
        for group in _chunk(lines, max_lines):
            text = _ass_escape("\n".join(group))
            dialogues.append(
                f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{text}"
            )
    return "".join(parts) + "\n".join(dialogues) + ("\n" if dialogues else "")


def write_text_no_bom(path, text: str) -> None:
    """UTF-8 無 BOM,暫存檔加 rename 避免中斷留下半個檔案。"""
    import os
    from pathlib import Path

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
