"""CLI:多檔輸入解析與萬用字元展開。"""

from vidtranssub.__main__ import build_parser, expand_inputs


def test_parser_accepts_multiple_inputs():
    args = build_parser().parse_args(["a.mp4", "b.mp4", "--target-lang", "zh-TW"])
    assert args.inputs == ["a.mp4", "b.mp4"]
    assert args.target_lang == "zh-TW"


def test_parser_single_input_still_works():
    args = build_parser().parse_args(["only.mp4"])
    assert args.inputs == ["only.mp4"]


def test_expand_literal_paths_preserved_and_deduped():
    # 不含萬用字元者原樣保留;重複路徑去重(保留順序)。
    assert expand_inputs(["x.mp4", "y.mp4", "x.mp4"]) == ["x.mp4", "y.mp4"]


def test_expand_glob(tmp_path):
    for name in ("c.mp4", "a.mp4", "b.mp4", "note.txt"):
        (tmp_path / name).write_text("x")
    got = expand_inputs([str(tmp_path / "*.mp4")])
    assert [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in got] == ["a.mp4", "b.mp4", "c.mp4"]


def test_expand_glob_no_match_skipped(tmp_path, capsys):
    got = expand_inputs([str(tmp_path / "*.mkv")])
    assert got == []
    assert "沒有符合的檔案" in capsys.readouterr().err
