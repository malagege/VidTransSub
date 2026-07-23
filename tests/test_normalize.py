from vidtranssub.normalize import (
    has_meaningful_text,
    match_text,
    normalize_display,
    similarity,
)


def test_has_meaningful_text():
    assert has_meaningful_text("今日")
    assert has_meaningful_text("A1")
    assert has_meaningful_text("あ")
    assert not has_meaningful_text("。！")   # 純標點
    assert not has_meaningful_text("★")      # 單一裝飾符號
    assert not has_meaningful_text("   ")     # 純空白
    assert not has_meaningful_text("")


def test_normalize_display_nfkc_and_whitespace():
    # 全形空白(NFKC 轉半形)、行內空白折疊、頭尾去空白
    assert normalize_display("  いい　天気  ") == "いい 天気"
    # 全形數字 NFKC -> 半形
    assert normalize_display("１２３") == "123"


def test_normalize_display_unifies_newlines_keeps_punct():
    assert normalize_display("今日は\r\nいい天気") == "今日は\nいい天気"
    # 中日文標點不可被移除
    assert normalize_display("こんにちは、世界。") == "こんにちは、世界。"


def test_match_text_removes_whitespace_and_casefolds():
    assert match_text("Hello World") == "helloworld"
    assert match_text("今日は 天気") == "今日は天気"


def test_similarity():
    assert similarity("今日はいい天気です", "今日はいい天気です") == 1.0
    # 輕微 OCR 差異(1 字)應高於 0.85
    assert similarity("今日はいい天気です", "今日はいい天気でず") > 0.85
    # 明確不同文字應偏低
    assert similarity("本文A", "本文B") < 0.85
    assert similarity("", "") == 1.0
    assert similarity("abc", "") == 0.0
