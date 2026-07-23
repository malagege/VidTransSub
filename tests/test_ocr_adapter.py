import pytest

from vidtranssub.ocr_provider import OCRBlock, OCRResult, blocks_from_parsing_list


def _item(content, bbox, label="text", order=1, **extra):
    d = {
        "block_content": content,
        "block_bbox": bbox,
        "block_label": label,
        "block_order": order,
    }
    d.update(extra)
    return d


def test_field_mapping_and_bbox_normalization():
    blocks = blocks_from_parsing_list(
        [_item("今日はいい天気です", [120, 720, 880, 910], "text", 3)],
        width=1000, height=1000,
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.source_text == "今日はいい天気です"
    assert b.label == "text"
    assert b.reading_order == 3
    assert b.bbox == pytest.approx((0.12, 0.72, 0.88, 0.91))
    assert b.confidence is None  # 沒有分數 -> None


def test_confidence_preserved_and_threshold():
    # 有分數且低於門檻 -> 丟棄
    assert blocks_from_parsing_list(
        [_item("あ", [0, 0, 10, 10], confidence=0.5)], 100, 100, confidence_threshold=0.8
    ) == []
    # 有分數且高於門檻 -> 保留
    kept = blocks_from_parsing_list(
        [_item("あ", [0, 0, 10, 10], confidence=0.9)], 100, 100, confidence_threshold=0.8
    )
    assert kept and kept[0].confidence == 0.9
    # 無分數 -> 不套用門檻,保留且 confidence 為 None
    kept2 = blocks_from_parsing_list(
        [_item("あ", [0, 0, 10, 10])], 100, 100, confidence_threshold=0.8
    )
    assert kept2 and kept2[0].confidence is None


def test_blank_and_punct_filtered():
    items = [
        _item("", [0, 0, 10, 10]),
        _item("   ", [0, 0, 10, 10]),
        _item("。", [0, 0, 10, 10]),
        _item("★", [0, 0, 10, 10]),
        _item("本文", [0, 0, 10, 10]),
    ]
    blocks = blocks_from_parsing_list(items, 100, 100)
    assert [b.source_text for b in blocks] == ["本文"]


def test_bbox_clamped_and_reordered():
    # 座標超界 -> 夾到 0~1;x1<x0 -> 交換
    blocks = blocks_from_parsing_list(
        [_item("x", [1200, -50, 100, 200])], 1000, 1000
    )
    x0, y0, x1, y1 = blocks[0].bbox
    assert 0.0 <= x0 <= x1 <= 1.0
    assert 0.0 <= y0 <= y1 <= 1.0
    assert y0 == 0.0  # -50 夾到 0


def test_alias_keys():
    blocks = blocks_from_parsing_list(
        [{"content": "hello", "bbox": [0, 0, 50, 50], "label": "t", "order": 2}],
        100, 100,
    )
    assert blocks[0].source_text == "hello"
    assert blocks[0].reading_order == 2


def test_missing_dimensions_raises():
    with pytest.raises(ValueError):
        blocks_from_parsing_list([_item("x", [0, 0, 10, 10])], 0, 100)


def test_ocrresult_roundtrip_and_reindex():
    res = OCRResult(0, 0.0, [OCRBlock("a", (0, 0, 1, 1), "x")], status="ok")
    res.reindex(17, 16.0)
    assert res.sample_index == 17 and res.timestamp == 16.0
    assert res.blocks[0].id == "17-1"
    d = res.to_dict()
    back = OCRResult.from_dict(d)
    assert back.sample_index == 17
    assert back.blocks[0].source_text == "x"
    assert "raw" not in d  # 正規化 JSON 不含 raw
