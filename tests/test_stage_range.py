import pytest

from vidtranssub.manifest import STAGES
from vidtranssub.pipeline import PipelineError, STAGE_INDEX, stage_bounds


def _in_range(stage, lo, hi):
    return lo <= STAGE_INDEX[stage] <= hi


def test_default_runs_all_stages():
    lo, hi = stage_bounds()
    assert (lo, hi) == (0, len(STAGES) - 1)
    assert all(_in_range(s, lo, hi) for s in STAGES)


def test_only_stage_single():
    lo, hi = stage_bounds(only_stage="ocr")
    assert lo == hi == STAGE_INDEX["ocr"]
    assert _in_range("ocr", lo, hi)
    assert not _in_range("track", lo, hi)


def test_cache_alias_maps_to_ocr():
    assert stage_bounds(only_stage="cache") == stage_bounds(only_stage="ocr")
    assert stage_bounds(from_stage="cache") == stage_bounds(from_stage="ocr")
    assert stage_bounds(to_stage="cache") == stage_bounds(to_stage="ocr")


def test_from_stage_runs_to_end():
    lo, hi = stage_bounds(from_stage="track")
    assert lo == STAGE_INDEX["track"]
    assert hi == len(STAGES) - 1
    # 關鍵:--from-stage track 時 ocr 不在區間 -> 不會載入 PaddleOCR
    assert not _in_range("ocr", lo, hi)
    assert _in_range("translate", lo, hi)


def test_to_stage_runs_from_start():
    lo, hi = stage_bounds(to_stage="ocr")
    assert lo == 0
    assert hi == STAGE_INDEX["ocr"]
    assert _in_range("probe", lo, hi) and _in_range("ocr", lo, hi)
    assert not _in_range("track", lo, hi)


def test_explicit_range():
    lo, hi = stage_bounds(from_stage="sample", to_stage="track")
    assert [s for s in STAGES if _in_range(s, lo, hi)] == ["sample", "ocr", "track"]


def test_inverted_range_raises():
    with pytest.raises(PipelineError):
        stage_bounds(from_stage="track", to_stage="sample")


def test_two_phase_workflow_split_is_contiguous():
    # 兩段(--to-stage ocr 與 --from-stage track)剛好無縫涵蓋全部階段
    lo1, hi1 = stage_bounds(to_stage="ocr")
    lo2, hi2 = stage_bounds(from_stage="track")
    covered = [s for s in STAGES if _in_range(s, lo1, hi1) or _in_range(s, lo2, hi2)]
    assert covered == STAGES
    assert hi1 + 1 == lo2  # 無重疊、無縫隙
