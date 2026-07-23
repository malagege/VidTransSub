import math

import pytest

from vidtranssub.ffmpeg import (
    _extract_rotation,
    _fmt,
    expected_sample_count,
    validate_interval,
)


def test_validate_interval_ok():
    for v in (0.1, 0.5, 1.0, 3.0, 60.0):
        validate_interval(v)


def test_validate_interval_rejects():
    for v in (0.0, -1.0, 60.5, 0.05):
        with pytest.raises(ValueError):
            validate_interval(v)
    with pytest.raises(ValueError):
        validate_interval(math.inf)
    with pytest.raises(ValueError):
        validate_interval(math.nan)


def test_expected_sample_count():
    assert expected_sample_count(20.0, 1.0) == 20
    assert expected_sample_count(20.0, 0.5) == 40
    assert expected_sample_count(20.0, 2.0) == 10
    assert expected_sample_count(20.0, 3.0) == 7   # ceil(6.67)
    assert expected_sample_count(0.0, 1.0) == 0


def test_fmt():
    assert _fmt(1.0) == "1"
    assert _fmt(0.5) == "0.5"
    assert _fmt(3.0) == "3"
    assert _fmt(2.25) == "2.25"


def test_extract_rotation():
    assert _extract_rotation({"side_data_list": [{"rotation": -90}]}) == 270
    assert _extract_rotation({"side_data_list": [{"rotation": 90}]}) == 90
    assert _extract_rotation({"tags": {"rotate": "180"}}) == 180
    assert _extract_rotation({}) == 0
