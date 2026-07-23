from vidtranssub.config import stable_hash
from vidtranssub.manifest import Manifest


def test_stage_done_roundtrip(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    h = stable_hash({"interval": 1.0})
    assert not m.stage_done("sample", h)
    m.mark_done("sample", h)
    assert m.stage_done("sample", h)
    # 重新載入仍保留
    assert Manifest(tmp_path / "manifest.json").stage_done("sample", h)


def test_param_change_invalidates_stage(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    m.mark_done("sample", stable_hash({"interval": 1.0}))
    assert not m.stage_done("sample", stable_hash({"interval": 0.5}))


def test_check_stage_invalidates_downstream(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    for stage in ["probe", "sample", "ocr", "track", "translate", "cleanup", "emit"]:
        m.mark_done(stage, stable_hash({"v": 1}))

    # track 參數變更 -> track/translate/cleanup/emit 失效,probe/sample/ocr 保留
    assert not m.check_stage("track", stable_hash({"v": 2}))
    assert m.stage_done("probe", stable_hash({"v": 1}))
    assert m.stage_done("sample", stable_hash({"v": 1}))
    assert m.stage_done("ocr", stable_hash({"v": 1}))
    assert not m.stage_done("translate", stable_hash({"v": 1}))
    assert not m.stage_done("cleanup", stable_hash({"v": 1}))
    assert not m.stage_done("emit", stable_hash({"v": 1}))


def test_ocr_change_invalidates_track_onward(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    for stage in ["probe", "sample", "ocr", "track", "translate", "cleanup", "emit"]:
        m.mark_done(stage, stable_hash({"v": 1}))
    assert not m.check_stage("ocr", stable_hash({"v": 2}))
    assert m.stage_done("sample", stable_hash({"v": 1}))
    for s in ["track", "translate", "cleanup", "emit"]:
        assert not m.stage_done(s, stable_hash({"v": 1}))


def test_emit_change_only_invalidates_emit(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    for stage in ["probe", "sample", "ocr", "track", "translate", "cleanup", "emit"]:
        m.mark_done(stage, stable_hash({"v": 1}))
    assert not m.check_stage("emit", stable_hash({"v": 2}))
    for s in ["probe", "sample", "ocr", "track", "translate", "cleanup"]:
        assert m.stage_done(s, stable_hash({"v": 1}))


def test_input_change_resets_everything(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    h = stable_hash({})
    m.mark_done("sample", h)
    m.set_video_info({"duration": 10})
    m.ensure_input("hash-b")
    assert not m.stage_done("sample", h)
    assert m.video_info == {}


def test_mark_running_records_params(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    h = stable_hash({"interval": 1.0})
    m.mark_running("ocr", h)
    assert not m.stage_done("ocr", h)
    assert m.stage_params_hash("ocr") == h
    assert Manifest(tmp_path / "manifest.json").stage_params_hash("ocr") == h
