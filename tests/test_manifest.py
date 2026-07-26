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


def test_migrate_params_hash_keeps_stage_done(tmp_path):
    """hash 組成改版時就地換 hash,已完成的階段不得因此重跑。"""
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    old, new, track_h = stable_hash({"v": 1}), stable_hash({"v": 2}), stable_hash({"t": 1})
    m.mark_done("ocr", old)
    m.mark_done("track", track_h)

    assert m.migrate_params_hash("ocr", old, new)
    assert m.stage_done("ocr", new)
    assert Manifest(tmp_path / "manifest.json").stage_done("ocr", new)
    # 只換 hash 表示法,結果沒變 -> 下游不該失效
    assert m.stage_done("track", track_h)


def test_migrate_params_hash_noop_when_not_matching(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.ensure_input("hash-a")
    old, new, other = stable_hash({"v": 1}), stable_hash({"v": 2}), stable_hash({"v": 3})
    m.mark_done("ocr", other)

    assert not m.migrate_params_hash("ocr", old, new)
    assert m.stage_done("ocr", other)  # 原記錄不動
    assert not m.migrate_params_hash("ocr", old, old)  # old == new 視為不需搬遷
    assert not m.migrate_params_hash("track", old, new)  # 沒有該階段紀錄
