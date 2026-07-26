"""整合測試:真實 ffmpeg 取樣 + 假 OCR provider + 假 LLM cache server 跑完整流程。"""

from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

from vidtranssub.config import Config
from vidtranssub.pipeline import run_pipeline

from fake_ocr import FakeOCRProvider, block
from mock_llm import create_mock_llm

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="需要 ffmpeg/ffprobe",
)

TOP = (0.1, 0.05, 0.9, 0.2)
BOTTOM = (0.1, 0.7, 0.9, 0.9)


def scene_blocks(index: int):
    """index(1-based)於 interval=1 對應時間 t = index-1。"""
    t = index - 1
    if 0 <= t < 4:
        return [block("こんにちは世界", BOTTOM, order=1)]
    if 4 <= t < 6:
        return []
    if 6 <= t < 9:
        return [block("タイトル", TOP, order=1), block("本文A", BOTTOM, order=2)]
    if 9 <= t < 15:
        return [block("タイトル", TOP, order=1), block("本文B", BOTTOM, order=2)]
    return []


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def llm_url():
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(create_mock_llm(), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("mock LLM server 未啟動")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def test_video(tmp_path):
    from testdata.make_test_video import build_video

    return build_video(tmp_path / "clip.mp4")


def _cfg(tmp_path, llm_url) -> Config:
    return Config(
        interval=1.0,
        work_dir=str(tmp_path / "work"),
        target_lang="zh-TW",
        source_lang="ja",
        llm_cache_url=llm_url,
    )


def test_full_pipeline(tmp_path, test_video, llm_url, capsys):
    provider = FakeOCRProvider(scene_blocks)
    cfg = _cfg(tmp_path, llm_url)
    stats = run_pipeline(cfg, test_video, provider=provider)

    # 產出 SRT/ASS 於影片同目錄
    srt = test_video.parent / "clip.zh-TW.srt"
    ass = test_video.parent / "clip.zh-TW.ass"
    assert srt.exists() and ass.exists()

    srt_text = srt.read_text(encoding="utf-8")
    assert "[譯] こんにちは世界" in srt_text
    assert "[譯] 本文A" in srt_text and "[譯] 本文B" in srt_text
    assert "[譯] タイトル" in srt_text

    # 取樣數約等於 ceil(20/1)
    assert 19 <= stats["sample_count"] <= 21

    # 事件:連續相同文字合併,同區域換字分裂
    work = Path(cfg.work_dir).resolve() / "clip-videosub"
    events = json.loads((work / "events.json").read_text(encoding="utf-8"))["events"]
    texts = ["".join(e["source_lines"]) for e in events]
    assert "こんにちは世界" in texts
    assert "本文A" in texts and "本文B" in texts

    greeting = [e for e in events if e["source_lines"] == ["こんにちは世界"]][0]
    assert greeting["start"] == 0.0
    assert greeting["end"] == pytest.approx(4.0, abs=1.0)

    # 所有事件滿足 0 <= start < end <= duration 且依時間排序
    duration = stats["duration_seconds"]
    starts = [e["start"] for e in events]
    assert starts == sorted(starts)
    for e in events:
        assert 0.0 <= e["start"] < e["end"] <= duration + 1e-6


def test_exact_image_cache_within_run(tmp_path, test_video, llm_url):
    provider = FakeOCRProvider(scene_blocks)
    cfg = _cfg(tmp_path, llm_url)
    stats = run_pipeline(cfg, test_video, provider=provider)

    ocr = stats["ocr"]
    # 完全相同的圖片走 cache,實際送 OCR 的張數少於樣本數
    assert ocr["cache_hits"] > 0
    assert ocr["images_sent_to_ocr"] < stats["sample_count"]
    # provider 只被叫在 miss 上
    assert provider.calls == ocr["images_sent_to_ocr"]


def test_resume_skips_completed_stages(tmp_path, test_video, llm_url):
    import httpx

    cfg = _cfg(tmp_path, llm_url)
    run_pipeline(cfg, test_video, provider=FakeOCRProvider(scene_blocks))

    # 重跑:OCR 不應再呼叫 provider,翻譯不應再打 LLM cache API(chat 請求數不變)。
    before = httpx.get(f"{llm_url}/vtf/stats").json()
    provider2 = FakeOCRProvider(scene_blocks)
    run_pipeline(cfg, test_video, provider=provider2)
    after = httpx.get(f"{llm_url}/vtf/stats").json()

    assert provider2.calls == 0  # OCR 已完成,不再辨識
    assert before["hits"] + before["misses"] == after["hits"] + after["misses"]  # 翻譯已完成
    assert (test_video.parent / "clip.zh-TW.srt").exists()


def test_batch_size_change_does_not_rerun_ocr(tmp_path, test_video, llm_url):
    """batch size 只決定一次送幾張,改了不得清空既有 ocr/*.json 或讓 cache 失效。"""
    import dataclasses

    cfg = _cfg(tmp_path, llm_url)
    run_pipeline(cfg, test_video, provider=FakeOCRProvider(scene_blocks), to_stage="ocr")

    work = Path(cfg.work_dir).resolve() / "clip-videosub"
    before = {p.name: p.read_text(encoding="utf-8") for p in (work / "ocr").glob("*.json")}
    assert before

    provider2 = FakeOCRProvider(scene_blocks)
    smaller_batch = dataclasses.replace(cfg, ocr_batch_size=1)
    run_pipeline(smaller_batch, test_video, provider=provider2, to_stage="ocr")

    assert provider2.calls == 0  # 已完成,不得重新辨識
    after = {p.name: p.read_text(encoding="utf-8") for p in (work / "ocr").glob("*.json")}
    assert after == before


def test_v1_ocr_hash_is_migrated_not_invalidated(tmp_path, test_video, llm_url, capsys):
    """升級後第一次執行:v1(hash 含 batch_size)的 manifest/cache 須就地搬遷,不重跑。"""
    from vidtranssub.config import stable_hash
    from vidtranssub.manifest import Manifest
    from vidtranssub.ocr_cache import OCRCache

    cfg = _cfg(tmp_path, llm_url)
    provider = FakeOCRProvider(scene_blocks)
    run_pipeline(cfg, test_video, provider=provider, to_stage="ocr")
    assert provider.calls > 0

    fingerprint = provider.fingerprint()
    new_hash = stable_hash(cfg.ocr_params(fingerprint))
    legacy_hash = stable_hash(cfg.legacy_ocr_params_v1(fingerprint))
    assert legacy_hash != new_hash

    # 把 manifest 與 cache 退回 v1 的樣子(key/hash 都含 batch_size)。
    work = Path(cfg.work_dir).resolve() / "clip-videosub"
    manifest = Manifest(work / "manifest.json")
    assert manifest.migrate_params_hash("ocr", new_hash, legacy_hash)
    db_path = Path(cfg.work_dir).resolve() / "videosub_ocr_cache.db"
    cache = OCRCache(db_path)
    assert cache.migrate_param_hash(new_hash, legacy_hash) > 0
    cache.close()

    capsys.readouterr()
    provider2 = FakeOCRProvider(scene_blocks)
    run_pipeline(cfg, test_video, provider=provider2, to_stage="ocr")

    assert provider2.calls == 0  # 舊結果被認得,不重跑
    assert "清除" not in capsys.readouterr().out  # 也沒觸發「參數已變更」清檔
    assert Manifest(work / "manifest.json").stage_done("ocr", new_hash)
    cache = OCRCache(db_path)
    entries = cache.db.execute(
        "SELECT COUNT(*) FROM ocr_cache WHERE substr(key, instr(key, ':') + 1) = ?",
        (new_hash,),
    ).fetchone()[0]
    cache.close()
    assert entries > 0  # cache 也搬到新 key,不是變成孤兒


def test_resume_progress_log_uses_total_samples(tmp_path, test_video, llm_url, capsys):
    """續跑時進度分母須是全部樣本數,否則看起來像整部重跑。"""
    cfg = _cfg(tmp_path, llm_url)
    run_pipeline(cfg, test_video, provider=FakeOCRProvider(scene_blocks), to_stage="ocr")

    # 砍掉後半段 OCR 結果並把階段退回 running,模擬中途被 Ctrl+C。
    import dataclasses

    work = Path(cfg.work_dir).resolve() / "clip-videosub"
    results = sorted(
        p for p in (work / "ocr").glob("*.json") if not p.name.endswith(".raw.json")
    )
    assert len(results) >= 4
    for p in results[len(results) // 2:]:
        p.unlink()
    manifest_path = work / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["stages"]["ocr"]["status"] = "running"
    manifest_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # 關掉 cache,讓剩下的都真的走一次 OCR,進度行才可觀測。
    resumed_cfg = dataclasses.replace(cfg, no_ocr_cache=True)
    capsys.readouterr()
    stats = run_pipeline(
        resumed_cfg, test_video, provider=FakeOCRProvider(scene_blocks), to_stage="ocr"
    )
    out = capsys.readouterr().out
    total = stats["sample_count"]

    assert f"/{total} 張樣本已完成" in out  # 續跑行帶總數
    progress = [ln for ln in out.splitlines() if ln.startswith("[ocr] 進度 ")]
    assert progress
    assert progress[-1].startswith(f"[ocr] 進度 {total}/{total} 張樣本")  # 收尾是整體完成


def test_two_phase_split_avoids_ocr_provider(tmp_path, test_video, llm_url):
    """--to-stage ocr 後 --from-stage track:第二段不得載入 OCR provider(VRAM 分離)。"""
    cfg = _cfg(tmp_path, llm_url)

    # 第一段:只跑到 ocr(用假 provider)。
    run_pipeline(cfg, test_video, provider=FakeOCRProvider(scene_blocks), to_stage="ocr")
    assert not (test_video.parent / "clip.zh-TW.srt").exists()  # 尚未 emit
    work = Path(cfg.work_dir).resolve() / "clip-videosub"
    assert (work / "ocr").exists() and any((work / "ocr").glob("*.json"))
    assert not (work / "tracks.json").exists()

    # 第二段:從 track 開始,provider=None。
    # 若誤入 ocr 階段,會嘗試 build_provider() 載入 PaddleOCR 而拋錯;能完成即證明沒碰 OCR。
    run_pipeline(cfg, test_video, provider=None, from_stage="track")
    assert (test_video.parent / "clip.zh-TW.srt").exists()
    assert (test_video.parent / "clip.zh-TW.ass").exists()


def test_translate_unique_cues(tmp_path, test_video, llm_url):
    cfg = _cfg(tmp_path, llm_url)
    stats = run_pipeline(cfg, test_video, provider=FakeOCRProvider(scene_blocks))
    tr = stats["translate"]
    # 4 個唯一文字:こんにちは世界 / タイトル / 本文A / 本文B
    assert tr["unique_cues"] == 4
    assert tr["cache_hits"] == 0  # 首次執行全 miss
    assert tr["upstream_calls"] == tr["cache_misses"]
