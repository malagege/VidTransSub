"""全流程串接與斷點續跑。

階段順序:probe -> sample -> ocr(含 exact-image cache) -> track -> translate -> cleanup -> emit。
中間資料一律以 JSON 落地;SRT/ASS 是可重新產生的最後輸出,不作為續跑依據。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import asr, ffmpeg, subtitle
from .cleanup import finalize_audio_events, finalize_cues
from .config import Config, stable_hash
from .llm_cache_client import LLMCacheClient, TranslationError, resolve_model
from .manifest import STAGES, Manifest
from .ocr_cache import OCRCache, ocr_cache_key, sha256_bytes
from .ocr_provider import OCRProvider, OCRResult
from .tracking import canonical_key, compose_cues, track_events

PROBE_HASH = stable_hash({"stage": "probe", "v": 1})

STAGE_INDEX = {name: i for i, name in enumerate(STAGES)}
STAGE_ALIAS = {"cache": "ocr"}  # cache 是 ocr 階段的一部分


class PipelineError(RuntimeError):
    pass


def stage_bounds(
    only_stage: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[int, int]:
    """把階段選項換算成要執行的 [from_idx, to_idx] 區間(含端點)。

    - only_stage:只跑單一階段(from=to)。
    - from_stage/to_stage:跑一段連續區間;省略者分別取頭/尾。
    - cache 視為 ocr。
    """
    def norm(s: str | None) -> str | None:
        return STAGE_ALIAS.get(s, s) if s else s

    only_stage, from_stage, to_stage = norm(only_stage), norm(from_stage), norm(to_stage)
    if only_stage is not None:
        i = STAGE_INDEX[only_stage]
        return i, i
    lo = STAGE_INDEX[from_stage] if from_stage else 0
    hi = STAGE_INDEX[to_stage] if to_stage else len(STAGES) - 1
    if lo > hi:
        raise PipelineError(
            f"--from-stage({from_stage})不可晚於 --to-stage({to_stage})"
        )
    return lo, hi


def build_provider(cfg: Config) -> OCRProvider:
    if cfg.ocr_provider == "paddleocr-vl":
        from .paddleocr_provider import PaddleOCRProvider

        # server API key 只從環境變數取,不進 Config/manifest/log。
        server_api_key = (
            os.environ.get(cfg.ocr_api_key_env) if cfg.ocr_server_url else None
        )
        return PaddleOCRProvider(
            model=cfg.paddleocr_model,
            engine=cfg.paddleocr_engine,
            device=cfg.ocr_device,
            batch_size=cfg.ocr_batch_size,
            confidence=cfg.ocr_confidence,
            server_url=cfg.ocr_server_url,
            server_backend=cfg.ocr_server_backend,
            server_model=cfg.ocr_server_model,
            server_api_key=server_api_key,
        )
    raise PipelineError(f"未知的 --ocr-provider:{cfg.ocr_provider}")


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _ocr_paths(work: Path, index: int) -> tuple[Path, Path]:
    stem = f"{index:08d}"
    return work / "ocr" / f"{stem}.json", work / "ocr" / f"{stem}.raw.json"


def run_pipeline(
    cfg: Config,
    input_path: str | Path,
    provider: OCRProvider | None = None,
    only_stage: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    log=print,
) -> dict:
    ffmpeg.check_tools()
    ffmpeg.validate_interval(cfg.interval)
    run_t0 = time.monotonic()

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise PipelineError(f"找不到輸入檔:{input_path}")

    work_root = Path(cfg.work_dir).resolve()
    work = work_root / f"{input_path.stem}-videosub"
    work.mkdir(parents=True, exist_ok=True)
    output_dir = input_path.parent

    manifest = Manifest(work / "manifest.json")
    log("[init] 計算輸入檔 hash…")
    input_hash = ffmpeg.file_sha256(input_path)
    manifest.ensure_input(input_hash)

    stats_path = work / "stats.json"
    stats: dict = (
        json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    )
    durations: dict = stats.setdefault("durations", {})

    def save_stats() -> None:
        _atomic_write_json(stats_path, stats)

    lo_idx, hi_idx = stage_bounds(only_stage, from_stage, to_stage)

    def wanted(stage: str) -> bool:
        return lo_idx <= STAGE_INDEX[stage] <= hi_idx

    # ---- Stage 1: probe ----
    if wanted("probe") and not manifest.check_stage("probe", PROBE_HASH):
        log("[probe] 讀取影片資訊…")
        t0 = time.monotonic()
        info = ffmpeg.probe_video(input_path)
        durations["probe"] = round(time.monotonic() - t0, 2)
        manifest.set_video_info(info)
        manifest.mark_done("probe", PROBE_HASH)
        log(f"[probe] 時長 {info['duration']:.2f}s,{info['width']}x{info['height']}"
            f",旋轉 {info['rotation']}°,{'含' if info['has_audio'] else '無'}音軌")
    elif wanted("probe"):
        log("[probe] 已完成,跳過")

    video_info = manifest.video_info
    if not video_info:
        raise PipelineError("manifest 缺 video_info,請先執行 probe 階段")
    duration = float(video_info.get("duration") or 0.0)

    stats.update({
        "input": str(input_path),
        "duration_seconds": round(duration, 3),
        "interval_seconds": cfg.interval,
    })

    # ---- 啟動摘要(在昂貴呼叫前) ----
    est = ffmpeg.expected_sample_count(duration, cfg.interval)
    log("")
    log("===== VideoTransSub =====")
    log(f"輸入影片   : {input_path.name}")
    log(f"影片時長   : {duration:.2f}s")
    log(f"取樣間隔   : {cfg.interval}s")
    log(f"預估樣本數 : ~{est}")
    log(f"OCR provider: {cfg.ocr_provider}")
    if cfg.ocr_server_url:
        log(f"OCR VLM server: {cfg.ocr_server_backend} @ {cfg.ocr_server_url}")
    if cfg.audio_transcribe:
        log(f"語音轉字幕 : 開啟(faster-whisper model={cfg.asr_model})")
    else:
        log("語音轉字幕 : 關閉")
    log(f"目標語言   : {cfg.target_lang}")
    log(f"工作目錄   : {work}")
    log("=========================")
    log("")

    # ---- Stage 2: sample ----
    sample_hash = stable_hash(cfg.sample_params())
    if wanted("sample") and not manifest.check_stage("sample", sample_hash):
        log(f"[sample] 以 interval={cfg.interval}s 取樣…")
        t0 = time.monotonic()
        samples = ffmpeg.sample_frames(
            work, input_path, video_info, cfg.interval, cfg.max_width, cfg.image_quality
        )
        ffmpeg.write_samples_json(work, samples, {
            "interval": cfg.interval, "max_width": cfg.max_width,
            "image_quality": cfg.image_quality, "duration": duration,
        })
        durations["sample"] = round(time.monotonic() - t0, 2)
        manifest.mark_done("sample", sample_hash)
        log(f"[sample] 完成:{len(samples)} 張樣本")
    elif wanted("sample"):
        log("[sample] 已完成,跳過")

    # 只有 sample 及其下游階段需要 samples.json。
    samples: list[dict] = []
    if hi_idx >= STAGE_INDEX["sample"]:
        samples_json = work / "samples.json"
        if not samples_json.exists():
            if wanted("sample"):
                raise PipelineError("samples.json 不存在,請先執行 sample 階段")
            raise PipelineError(
                "缺少 samples.json,無法執行選定的階段區間(請先跑到 sample 階段)"
            )
        samples = json.loads(samples_json.read_text(encoding="utf-8"))["samples"]
        stats["sample_count"] = len(samples)
        save_stats()

    # ASR 與 OCR 的原生 DLL 在 Windows 上會依載入順序衝突(WinError 127);
    # 若啟用 ASR,在載入 paddle(OCR)前先暖身 ctranslate2,確保其 DLL 先就位。
    if cfg.audio_transcribe and wanted("asr"):
        asr.preload_backend(log=log)

    # ---- Stage 3+4: ocr(含 exact-image cache)----
    if wanted("ocr"):
        provider = _run_ocr_stage(cfg, work, work_root, samples, provider, manifest, stats, log)
        save_stats()

    # ---- Stage 5: track ----
    track_hash = stable_hash(cfg.track_params())
    if wanted("track") and not manifest.check_stage("track", track_hash):
        log("[track] 正規化與跨樣本合併…")
        t0 = time.monotonic()
        results = _load_ocr_results(work, samples)
        events = track_events(results, cfg, duration)
        _atomic_write_json(work / "tracks.json", {"events": events})
        durations["track"] = round(time.monotonic() - t0, 2)
        manifest.mark_done("track", track_hash)
        stats["track_count"] = len(events)
        log(f"[track] 完成:{len(events)} 個事件")
    elif wanted("track"):
        log("[track] 已完成,跳過")

    # ---- Stage: asr(語音轉字幕;預設關閉,與 OCR 平行的獨立分支) ----
    if wanted("asr"):
        # 同一行程接著跑 ASR 前,先釋放 OCR 佔用的顯存,避免 large-v3 等 CUDA OOM。
        if cfg.audio_transcribe and provider is not None and hasattr(provider, "release"):
            provider.release(log=log)
        _run_asr_stage(cfg, work, input_path, manifest, stats, durations, log)
        save_stats()

    # ---- Stage 6: translate ----
    if wanted("translate"):
        _run_translate_stage(cfg, work, manifest, stats, durations, log)
        save_stats()

    # ---- Stage 7: cleanup ----
    cleanup_hash = stable_hash(cfg.cleanup_params())
    if wanted("cleanup") and not manifest.check_stage("cleanup", cleanup_hash):
        log("[cleanup] 整理時間軸…")
        t0 = time.monotonic()
        tracks = json.loads((work / "tracks.json").read_text(encoding="utf-8"))["events"]
        translations = _load_translations(work)
        cues = compose_cues(tracks, cfg)
        events = finalize_cues(cues, translations, cfg, duration)
        # 併入語音字幕事件(來源標記 source="audio";無語音時為空,不影響 OCR 結果)。
        audio_events = finalize_audio_events(
            asr.read_segments(work), translations, cfg, duration
        )
        if audio_events:
            events = sorted(events + audio_events, key=lambda c: (c["start"], c["end"]))
        _atomic_write_json(work / "events.json", {"events": events})
        durations["cleanup"] = round(time.monotonic() - t0, 2)
        manifest.mark_done("cleanup", cleanup_hash)
        stats["event_count"] = len(events)
        log(f"[cleanup] 完成:{len(events)} 條字幕事件")
    elif wanted("cleanup"):
        log("[cleanup] 已完成,跳過")

    # ---- Stage 8: emit ----
    emit_hash = stable_hash(cfg.emit_params())
    if wanted("emit") and not manifest.check_stage("emit", emit_hash):
        log("[emit] 輸出 SRT/ASS…")
        t0 = time.monotonic()
        events = json.loads((work / "events.json").read_text(encoding="utf-8"))["events"]
        srt_path = output_dir / f"{input_path.stem}.{cfg.target_lang}.srt"
        ass_path = output_dir / f"{input_path.stem}.{cfg.target_lang}.ass"
        subtitle.write_text_no_bom(srt_path, subtitle.to_srt(events, cfg.bilingual))
        subtitle.write_text_no_bom(ass_path, subtitle.to_ass(
            events, int(video_info.get("width") or 0), int(video_info.get("height") or 0),
            cfg.subtitle_position, cfg.bilingual, cfg.max_lines,
            audio_position=cfg.audio_subtitle_position,
            audio_color=cfg.audio_subtitle_color,
        ))
        durations["emit"] = round(time.monotonic() - t0, 2)
        manifest.mark_done("emit", emit_hash)
        stats["outputs"] = {"srt": str(srt_path), "ass": str(ass_path)}
        log(f"[emit] 產出 SRT: {srt_path}")
        log(f"[emit] 產出 ASS: {ass_path}")
    elif wanted("emit"):
        log("[emit] 已完成,跳過")

    stats["total_wall_seconds"] = round(time.monotonic() - run_t0, 2)
    save_stats()
    _print_summary(stats, duration, video_info, log)
    return stats


def _migrate_v1_ocr_hash(cfg, fingerprint, ocr_hash, cache_path, manifest, log) -> None:
    """一次性搬遷:v1 的 OCR params hash 含 ocr_batch_size,現已移除(見 Config.ocr_params)。

    直接讓舊 hash 過期的話,升級後第一次執行就會清空所有 ocr/*.json 並讓整份
    exact-image cache 對不上 key,等於整部影片重跑。這裡改成就地換 hash。
    批次大小若與當初那次執行不同就比對不到,屆時的行為與升級前相同(自然失效)。
    """
    legacy_hash = stable_hash(cfg.legacy_ocr_params_v1(fingerprint))
    if legacy_hash == ocr_hash:
        return
    if manifest.migrate_params_hash("ocr", legacy_hash, ocr_hash):
        log("[ocr] manifest 的 OCR params hash 已更新格式(batch_size 不再參與),保留既有結果")
    if cache_path is None or not cache_path.exists():
        return
    cache = OCRCache(cache_path)
    try:
        moved = cache.migrate_param_hash(legacy_hash, ocr_hash)
    finally:
        cache.close()
    if moved:
        log(f"[ocr] 已搬遷 {moved} 筆 exact-image cache 到新 key(batch_size 不再參與)")


def _run_ocr_stage(cfg, work, work_root, samples, provider, manifest, stats, log) -> OCRProvider | None:
    (work / "ocr").mkdir(parents=True, exist_ok=True)

    if provider is None:
        log(f"[ocr] 初始化 {cfg.ocr_provider} pipeline…")
        provider = build_provider(cfg)

    fingerprint = provider.fingerprint() if hasattr(provider, "fingerprint") else {}
    ocr_params = cfg.ocr_params(fingerprint)
    ocr_hash = stable_hash(ocr_params)
    cache_path = None if cfg.no_ocr_cache else work_root / "videosub_ocr_cache.db"
    _migrate_v1_ocr_hash(cfg, fingerprint, ocr_hash, cache_path, manifest, log)

    if manifest.stage_done("ocr", ocr_hash):
        log("[ocr] 已完成,跳過")
        return provider

    # 參數變更 -> 既有 ocr/*.json 過期,清除後重跑。
    prev_hash = manifest.stage_params_hash("ocr")
    if prev_hash is not None and prev_hash != ocr_hash:
        stale = list((work / "ocr").glob("*.json"))
        if stale:
            log(f"[ocr] OCR 參數已變更,清除 {len(stale)} 個舊 OCR 結果")
            for p in stale:
                p.unlink()
    manifest.invalidate("ocr")
    manifest.mark_running("ocr", ocr_hash)

    cache = OCRCache(cache_path) if cache_path is not None else None

    ocr_stats = stats.setdefault("ocr", {})
    cache_hits = 0
    cache_misses = 0
    failed: list[int] = []
    no_text = 0
    batches = 0
    t0 = time.monotonic()

    def _is_no_text(normalized: dict) -> bool:
        return normalized.get("status") == "ok" and not normalized.get("blocks")

    try:
        # 決定待處理樣本(續跑:已有 ocr json 的跳過)。
        pending = []
        for s in samples:
            json_path, _ = _ocr_paths(work, s["index"])
            if json_path.exists():
                continue
            pending.append(s)
        resumed = len(samples) - len(pending)
        done = resumed  # 已有 OCR 結果的樣本數(續跑起點 + 本次寫出的)
        if resumed:
            log(f"[ocr] 續跑:{resumed}/{len(samples)} 張樣本已完成,{len(pending)} 張待處理")

        # 過 cache 並在本次執行內對「完全相同圖片」去重:
        #   - 已解析過的相同圖片(本次執行或 cross-video cache)直接沿用,算 cache 命中。
        #   - 未命中的每個唯一圖片只送一次 OCR(代表),同組其餘沿用結果。
        resolved: dict[str, tuple[dict, dict | None]] = {}
        groups: dict[str, list[dict]] = {}
        ocr_order: list[tuple[str, dict, str | None]] = []  # (dedup_key, rep_sample, cv_key)

        def _apply(sample: dict, normalized: dict, raw) -> None:
            nonlocal done
            dnorm = dict(normalized)
            dnorm["sample_index"] = sample["index"]
            dnorm["timestamp"] = sample["timestamp"]
            dnorm["blocks"] = [
                {**b, "id": f'{sample["index"]}-{n}'}
                for n, b in enumerate(normalized.get("blocks", []), start=1)
            ]
            _write_ocr(work, sample, dnorm, raw)
            done += 1

        for s in pending:
            # sample 階段已把每張的 sha256 寫進 samples.json;沒有才回頭讀檔重算,
            # 否則每次(續跑)都要把待處理的圖片全部讀過一遍才會開始 OCR。
            sha = s.get("sha256") or sha256_bytes((work / s["path"]).read_bytes())
            cv_key = ocr_cache_key(sha, ocr_params) if cache else None
            # 停用 cache 時不去重,每張各自 OCR。
            dedup_key = cv_key if cache else f"__nodedup__{s['index']}"

            if dedup_key in resolved:
                normalized, raw = resolved[dedup_key]
                _apply(s, normalized, raw)
                cache_hits += 1
                if _is_no_text(normalized):
                    no_text += 1
                continue
            if dedup_key in groups:
                groups[dedup_key].append(s)  # 本次執行內的重複,待代表 OCR 完成後沿用
                continue
            cached = cache.get(cv_key) if cache else None
            if cached is not None:
                normalized, raw = cached
                resolved[dedup_key] = (normalized, raw)
                _apply(s, normalized, raw)
                cache_hits += 1
                if _is_no_text(normalized):
                    no_text += 1
                continue
            groups[dedup_key] = [s]
            ocr_order.append((dedup_key, s, cv_key))

        if pending:
            log(f"[ocr] 本次需辨識 {len(ocr_order)} 張唯一圖片"
                f"(待處理 {len(pending)} 張,去重與 cache 省下 {len(pending) - len(ocr_order)} 張)")

        # 對每個唯一圖片(代表)分批送 OCR,逐批寫入(斷點續跑)。
        bs = max(1, cfg.ocr_batch_size)
        for start in range(0, len(ocr_order), bs):
            chunk = ocr_order[start : start + bs]
            paths = [work / rep["path"] for (_k, rep, _cv) in chunk]
            results = provider.recognize(paths)
            if len(results) != len(chunk):
                raise PipelineError(
                    f"provider 回傳數量不符:輸入 {len(chunk)}、回傳 {len(results)}"
                )
            batches += 1
            for (dkey, rep, cv_key), res in zip(chunk, results):
                res.reindex(rep["index"], rep["timestamp"])
                normalized = res.to_dict()
                _write_ocr(work, rep, normalized, res.raw)
                done += 1
                cache_misses += 1  # 唯一圖片實際送 OCR
                group = groups[dkey]
                if res.status == "failed":
                    failed.extend(g["index"] for g in group)
                else:
                    if cache is not None and cv_key is not None:
                        cache.put(cv_key, normalized, res.raw)
                    if not res.blocks:
                        no_text += 1
                # 同組其餘(本次執行內完全相同的圖片)沿用結果,算命中。
                for dup in group[1:]:
                    _apply(dup, normalized, res.raw)
                    if res.status == "failed":
                        continue
                    cache_hits += 1
                    if not res.blocks:
                        no_text += 1
            # 分母用「全部樣本」,續跑時才看得出整體進度;括號內是本次實際送 OCR 的張數。
            log(f"[ocr] 進度 {done}/{len(samples)} 張樣本"
                f"(本次已辨識 {min(start + bs, len(ocr_order))}/{len(ocr_order)} 張唯一圖片)")
    finally:
        if cache is not None:
            cache.close()

    # no_text / failed 以全部 sample 的 ocr JSON 為準,續跑後才不會漏算之前完成的部分。
    all_failed: list[int] = []
    all_no_text = 0
    for s in samples:
        json_path, _ = _ocr_paths(work, s["index"])
        if not json_path.exists():
            continue
        d = json.loads(json_path.read_text(encoding="utf-8"))
        if d.get("status") == "failed":
            all_failed.append(s["index"])
        elif not d.get("blocks"):
            all_no_text += 1

    ocr_stats.update({
        # cache_hits/misses 為「本次執行」的 cache 效益;no_text/failed 為全部樣本的實際結果。
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "images_sent_to_ocr": cache_misses,
        "no_text_images": all_no_text,
        "batches": batches,
        "failed_samples": all_failed,
        "seconds": round(time.monotonic() - t0, 2),
    })
    failed = all_failed
    stats.setdefault("durations", {})["ocr"] = ocr_stats["seconds"]

    if failed and cfg.strict:
        raise PipelineError(f"strict 模式:{len(failed)} 張 OCR 失敗:{failed}")

    manifest.mark_done("ocr", ocr_hash)
    log(f"[ocr] 完成:cache 命中 {cache_hits}、實送 OCR {cache_misses}、"
        f"無文字 {no_text}、失敗 {len(failed)}")
    return provider


def _write_ocr(work: Path, sample: dict, normalized: dict, raw) -> None:
    json_path, raw_path = _ocr_paths(work, sample["index"])
    _atomic_write_json(json_path, normalized)
    if raw is not None:
        _atomic_write_json(raw_path, raw)


def _load_ocr_results(work: Path, samples: list[dict]) -> list[OCRResult]:
    results = []
    for s in samples:
        json_path, _ = _ocr_paths(work, s["index"])
        if not json_path.exists():
            raise PipelineError(f"缺少 OCR 結果:{json_path}(請先執行 ocr 階段)")
        results.append(OCRResult.from_dict(json.loads(json_path.read_text(encoding="utf-8"))))
    return results


def _run_asr_stage(cfg, work, input_path, manifest, stats, durations, log) -> None:
    asr_hash = stable_hash(cfg.asr_params())
    if manifest.stage_done("asr", asr_hash):
        log("[asr] 已完成,跳過")
        return
    manifest.invalidate("asr")
    manifest.mark_running("asr", asr_hash)
    t0 = time.monotonic()

    if not cfg.audio_transcribe:
        # 預設關閉:輸出空片段,後續 translate/cleanup/emit 視為無語音字幕。
        asr.write_segments(work, asr.empty_result())
        durations["asr"] = round(time.monotonic() - t0, 2)
        manifest.mark_done("asr", asr_hash)
        log("[asr] 語音轉字幕未啟用(加 --audio-transcribe 以啟用)")
        return

    if not manifest.video_info.get("has_audio", True):
        asr.write_segments(work, asr.empty_result())
        durations["asr"] = round(time.monotonic() - t0, 2)
        manifest.mark_done("asr", asr_hash)
        log("[asr] 影片無音軌,略過語音辨識")
        return

    result = asr.transcribe(input_path, work, cfg, log=log)
    asr.write_segments(work, result)
    durations["asr"] = round(time.monotonic() - t0, 2)
    stats["asr"] = {
        "segments": len(result["segments"]),
        "language": result.get("language"),
        "seconds": durations["asr"],
    }
    manifest.mark_done("asr", asr_hash)
    log(f"[asr] 完成:{len(result['segments'])} 段語音字幕,偵測語言={result.get('language')}")


def _run_translate_stage(cfg, work, manifest, stats, durations, log) -> None:
    resolved_model = resolve_model(cfg.llm_cache_url, cfg.llm_model)
    translate_hash = stable_hash(cfg.translate_params(resolved_model))
    if manifest.stage_done("translate", translate_hash):
        log("[translate] 已完成,跳過")
        return
    manifest.invalidate("translate")

    tracks = json.loads((work / "tracks.json").read_text(encoding="utf-8"))["events"]
    cues = compose_cues(tracks, cfg)

    # 去重:相同 canonical cue 只翻譯一次。
    unique: dict[str, list[str]] = {}
    for cue in cues:
        unique.setdefault(canonical_key(cue["source_texts"]), cue["source_texts"])

    # 併入語音字幕文字,與 OCR 共用同一套 LLM cache 去重翻譯(單行 cue)。
    audio_segments = asr.read_segments(work)
    for seg in audio_segments:
        txt = asr.clean_text(seg.get("text", ""))
        if txt:
            unique.setdefault(canonical_key([txt]), [txt])

    log(f"[translate] {len(cues)} 條 OCR cue、{len(audio_segments)} 段語音,"
        f"{len(unique)} 個唯一 cue,model={resolved_model}")
    t0 = time.monotonic()
    translations: dict[str, list[str]] = {}
    failed_cues: list[str] = []

    total = len(unique)
    # 約每 5% 印一次進度(cue 多時才印,避免少量時洗版)。
    step = max(1, total // 20)
    with LLMCacheClient(cfg.llm_cache_url, resolved_model) as client:
        for i, (key, texts) in enumerate(unique.items(), 1):
            translated, failed_idx = client.translate_cue_detailed(
                texts, cfg.source_lang, cfg.target_lang
            )
            translations[key] = translated
            if failed_idx:
                failed_cues.append(key)
            if total >= 25 and (i % step == 0 or i == total):
                elapsed = time.monotonic() - t0
                log(f"[translate] 已處理 {i}/{total} 個唯一 cue"
                    f"(cache 命中 {client.cache_hits}、上游 {client.upstream_calls}、"
                    f"{elapsed:.0f}s)")
        cache_hits = client.cache_hits
        cache_misses = client.cache_misses
        upstream_calls = client.upstream_calls

    durations["translate"] = round(time.monotonic() - t0, 2)
    _atomic_write_json(work / "translations.json", {
        "model": resolved_model,
        "source_lang": cfg.source_lang,
        "target_lang": cfg.target_lang,
        "translations": translations,
        "failed_cues": failed_cues,
    })
    stats["translate"] = {
        "unique_cues": len(unique),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "upstream_calls": upstream_calls,
        "failed_cues": len(failed_cues),
        "seconds": durations["translate"],
    }

    if failed_cues and cfg.strict:
        raise PipelineError(f"strict 模式:{len(failed_cues)} 個 cue 翻譯失敗")

    manifest.mark_done("translate", translate_hash)
    log(f"[translate] 完成:cache 命中 {cache_hits}、上游呼叫 {upstream_calls}"
        + (f"、{len(failed_cues)} 個 cue 失敗保留原文" if failed_cues else ""))


def _load_translations(work: Path) -> dict[str, list[str]]:
    path = work / "translations.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("translations", {})


def _print_summary(stats: dict, duration: float, video_info: dict, log) -> None:
    log("")
    log("===== 摘要 =====")
    log(f"影片時長   : {stats.get('duration_seconds', duration)}s")
    log(f"取樣間隔   : {stats.get('interval_seconds')}s")
    log(f"樣本數     : {stats.get('sample_count', 0)}")
    ocr = stats.get("ocr", {})
    if ocr:
        log(f"OCR cache  : 命中 {ocr.get('cache_hits', 0)} / 實送 {ocr.get('images_sent_to_ocr', 0)}"
            f" / 無文字 {ocr.get('no_text_images', 0)} / 失敗 {len(ocr.get('failed_samples', []))}")
    asr_stats = stats.get("asr", {})
    if asr_stats:
        log(f"語音字幕   : {asr_stats.get('segments', 0)} 段(語言 {asr_stats.get('language')})")
    tr = stats.get("translate", {})
    if tr:
        reqs = tr.get("cache_hits", 0) + tr.get("cache_misses", 0)
        rate = tr.get("cache_hits", 0) / reqs if reqs else 0.0
        log(f"唯一 cue   : {tr.get('unique_cues', 0)}")
        log(f"LLM cache  : 命中 {tr.get('cache_hits', 0)}/{reqs}({rate:.1%})、上游 {tr.get('upstream_calls', 0)}")
    log(f"追蹤事件   : {stats.get('track_count', 0)}")
    log(f"字幕事件   : {stats.get('event_count', 0)}")

    # 與理論全幀數比較縮減比例。
    avg_fps = float(video_info.get("avg_fps") or 0.0)
    if avg_fps > 0 and duration > 0:
        full_frames = duration * avg_fps
        sent = stats.get("ocr", {}).get("images_sent_to_ocr", stats.get("sample_count", 0))
        if full_frames > 0:
            log(f"重型處理張數: {sent}(全幀模式約 {int(full_frames)},"
                f"縮減至 {sent / full_frames:.1%})")
    for stage, secs in stats.get("durations", {}).items():
        log(f"耗時[{stage:<9s}]: {secs}s")
    if "total_wall_seconds" in stats:
        log(f"總耗時     : {stats['total_wall_seconds']}s")
