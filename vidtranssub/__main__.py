"""VideoTransSub CLI。

主要流程直接接收影片路徑,不使用子命令:

    vidtranssub input.mp4 --target-lang zh-TW
    vidtranssub input.mp4 --interval 0.5 --target-lang zh-TW
    vidtranssub input.mp4 --subtitle-position top
    vidtranssub input.mp4 --stage ocr
"""

from __future__ import annotations

import argparse
import sys

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidtranssub",
        description="固定秒數取樣畫面文字,PaddleOCR-VL 辨識 + 外部 LLM cache API 翻譯,輸出 SRT/ASS",
    )
    p.add_argument("input", help="輸入影片路徑")
    p.add_argument("--interval", type=float, default=1.0, help="每隔幾秒取樣,可用小數(預設 1.0)")
    p.add_argument("--max-width", type=int, default=1920, help="OCR 圖片最大寬度,0 為不縮放")
    p.add_argument("--image-quality", type=int, default=3, help="取樣 JPEG 品質(ffmpeg -q:v,2 最佳 31 最差)")
    p.add_argument("--work-dir", default="./work", help="工作目錄與 OCR cache")
    p.add_argument("--target-lang", default="zh-TW", help="翻譯目標語言")
    p.add_argument("--source-lang", default=None, help="原文語言(預設 auto)")
    p.add_argument("--ocr-provider", choices=["paddleocr-vl"], default="paddleocr-vl")
    p.add_argument("--paddleocr-model", default="PaddleOCR-VL", help="模型名稱或本機路徑")
    p.add_argument("--paddleocr-engine", default=None, help="PaddleOCR-VL inference engine")
    p.add_argument("--ocr-device", default="auto", help="PaddleOCR-VL 執行裝置")
    p.add_argument("--ocr-batch-size", type=int, default=8, help="每批送入 PaddleOCR-VL 的 sample 數")
    p.add_argument("--ocr-confidence", type=float, default=None, help="provider 有分數時才啟用的最低門檻")
    p.add_argument(
        "--ocr-server-url", default=None,
        help="PaddleOCR genai_server 的 OpenAI-compatible base URL(例如 http://GPU_HOST:8118/v1);"
        "指定後 VLM 辨識走遠端 server,layout 仍在本地(未指定則維持 in-process)",
    )
    p.add_argument(
        "--ocr-server-backend",
        choices=["vllm-server", "sglang-server", "fastdeploy-server", "mlx-vlm-server", "llama-cpp-server"],
        default="vllm-server",
        help="OCR VLM server 後端類型(對應 PaddleOCR-VL 的 vl_rec_backend)",
    )
    p.add_argument(
        "--ocr-server-model", default=None,
        help="server 端模型名稱(vl_rec_api_model_name;預設沿用 --paddleocr-model)",
    )
    p.add_argument(
        "--ocr-api-key-env", default="PADDLEOCR_VL_API_KEY",
        help="存放 OCR server API key 的環境變數名稱(server 免金鑰時可忽略)",
    )
    p.add_argument("--llm-model", default=None, help="翻譯模型名稱(預設取上游第一個)")
    p.add_argument("--llm-cache-url", default="http://127.0.0.1:8790",
                   help="外部 OpenAI-compatible LLM cache API base URL")
    p.add_argument("--text-similarity", type=float, default=0.85, help="相鄰文字合併門檻")
    p.add_argument("--gap-tolerance", type=int, default=1, help="容許短暫消失的樣本數")
    p.add_argument("--reading-order", choices=["auto", "ltr", "rtl", "ttb"], default="auto")
    p.add_argument("--bilingual", action="store_true", help="譯文加原文雙語輸出")
    p.add_argument("--subtitle-position", choices=["bottom", "top"], default="bottom",
                   help="ASS 固定位置(SRT 位置由播放器決定)")
    p.add_argument("--no-ocr-cache", action="store_true", help="停用完全相同圖片的 OCR cache")
    p.add_argument("--strict", action="store_true", help="任一樣本或翻譯失敗即中止")
    stage_choices = ["probe", "sample", "cache", "ocr", "track", "translate", "cleanup", "emit"]
    p.add_argument(
        "--stage", choices=stage_choices, default=None,
        help="只重跑指定的單一階段(cache 等同 ocr);與 --from-stage/--to-stage 互斥",
    )
    p.add_argument(
        "--from-stage", choices=stage_choices, default=None,
        help="從指定階段跑到最後(可與 --to-stage 合用成區間)",
    )
    p.add_argument(
        "--to-stage", choices=stage_choices, default=None,
        help="從頭跑到指定階段(可與 --from-stage 合用成區間)",
    )
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    source_lang = None if (args.source_lang in (None, "", "auto")) else args.source_lang
    return Config(
        interval=args.interval,
        max_width=args.max_width,
        image_quality=args.image_quality,
        work_dir=args.work_dir,
        target_lang=args.target_lang,
        source_lang=source_lang,
        ocr_provider=args.ocr_provider,
        paddleocr_model=args.paddleocr_model,
        paddleocr_engine=args.paddleocr_engine,
        ocr_device=args.ocr_device,
        ocr_batch_size=args.ocr_batch_size,
        ocr_confidence=args.ocr_confidence,
        no_ocr_cache=args.no_ocr_cache,
        ocr_server_url=args.ocr_server_url,
        ocr_server_backend=args.ocr_server_backend,
        ocr_server_model=args.ocr_server_model,
        ocr_api_key_env=args.ocr_api_key_env,
        llm_model=args.llm_model,
        llm_cache_url=args.llm_cache_url,
        text_similarity=args.text_similarity,
        gap_tolerance=args.gap_tolerance,
        reading_order=args.reading_order,
        bilingual=args.bilingual,
        subtitle_position=args.subtitle_position,
        strict=args.strict,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage is not None and (args.from_stage is not None or args.to_stage is not None):
        parser.error("--stage 不可與 --from-stage/--to-stage 併用")

    from .pipeline import PipelineError, run_pipeline
    from .ffmpeg import FFmpegError
    from .llm_cache_client import TranslationError
    from .paddleocr_provider import PaddleOCRInitError

    cfg = config_from_args(args)
    try:
        run_pipeline(
            cfg, args.input,
            only_stage=args.stage,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
        )
    except KeyboardInterrupt:
        print("\n已中斷;重跑相同指令即可從斷點續跑。", file=sys.stderr)
        sys.exit(130)
    except (PipelineError, FFmpegError, TranslationError, PaddleOCRInitError, ValueError) as e:
        print(f"\n錯誤:{e}", file=sys.stderr)
        print("(已保留既有斷點,修正後重跑相同指令即可續跑)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
