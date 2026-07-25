"""Stage 4:PaddleOCR-VL 完整 pipeline adapter。

- 使用含 layout analysis 與 VLM recognition 的完整 pipeline,不直接呼叫裸 VLM endpoint。
- pipeline 只在程序啟動時初始化一次,不為每張 sample 重新載入模型。
- 以圖片路徑 list 分批呼叫 predict;一批失敗先重試,再拆成單張找出失敗 sample。
- baseline 關閉影片不需要的 document orientation、unwarping 與 chart recognition,保留 layout detection。

注意:PaddleOCR-VL 的實際建構參數與結果 schema 需對鎖定版本做整合測試
(見官方文件 https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)。
本 adapter 以官方文件的 parsing_res_list 欄位為準,並容許常見別名。
"""

from __future__ import annotations

from pathlib import Path

from .ocr_provider import OCRResult, blocks_from_parsing_list

# baseline pipeline 選項:關閉影片不需要的前處理,保留版面偵測。
BASELINE_OPTIONS = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_chart_recognition": False,
}

# 支援的 VLM server 後端(對應 PaddleOCR-VL 的 vl_rec_backend)。
SERVER_BACKENDS = (
    "vllm-server",
    "sglang-server",
    "fastdeploy-server",
    "mlx-vlm-server",
    "llama-cpp-server",
)


class PaddleOCRInitError(RuntimeError):
    pass


class PaddleOCRProvider:
    """PaddleOCR-VL provider;PaddleOCR 是必要元件,於此 lazy import 並初始化一次。"""

    def __init__(
        self,
        model: str = "PaddleOCR-VL",
        engine: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
        confidence: float | None = None,
        pipeline_options: dict | None = None,
        server_url: str | None = None,
        server_backend: str = "vllm-server",
        server_model: str | None = None,
        server_api_key: str | None = None,
    ):
        self.model = model
        self.engine = engine
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.confidence = confidence
        self.options = {**BASELINE_OPTIONS, **(pipeline_options or {})}
        # VLM server 橋接:給了 server_url 才啟用(layout 仍在本地跑)。
        self.server_url = server_url or None
        self.server_backend = server_backend
        self.server_model = server_model or None
        self.server_api_key = server_api_key or None
        if self.server_url and self.server_backend not in SERVER_BACKENDS:
            raise PaddleOCRInitError(
                f"未知的 OCR server backend:{self.server_backend};"
                f"可用:{', '.join(SERVER_BACKENDS)}"
            )
        self._pipeline = None
        self._paddleocr_version: str | None = None

    # ---- 初始化 ----

    def _pipeline_kwargs(self) -> dict:
        """組出傳給 PaddleOCRVL 的 kwargs(抽出以便測試,不需載入 paddleocr)。

        server 模式只多轉發 vl_rec_* 參數,把 VLM 辨識委派給已啟動的 genai_server;
        layout 偵測與其他選項一律沿用 in-process 行為。
        """
        kwargs = dict(self.options)
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        if self.engine:
            kwargs["backend"] = self.engine
        if self.server_url:
            kwargs["vl_rec_backend"] = self.server_backend
            kwargs["vl_rec_server_url"] = self.server_url
            kwargs["vl_rec_api_model_name"] = self.server_model or self.model
            if self.server_api_key:
                kwargs["vl_rec_api_key"] = self.server_api_key
        return kwargs

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import paddleocr  # type: ignore

            self._paddleocr_version = getattr(paddleocr, "__version__", "unknown")
            from paddleocr import PaddleOCRVL  # type: ignore
        except Exception as e:  # pragma: no cover - 需實機安裝 paddle
            raise PaddleOCRInitError(
                "無法載入 PaddleOCR-VL。請依 CPU/GPU 硬體安裝 paddlepaddle 與 paddleocr"
                f"(pip install 'vidtranssub[ocr]')。原始錯誤:{e}"
            ) from e

        try:
            self._pipeline = PaddleOCRVL(**self._pipeline_kwargs())
        except Exception as e:  # pragma: no cover
            raise PaddleOCRInitError(
                f"PaddleOCR-VL 初始化失敗(engine={self.engine} device={self.device}"
                f" model={self.model} server={self.server_url or '無'}):{e}"
            ) from e
        return self._pipeline

    def release(self, log=print) -> None:
        """釋放 GPU 資源:丟棄已載入的 pipeline 並清空 paddle 顯存池。

        OCR/track 之後的階段(ASR、translate)不再需要 OCR 模型。在同一行程接著
        跑 ASR(faster-whisper)前呼叫本方法,把 PaddleOCR-VL 佔用的顯存還給
        驅動,避免載入大型 ASR 模型(如 large-v3)時 CUDA out of memory。

        可安全重複呼叫;未初始化或非 GPU 時為 no-op。釋放後若再辨識會自動
        重新初始化(見 :meth:`_ensure_pipeline`)。
        """
        import gc
        import sys

        had_pipeline = self._pipeline is not None
        self._pipeline = None
        gc.collect()
        paddle = sys.modules.get("paddle")
        if paddle is not None:
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
        if had_pipeline:
            log("[ocr] 已釋放 PaddleOCR-VL 顯存,供後續階段使用")

    def _static_fingerprint(self) -> dict:
        """不需啟動 pipeline 的指紋內容(server 資訊只放後端與服務端模型名)。

        刻意不含 server_url 與 api key:同一顆模型換一台機器不該讓整份 cache 失效,
        金鑰更不可寫入 manifest/log。
        """
        fp = {
            "model": self.model,
            "engine": self.engine,
            "device": self.device,
            "options": self.options,
        }
        if self.server_url:
            fp["vl_rec_backend"] = self.server_backend
            fp["vl_rec_model"] = self.server_model or self.model
        return fp

    def fingerprint(self) -> dict:
        """寫入 manifest 與 cache key 的 OCR 指紋。啟動 pipeline 以取得實際版本。"""
        self._ensure_pipeline()
        return {"paddleocr_version": self._paddleocr_version, **self._static_fingerprint()}

    # ---- 辨識 ----

    def _image_size(self, path: Path) -> tuple[int, int]:
        from PIL import Image  # PaddleOCR 依賴 PIL,此處可安全匯入

        with Image.open(path) as im:
            return im.width, im.height

    def _predict_one(self, path: Path) -> OCRResult:
        pipeline = self._ensure_pipeline()
        width, height = self._image_size(path)
        outputs = list(pipeline.predict(str(path)))
        raw = self._result_to_json(outputs[0]) if outputs else {}
        parsing = raw.get("parsing_res_list", []) if isinstance(raw, dict) else []
        blocks = blocks_from_parsing_list(parsing, width, height, self.confidence)
        return OCRResult(sample_index=0, timestamp=0.0, blocks=blocks, status="ok", raw=raw)

    @staticmethod
    def _result_to_json(result) -> dict:
        """PaddleOCR 結果物件轉 dict(不同版本 API 略有差異)。

        PaddleOCR-VL 的 .json 會把內容包在 {"res": {...}} 之下,此處一併解包,
        讓 parsing_res_list / width / height 等欄位回到頂層(對齊規格 §4 結果轉換表)。
        """
        raw: dict | None = None
        for attr in ("json", "res"):
            val = getattr(result, attr, None)
            if isinstance(val, dict):
                raw = val
                break
        if raw is None and isinstance(result, dict):
            raw = result
        if raw is None:
            return {}
        # 解包 {"res": {...}} 外層封裝
        inner = raw.get("res")
        if isinstance(inner, dict) and "parsing_res_list" in inner:
            return inner
        return raw

    def recognize(self, images: list[Path]) -> list[OCRResult]:
        """對整批圖片辨識;一批失敗先重試,再拆成單張標記失敗 sample。"""
        pipeline = self._ensure_pipeline()
        images = [Path(p) for p in images]
        results: list[OCRResult] = []

        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            batch_results = self._recognize_batch(pipeline, batch)
            results.extend(batch_results)
        return results

    def _recognize_batch(self, pipeline, batch: list[Path]) -> list[OCRResult]:
        # 先嘗試整批;失敗重試一次;仍失敗才拆單張。
        for _ in range(2):
            try:
                outputs = list(pipeline.predict([str(p) for p in batch]))
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"predict 回傳數量不符:輸入 {len(batch)}、回傳 {len(outputs)}"
                    )
                out: list[OCRResult] = []
                for path, result in zip(batch, outputs):
                    width, height = self._image_size(path)
                    raw = self._result_to_json(result)
                    parsing = raw.get("parsing_res_list", []) if isinstance(raw, dict) else []
                    blocks = blocks_from_parsing_list(parsing, width, height, self.confidence)
                    out.append(OCRResult(0, 0.0, blocks, status="ok", raw=raw))
                return out
            except Exception:
                continue

        # 拆單張:確定失敗的 sample 標記 failed(不可當成無文字)。
        out = []
        for path in batch:
            try:
                out.append(self._predict_one(path))
            except Exception:
                out.append(OCRResult(0, 0.0, [], status="failed", raw=None))
        return out
