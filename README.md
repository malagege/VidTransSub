# VideoTransSub

只產生**外掛字幕**的影片畫面文字翻譯工具:每隔固定秒數用 `ffmpeg` 擷取一張畫面,
以 [PaddleOCR-VL](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
完整 pipeline 辨識文字,合併連續出現的文字為字幕事件,再透過外部 OpenAI-compatible
**LLM cache API** 翻譯,最後輸出 `.srt` 與 `.ass`。**不修改、不重新編碼原影片。**

規格見 [VideoTransSub_spec.md](VideoTransSub_spec.md)。這是獨立於 VidTransFlow 的新專案:
VidTransFlow 是把「翻譯後文字嵌回畫面並重編碼影片」,VideoTransSub 只解決「快速產生畫面文字字幕」。

## 為什麼更快

一小時、30 fps 的影片有 108,000 幀。VideoTransSub 只需檢查 `ceil(時長 / 取樣間隔)` 張:

| 模式 | 需處理圖片數 |
|---|---:|
| 全幀 | 108,000 |
| 每 1 秒 | 3,600 |
| 每 2 秒 | 1,800 |

完全相同的圖片還會命中 OCR cache,實際 OCR 呼叫更少。代價是顯示時間短於取樣間隔的文字可能漏抓。

## 安裝需求

- Python 3.11+
- `ffmpeg` / `ffprobe` 在 PATH 上
- PaddleOCR-VL(依 CPU/GPU 硬體安裝):`pip install "vidtranssub[ocr]"`
- 一個 OpenAI-compatible 的 LLM cache API(預設 `http://127.0.0.1:8790`)

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"      # 開發/測試
uv pip install -e ".[ocr]"      # 實際辨識需要的 PaddleOCR-VL
```

## 快速開始

```bash
# 預設每 1 秒取樣
vidtranssub input.mp4 --target-lang zh-TW

# 動作較快、短字幕較多
vidtranssub input.mp4 --interval 0.5 --target-lang zh-TW

# 簡報或長時間靜態畫面
vidtranssub input.mp4 --interval 3 --target-lang zh-TW

# ASS 固定顯示在影片上方
vidtranssub input.mp4 --subtitle-position top

# 只重跑指定階段
vidtranssub input.mp4 --stage ocr
```

產出(在輸入檔同目錄):`input.zh-TW.srt` 與 `input.zh-TW.ass`(UTF-8,不加 BOM)。

任何時刻 Ctrl+C 中斷後,重跑相同指令即從斷點續跑。

## 處理階段

```
probe → sample → ocr(含 exact-image cache)→ track → translate → cleanup → emit
```

1. **probe**:`ffprobe` 讀時長/寬高/旋轉;算出預估樣本數。
2. **sample**:`ffmpeg fps=1/interval` 取樣為 JPEG,樣本時間由序號與 interval 計算。
3. **ocr**:完全相同圖片(bytes + OCR 參數皆同)走 cache,其餘送 PaddleOCR-VL 完整 pipeline。
4. **track**:正規化文字,依 bbox 與相似度把連續出現的文字合併為事件。
5. **translate**:每個唯一 cue 一個穩定 request,呼叫外部 LLM cache API(`temperature=0`)。
6. **cleanup**:排序、去重、時間裁切。
7. **emit**:輸出 SRT/ASS。

## 常用參數

| 參數 | 預設 | 說明 |
|---|---:|---|
| `--interval` | `1.0` | 每隔幾秒取樣,可用小數(有效 0.1~60) |
| `--max-width` | `1920` | OCR 圖片最大寬度,0 為不縮放 |
| `--work-dir` | `./work` | 工作目錄與跨影片 OCR cache |
| `--target-lang` | `zh-TW` | 翻譯目標語言 |
| `--source-lang` | auto | 原文語言 |
| `--ocr-batch-size` | `8` | 每批送入 PaddleOCR-VL 的 sample 數 |
| `--ocr-confidence` | 停用 | provider 有分數時才啟用的最低門檻 |
| `--llm-model` | 上游第一個 | 翻譯模型名稱 |
| `--llm-cache-url` | `http://127.0.0.1:8790` | 外部 LLM cache API base URL |
| `--text-similarity` | `0.85` | 相鄰文字合併門檻 |
| `--gap-tolerance` | `1` | 容許短暫消失的樣本數 |
| `--reading-order` | `auto` | `auto/ltr/rtl/ttb` |
| `--bilingual` | false | 譯文加原文雙語輸出 |
| `--subtitle-position` | `bottom` | ASS 固定位置:`bottom/top` |
| `--no-ocr-cache` | false | 停用完全相同圖片的 OCR cache |
| `--strict` | false | 任一樣本或翻譯失敗即中止 |
| `--stage` | 全流程 | `probe/sample/cache/ocr/track/translate/cleanup/emit` |

## 工作目錄與續跑

```
work/
├── videosub_ocr_cache.db              # 跨影片共用的 exact-image OCR cache
└── <video-stem>-videosub/
    ├── manifest.json                  # 各階段完成狀態 + 輸入/參數 hash
    ├── samples.json
    ├── samples/00000001.jpg …
    ├── ocr/00000001.json + .raw.json  # 正規化 + PaddleOCR-VL 原始 JSON
    ├── tracks.json
    ├── translations.json
    ├── events.json
    └── stats.json                     # 耗時、命中率、事件數
```

參數變更只使必要階段失效:改 interval → sample 之後全部;改 OCR 參數 → OCR 之後;
改翻譯模型/語言/prompt 版本 → translate 之後;改字幕樣式/位置 → 只重跑 emit。

## 測試

```bash
uv run pytest tests -q                              # 全部單元 + 整合測試(需 ffmpeg)
uv run python tests/mock_llm.py --port 8790         # 測試用假 LLM cache API
uv run python testdata/make_test_video.py out.mp4   # 產生 20s 測試影片
```

整合測試以真實 ffmpeg 取樣、假 OCR provider 與假 LLM cache server 跑完整流程,
驗證取樣時間、事件合併、exact-image cache、續跑與 SRT/ASS 輸出。

實際 PaddleOCR-VL 讀字正確率與每張耗時屬於開發驗證,請以整合測試影片實測記錄。
