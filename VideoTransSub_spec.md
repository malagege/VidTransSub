# VideoTransSub 規格書

Last updated: 2026-07-22

## 1. 背景與定位

現有 VidTransFlow 會把影片的每一個原始影格送進 Koharu，執行偵測、OCR、翻譯、修補與嵌字，最後再重新編碼影片。30 fps 的一小時影片有 108,000 幀，即使畫面長時間沒有變化，仍會產生大量重複工作。

VideoTransSub 改成只產生外掛字幕：

1. `ffmpeg` 每隔固定秒數擷取一張圖片，預設每 1 秒一張，使用者可調整。
2. 完全相同的圖片直接沿用 OCR cache，其餘取樣圖片全部辨識。
3. 以 PaddleOCR-VL 完整 pipeline 解析其餘圖片，取得文字內容、區域與閱讀順序。
4. 比對相鄰取樣結果，把重複文字合併為有開始、結束時間的字幕事件。
5. 對唯一字幕事件直接呼叫外部 LLM cache API 翻譯。
6. 輸出 `.srt` 與 `.ass`，不修改也不重新編碼原影片。

VideoTransSub 是獨立新專案，有自己的套件與 CLI。它解決快速產生畫面文字字幕的需求，不取代 VidTransFlow「翻譯文字直接嵌回畫面」的既有流程。

## 2. 目標

- 接受常見本機影片格式，輸出 UTF-8 編碼的 SRT/ASS 字幕。
- 取樣間隔可由使用者設定，支援小數秒，預設 `1.0` 秒。
- v1 使用 PaddleOCR-VL 辨識文字，並以 adapter 隔離模型實作。
- 翻譯透過 HTTP 呼叫既有 OpenAI-compatible LLM cache API。
- bytes 完全相同的取樣圖片不重複 OCR，相同文字事件不重複翻譯。
- 合併連續出現的文字，避免每一張取樣圖片各產生一條字幕。
- 全流程可斷點續跑；改變取樣、OCR 或翻譯參數時，只使必要階段失效。
- 記錄耗時、呼叫次數、快取命中率與字幕事件數，能客觀比較 VidTransFlow。

## 3. 非目標（v1）

- 不翻譯語音；本規格處理的是影片畫面中可見的文字。
- 不修補原文、不把譯文嵌回影片、不輸出翻譯後影片。
- 不依賴 Koharu；Koharu 的偵測、修補、renderer 與圖片 export 均不在此流程內。
- 不承諾抓到顯示時間短於取樣間隔的文字。
- 不做場景偵測、動態取樣或字幕時間邊界精修；時間精度由 interval 決定。
- 不因圖片「看起來很像」或程式猜測「可能沒有文字」而跳過 sample，避免誤漏小字幕。
- 不做即時串流與 GUI，v1 提供 CLI。
- SRT 不保留畫面座標；ASS 只提供固定上方或固定底部。
- 不在第一版解決複雜追蹤，例如移動文字、旋轉文字或逐字動畫。

## 4. 核心取捨

### 4.1 速度與漏字率

若影片長度為 `D` 秒、取樣間隔為 `I` 秒，最多只需檢查：

```text
sample_count = ceil(D / I)
```

例如一小時、30 fps 的影片：

| 模式 | 需處理圖片數 |
|---|---:|
| VidTransFlow 全幀 | 108,000 |
| VideoTransSub，每 0.5 秒 | 7,200 |
| VideoTransSub，每 1 秒 | 3,600 |
| VideoTransSub，每 2 秒 | 1,800 |

以上是 OCR 圖片數上限；完全相同圖片的 cache 命中後，實際 OCR 呼叫可能更少。代價是顯示時間小於 `I` 的文字可能完全落在兩個取樣點之間。一般字幕建議 `0.5` 到 `1.0` 秒，長時間靜態簡報可用 `2.0` 到 `5.0` 秒。

### 4.2 已決策的辨識與翻譯邊界

VideoTransSub v1 使用 PaddleOCR-VL 完整 pipeline 取得結構化原文，再由外部 LLM cache API 翻譯，不使用 Koharu。字幕流程必要欄位為 `source_text`、`bbox` 與 `reading_order`；`confidence` 為可選欄位，不得假設 PaddleOCR-VL 每個 parsing block 都會提供信心值。

PaddleOCR-VL 的正式結果可提供 `parsing_res_list`，其中包含 `block_bbox`、`block_label`、`block_content` 與 `block_order`。VideoTransSub 使用完整 pipeline 的 layout analysis 與 VLM recognition，不把圖片直接送到裸 VLM endpoint。介面依據見 [PaddleOCR-VL 官方文件](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)。

責任劃分：

- PaddleOCR-VL：圖片布局分析、文字區域定位、原文辨識與閱讀順序。
- VideoTransSub：欄位正規化、跨樣本追蹤、事件去重、時間軸與字幕輸出。
- LLM cache API：翻譯請求快取、上游 LLM 呼叫與錯誤轉送。

PaddleOCR-VL 不負責翻譯；LLM 不接收圖片。這可讓 OCR 結果與翻譯結果分別快取及重跑。

## 5. 整體架構

```text
input.mp4
   |
   +-- ffprobe --> duration / resolution / time base
   |
   +-- ffmpeg fps=1/interval --> samples/00000001.jpg ...
                                  |
                         exact-image OCR cache
                                  |
                    PaddleOCR-VL full pipeline
                  (bbox / content / reading order)
                                  |
                    normalize / region and text match
                                  |
                         unique cue events
                                  |
                      LLM cache API translation
                                  |
                       timeline cleanup and emit
                                  |
                         +--------+--------+
                         |                 |
                    output.srt        output.ass
```

PaddleOCR-VL adapter、事件追蹤與翻譯 client 分開。未來更換 OCR 模型時，不需要重寫取樣、LLM cache 或字幕輸出邏輯。

### 5.1 獨立專案邊界

```text
VideoTransSub/
├── pyproject.toml
├── vidtranssub/
│   ├── __main__.py
│   ├── config.py
│   ├── paddleocr_provider.py
│   ├── llm_cache_client.py
│   ├── tracking.py
│   ├── subtitle.py
│   └── manifest.py
└── tests/
```

- Python distribution 與 import package 使用 `vidtranssub`。
- console script 為 `vidtranssub = "vidtranssub.__main__:main"`。
- 不 import `vidtransflow`，也不共用 VidTransFlow 的工作目錄或 manifest。
- LLM cache 是外部 HTTP API；VideoTransSub 只呼叫 API，不管理該服務的程序或資料庫。

## 6. 資料模型

### 6.1 取樣圖片

```json
{
  "index": 17,
  "timestamp": 16.0,
  "path": "samples/00000017.jpg",
  "sha256": "...",
  "status": "pending"
}
```

`timestamp` 是影片時間，不使用檔案建立時間。第 `n` 張樣本的預期時間為 `(n - 1) * interval`，最後一張不得超過影片時長。

### 6.2 OCR 區塊

座標使用 0 到 1 的正規化值，避免依賴影片解析度。

```json
{
  "sample_index": 17,
  "timestamp": 16.0,
  "blocks": [
    {
      "id": "17-1",
      "bbox": [0.12, 0.72, 0.88, 0.91],
      "source_text": "今日はいい天気です",
      "label": "text",
      "reading_order": 3,
      "confidence": null,
      "language": null
    }
  ]
}
```

### 6.3 字幕事件

```json
{
  "start": 16.0,
  "end": 19.0,
  "source_text": "今日はいい天気です",
  "translated_text": "今天天氣很好",
  "bbox": [0.12, 0.72, 0.88, 0.91],
  "sample_indices": [17, 18, 19],
  "observation_count": 3,
  "confidence": null
}
```

中間資料一律以 JSON 落地。SRT/ASS 是可重新產生的最後輸出，不作為續跑依據。

## 7. 處理階段

### Stage 1: probe

- 用 `ffprobe` 讀取影片時長、寬高、旋轉資訊與起始時間。
- 計算預期取樣數，在執行前顯示給使用者。
- 對輸入檔計算 SHA-256，寫入 manifest。
- 影片無音軌仍可正常處理；此流程完全不需要音軌。

### Stage 2: sample

- 預設 `interval=1.0`，有效範圍 `0.1` 到 `60.0` 秒。
- 建議命令概念如下，實作時使用參數陣列呼叫，不拼 shell 字串：

```bash
ffmpeg -i input.mp4 -vf "fps=1/1.0,scale='min(1920,iw)':-2" \
  -q:v 3 samples/%08d.jpg
```

- `--max-width` 預設 `1920`；OCR 通常不需要保留 4K，縮圖可顯著降低傳輸與推論成本。設為 `0` 表示保留原尺寸。
- 套用影片旋轉資訊後再取樣，確保 OCR 方向正確。
- 樣本時間用序號與 interval 建立並寫進 `samples.json`。不得用輸出 JPEG 的檔案時間推算。
- 若使用者改變 interval、縮放或圖片品質，sample 與全部下游階段失效。

### Stage 3: exact-image OCR cache

這一階段只處理百分之百相同的圖片，不做智慧判斷：

1. 計算 sample 原始 bytes 的 SHA-256。
2. cache key 同時包含圖片 SHA-256 與完整 PaddleOCR-VL 參數 hash。
3. key 命中時沿用先前的原始與正規化 OCR JSON。
4. key 未命中時一定送進 PaddleOCR-VL，不因畫面看起來相似或可能沒有文字而跳過。

不使用「相似圖片就跳過」、場景偵測或「先猜圖片有沒有文字」等功能。這會多做一些 OCR，但不會因小範圍字幕變化而誤漏。`--no-ocr-cache` 可停用完全相同圖片 cache，作為問題排查手段。

### Stage 4: PaddleOCR-VL

v1 的 OCR provider 固定為 PaddleOCR-VL，仍透過 adapter 隔離：

```python
class OCRProvider(Protocol):
    def recognize(self, images: list[Path]) -> list[OCRResult]: ...
```

執行規則：

- 使用包含 layout analysis 與 VLM recognition 的完整 PaddleOCR-VL pipeline，不直接呼叫裸 VLM endpoint。
- 程序啟動時只初始化一次 pipeline；不得為每張 sample 重新載入模型。
- 以圖片路徑 list 分批呼叫 `predict`，預設 `--ocr-batch-size 8`；批次大小依 GPU/CPU 記憶體調整。
- baseline 關閉影片不需要的 document orientation、unwarping 與 chart recognition，保留 layout detection；實際參數名稱與行為須對鎖定的 PaddleOCR 版本做整合測試。
- manifest 記錄 PaddleOCR 套件版本、模型、engine、device、batch size 與所有 pipeline options，任一項改變即使得 OCR 與下游失效。
- PaddleOCR 是 VideoTransSub 的必要元件；PaddlePaddle engine 依 CPU/GPU 硬體選擇安裝版本，啟動時檢查。離線執行可使用本機模型路徑。

結果轉換：

| PaddleOCR-VL | VideoTransSub |
|---|---|
| `block_content` | `source_text` |
| `block_bbox` | `bbox`，除以 sample 寬高後正規化到 0–1 |
| `block_label` | `label` |
| `block_order` | `reading_order` |

- adapter 保存 PaddleOCR-VL 原始 JSON，另輸出 VideoTransSub 正規化 JSON。
- `block_content` 不是空白就先保留為候選文字，`block_label` 只保存供除錯；v1 不增加 label allowlist。
- `confidence` 為 optional。provider 有提供時才套用 `--ocr-confidence`；沒有時寫 `null`，不得自行捏造分數或直接丟棄 block。
- 空白、純標點、單一裝飾符號不建立字幕。
- 一批失敗時先重試；仍失敗則拆成單張找出失敗 sample。失敗 sample 要記錄為 `failed`，不可當成「無文字」。
- 本機 GPU 預設只維持一個推論工作者，靠 batch 提升吞吐量，避免多份模型占滿記憶體。

官方文件建議多張圖片傳入 list，效率優於逐張呼叫；實作與結果 schema 以 [PaddleOCR-VL 官方文件](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) 為準。

### Stage 5: normalize and track

先正規化 OCR 原文，保留原始值供輸出與除錯：

- Unicode NFKC。
- 去除頭尾空白，連續空白折疊為一個。
- 統一換行，但不要任意移除中日文標點。
- 產生只供比對使用的 `match_text`；顯示文字不直接使用破壞性的正規化結果。

相鄰樣本的 block 在符合以下條件時視為同一事件：

- bbox IoU 大於預設 `0.3`，或中心點距離小於畫面對角線的 `8%`。
- 文字正規化相似度大於預設 `0.85`。
- 中間消失不超過 `--gap-tolerance` 個樣本，預設 `1`。

合併規則：

- 第一次看到文字時建立 active event，`start = sample.timestamp`。
- 相同文字持續出現時只延伸事件，不重複翻譯。
- 同區域文字明確改變時，舊事件在當前 sample timestamp 結束，新事件同時開始。
- 文字消失超過容忍值時，`end = last_seen + interval`，但不得超過影片時長。
- 最短事件預設 `0.4` 秒。confidence 存在時可納入過濾；不存在時使用 `observation_count`、文字內容與位置一致性判斷，不得把 `null` 當成低信心。
- 相同時間內多個區塊優先依 PaddleOCR-VL 的 `block_order` 組成一條多行字幕；缺少順序時才依 `auto/ltr/rtl/ttb` 幾何規則推算。

固定取樣只能提供 interval 等級的時間精度。v1 不宣稱字幕邊界精確到幀。

### Stage 6: translate

翻譯 client 直接呼叫外部 OpenAI-compatible LLM cache API：

```python
class TranslationClient(Protocol):
    def translate_cue(self, texts: list[str], source_lang: str | None,
                      target_lang: str) -> list[str]: ...
```

- 先完成 track，再以按閱讀順序排列的 `texts` 建立 canonical cue；相同 canonical cue 只翻譯一次。
- 一個 request 對應一個唯一 cue。多個相關 block 可放在同一 JSON 陣列，但不同時間、互不相關的 cue 不混成 30 句批次。
- 呼叫 `POST <llm_cache_url>/v1/chat/completions`，固定 `temperature=0`、`stream=false`，要求回傳等長 JSON 字串陣列。
- system message 包含固定 prompt 及明確版本，例如 `VideoTransSub translation prompt v1`；source/target language 也必須在穩定內容中。
- messages 不得包含 timestamp、bbox、sample index、檔名、request id 或其他每次執行會改變的內容。
- LLM cache API 以正規化後的完整 request body 計算 SHA-256；因此 model、prompt version、語言或 canonical text 改變會自然產生不同 cache key。
- 回傳數量不符或內容無法解析時重試一次，再降級成每個 block 一個穩定 request。
- 翻譯失敗預設保留原文並在 stats 記錄；`--strict` 則中止。
- API key 只從環境變數或 LLM cache/upstream 自身設定取得，不寫入 manifest/log。
- 以 response header `x-vtf-cache` 與 `/vtf/stats` 前後差值記錄命中率。

穩定請求範例：

```json
{
  "model": "translation-model",
  "messages": [
    {
      "role": "system",
      "content": "VideoTransSub translation prompt v1. Translate ja into zh-TW. Return only a JSON array with exactly the same number of strings."
    },
    {
      "role": "user",
      "content": "[\"今日はいい天気です\"]"
    }
  ],
  "temperature": 0,
  "stream": false
}
```

MVP 不使用「30 句加前後文」請求，因為任何批次成員、順序或 context 改變都會使完整 request cache miss。每個唯一 cue 使用一個穩定 request，確保跨影片也能命中相同翻譯。

可選 `--bilingual`：每個 cue 先顯示譯文，下一行顯示原文。預設只輸出譯文。

### Stage 7: timeline cleanup

- 按 start 排序，移除完全重複事件。
- 同一 track 不允許事件時間倒置或 `end <= start`。
- 同一時間多區塊合併後限制最大行數，預設 2；超出時依 bbox 分成同時存在的 ASS events，SRT 則依閱讀順序串接。
- 單一 cue 過長時優先依 OCR block 邊界換行，不任意切斷中日文詞句。
- 所有時間裁切到 `[0, duration]`。

### Stage 8: emit

- 輸出 `<video-stem>.<target-lang>.srt` 與 `.ass`。
- SRT 使用 `HH:MM:SS,mmm`，ASS 使用 `H:MM:SS.cc`。
- 檔案使用 UTF-8，不加 BOM。
- `--subtitle-position bottom|top` 控制字幕位置，預設 `bottom`。
- ASS 提供標準樣式、外框與安全邊距；解析度使用原影片寬高。`bottom` 使用底部置中，`top` 使用頂部置中。
- 標準 SRT 沒有跨播放器可靠的上方定位能力，因此維持純文字並由播放器決定位置；需要保證上方顯示時使用 ASS。
- v1 不依 OCR bbox 任意定位字幕，只支援頂部或底部兩種固定位置。

## 8. 工作目錄與續跑

```text
work/<video-stem>-videosub/
├── manifest.json
├── samples.json
├── samples/
│   └── 00000001.jpg
├── ocr/
│   ├── 00000001.json
│   └── 00000001.raw.json
├── tracks.json
├── translations.json
├── events.json
└── stats.json
```

VideoTransSub 自己的跨影片共用 cache：

```text
work/
└── videosub_ocr_cache.db
```

LLM cache DB 由外部 LLM cache API 服務管理，不屬於 VideoTransSub 工作目錄。

失效範圍：

| 變更 | 需重跑 |
|---|---|
| interval、max-width、圖片品質 | sample 之後全部 |
| PaddleOCR 版本/model/engine/device/options | OCR、track、translate、cleanup、emit |
| track 相似度或 gap | track、translate、cleanup、emit |
| LLM model、語言、prompt version | translate、cleanup、emit |
| SRT/ASS 樣式、subtitle position | emit |

每個 sample 完成 OCR 後立即原子寫入 JSON，Ctrl+C 後可從未完成 sample 繼續。

## 9. CLI 設計

安裝 VideoTransSub 後提供獨立的 `vidtranssub` 指令。主要流程直接接收影片路徑，不使用 VidTransFlow 子命令：

```bash
# 預設每 1 秒取樣
vidtranssub input.mp4 --target-lang zh-TW

# 動作較快、短字幕較多的影片
vidtranssub input.mp4 --interval 0.5 --target-lang zh-TW

# 簡報或長時間靜態畫面
vidtranssub input.mp4 --interval 3 --target-lang zh-TW

# ASS 固定顯示在影片上方
vidtranssub input.mp4 --subtitle-position top --target-lang zh-TW

# 只重跑指定階段
vidtranssub input.mp4 --stage ocr
```

主要參數：

| 參數 | 預設 | 說明 |
|---|---:|---|
| `--interval` | `1.0` | 每隔幾秒取樣，可用小數 |
| `--max-width` | `1920` | OCR 圖片最大寬度，0 為不縮放 |
| `--work-dir` | `./work` | VideoTransSub 自己的工作目錄與 OCR cache |
| `--target-lang` | `zh-TW` | 翻譯目標語言 |
| `--source-lang` | auto | 原文語言 |
| `--ocr-provider` | `paddleocr-vl` | v1 固定 provider，保留 adapter 介面 |
| `--paddleocr-model` | `PaddleOCR-VL` | 模型名稱或本機路徑；resolved version 寫入 manifest |
| `--paddleocr-engine` | 套件預設 | PaddleOCR-VL inference engine |
| `--ocr-device` | `auto` | PaddleOCR-VL 執行裝置 |
| `--ocr-batch-size` | `8` | 每批送入 PaddleOCR-VL 的 sample 數 |
| `--ocr-confidence` | disabled | provider 有分數時才啟用的最低門檻 |
| `--llm-model` | 上游第一個模型 | 翻譯模型名稱 |
| `--llm-cache-url` | `http://127.0.0.1:8790` | 外部 OpenAI-compatible LLM cache API base URL |
| `--text-similarity` | `0.85` | 相鄰文字合併門檻 |
| `--gap-tolerance` | `1` | 容許短暫消失的樣本數 |
| `--reading-order` | `auto` | `auto/ltr/rtl/ttb` |
| `--bilingual` | false | 譯文加原文雙語輸出 |
| `--subtitle-position` | `bottom` | ASS 固定位置：`bottom/top` |
| `--no-ocr-cache` | false | 停用完全相同圖片的 OCR cache |
| `--strict` | false | 任一樣本或翻譯失敗即中止 |
| `--stage` | 全流程 | `probe/sample/cache/ocr/track/translate/cleanup/emit` |

CLI 啟動時應印出影片時長、取樣間隔、預估樣本數、provider 與工作目錄，讓使用者在昂貴呼叫前發現設定錯誤。

## 10. 錯誤處理

- 找不到 ffmpeg/ffprobe：啟動時立即失敗並顯示缺少的執行檔。
- interval 非有限數、`<= 0` 或超過 60：參數驗證失敗，不執行 ffmpeg。
- PaddleOCR-VL 初始化或模型載入失敗：在處理 sample 前退出，顯示 engine/device/model，保留既有斷點。
- PaddleOCR-VL 批次失敗：有限重試後拆成單張；只將確定失敗的 sample 標記為 `failed`。
- LLM cache API 無法連線：有限重試後保留 tracks 與斷點並退出，不需要重跑 OCR。
- 單張 OCR 失敗：記錄 `failed`，預設繼續；不可把失敗當成空畫面參與 track 合併。
- 翻譯失敗：預設保留原文並在輸出摘要警告；strict 模式退出。
- 無偵測文字：正常產生空的 SRT/ASS，exit code 為 0，摘要顯示 0 events。
- JSON/字幕輸出採暫存檔加 rename，避免中斷留下半個檔案。
- LLM cache API/upstream 的 429 遵守 `Retry-After`；401/403 不重試。

## 11. 效能統計

`stats.json` 至少記錄：

- `duration_seconds`、`interval_seconds`、`sample_count`。
- 完全相同圖片 cache 命中數、實際送 OCR 圖片數、OCR 無文字圖片數。
- OCR cache hit/miss、PaddleOCR-VL batch 數、實際圖片數與耗時。
- 唯一 cue 數、LLM cache API hit/miss、上游實際呼叫數與耗時。
- track 數、最後字幕事件數、OCR/翻譯失敗清單。
- 各 stage wall time 與總 wall time。

完成時 CLI 顯示摘要，並與理論全幀數 `duration * avg_fps` 比較縮減比例。效能驗收不能只看「感覺很快」。

## 12. 測試計畫

### 單元測試

- interval 為 `0.5/1/3` 時，樣本 timestamp 與數量正確，最後時間不超過 duration。
- PaddleOCR-VL adapter 能把 `block_content/block_bbox/block_label/block_order` 正確轉成 VideoTransSub schema。
- PaddleOCR-VL 未回 confidence 時保留 block 並寫 `null`，不套用 confidence 門檻。
- 只有圖片 bytes 與 PaddleOCR-VL 參數都完全相同時才命中 OCR cache；只有畫面相似不得跳過。
- Unicode、空白與換行正規化不破壞顯示原文。
- 相同文字、輕微 OCR 差異、同文不同位置、同位置換字的 track 行為。
- 文字短暫消失一個 sample 時，gap tolerance 能正確合併。
- OCR 失敗不被視為文字消失。
- 相同 canonical cue 產生完全相同的 LLM request；timestamp、bbox、sample index 不得進入 messages。
- LLM cache 在模型、語言或 prompt version 改變後失效。
- SRT/ASS 時間格式、跳脫、多行、空輸出正確。
- ASS 的 `bottom/top` 分別產生底部置中與頂部置中的樣式。
- manifest 參數變更只使正確的下游階段失效。

### 整合測試

建立一支約 20 秒的測試影片，包含：

1. 固定 4 秒的日文字幕。
2. 只出現 0.4 秒的短文字，用來驗證 interval 的已知限制。
3. 同一背景上文字改變。
4. 畫面切換但文字不變。
5. 同時位於上、下方的兩個文字區塊。
6. 一段完全沒有文字的畫面。

PaddleOCR-VL 實測屬於開發驗證，不是使用者需要決定的功能選項。先人工列出測試影片中應該讀到的文字，再記錄「應有幾段、實際成功讀到幾段」以及「平均每張圖片花幾秒」。這能直接確認 PaddleOCR-VL 是否適合目前影片與硬體。

分別以 interval `0.5`、`1.0`、`2.0` 執行，比較成功讀到的文字段數、時間誤差、平均每張 OCR 秒數、LLM 上游呼叫數與總耗時。另需測試模型只初始化一次、list batch、中途終止、續跑、cache hit、429 retry 與空字幕輸出。

## 13. 驗收標準

- 一支 10 分鐘 CFR/VFR 測試影片都能輸出可由常見播放器載入的 SRT 與 ASS。
- interval 參數確實控制取樣數，誤差不超過 1 張。
- 連續三個以上 sample 出現的相同文字只形成一個字幕事件，且只翻譯一次。
- 重跑相同命令時不重新執行已完成的 PaddleOCR-VL sample，且相同翻譯 request 由 LLM cache API 回覆。
- 字幕事件全部滿足 `0 <= start < end <= duration`，並依時間排序。
- 單張 PaddleOCR-VL 失敗不會錯誤地切斷相鄰的同一字幕事件。
- 相對全幀模式，送入重型圖片處理的張數符合 `ceil(duration / interval)` 上限；實際 OCR 呼叫數不得高於樣本數。
- ASS 使用 `bottom` 時顯示於底部、使用 `top` 時顯示於上方；SRT 位置由播放器決定。
- CLI 與 stats 可清楚顯示漏字風險、失敗數和各階段耗時。

## 14. 建議開發順序

### Milestone 1: 可用 MVP

- probe、固定 interval 取樣與 samples manifest。
- PaddleOCR-VL 完整 pipeline adapter、批次推論與原始 JSON 保存。
- exact-image cache。
- 基本文字正規化、相鄰 track 合併。
- 以每個唯一 cue 的穩定 request 直接串接外部 LLM cache API。
- SRT 及 ASS `bottom/top` 字幕輸出、續跑與基本統計。

### Milestone 2: 速度與品質

- PaddleOCR-VL batch size/device/engine 效能調校。
- bbox + 文字 fuzzy matching、gap tolerance 調校。
- 雙語字幕與閱讀順序改善。
- 以整合測試影片建立 interval 建議值與品質基準。

## 15. 已確認決策

1. OCR 使用 PaddleOCR-VL 完整 pipeline；翻譯直接走外部 LLM cache API，不使用 Koharu。
2. 只使用使用者設定的固定 interval，不做場景偵測或動態取樣。
3. 每一張取樣圖片都進行 OCR；只有檔案內容與 OCR 參數完全相同時才讀 cache，不用「圖片看起來相似」或「可能沒文字」作為跳過條件。
4. 不做開始與結束時間精修。字幕時間直接由 sample timestamp、interval 與相鄰文字合併結果決定。
5. ASS 只支援固定底部或固定上方，預設底部；不依畫面文字 bbox 任意定位。SRT 位置交由播放器處理。
6. PaddleOCR-VL 對實際影片讀字是否正確、每張耗時多久，由整合測試記錄；這是開發驗證，不是額外的使用者設定或待決策功能。
