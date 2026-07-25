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
- PaddleOCR-VL:CPU 版用 `pip install "vidtranssub[ocr]"`;**GPU 版另有裝法**,見下方〈[用 NVIDIA 顯卡加速](#用-nvidia-顯卡加速windows)〉
- 一個 OpenAI-compatible 的 LLM cache API(預設 `http://127.0.0.1:8790`)

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"      # 開發/測試
uv pip install -e ".[ocr]"      # 實際辨識需要的 PaddleOCR-VL
uv pip install -e ".[server]"   # 內建翻譯快取 proxy(接 Ollama/llama.cpp 等,選用)
uv pip install -e ".[asr]"      # 語音轉字幕(faster-whisper,選用;預設關閉)
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

### 把聲音也轉成字幕(語音辨識,預設關閉)

除了畫面上的文字(OCR),還能把**聲音**用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
轉成字幕並一併翻譯。此功能**預設關閉**——不加 `--audio-transcribe` 時,輸出與只做 OCR 完全相同,
不受任何影響。

```bash
# 先安裝語音相依(只需一次)
uv pip install -e ".[asr]"

# 啟用語音轉字幕(自動偵測語言,翻成 zh-TW)
vidtranssub input.mp4 --target-lang zh-TW --audio-transcribe

# VRAM 吃緊可換小模型;也能指定原文語言、字幕顏色/位置
vidtranssub input.mp4 --audio-transcribe --asr-model medium \
  --asr-language ja --audio-color cyan --audio-subtitle-position top
```

- **顏色區分**:在 **ASS** 字幕中,OCR 字幕維持白色、語音字幕用 `--audio-color`(預設黃色),
  且預設放**頂部**(`--audio-subtitle-position`,與 OCR 底部區隔);兩者可同時顯示、不混淆。
  **SRT 不分色**(格式限制),只依時間合併兩種來源。
- 語音辨識與 OCR 是**平行的獨立分支**,只在翻譯/輸出階段合流;調整 OCR 參數不會重跑語音辨識,
  反之亦然。語音文字與 OCR 共用同一套 LLM 翻譯快取去重。
- 影片無音軌時自動略過。faster-whisper 首次會下載模型(`large-v3` 約 3GB;可改 `medium`/`small`)。

### 讓 OCR 與翻譯分開吃 VRAM

PaddleOCR-VL 與 llama.cpp 這類 LLM 上游若共用同一張顯卡,同時常駐會互相搶 VRAM。
由於本工具階段循序、且中間結果落地可續跑,可用 `--to-stage` / `--from-stage` 把流程切成兩段,
**每段只載入一個模型**:

```bash
# 第一段:llama.cpp 關著,只有 PaddleOCR 吃 VRAM
vidtranssub input.mp4 --to-stage ocr

# 開 llama.cpp 後,第二段:ocr 不在區間 -> 完全不載入 PaddleOCR,VRAM 全留給 llama.cpp
vidtranssub input.mp4 --from-stage track --target-lang zh-TW
```

`--stage X` 仍表示只跑單一階段;`--from-stage`/`--to-stage` 表示跑一段連續區間(可省略其一取頭/尾),
兩者互斥。

### 把 OCR 的 VLM 推到另一台 GPU server(genai_server 橋接)

PaddleOCR-VL 分成「本地 layout 偵測 + VLM 辨識」兩段。吃 VRAM 的是 VLM;可用 PaddleOCR 內建的
`genai_server` 把 VLM 獨立成一個 OpenAI-compatible 服務,本工具只在本地跑輕量 layout,再透過 HTTP
呼叫遠端 VLM。適合把 VLM 常駐在一台 Linux GPU server、其餘流程留在你的機器。

```bash
# ① Linux GPU server 上:啟動 VLM 服務(吃顯卡的是這支)
paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --backend vllm --port 8118

# ② 本工具端:layout 在本地,VLM 走遠端 server
vidtranssub input.mp4 --target-lang zh-TW \
  --ocr-server-url http://GPU_HOST:8118/v1 \
  --ocr-server-backend vllm-server \
  --ocr-server-model PaddleOCR-VL-1.6-0.9B
```

- 未指定 `--ocr-server-url` 時維持原本的 in-process 行為,完全不受影響。
- `--ocr-server-backend` 可選 `vllm-server` / `sglang-server` / `fastdeploy-server` /
  `mlx-vlm-server` / `llama-cpp-server`(對應官方文件的 `vl_rec_backend`)。
- server 需要金鑰時,把金鑰放進環境變數(預設讀 `PADDLEOCR_VL_API_KEY`,可用 `--ocr-api-key-env`
  改名);金鑰只在執行時讀取,不寫入 manifest/log,也不進 OCR cache key。
- server 後端與服務端模型名會併入 OCR cache 指紋,因此「本地跑」與「server 跑」的結果會分開快取、
  不互相污染;但更換 server 主機(URL)不會讓既有 cache 失效。
- **官方提醒**:layout 用的 transformers 與 vLLM 在同一環境相依會衝突,建議如上「本地 layout /
  遠端 VLM」分開部署。實際 `vl_rec_*` 參數名請以你 pin 的 paddleocr 版本為準(見 [官方文件](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html))。

> **Windows 使用者注意**:`paddleocr genai_server`(vLLM/SGLang 後端)實務上跑在 Linux + CUDA。
> 本工具的 **client 端(`--ocr-server-url ...`)在 Windows 可正常使用**,只要把 server 那支開在
> Linux GPU 機器上、Windows 這端連過去即可;不需在 Windows 本機啟動 genai_server。

## 新手快速上手(含 GPU 安裝與疑難排解)

### 用 NVIDIA 顯卡加速(Windows)

`nvidia-smi` 右上角的 `CUDA Version` 是**驅動支援的最高版本**,而 NVIDIA 驅動向下相容,
所以裝「不高於」它的 CUDA 版 paddle 即可。Windows 上 PaddlePaddle GPU 版目前最高只提供到
**CUDA 12.9(cu129)**;CUDA 13.x(cu130/cu132)只有 Linux 版。因此 Windows(即使驅動是
CUDA 13.x)一律裝 cu129,驅動會相容。

> ⚠️ GPU 版**不要**用 `pip install -e ".[ocr]"`(那個 extra 會裝 CPU 版 paddle)。
> 請改成下面兩步,且**不要**同時裝 CPU 版,以免衝突:

```powershell
# 1) 裝 GPU 版 paddle(Windows 最高 cu129;CUDA 13.x 驅動可相容)
uv pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# 2) 裝 PaddleOCR 與 PaddleOCR-VL 需要的相依(不經 .[ocr] extra,避免又拉進 CPU 版 paddle)
#    paddlex[ocr] 提供 doc-parser 相依,少了它初始化會報 DependencyError;它不會拉進 paddlepaddle。
uv pip install "paddleocr>=3.0" "paddlex[ocr]>=3.0"
```

驗證有吃到 GPU:

```powershell
uv run python -c "import paddle; print(paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```

印出 `True 1`(或更多張)就對了。執行主程式時預設 `--ocr-device auto` 會自動選 GPU,
也可明確指定 `--ocr-device gpu:0`。

> 高效能服務化後端(vLLM / SGLang / FastDeploy)僅支援 **Linux**;Windows 走的是 PaddlePaddle
> 內建 GPU 推理(一樣有 GPU 加速,只是非最快的服務化堆疊)。要極致吞吐,建議在 Linux 或 WSL2
> 用 cu129/cu132 + FastDeploy。

### 實際跑一支影片(兩個終端機)

翻譯需要一個 OpenAI-compatible 的 LLM cache API 在背景執行。想先看整條流程跑通、還不接真翻譯時,
可用內建的**假翻譯伺服器**(譯文會是「[譯] 原文」):

```powershell
# 終端機 A:啟動假翻譯伺服器,保持開著
uv run python tests/mock_llm.py --port 8790

# 終端機 B:產生測試影片並執行(產出 test.zh-TW.srt / test.zh-TW.ass)
uv run python testdata/make_test_video.py test.mp4
uv run vidtranssub test.mp4 --target-lang zh-TW
```

要接真翻譯時,改用下面的內建 proxy(或你自己的 OpenAI-compatible 端點)。

### 用真翻譯:內建 LLM cache proxy(接 Ollama / llama.cpp)

`mock_llm.py` 只是假翻譯。要真翻譯,本專案內建一個 **OpenAI-compatible 的翻譯快取 proxy**
([vidtranssub/llm_cache_server.py](vidtranssub/llm_cache_server.py)):把每個唯一字幕的翻譯
快取到 SQLite(跨影片重複字幕不再重打 LLM),cache miss 時轉發到你指定的上游;
命中回傳 `x-vtf-cache: hit`、統計走 `/vtf/stats`,完全對齊 client 介面。

先安裝 server 相依:`uv pip install -e ".[server]"`。以 **Ollama** 為例(先 `ollama pull qwen2.5`):

```powershell
# 終端機 A:啟動翻譯 proxy(預設埠 8790,對齊 client 預設)
uv run vidtranssub-llm-cache --upstream-url http://127.0.0.1:11434 --default-model qwen2.5

# 終端機 B:跑影片。proxy 在預設 8790,故不必加 --llm-cache-url;
#           以 --llm-model 指定模型,避免自動抓到 /v1/models 清單的第一個
uv run vidtranssub 你的影片.mp4 --target-lang zh-TW --llm-model qwen2.5
```

- **llama.cpp server**:把 `--upstream-url` 換成 `http://127.0.0.1:8080` 即可。
- **需要金鑰的上游**(商用 OpenAI-compatible 端點):把金鑰放進環境變數(預設讀 `LLM_API_KEY`,
  可用 `--api-key-env` 改名),proxy 會以 `Authorization: Bearer` 轉發;client → proxy 之間不需金鑰。
- proxy 選項:`--host` / `--port` / `--db`(SQLite 快取檔)/ `--api-key-env` / `--default-model`。

### 常見卡關

| 症狀 | 原因 / 解法 |
|---|---|
| `LLM cache API 連線失敗` | 翻譯伺服器沒開;確認終端機 A 的 mock_llm(或你的 API)還在跑,或 `--llm-cache-url` 是否正確。 |
| 裝 GPU 版後仍跑 CPU | 可能同時裝了 CPU 版 paddle;先 `uv pip uninstall paddlepaddle`,再重裝 `paddlepaddle-gpu`。用上面的驗證指令確認為 `True`。 |
| `ffmpeg not found` | `ffmpeg` / `ffprobe` 需在 PATH 上(本機已確認具備)。 |
| 中途中斷了 | 直接重跑**同一行**指令即可從斷點續跑,不會重來。 |
| 想清空重跑 | 刪掉 `work/` 資料夾(暫存與跨影片 OCR cache)即可。 |

## 處理階段

```
probe → sample → ocr(含 exact-image cache)→ track → asr → translate → cleanup → emit
                                                      └ 語音分支,預設關閉
```

1. **probe**:`ffprobe` 讀時長/寬高/旋轉/是否有音軌;算出預估樣本數。
2. **sample**:`ffmpeg fps=1/interval` 取樣為 JPEG,樣本時間由序號與 interval 計算。
3. **ocr**:完全相同圖片(bytes + OCR 參數皆同)走 cache,其餘送 PaddleOCR-VL 完整 pipeline。
4. **track**:正規化文字,依 bbox 與相似度把連續出現的文字合併為事件。
5. **asr**(預設關閉):`--audio-transcribe` 時抽音訊交 faster-whisper 轉逐段字幕;與 OCR 平行,只依賴 probe。
6. **translate**:每個唯一 cue(含語音)一個穩定 request,呼叫外部 LLM cache API(`temperature=0`)。
7. **cleanup**:排序、去重、時間裁切;OCR 與語音事件標記 `source` 後合併。
8. **emit**:輸出 SRT/ASS(ASS 依 `source` 為語音字幕著色)。

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
| `--ocr-server-url` | in-process | 指定 PaddleOCR genai_server base URL(如 `http://GPU_HOST:8118/v1`)後,VLM 辨識走遠端 server |
| `--ocr-server-backend` | `vllm-server` | server 後端:`vllm/sglang/fastdeploy/mlx-vlm/llama-cpp -server` |
| `--ocr-server-model` | 同 `--paddleocr-model` | server 端模型名(`vl_rec_api_model_name`) |
| `--ocr-api-key-env` | `PADDLEOCR_VL_API_KEY` | 存放 OCR server API key 的環境變數名稱 |
| `--llm-model` | 上游第一個 | 翻譯模型名稱 |
| `--llm-cache-url` | `http://127.0.0.1:8790` | 外部 LLM cache API base URL |
| `--text-similarity` | `0.85` | 相鄰文字合併門檻 |
| `--gap-tolerance` | `1` | 容許短暫消失的樣本數 |
| `--reading-order` | `auto` | `auto/ltr/rtl/ttb` |
| `--bilingual` | false | 譯文加原文雙語輸出 |
| `--subtitle-position` | `bottom` | ASS 固定位置:`bottom/top` |
| `--audio-transcribe` | false | 啟用語音轉字幕(需 `[asr]` extra) |
| `--asr-model` | `large-v3` | faster-whisper 模型(VRAM 吃緊可用 `medium`/`small`) |
| `--asr-language` | auto | 語音原文語言 |
| `--audio-color` | `yellow` | ASS 語音字幕顏色(顏色名/`#RRGGBB`/`&H..`;SRT 不分色) |
| `--audio-subtitle-position` | `top` | ASS 語音字幕位置 `bottom/top` |
| `--no-ocr-cache` | false | 停用完全相同圖片的 OCR cache |
| `--strict` | false | 任一樣本或翻譯失敗即中止 |
| `--stage` | 全流程 | 只跑單一階段 `probe/sample/cache/ocr/track/asr/translate/cleanup/emit` |
| `--from-stage` | 頭 | 從指定階段跑到最後(與 `--to-stage` 合用成區間) |
| `--to-stage` | 尾 | 從頭跑到指定階段 |

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
    ├── audio_segments.json            # 語音辨識結果(關閉時為空片段)
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
