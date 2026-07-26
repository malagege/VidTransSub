"""Stage 3:完全相同圖片的 OCR cache(跨影片共用)。

只處理百分之百相同的圖片,不做任何智慧判斷:
- cache key = 圖片 bytes 的 SHA-256 + 完整 PaddleOCR-VL 參數 hash。
- key 命中時沿用先前的原始與正規化 OCR JSON。
- key 未命中時一定送進 PaddleOCR-VL,不因畫面相似或可能沒文字而跳過。

不使用「相似圖片就跳過」、場景偵測或「先猜有沒有文字」。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .config import stable_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_cache (
    key TEXT PRIMARY KEY,
    normalized_json TEXT NOT NULL,
    raw_json TEXT,
    created_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
)
"""


def ocr_cache_key(image_sha256: str, ocr_params: dict) -> str:
    """圖片 SHA-256 與 OCR 參數 hash 一起構成 key。"""
    return f"{image_sha256}:{stable_hash(ocr_params)}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class OCRCache:
    def __init__(self, db_path: Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute(SCHEMA)
        self.db.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        self.db.close()

    def get(self, key: str) -> tuple[dict, dict | None] | None:
        """回傳 (normalized_dict, raw_dict) 或 None。"""
        with self._lock:
            row = self.db.execute(
                "SELECT normalized_json, raw_json FROM ocr_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.db.execute(
                "UPDATE ocr_cache SET hit_count = hit_count + 1 WHERE key = ?", (key,)
            )
            self.db.commit()
            self.hits += 1
            normalized = json.loads(row[0])
            raw = json.loads(row[1]) if row[1] else None
            return normalized, raw

    def put(self, key: str, normalized: dict, raw: dict | None) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO ocr_cache"
                " (key, normalized_json, raw_json, created_at, hit_count)"
                " VALUES (?, ?, ?, ?,"
                "  COALESCE((SELECT hit_count FROM ocr_cache WHERE key = ?), 0))",
                (
                    key,
                    json.dumps(normalized, ensure_ascii=False),
                    json.dumps(raw, ensure_ascii=False) if raw is not None else None,
                    time.time(),
                    key,
                ),
            )
            self.db.commit()

    def migrate_param_hash(self, old_hash: str, new_hash: str) -> int:
        """把 key 尾端的參數 hash 由 old_hash 改寫成 new_hash,回傳搬遷筆數。

        參數 hash 的「組成」改版(而非參數真的變了)時使用,讓既有 cache 不必重跑。
        key 格式為 `<image_sha256>:<params_hash>`,只換冒號之後的部分;若新 key 已存在
        則保留既有那筆並丟棄舊的。
        """
        if old_hash == new_hash:
            return 0
        with self._lock:
            cur = self.db.execute(
                "UPDATE OR IGNORE ocr_cache"
                " SET key = substr(key, 1, instr(key, ':')) || ?"
                " WHERE substr(key, instr(key, ':') + 1) = ?",
                (new_hash, old_hash),
            )
            moved = cur.rowcount
            self.db.execute(
                "DELETE FROM ocr_cache WHERE substr(key, instr(key, ':') + 1) = ?",
                (old_hash,),
            )
            self.db.commit()
        return moved

    def stats(self) -> dict:
        with self._lock:
            (entries,) = self.db.execute("SELECT COUNT(*) FROM ocr_cache").fetchone()
        return {"hits": self.hits, "misses": self.misses, "entries": entries}
