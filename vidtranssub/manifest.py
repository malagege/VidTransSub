"""Manifest:記錄各階段完成狀態,支援斷點續跑。

一個階段能被跳過的條件是:記錄的 input hash 與 params hash 都與本次執行相符。
階段(重新)失效時,所有下游階段一併失效(見規格 §8 失效範圍表)。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

STAGES = ["probe", "sample", "ocr", "track", "asr", "translate", "cleanup", "emit"]

# stage -> 若它重跑,必須一起重跑的下游階段。
# asr(語音)是與 OCR 平行的獨立分支:只依賴 probe(音訊來源),不受 sample/ocr/track 影響,
# 兩條分支在 translate/cleanup/emit 合流。故 sample/ocr/track 的下游刻意不含 asr。
DOWNSTREAM = {
    "probe": ["sample", "ocr", "track", "asr", "translate", "cleanup", "emit"],
    "sample": ["ocr", "track", "translate", "cleanup", "emit"],
    "ocr": ["track", "translate", "cleanup", "emit"],
    "track": ["translate", "cleanup", "emit"],
    "asr": ["translate", "cleanup", "emit"],
    "translate": ["cleanup", "emit"],
    "cleanup": ["emit"],
    "emit": [],
}


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"input_hash": None, "video_info": {}, "stages": {}}

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.path)

    def ensure_input(self, input_hash: str) -> None:
        """輸入檔改變時整個重設。"""
        if self.data.get("input_hash") != input_hash:
            self.data = {"input_hash": input_hash, "video_info": {}, "stages": {}}
            self.save()

    @property
    def video_info(self) -> dict:
        return self.data.setdefault("video_info", {})

    def set_video_info(self, info: dict) -> None:
        self.data["video_info"] = info
        self.save()

    def stage_done(self, name: str, params_hash: str) -> bool:
        st = self.data["stages"].get(name)
        return bool(
            st
            and st.get("status") == "completed"
            and st.get("params_hash") == params_hash
        )

    def stage_params_hash(self, name: str) -> str | None:
        """階段(running 或 completed)已記錄的 params hash。"""
        st = self.data["stages"].get(name)
        return st.get("params_hash") if st else None

    def mark_running(self, name: str, params_hash: str) -> None:
        """記錄階段以這組參數開始執行,讓中斷後能分辨可續跑的部分產物與過期產物。"""
        self.data["stages"][name] = {"status": "running", "params_hash": params_hash}
        self.save()

    def mark_done(self, name: str, params_hash: str) -> None:
        self.data["stages"][name] = {
            "status": "completed",
            "params_hash": params_hash,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save()

    def invalidate(self, name: str) -> None:
        """移除 name 與其所有下游階段。"""
        for stage in [name, *DOWNSTREAM.get(name, [])]:
            self.data["stages"].pop(stage, None)
        self.save()

    def check_stage(self, name: str, params_hash: str) -> bool:
        """True 表示可跳過;否則使該階段(與下游)失效以便重新執行。"""
        if self.stage_done(name, params_hash):
            return True
        self.invalidate(name)
        return False
