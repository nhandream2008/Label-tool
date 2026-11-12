"""Tiện ích dùng chung cho giao diện."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SETTINGS_FILE = Path.home() / ".labeltool" / "settings.json"


def open_folder(path: Path) -> None:
    """Mở thư mục bằng trình quản lý tệp của hệ điều hành."""
    path = Path(path)
    if os.name == "nt":
        os.startfile(str(path))  # noqa: S606 - chỉ mở thư mục cục bộ
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # không lưu được cài đặt thì bỏ qua, không ảnh hưởng chức năng chính
