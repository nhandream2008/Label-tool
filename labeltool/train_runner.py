"""Chạy ``train_worker.py`` trong tiến trình con và đọc log theo thời gian thực.

Lớp ``WorkerProcess`` không đụng tới giao diện: một luồng nền đọc stdout của
tiến trình con và đẩy sự kiện vào hàng đợi (``queue.Queue``). Giao diện lấy sự
kiện ra bằng ``after()`` trên luồng chính, nên không có lời gọi Tk nào từ luồng phụ.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

WORKER = Path(__file__).with_name("train_worker.py")
# Thư mục chứa trọng số gốc tải về (yolo11n.pt, ...). Đã có trong .gitignore.
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
# Mã điều khiển màu/xoá dòng của terminal (ESC [ ... chữ cái)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def missing_packages() -> list[str]:
    """Trả về các thư viện huấn luyện còn thiếu (không import thật, chỉ tìm)."""
    return [m for m in ("torch", "ultralytics") if importlib.util.find_spec(m) is None]


def cuda_available() -> bool | None:
    """True/False nếu kiểm tra được, None nếu chưa cài torch."""
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class WorkerProcess:
    """Một lần chạy train hoặc export. Sự kiện trong ``events``:

    ("log", text) | ("BATCH"/"EPOCH"/"DONE"/"EXPORTED"/"ERROR", dict) | ("exit", returncode)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.events: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.stopped_by_user = False
        self._cfg_file: str | None = None

    def start(self) -> None:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        fd, self._cfg_file = tempfile.mkstemp(prefix="labeltool-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW  # không bật cửa sổ console đen
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(WORKER), self._cfg_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(WEIGHTS_DIR),  # trọng số gốc được tải về đây, không rơi vào repo
            env=env,
            creationflags=creationflags,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        """Đọc stdout theo từng khối. Thanh tiến trình của ultralytics dùng ký tự
        '\\r' để vẽ đè, nên với mỗi dòng ta chỉ giữ trạng thái cuối cùng sau '\\r'."""
        assert self.proc and self.proc.stdout
        buf = ""
        stream = self.proc.stdout
        while True:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._handle_line(self._last_segment(line))
        if buf.strip():
            self._handle_line(self._last_segment(buf))
        code = self.proc.wait()
        if self._cfg_file:
            try:
                os.unlink(self._cfg_file)
            except OSError:
                pass
        self.events.put(("exit", code))

    @staticmethod
    def _last_segment(line: str) -> str:
        # Windows kết thúc dòng bằng CR LF: bỏ CR ở cuối trước, rồi mới lấy phần sau CR cuối cùng
        return line.rstrip("\r").split("\r")[-1].rstrip()

    def _handle_line(self, line: str) -> None:
        line = ANSI_RE.sub("", line)  # bỏ mã màu của terminal
        if not line.strip():
            return
        # Sự kiện "@@..." có thể dính vào cuối một dòng thanh tiến trình chưa xuống dòng
        idx = line.find("@@")
        if idx >= 0:
            tag, _, payload = line[idx + 2:].partition(" ")
            try:
                self.events.put((tag, json.loads(payload)))
                return  # phần trước "@@" chỉ là thanh tiến trình dở dang, bỏ qua
            except json.JSONDecodeError:
                pass
        self.events.put(("log", line))

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Dừng tiến trình con và các tiến trình DataLoader con của nó."""
        if not self.running():
            return
        self.stopped_by_user = True
        assert self.proc
        if os.name == "nt":
            # /T: dừng cả cây tiến trình, /F: bắt buộc
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self.proc.terminate()
