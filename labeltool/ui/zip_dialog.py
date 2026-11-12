"""Hộp thoại "Xuất dự án ra .zip" — đóng gói để chuyển sang máy khác."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QProgressBar, QVBoxLayout

from ..archive import default_zip_name, export_project_zip, human_size, part_sizes
from ..project import Project
from . import dialogs
from .common import open_folder
from .widgets import GlassDialog, button, label


class ZipExportDialog(GlassDialog):
    def __init__(self, parent, project: Project, start_dir: str = ""):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Xuất dự án ra tệp .zip")
        self.resize(620, 420)
        self.setMinimumWidth(560)
        self.setModal(True)
        self._events: queue.Queue = queue.Queue()
        self._busy = False
        self.result_path: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        st = project.stats()
        sizes = part_sizes(project)
        root.addWidget(label(f"Dự án: {project.name}", "title"))
        root.addWidget(label(
            f"{st['images']} ảnh ({st['labeled']} đã gán nhãn) • {st['boxes']} khung • {len(project.classes)} lớp"
            f" • {human_size(sizes['data'])}", "muted", wrap=True))
        root.addWidget(label("Tệp .zip giữ nguyên ảnh, nhãn, tên lớp và màu lớp. Trên máy khác, bấm "
                             "\"Nhập dự án từ .zip\" để mở lại và làm tiếp.", wrap=True))

        card = QFrame()
        card.setObjectName("panel")
        g = QGridLayout(card)
        g.setContentsMargins(14, 12, 14, 12)
        g.setVerticalSpacing(10)
        g.setColumnStretch(1, 1)
        g.addWidget(label("Lưu thành"), 0, 0)
        start = Path(start_dir) if start_dir and Path(start_dir).is_dir() else project.root.parent
        self.out_entry = QLineEdit(str(start / default_zip_name(project)))
        g.addWidget(self.out_entry, 0, 1)
        g.addWidget(button("Chọn…", self._pick), 0, 2)
        self.v_exports = QCheckBox(f"Gồm các dataset đã xuất (exports/, {human_size(sizes['exports'])})")
        self.v_runs = QCheckBox(f"Gồm kết quả huấn luyện và trọng số (runs/, {human_size(sizes['runs'])})")
        self.v_exports.setEnabled(sizes["exports"] > 0)
        self.v_runs.setEnabled(sizes["runs"] > 0)
        g.addWidget(self.v_exports, 1, 0, 1, 3)
        g.addWidget(self.v_runs, 2, 0, 1, 3)
        root.addWidget(card)

        bar = QHBoxLayout()
        self.go_btn = button("Xuất .zip", self._start, obj="primary", focus=True)
        bar.addWidget(self.go_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        bar.addWidget(self.progress, 1)
        root.addLayout(bar)
        self.status = label("", "muted", wrap=True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)
        root.addStretch(1)

        after = QHBoxLayout()
        self.open_btn = button("Mở thư mục chứa", self._open)
        self.open_btn.setEnabled(False)
        after.addWidget(self.open_btn)
        after.addStretch(1)
        after.addWidget(button("Đóng", self.close))
        root.addLayout(after)
        self._timer = QTimer(self, interval=50, timeout=self._poll)

    def _pick(self) -> None:
        f = dialogs.ask_save_file(self, "Lưu tệp .zip của dự án", self.out_entry.text(), "Tệp zip (*.zip)")
        if f:
            self.out_entry.setText(f)

    def _start(self) -> None:
        out = self.out_entry.text().strip()
        if not out:
            dialogs.error(self, "Chưa chọn nơi lưu", "Hãy chọn nơi lưu tệp .zip.")
            return
        path = Path(out)
        if path.suffix.lower() != ".zip":
            path = path.with_name(path.name + ".zip")
        if path.exists() and not dialogs.ask_yes_no(self, "Tệp đã có", f"{path.name} đã tồn tại. Ghi đè?"):
            return
        self._busy = True
        self.go_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Đang đóng gói…")
        events, project = self._events, self.project
        exports, runs = self.v_exports.isChecked(), self.v_runs.isChecked()

        def work():
            try:
                z = export_project_zip(project, path, include_exports=exports, include_runs=runs,
                                       progress=lambda i, n, name: events.put(("p", i / max(n, 1))))
                events.put(("done", z))
            except Exception as e:
                events.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()
        self._timer.start()

    def _poll(self) -> None:
        try:
            while True:
                kind, val = self._events.get_nowait()
                if kind == "p":
                    self.progress.setValue(int(val * 1000))
                    continue
                self._timer.stop()
                self._busy = False
                self.go_btn.setEnabled(True)
                if kind == "error":
                    self.status.setText("")
                    dialogs.error(self, "Xuất .zip thất bại", val)
                else:
                    self.result_path = val
                    self.progress.setValue(1000)
                    self.status.setText(f"Đã lưu: {val}  ({human_size(val.stat().st_size)})")
                    self.open_btn.setEnabled(True)
                return
        except queue.Empty:
            pass

    def _open(self) -> None:
        if self.result_path:
            open_folder(self.result_path.parent)

    def closeEvent(self, e) -> None:
        if self._busy:
            dialogs.info(self, "Đang xuất", "Vui lòng đợi đóng gói xong.")
            e.ignore()
            return
        super().closeEvent(e)

    def reject(self) -> None:
        if not self._busy:
            super().reject()
