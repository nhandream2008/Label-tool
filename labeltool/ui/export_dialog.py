"""Hộp thoại "Hoàn thành / Xuất dataset"."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLineEdit,
                               QPlainTextEdit, QProgressBar, QVBoxLayout)

from ..exporter import ExportOptions, default_export_dir, export_dataset
from ..project import Project
from . import dialogs
from .common import open_folder
from .widgets import GlassDialog, button, label

RESIZE_CHOICES = {
    "Giữ nguyên kích thước": "none",
    "Thu nhỏ vừa khung (giữ tỉ lệ)": "fit",
    "Kéo giãn về W×H": "stretch",
}


def _small(value: str, width: int = 64) -> QLineEdit:
    e = QLineEdit(value)
    e.setFixedWidth(width)
    e.setAlignment(Qt.AlignCenter)
    return e


class ExportDialog(GlassDialog):
    def __init__(self, parent, project: Project, on_use_for_training: Callable[[Path], None]):
        super().__init__(parent)
        self.project = project
        self.on_use_for_training = on_use_for_training
        self.setWindowTitle("Hoàn thành / Xuất dataset")
        self.resize(780, 720)
        self.setMinimumSize(660, 580)
        self.setModal(True)
        self._events: queue.Queue = queue.Queue()
        self._summary = None
        self._busy = False

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        st = project.stats()
        head = label(f"Dự án: {project.name}", "title")
        root.addWidget(head)
        per_class = "   ".join(f"{c.name}: {n}" for c, n in zip(project.classes, st["per_class"]))
        root.addWidget(label(f"{st['labeled']}/{st['images']} ảnh đã gán nhãn   •   {st['boxes']} khung"
                             + (f"\n{per_class}" if per_class else ""), "muted", wrap=True))

        card = QFrame()
        card.setObjectName("panel")
        g = QGridLayout(card)
        g.setContentsMargins(14, 12, 14, 12)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)
        g.setColumnStretch(1, 1)
        r = 0

        g.addWidget(label("Thư mục xuất"), r, 0)
        self.out_entry = QLineEdit(str(default_export_dir(project)))
        g.addWidget(self.out_entry, r, 1)
        g.addWidget(button("Chọn…", self._pick_out), r, 2)
        r += 1

        g.addWidget(label("Tỉ lệ train / valid / test (%)"), r, 0)
        ratio = QHBoxLayout()
        self.e_train, self.e_valid, self.e_test = _small("70"), _small("20"), _small("10")
        for i, e in enumerate((self.e_train, self.e_valid, self.e_test)):
            ratio.addWidget(e)
            if i < 2:
                ratio.addWidget(label("/", "muted"))
        ratio.addStretch()
        g.addLayout(ratio, r, 1, 1, 2)
        r += 1

        g.addWidget(label("Seed (để trống = ngẫu nhiên)"), r, 0)
        self.e_seed = _small("42")
        g.addWidget(self.e_seed, r, 1, Qt.AlignLeft)
        r += 1

        g.addWidget(label("Đổi kích thước"), r, 0)
        rs = QHBoxLayout()
        self.resize_menu = QComboBox()
        self.resize_menu.addItems(list(RESIZE_CHOICES))
        self.resize_menu.setMinimumWidth(240)
        rs.addWidget(self.resize_menu)
        self.e_w = _small("640")
        self.e_h = _small("640")
        rs.addSpacing(6)
        rs.addWidget(self.e_w)
        rs.addWidget(label("×", "muted"))
        rs.addWidget(self.e_h)
        rs.addStretch()
        g.addLayout(rs, r, 1, 1, 2)
        self.resize_menu.currentIndexChanged.connect(self._sync_resize)
        r += 1

        g.addWidget(label("Tăng cường (chỉ tập train)"), r, 0, Qt.AlignTop)
        aug = QHBoxLayout()
        self.v_flip = QCheckBox("Lật ngang")
        aug.addWidget(self.v_flip)
        aug.addSpacing(14)
        self.v_bright = QCheckBox("Độ sáng ±")
        aug.addWidget(self.v_bright)
        self.e_bright = _small("25", 52)
        aug.addWidget(self.e_bright)
        aug.addWidget(label("%", "muted"))
        aug.addSpacing(14)
        aug.addWidget(label("Số bản/ảnh"))
        self.e_copies = _small("1", 52)
        aug.addWidget(self.e_copies)
        aug.addStretch()
        g.addLayout(aug, r, 1, 1, 2)
        r += 1

        g.addWidget(label("Tuỳ chọn khác"), r, 0, Qt.AlignTop)
        other = QVBoxLayout()
        self.v_coco = QCheckBox("Xuất thêm COCO JSON")
        self.v_unlabeled = QCheckBox("Gồm cả ảnh chưa gán nhãn (làm ảnh nền)")
        other.addWidget(self.v_coco)
        other.addWidget(self.v_unlabeled)
        g.addLayout(other, r, 1, 1, 2)
        root.addWidget(card)
        self._sync_resize()

        bar = QHBoxLayout()
        self.export_btn = button("Xuất dataset", self._start, obj="primary", focus=True)
        bar.addWidget(self.export_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        bar.addWidget(self.progress, 1)
        root.addLayout(bar)

        self.result = QPlainTextEdit()
        self.result.setObjectName("mono")
        self.result.setReadOnly(True)
        self.result.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.result.setPlainText("Bảng tổng kết sẽ hiện ở đây sau khi xuất.")
        root.addWidget(self.result, 1)

        after = QHBoxLayout()
        self.open_btn = button("Mở thư mục", self._open)
        self.open_btn.setEnabled(False)
        after.addWidget(self.open_btn)
        self.train_btn = button("Dùng để huấn luyện →", self._use)
        self.train_btn.setEnabled(False)
        after.addWidget(self.train_btn)
        after.addStretch()
        after.addWidget(button("Đóng", self.close))
        root.addLayout(after)

        self._timer = QTimer(self, interval=50, timeout=self._poll)

    def _sync_resize(self) -> None:
        on = RESIZE_CHOICES[self.resize_menu.currentText()] != "none"
        self.e_w.setEnabled(on)
        self.e_h.setEnabled(on)

    def _pick_out(self) -> None:
        d = dialogs.ask_dir(self, "Chọn thư mục trống để xuất dataset")
        if d:
            self.out_entry.setText(d)

    def _read_options(self) -> ExportOptions:
        def num(entry, name, cast=float, minimum=0):
            try:
                v = cast(entry.text().strip())
            except ValueError:
                raise ValueError(f"Giá trị \"{name}\" không hợp lệ")
            if v < minimum:
                raise ValueError(f"\"{name}\" phải ≥ {minimum}")
            return v

        tr = num(self.e_train, "train")
        va = num(self.e_valid, "valid")
        te = num(self.e_test, "test")
        if tr <= 0 or va <= 0:
            raise ValueError("Tỉ lệ train và valid phải lớn hơn 0 (Ultralytics cần tập valid).")
        seed_text = self.e_seed.text().strip()
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError:
            raise ValueError("Seed phải là số nguyên (hoặc để trống)")
        mode = RESIZE_CHOICES[self.resize_menu.currentText()]
        w = num(self.e_w, "chiều rộng", int, 32)
        h = num(self.e_h, "chiều cao", int, 32)
        bright = num(self.e_bright, "độ sáng", float, 0) / 100 if self.v_bright.isChecked() else 0.0
        if bright >= 1:
            raise ValueError("Độ sáng ± phải nhỏ hơn 100%")
        copies = num(self.e_copies, "số bản", int, 1)
        out = self.out_entry.text().strip()
        if not out:
            raise ValueError("Hãy chọn thư mục xuất")
        return ExportOptions(out_dir=Path(out), train=tr, valid=va, test=te, seed=seed,
                             resize_mode=mode, resize_w=w, resize_h=h,
                             aug_flip=self.v_flip.isChecked(), aug_brightness=bright, aug_copies=min(copies, 10),
                             include_unlabeled=self.v_unlabeled.isChecked(), coco=self.v_coco.isChecked())

    def _start(self) -> None:
        try:
            opts = self._read_options()
        except ValueError as e:
            dialogs.error(self, "Tuỳ chọn chưa hợp lệ", str(e))
            return
        self._busy = True
        self.export_btn.setEnabled(False)
        self.progress.setValue(0)
        events = self._events
        project = self.project

        def work():
            try:
                s = export_dataset(project, opts, progress=lambda i, n, t: events.put(("progress", i / max(n, 1))))
                events.put(("done", s))
            except Exception as e:
                events.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()
        self._timer.start()

    def _poll(self) -> None:
        try:
            while True:
                kind, val = self._events.get_nowait()
                if kind == "progress":
                    self.progress.setValue(int(val * 1000))
                elif kind == "error":
                    self._timer.stop()
                    self._busy = False
                    self.export_btn.setEnabled(True)
                    dialogs.error(self, "Xuất dataset thất bại", val)
                    return
                elif kind == "done":
                    self._timer.stop()
                    self._busy = False
                    self._summary = val
                    self.progress.setValue(1000)
                    self.export_btn.setEnabled(True)
                    self.result.setPlainText("Xuất dataset xong!\n\n" + val.to_text())
                    self.open_btn.setEnabled(True)
                    self.train_btn.setEnabled(True)
                    # Gợi ý thư mục mới cho lần xuất sau
                    self.out_entry.setText(str(default_export_dir(self.project)))
                    return
        except queue.Empty:
            pass

    def _open(self) -> None:
        if self._summary:
            open_folder(self._summary.out_dir)

    def _use(self) -> None:
        if self._summary:
            self.on_use_for_training(self._summary.yaml_path)
            self.close()

    def closeEvent(self, e) -> None:
        if self._busy:
            dialogs.info(self, "Đang xuất", "Vui lòng đợi xuất dataset xong.")
            e.ignore()
            return
        super().closeEvent(e)

    def reject(self) -> None:  # phím Esc
        if self._busy:
            return
        super().reject()
