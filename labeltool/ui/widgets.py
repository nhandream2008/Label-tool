"""Các thành phần giao diện nhỏ dùng lại nhiều nơi."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from . import glass

# Kiểu nền đang dùng cho toàn ứng dụng ("acrylic" | "mica" | "off"); cửa sổ chính cập nhật giá trị này
current_style = "acrylic"


def button(text: str, slot: Callable | None = None, obj: str | None = None, tip: str | None = None,
           focus: bool = False) -> QPushButton:
    """Nút bấm. Mặc định không nhận tiêu điểm bàn phím để phím Space/Enter luôn dành cho khung vẽ."""
    b = QPushButton(text)
    if obj:
        b.setObjectName(obj)
    if slot:
        b.clicked.connect(lambda _checked=False: slot())
    if tip:
        b.setToolTip(tip)
    if not focus:
        b.setFocusPolicy(Qt.NoFocus)
    b.setCursor(Qt.PointingHandCursor)
    return b


def label(text: str = "", obj: str | None = None, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


def vsep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    return f


def hsep() -> QFrame:
    f = QFrame()
    f.setObjectName("hsep")
    return f


class Segmented(QFrame):
    """Nhóm nút chọn một trong nhiều (kiểu Windows 11)."""

    def __init__(self, options: list[str], on_change: Callable[[int], None] | None = None):
        super().__init__()
        self.setObjectName("segbox")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.buttons: list[QPushButton] = []
        self.on_change = on_change
        for i, text in enumerate(options):
            b = button(text, obj="seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, i=i: self._clicked(i))
            lay.addWidget(b)
            self.buttons.append(b)
        self.set_index(0)

    def _clicked(self, i: int) -> None:
        self.set_index(i)
        if self.on_change:
            self.on_change(i)

    def set_index(self, i: int) -> None:
        for j, b in enumerate(self.buttons):
            b.setChecked(i == j)

    def index(self) -> int:
        return next((i for i, b in enumerate(self.buttons) if b.isChecked()), 0)


class GlassDialog(QDialog):
    """Hộp thoại có nền kính giống cửa sổ chính (nếu đang bật)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._glass = current_style != "off" and glass.supported()
        if self._glass:
            self.setAttribute(Qt.WA_TranslucentBackground)
        self.setProperty("glass", "true" if self._glass else "false")
        self._applied = False

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not self._applied:
            self._applied = True
            if self._glass and not glass.apply(self, current_style):
                # Không bật được kính: quay về nền đặc
                self._glass = False
                self.setProperty("glass", "false")
                self.style().unpolish(self)
                self.style().polish(self)
            glass.dark_title_bar(self)
