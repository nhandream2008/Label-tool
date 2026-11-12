"""Hộp thoại dùng chung (nút tiếng Việt). Gom về một chỗ để dễ thay thế khi kiểm thử tự động."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QFileDialog, QInputDialog, QMessageBox, QWidget


def _box(parent: QWidget | None, icon, title: str, text: str) -> QMessageBox:
    box = QMessageBox(icon, title, text, QMessageBox.NoButton, parent)
    return box


def error(parent, title: str, text: str) -> None:
    box = _box(parent, QMessageBox.Critical, title, text)
    box.addButton("Đóng", QMessageBox.AcceptRole)
    box.exec()


def warning(parent, title: str, text: str) -> None:
    box = _box(parent, QMessageBox.Warning, title, text)
    box.addButton("Đã hiểu", QMessageBox.AcceptRole)
    box.exec()


def info(parent, title: str, text: str) -> None:
    box = _box(parent, QMessageBox.Information, title, text)
    box.addButton("OK", QMessageBox.AcceptRole)
    box.exec()


def ask_yes_no(parent, title: str, text: str) -> bool:
    box = _box(parent, QMessageBox.Question, title, text)
    yes = box.addButton("Có", QMessageBox.YesRole)
    no = box.addButton("Không", QMessageBox.NoRole)
    box.setDefaultButton(no)
    box.setEscapeButton(no)
    box.exec()
    return box.clickedButton() is yes


def ask_choice(parent, title: str, text: str, options: list[str]) -> int | None:
    """Hỏi chọn một trong các nút. Trả về chỉ số nút được bấm, hoặc None nếu huỷ."""
    box = _box(parent, QMessageBox.Question, title, text)
    buttons = [box.addButton(o, QMessageBox.AcceptRole) for o in options]
    cancel = box.addButton("Huỷ", QMessageBox.RejectRole)
    box.setDefaultButton(buttons[0] if buttons else cancel)
    box.setEscapeButton(cancel)
    box.exec()
    clicked = box.clickedButton()
    return buttons.index(clicked) if clicked in buttons else None


def ask_text(parent, title: str, label: str, value: str = "") -> str | None:
    dlg = QInputDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setLabelText(label)
    dlg.setTextValue(value)
    dlg.setOkButtonText("Đồng ý")
    dlg.setCancelButtonText("Huỷ")
    dlg.resize(380, dlg.sizeHint().height())
    if dlg.exec():
        text = dlg.textValue().strip()
        return text or None
    return None


def ask_color(parent, title: str, initial: str) -> str | None:
    c = QColorDialog.getColor(QColor(initial), parent, title)
    return c.name() if c.isValid() else None


def ask_dir(parent, title: str, start: str = "") -> str:
    return QFileDialog.getExistingDirectory(parent, title, start)


def ask_open_file(parent, title: str, filters: str) -> str:
    return QFileDialog.getOpenFileName(parent, title, "", filters)[0]


def ask_save_file(parent, title: str, start: str, filters: str) -> str:
    return QFileDialog.getSaveFileName(parent, title, start, filters)[0]


def ask_open_files(parent, title: str, filters: str) -> list[str]:
    return QFileDialog.getOpenFileNames(parent, title, "", filters)[0]
