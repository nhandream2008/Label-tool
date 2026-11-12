"""Bảng màu và stylesheet kiểu Windows 11 (Fluent, chế độ tối) cho toàn ứng dụng."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import QApplication

ACCENT = "#60cdff"          # màu nhấn kiểu Windows 11 tối
ACCENT_HOVER = "#7fd6ff"
ACCENT_TEXT = "#002a3d"
SUCCESS = "#6ccb5f"
DANGER = "#ff99a4"
WARNING = "#fce100"
TEXT = "#f3f3f3"
TEXT_MUTED = "rgba(255,255,255,0.62)"
SOLID_BG = "#202020"        # nền khi tắt hiệu ứng kính
POPUP_BG = "#2c2c2c"        # menu xổ xuống, hộp thoại (luôn đặc)

FONT_FAMILY = "Segoe UI Variable Text"

# Độ đục của các lớp nền phủ lên kính (0 = trong nhất). Người dùng chỉnh bằng thanh trượt "Độ đục".
DEFAULT_OPACITY = 0.20
current_opacity = DEFAULT_OPACITY
_chevron = ""

QSS = f"""
* {{
    color: {TEXT};
    font-size: 10pt;
    outline: none;
}}
QWidget#root {{ background: rgba(18,18,18,@TINT@); }}
QWidget#root[glass="false"] {{ background: {SOLID_BG}; }}
QDialog {{ background: {POPUP_BG}; }}
QDialog[glass="true"] {{ background: rgba(18,18,18,@DIALOG@); }}
QToolTip {{ background: {POPUP_BG}; color: {TEXT}; border: 1px solid rgba(255,255,255,0.12); padding: 4px 8px; }}

/* ---------- Khung / bảng ---------- */
QFrame#panel {{
    background: rgba(28,28,28,@PANEL@);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
}}
QFrame#card {{
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
}}
QFrame#sep {{ background: rgba(255,255,255,0.10); max-width: 1px; min-width: 1px; margin: 6px 4px; }}
QFrame#hsep {{ background: rgba(255,255,255,0.08); max-height: 1px; min-height: 1px; }}

QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 11pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 10.5pt; font-weight: 600; }}
QLabel#muted {{ color: {TEXT_MUTED}; }}
QLabel#small {{ color: {TEXT_MUTED}; font-size: 9pt; }}
QLabel#chip {{
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 3px 12px;
}}
QLabel#warn {{ color: {WARNING}; }}
QLabel#big {{ font-size: 12pt; font-weight: 600; }}

/* ---------- Nút ---------- */
QPushButton, QToolButton {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-top-color: rgba(255,255,255,0.11);
    border-radius: 5px;
    padding: 5px 12px;
    min-height: 20px;
}}
QPushButton:hover, QToolButton:hover {{ background: rgba(255,255,255,0.10); }}
QPushButton:pressed, QToolButton:pressed {{ background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.75); }}
QPushButton:disabled, QToolButton:disabled {{ color: rgba(255,255,255,0.36); background: rgba(255,255,255,0.03); }}
QPushButton:checked {{ background: rgba(255,255,255,0.13); }}

/* Nút trên thanh lệnh: phẳng như File Explorer */
QToolButton#cmd, QPushButton#cmd {{
    background: transparent; border: 1px solid transparent; padding: 5px 10px;
}}
QToolButton#cmd:hover, QPushButton#cmd:hover {{ background: rgba(255,255,255,0.08); }}
QToolButton#cmd:pressed, QPushButton#cmd:pressed {{ background: rgba(255,255,255,0.05); }}
QToolButton#cmd:disabled, QPushButton#cmd:disabled {{ background: transparent; }}

QPushButton#primary {{
    background: {ACCENT}; color: {ACCENT_TEXT}; font-weight: 600;
    border: 1px solid rgba(255,255,255,0.18); border-bottom-color: rgba(0,0,0,0.25);
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: #4ab8e8; color: rgba(0,42,61,0.8); }}
QPushButton#primary:disabled {{ background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.36); }}
QPushButton#danger {{ color: {DANGER}; }}
QPushButton#danger:hover {{ background: rgba(255,153,164,0.14); }}

/* Nút phân đoạn (chế độ vẽ, chuyển thẻ) */
QPushButton#seg {{
    background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 5px 14px;
    color: rgba(255,255,255,0.72);
}}
QPushButton#seg:hover {{ background: rgba(255,255,255,0.07); color: {TEXT}; }}
QPushButton#seg:checked {{
    background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.10); color: {TEXT};
}}
QFrame#segbox {{
    background: rgba(0,0,0,0.20); border: 1px solid rgba(255,255,255,0.07); border-radius: 7px;
}}

/* ---------- Ô nhập ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom: 1px solid rgba(255,255,255,0.45);
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT};
}}
QLineEdit:hover, QSpinBox:hover, QComboBox:hover {{ background: rgba(255,255,255,0.09); }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    background: rgba(0,0,0,0.25); border-bottom: 2px solid {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: rgba(255,255,255,0.36); }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ border: none; width: 26px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url("@CHEVRON@"); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background: {POPUP_BG}; border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; padding: 4px;
    selection-background-color: rgba(255,255,255,0.10); selection-color: {TEXT};
}}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid rgba(255,255,255,0.55); background: rgba(0,0,0,0.15);
}}
QCheckBox::indicator:hover {{ background: rgba(255,255,255,0.08); }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

/* ---------- Danh sách, chữ, thanh cuộn ---------- */
QListView, QListWidget, QPlainTextEdit, QTextEdit {{
    background: transparent; border: none;
}}
QPlainTextEdit#log, QPlainTextEdit#mono {{
    background: rgba(0,0,0,0.30); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px;
    font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; padding: 6px;
    selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT};
}}
QListWidget::item {{ border-radius: 5px; padding: 5px 6px; margin: 1px 2px; }}
QListWidget::item:hover {{ background: rgba(255,255,255,0.06); }}
QListWidget::item:selected {{ background: rgba(255,255,255,0.11); color: {TEXT}; }}

QScrollArea, QScrollArea > QWidget > QWidget#scrollbody {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: rgba(255,255,255,0.22); border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: rgba(255,255,255,0.38); }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{
    background: rgba(255,255,255,0.10); border: none; border-radius: 2px;
    max-height: 4px; min-height: 4px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QSlider {{ background: transparent; min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: rgba(255,255,255,0.28); }}
QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {ACCENT}; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0; border-radius: 8px;
    background: {ACCENT}; border: 3px solid #454545;
}}
QSlider::handle:horizontal:hover {{ border-width: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: rgba(255,255,255,0.25); }}
QSlider::handle:horizontal:disabled {{ background: rgba(255,255,255,0.4); }}

QMenu {{ background: {POPUP_BG}; border: 1px solid rgba(255,255,255,0.10); padding: 4px; border-radius: 6px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: rgba(255,255,255,0.09); }}
QMessageBox QLabel {{ min-width: 280px; }}
"""


def setup_app(app: QApplication) -> None:
    """Kiểu Fusion + bảng màu tối (cho hộp thoại hệ thống) + stylesheet Fluent."""
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(POPUP_BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor("#1c1c1c"))
    pal.setColor(QPalette.AlternateBase, QColor("#262626"))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor("#333333"))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(ACCENT_TEXT))
    pal.setColor(QPalette.ToolTipBase, QColor(POPUP_BG))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(255, 255, 255, 110))
    pal.setColor(QPalette.Link, QColor(ACCENT))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(255, 255, 255, 92))
    app.setPalette(pal)
    font = QFont(FONT_FAMILY)
    if not font.exactMatch():
        font = QFont("Segoe UI")
    font.setPointSizeF(10)
    app.setFont(font)
    global _chevron
    _chevron = _chevron_file()
    set_opacity(app, current_opacity)


def build_qss(opacity: float) -> str:
    """Stylesheet với độ đục nền cho trước (0–0.9)."""
    op = max(0.0, min(0.9, opacity))
    return (QSS.replace("@CHEVRON@", _chevron)
            .replace("@TINT@", f"{op * 0.45:.3f}")
            .replace("@PANEL@", f"{op:.3f}")
            .replace("@DIALOG@", f"{0.25 + op * 0.6:.3f}"))


def set_opacity(app: QApplication, opacity: float) -> None:
    """Đổi độ đục của nền (chỉ có tác dụng khi đang bật kính)."""
    global current_opacity
    current_opacity = max(0.0, min(0.9, opacity))
    app.setStyleSheet(build_qss(current_opacity))


def _chevron_file() -> str:
    """Vẽ mũi tên ▾ cho ô chọn (QSS không vẽ được tam giác bằng viền như CSS)."""
    path = Path(tempfile.gettempdir()) / "labeltool-ui" / "chevron.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        img = QImage(40, 40, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 255, 255, 200), 4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline([QPointF(8, 14), QPointF(20, 26), QPointF(32, 14)])
        p.end()
        img.save(str(path))
    return path.as_posix()
