"""Biểu tượng lấy từ phông chữ biểu tượng có sẵn của Windows (Segoe Fluent Icons / Segoe MDL2 Assets).

Không cần tệp ảnh nào. Nếu máy không có các phông này (macOS/Linux), nút chỉ hiện chữ.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap

ADD = "\ue710"
FOLDER = "\ue8b7"
FOLDER_OPEN = "\ue838"
PHOTO = "\ue91b"
OPEN_FILE = "\ue8e5"
CHECK = "\ue73e"
PLAY = "\ue768"
STOP = "\ue71a"
BACK = "\ue72b"
FORWARD = "\ue72a"
ZOOM_IN = "\ue8a3"
ZOOM_OUT = "\ue71f"
DELETE = "\ue74d"
COLOR = "\ue790"
RENAME = "\ue8ac"
CLEAR = "\ue894"
EDIT = "\ue70f"
EXPORT = "\uede1"
FIT = "\ue9a6"
CONTACT = "\ue77b"  # h\u00ecnh ng\u01b0\u1eddi (n\u00fat About me)


@lru_cache(maxsize=1)
def _family() -> str | None:
    families = set(QFontDatabase.families())
    for name in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if name in families:
            return name
    return None


@lru_cache(maxsize=128)
def icon(glyph: str, color: str = "#f3f3f3", size: int = 16) -> QIcon:
    fam = _family()
    if fam is None:
        return QIcon()
    scale = 2  # vẽ to gấp đôi cho nét trên màn hình độ phân giải cao
    pm = QPixmap(size * scale, size * scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.TextAntialiasing)
    f = QFont(fam)
    f.setPixelSize(int(size * scale * 0.85))
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(QRectF(0, 0, size * scale, size * scale), Qt.AlignCenter, glyph)
    p.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)
