"""Nền kính mờ kiểu Windows 11 (Acrylic / Mica) cho cửa sổ Qt.

Dùng thẳng API chính thức của Windows qua ctypes, không cần thư viện ngoài:
- DwmSetWindowAttribute(DWMWA_SYSTEMBACKDROP_TYPE) — Windows 11 bản 22621 trở lên.
- DwmExtendFrameIntoClientArea — cho lớp kính phủ cả vùng nội dung.
Cửa sổ Qt phải bật Qt.WA_TranslucentBackground để các vùng nền trong suốt nhìn xuyên được.
Trên Windows cũ hơn hoặc hệ điều hành khác: tự chuyển sang nền đặc.
"""

from __future__ import annotations

import sys

STYLES = {
    "Kính trong (Acrylic)": "acrylic",
    "Kính mờ (Mica)": "mica",
    "Nền đặc": "off",
}

_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_SYSTEMBACKDROP_TYPE = 38
_BACKDROP = {"off": 1, "mica": 2, "acrylic": 3}


def supported() -> bool:
    return sys.platform == "win32" and sys.getwindowsversion().build >= 22621


def _dwm_set(hwnd: int, attr: int, value: int) -> bool:
    import ctypes
    v = ctypes.c_int(value)
    return ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v)) == 0


def _extend_frame(hwnd: int, on: bool) -> None:
    import ctypes

    class MARGINS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]

    m = MARGINS(-1, -1, -1, -1) if on else MARGINS(0, 0, 0, 0)
    ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(m))


def dark_title_bar(widget) -> None:
    """Thanh tiêu đề tối (Windows 10 20H1+ / Windows 11)."""
    if sys.platform == "win32":
        try:
            _dwm_set(int(widget.winId()), _DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
        except (OSError, AttributeError):
            pass


def apply(widget, style: str) -> bool:
    """Bật kiểu nền cho cửa sổ. Trả về True nếu đang dùng kính (nền phía sau nhìn xuyên được)."""
    dark_title_bar(widget)
    if not supported():
        return False
    try:
        hwnd = int(widget.winId())
        if style not in ("acrylic", "mica"):
            _dwm_set(hwnd, _DWMWA_SYSTEMBACKDROP_TYPE, _BACKDROP["off"])
            _extend_frame(hwnd, False)
            return False
        _extend_frame(hwnd, True)
        return _dwm_set(hwnd, _DWMWA_SYSTEMBACKDROP_TYPE, _BACKDROP[style])
    except (OSError, AttributeError):
        return False
