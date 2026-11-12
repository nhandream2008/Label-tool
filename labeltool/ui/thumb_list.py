"""Danh sách ảnh của dự án (bên trái): ảnh nhỏ + trạng thái gán nhãn + bộ lọc.

Dùng QListView ảo hoá (chỉ vẽ các dòng đang thấy) nên dự án hàng nghìn ảnh vẫn mượt.
Ảnh nhỏ được tạo ở luồng nền bằng Pillow, chỉ khi dòng đó hiện ra lần đầu.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QAbstractListModel, QModelIndex, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (QComboBox, QFrame, QLabel, QListView, QStyle, QStyledItemDelegate,
                               QVBoxLayout)

from . import theme

ROW_H = 60
THUMB_W, THUMB_H = 64, 48
FILTERS = ["Tất cả ảnh", "Chưa gán nhãn", "Đã gán nhãn"]
STATUS_TEXT = {"none": "Chưa gán", "labeled": "{n} khung", "empty": "Ảnh trống"}
STATUS_COLOR = {"none": QColor(255, 255, 255, 120), "labeled": QColor(theme.SUCCESS), "empty": QColor("#f7c948")}


def pil_to_qimage(im: Image.Image) -> QImage:
    """Chuyển ảnh Pillow sang QImage (bản sao độc lập, an toàn khi dùng ở luồng khác)."""
    if im.mode in ("RGBA", "LA", "P", "PA") and ("A" in im.mode or "transparency" in im.info):
        im = im.convert("RGBA")
        fmt, bpp = QImage.Format_RGBA8888, 4
    else:
        im = im.convert("RGB")
        fmt, bpp = QImage.Format_RGB888, 3
    data = im.tobytes("raw", im.mode)
    return QImage(data, im.width, im.height, im.width * bpp, fmt).copy()


class _ThumbLoader:
    """Một luồng nền tạo ảnh nhỏ; yêu cầu mới nhất được làm trước (LIFO)."""

    def __init__(self):
        self.requests: queue.LifoQueue = queue.LifoQueue()
        self.results: queue.Queue = queue.Queue()
        self.generation = 0
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, path: Path) -> None:
        self.requests.put((self.generation, path))

    def _run(self) -> None:
        while True:
            gen, path = self.requests.get()
            if gen != self.generation:
                continue
            try:
                with Image.open(path) as im:
                    im.draft("RGB", (THUMB_W * 2, THUMB_H * 2))  # JPEG: giải mã nhanh ở độ phân giải thấp
                    im.thumbnail((THUMB_W * 2, THUMB_H * 2))
                    qimg = pil_to_qimage(im)
            except Exception:
                qimg = None
            self.results.put((gen, path, qimg))


class _Model(QAbstractListModel):
    def __init__(self, owner: "ThumbList"):
        super().__init__()
        self.owner = owner

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.owner.visible)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.UserRole:
            return self.owner.visible[index.row()]
        if role == Qt.ToolTipRole:
            return str(self.owner.items[self.owner.visible[index.row()]])
        return None

    def reset(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class _Delegate(QStyledItemDelegate):
    def __init__(self, owner: "ThumbList"):
        super().__init__()
        self.owner = owner

    def sizeHint(self, option, index) -> QSize:
        return QSize(200, ROW_H)

    def paint(self, p: QPainter, option, index) -> None:
        o = self.owner
        i = index.data(Qt.UserRole)
        path = o.items[i]
        r = QRectF(option.rect).adjusted(3, 2, -3, -2)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        current = i == o.current
        if current:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawRoundedRect(r, 6, 6)
            p.setBrush(QColor(theme.ACCENT))
            p.drawRoundedRect(QRectF(r.left(), r.center().y() - 8, 3, 16), 1.5, 1.5)
        elif option.state & QStyle.State_MouseOver:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 14))
            p.drawRoundedRect(r, 6, 6)

        # Ảnh nhỏ
        tr = QRectF(r.left() + 10, r.center().y() - THUMB_H / 2, THUMB_W, THUMB_H)
        clip = QPainterPath()
        clip.addRoundedRect(tr, 4, 4)
        p.setClipPath(clip)
        p.fillRect(tr, QColor(0, 0, 0, 70))
        pix = o.thumb(path)
        if pix is not None and not pix.isNull():
            s = min(tr.width() / pix.width(), tr.height() / pix.height())
            w, h = pix.width() * s, pix.height() * s
            p.drawPixmap(QRectF(tr.center().x() - w / 2, tr.center().y() - h / 2, w, h), pix, QRectF(pix.rect()))
        p.setClipping(False)

        # Tên và trạng thái
        x = tr.right() + 10
        tw = r.right() - x - 6
        f = QFont(option.font)
        f.setWeight(QFont.DemiBold if current else QFont.Normal)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        fm = QFontMetrics(f)
        name = fm.elidedText(path.name, Qt.ElideMiddle, int(tw))
        p.drawText(QRectF(x, r.top() + 8, tw, 20), Qt.AlignLeft | Qt.AlignVCenter, name)
        state, n = o.status.get(path, ("none", 0))
        f2 = QFont(option.font)
        f2.setPointSizeF(8.5)
        p.setFont(f2)
        p.setPen(QColor(255, 255, 255, 110))
        num = f"#{i + 1}"
        p.drawText(QRectF(x, r.top() + 29, tw, 18), Qt.AlignRight | Qt.AlignVCenter, num)
        p.setPen(STATUS_COLOR[state])
        p.drawText(QRectF(x, r.top() + 29, tw, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   "● " + STATUS_TEXT[state].format(n=n))
        p.restore()


class ThumbList(QFrame):
    image_clicked = Signal(int)   # chỉ số ảnh trong toàn bộ danh sách

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.items: list[Path] = []          # toàn bộ ảnh của dự án
        self.visible: list[int] = []         # chỉ số ảnh đang hiển thị (sau khi lọc)
        self.status: dict[Path, tuple[str, int]] = {}
        self.current: int | None = None
        self._cache: dict[Path, QPixmap | None] = {}
        self._pending: set[Path] = set()
        self._loader = _ThumbLoader()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 10, 6, 8)
        lay.setSpacing(6)
        self.count_label = QLabel("Chưa mở dự án")
        self.count_label.setObjectName("h2")
        lay.addWidget(self.count_label)
        self.filter_box = QComboBox()
        self.filter_box.addItems(FILTERS)
        self.filter_box.setFocusPolicy(Qt.NoFocus)
        self.filter_box.currentIndexChanged.connect(lambda _i: self.refresh_filter())
        lay.addWidget(self.filter_box)

        self.model = _Model(self)
        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setItemDelegate(_Delegate(self))
        self.view.setUniformItemSizes(True)
        self.view.setSelectionMode(QListView.NoSelection)
        self.view.setFocusPolicy(Qt.NoFocus)
        self.view.setMouseTracking(True)
        self.view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.viewport().setAutoFillBackground(False)
        self.view.clicked.connect(lambda idx: self.image_clicked.emit(idx.data(Qt.UserRole)))
        lay.addWidget(self.view, 1)

        self._timer = QTimer(self, interval=60, timeout=self._poll_thumbs)
        self._timer.start()

    # ---------- Ảnh nhỏ ----------
    def thumb(self, path: Path) -> QPixmap | None:
        if path in self._cache:
            return self._cache[path]
        if path not in self._pending:
            self._pending.add(path)
            self._loader.request(path)
        return None

    def _poll_thumbs(self) -> None:
        got = False
        try:
            while True:
                gen, path, qimg = self._loader.results.get_nowait()
                if gen != self._loader.generation:
                    continue
                self._pending.discard(path)
                self._cache[path] = QPixmap.fromImage(qimg) if qimg is not None else None
                got = True
        except queue.Empty:
            pass
        if got:
            self.view.viewport().update()

    # ---------- Dữ liệu ----------
    def set_items(self, items: list[Path], status: dict[Path, tuple[str, int]]) -> None:
        self.items = items
        self.status = status
        self._loader.generation += 1
        self._cache.clear()
        self._pending.clear()
        self.refresh_filter()

    def set_status(self, path: Path, state: str, n: int) -> None:
        self.status[path] = (state, n)
        self._update_count()
        self.view.viewport().update()

    def refresh_filter(self) -> None:
        f = self.filter_box.currentText()
        if f == "Chưa gán nhãn":
            self.visible = [i for i, p in enumerate(self.items) if self.status.get(p, ("none", 0))[0] == "none"]
        elif f == "Đã gán nhãn":
            self.visible = [i for i, p in enumerate(self.items) if self.status.get(p, ("none", 0))[0] != "none"]
        else:
            self.visible = list(range(len(self.items)))
        self.model.reset()
        self._update_count()
        self._scroll_to_current()

    def _update_count(self) -> None:
        if not self.items:
            self.count_label.setText("Chưa có ảnh — bấm \"Nhập ảnh\"")
            return
        done = sum(1 for p in self.items if self.status.get(p, ("none", 0))[0] != "none")
        self.count_label.setText(f"Đã gán nhãn: {done}/{len(self.items)} ảnh")

    def set_current(self, index: int | None) -> None:
        self.current = index
        self._scroll_to_current()
        self.view.viewport().update()

    def _scroll_to_current(self) -> None:
        if self.current is not None and self.current in self.visible:
            row = self.visible.index(self.current)
            self.view.scrollTo(self.model.index(row), QListView.EnsureVisible)

    def neighbor(self, index: int | None, step: int) -> int | None:
        """Ảnh kế tiếp / trước đó trong danh sách đang lọc."""
        if not self.visible:
            return None
        if index is None or index not in self.visible:
            return self.visible[0]
        pos = self.visible.index(index) + step
        if 0 <= pos < len(self.visible):
            return self.visible[pos]
        return None
