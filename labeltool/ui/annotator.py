"""Khung vẽ gán nhãn: vẽ, chọn, di chuyển, đổi kích thước, xoá khung; phóng to/thu nhỏ, kéo ảnh.

Toạ độ khung luôn lưu theo pixel của ảnh gốc. Phép biến đổi hiển thị:
    toạ_độ_màn_hình = toạ_độ_ảnh * scale + offset

Thao tác chuột:
- Chế độ "Vẽ": kéo chuột trái trên ảnh để tạo khung; nhấp (không kéo) để chọn khung.
- Chế độ "Chọn": nhấp để chọn khung, kéo để di chuyển.
- Ở cả hai chế độ: kéo các ô vuông nhỏ ở góc/cạnh của khung đang chọn để đổi kích thước.
- Kéo bằng chuột phải / chuột giữa, hoặc giữ phím Space + kéo chuột trái: di chuyển ảnh.
- Lăn chuột: phóng to / thu nhỏ quanh vị trí con trỏ.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..project import Box
from . import theme

HANDLE = 5          # nửa cạnh ô điều khiển (pixel màn hình)
MIN_BOX_SCREEN = 4  # khung nhỏ hơn mức này (pixel màn hình) coi như cú nhấp
MAX_UNDO = 100
PLACEHOLDER = "Mở hoặc tạo dự án, nhập ảnh, rồi chọn một ảnh ở danh sách bên trái"

_HANDLE_CURSORS = {
    "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
    "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
    "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
    "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
}


def _text_color_on(bg: QColor) -> QColor:
    """Chữ đen trên nền sáng, chữ trắng trên nền tối."""
    lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
    return QColor("#000000") if lum > 140 else QColor("#ffffff")


class Annotator(QWidget):
    changed = Signal()            # danh sách khung thay đổi (để tự lưu)
    selection_changed = Signal()  # khung đang chọn thay đổi

    def __init__(self, get_color: Callable[[int], str], get_name: Callable[[int], str], parent=None):
        super().__init__(parent)
        self.get_color = get_color
        self.get_name = get_name
        self.before_draw: Callable[[], bool] = lambda: True  # trả False để chặn vẽ khung mới

        self.pixmap: QPixmap | None = None
        self.img_w = 0
        self.img_h = 0
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self.mode = "draw"              # "draw" | "select"
        self.current_class = 0
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self.show_labels = True
        self._undo: list[list[Box]] = []
        self._drag: dict | None = None
        self._space = False
        self._mouse: QPointF | None = None
        self._scaled: tuple[float, QPixmap] | None = None   # ảnh đã thu nhỏ sẵn cho mức zoom hiện tại

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)
        self._update_cursor()

    # ================= Dữ liệu =================
    @property
    def has_image(self) -> bool:
        return self.pixmap is not None

    def load(self, image: QImage | None, boxes: list[Box]) -> None:
        self.pixmap = QPixmap.fromImage(image) if image is not None else None
        self.img_w = image.width() if image is not None else 0
        self.img_h = image.height() if image is not None else 0
        self._scaled = None
        self.boxes = [b.copy() for b in boxes]
        self.selected = None
        self._undo.clear()
        self._drag = None
        self.fit()
        self.selection_changed.emit()

    def _push_undo(self) -> None:
        self._undo.append([b.copy() for b in self.boxes])
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)

    def undo(self) -> None:
        if not self._undo:
            return
        self.boxes = self._undo.pop()
        self.selected = None
        self.update()
        self.changed.emit()
        self.selection_changed.emit()

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self._push_undo()
        del self.boxes[self.selected]
        self.selected = None
        self.update()
        self.changed.emit()
        self.selection_changed.emit()

    def set_selected_class(self, cls: int) -> None:
        if self.selected is not None and self.boxes[self.selected].cls != cls:
            self._push_undo()
            self.boxes[self.selected].cls = cls
            self.update()
            self.changed.emit()

    def select_next(self) -> None:
        if not self.boxes:
            return
        self.selected = 0 if self.selected is None else (self.selected + 1) % len(self.boxes)
        self.update()
        self.selection_changed.emit()

    def deselect(self) -> None:
        if self._drag and self._drag.get("kind") == "draw":
            self._drag = None
            self.update()
        if self.selected is not None:
            self.selected = None
            self.update()
            self.selection_changed.emit()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._update_cursor()
        self.update()

    def redraw_boxes(self) -> None:
        self.update()

    # ================= Biến đổi toạ độ =================
    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x * self.scale, self.oy + y * self.scale

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.ox) / self.scale, (cy - self.oy) / self.scale

    def _clamp_img(self, x: float, y: float) -> tuple[float, float]:
        return min(max(x, 0.0), float(self.img_w)), min(max(y, 0.0), float(self.img_h))

    def fit(self) -> None:
        """Thu/phóng để cả ảnh vừa khung nhìn."""
        if self.has_image:
            cw, ch = max(self.width(), 50), max(self.height(), 50)
            self.scale = max(min((cw - 20) / self.img_w, (ch - 20) / self.img_h), 0.02)
            self.ox = (cw - self.img_w * self.scale) / 2
            self.oy = (ch - self.img_h * self.scale) / 2
        self.update()

    def zoom_at(self, factor: float, cx: float, cy: float) -> None:
        if not self.has_image:
            return
        new_scale = min(max(self.scale * factor, 0.02), 40.0)
        factor = new_scale / self.scale
        # Giữ nguyên điểm ảnh nằm dưới con trỏ chuột
        self.ox = cx - (cx - self.ox) * factor
        self.oy = cy - (cy - self.oy) * factor
        self.scale = new_scale
        self.update()

    def zoom_center(self, factor: float) -> None:
        self.zoom_at(factor, self.width() / 2, self.height() / 2)

    # ================= Vẽ =================
    def _handles(self, b: Box) -> dict[str, tuple[float, float]]:
        x1, y1 = self.to_canvas(b.x1, b.y1)
        x2, y2 = self.to_canvas(b.x2, b.y2)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        return {"nw": (x1, y1), "n": (xm, y1), "ne": (x2, y1), "e": (x2, ym),
                "se": (x2, y2), "s": (xm, y2), "sw": (x1, y2), "w": (x1, ym)}

    def _image_for_scale(self) -> QPixmap:
        """Khi thu nhỏ, dùng bản ảnh đã thu nhỏ mượt (đỡ răng cưa, vẽ nhanh hơn)."""
        if self.scale >= 1:
            return self.pixmap
        if self._scaled is None or abs(self._scaled[0] - self.scale) > 1e-9:
            w = max(1, round(self.img_w * self.scale))
            h = max(1, round(self.img_h * self.scale))
            self._scaled = (self.scale, self.pixmap.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        return self._scaled[1]

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        area = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(255, 255, 255, 16), 1))
        # Nền vùng vẽ theo thanh trượt "Độ đục" (ảnh luôn vẽ đặc, không bị nhoè)
        p.setBrush(QColor(0, 0, 0, int(40 + theme.current_opacity * 160)))
        p.drawRoundedRect(area, 8, 8)

        if not self.has_image:
            p.setPen(QColor(255, 255, 255, 140))
            p.drawText(self.rect().adjusted(30, 0, -30, 0), Qt.AlignCenter | Qt.TextWordWrap, PLACEHOLDER)
            return

        # Ảnh luôn vẽ đặc (không trong suốt) để màu chính xác
        target = QRectF(self.ox, self.oy, self.img_w * self.scale, self.img_h * self.scale)
        src = self._image_for_scale()
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2)
        p.drawPixmap(target, src, QRectF(src.rect()))

        # Khung
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(9)
        fm = QFontMetricsF(font)
        p.setFont(font)
        for i, b in enumerate(self.boxes):
            color = QColor(self.get_color(b.cls))
            x1, y1 = self.to_canvas(b.x1, b.y1)
            x2, y2 = self.to_canvas(b.x2, b.y2)
            r = QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized()
            sel = i == self.selected
            fill = QColor(color)
            fill.setAlpha(55 if sel else 0)
            p.setBrush(fill)
            p.setPen(QPen(color, 2.5 if sel else 2))
            p.drawRect(r)
            if self.show_labels:
                text = self.get_name(b.cls)
                tw, th = fm.horizontalAdvance(text) + 8, fm.height() + 2
                ty = r.top() - th if r.top() - th >= 0 else r.top()
                lab = QRectF(r.left() - 1, ty, tw, th)
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawRect(lab)
                p.setPen(_text_color_on(color))
                p.drawText(lab, Qt.AlignCenter, text)
        if self.selected is not None and self.selected < len(self.boxes):
            p.setPen(QPen(QColor(0, 0, 0, 200), 1))
            p.setBrush(QColor("#ffffff"))
            for hx, hy in self._handles(self.boxes[self.selected]).values():
                p.drawRect(QRectF(hx - HANDLE, hy - HANDLE, HANDLE * 2, HANDLE * 2))

        # Khung đang vẽ dở
        d = self._drag
        if d and d["kind"] == "draw" and self._mouse is not None:
            x0, y0 = self.to_canvas(*d["p0"])
            x, y = self.to_canvas(*self._clamp_img(*self.to_image(self._mouse.x(), self._mouse.y())))
            pen = QPen(QColor(self.get_color(self.current_class)), 2, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(QPointF(x0, y0), QPointF(x, y)).normalized())

        # Đường gióng chữ thập theo màu lớp đang chọn
        if (self.mode == "draw" and self._mouse is not None and not self._space
                and not (d and d["kind"] == "pan")):
            c = QColor(self.get_color(self.current_class))
            c.setAlpha(200)
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(QPen(c, 1, Qt.DashLine))
            mx, my = self._mouse.x(), self._mouse.y()
            p.drawLine(QPointF(0, my), QPointF(self.width(), my))
            p.drawLine(QPointF(mx, 0), QPointF(mx, self.height()))

    # ================= Dò trúng =================
    def _hit_handle(self, cx: float, cy: float) -> str | None:
        if self.selected is None:
            return None
        for name, (hx, hy) in self._handles(self.boxes[self.selected]).items():
            if abs(cx - hx) <= HANDLE + 3 and abs(cy - hy) <= HANDLE + 3:
                return name
        return None

    def _hit_box(self, cx: float, cy: float) -> int | None:
        """Khung chứa điểm; nếu nhiều khung chồng nhau thì lấy khung nhỏ nhất."""
        x, y = self.to_image(cx, cy)
        best, best_area = None, None
        for i, b in enumerate(self.boxes):
            if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                area = b.width * b.height
                if best_area is None or area < best_area:
                    best, best_area = i, area
        return best

    # ================= Chuột =================
    def mousePressEvent(self, e) -> None:
        self.setFocus(Qt.MouseFocusReason)
        if not self.has_image:
            return
        pos = e.position()
        cx, cy = pos.x(), pos.y()
        if e.button() in (Qt.RightButton, Qt.MiddleButton) or (e.button() == Qt.LeftButton and self._space):
            self._drag = {"kind": "pan", "last": (cx, cy)}
            self.setCursor(Qt.ClosedHandCursor)
            self.update()
            return
        if e.button() != Qt.LeftButton:
            return
        handle = self._hit_handle(cx, cy)
        if handle:
            self._drag = {"kind": "resize", "handle": handle, "orig": self.boxes[self.selected].copy(),
                          "start": (cx, cy), "changed": False}
            return
        if self.mode == "select":
            hit = self._hit_box(cx, cy)
            if hit is not None:
                if hit != self.selected:
                    self.selected = hit
                    self.update()
                    self.selection_changed.emit()
                self._drag = {"kind": "move", "orig": self.boxes[hit].copy(), "start": (cx, cy),
                              "changed": False}
            else:
                self.deselect()
            return
        # Chế độ Vẽ: bắt đầu một khung mới (nếu đã có lớp để gán)
        if not self.before_draw():
            return
        x, y = self._clamp_img(*self.to_image(cx, cy))
        self._drag = {"kind": "draw", "p0": (x, y), "start": (cx, cy)}

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        self._mouse = pos
        cx, cy = pos.x(), pos.y()
        d = self._drag
        if d is None:
            self._hover(cx, cy)
            self.update()
            return
        if d["kind"] == "pan":
            lx, ly = d["last"]
            self.ox += cx - lx
            self.oy += cy - ly
            d["last"] = (cx, cy)
            self.update()
            return
        if d["kind"] == "draw":
            self.update()
            return
        dx = (cx - d["start"][0]) / self.scale
        dy = (cy - d["start"][1]) / self.scale
        if not d["changed"]:
            if abs(cx - d["start"][0]) < 2 and abs(cy - d["start"][1]) < 2:
                return
            self._push_undo()
            d["changed"] = True
        o: Box = d["orig"]
        w, h = self.img_w, self.img_h
        if d["kind"] == "move":
            # Giữ khung nằm trong ảnh khi kéo
            dx = min(max(dx, -o.x1), w - o.x2)
            dy = min(max(dy, -o.y1), h - o.y2)
            self.boxes[self.selected] = Box(o.cls, o.x1 + dx, o.y1 + dy, o.x2 + dx, o.y2 + dy)
        else:
            hd = d["handle"]
            x1, y1, x2, y2 = o.x1, o.y1, o.x2, o.y2
            if "w" in hd:
                x1 = min(max(o.x1 + dx, 0), w)
            if "e" in hd:
                x2 = min(max(o.x2 + dx, 0), w)
            if "n" in hd:
                y1 = min(max(o.y1 + dy, 0), h)
            if "s" in hd:
                y2 = min(max(o.y2 + dy, 0), h)
            self.boxes[self.selected] = Box(o.cls, x1, y1, x2, y2)
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        d = self._drag
        if d is None:
            return
        pos = e.position()
        cx, cy = pos.x(), pos.y()
        if d["kind"] == "pan":
            if e.button() == Qt.LeftButton or e.button() in (Qt.RightButton, Qt.MiddleButton):
                self._drag = None
                self._update_cursor()
                self.update()
            return
        if e.button() != Qt.LeftButton:
            return
        self._drag = None
        if d["kind"] == "draw":
            self.update()
            sx, sy = d["start"]
            if abs(cx - sx) < MIN_BOX_SCREEN or abs(cy - sy) < MIN_BOX_SCREEN:
                # Chỉ là cú nhấp: chọn khung dưới con trỏ (nếu có)
                hit = self._hit_box(cx, cy)
                if hit != self.selected:
                    self.selected = hit
                    self.update()
                    self.selection_changed.emit()
                return
            x0, y0 = d["p0"]
            x1, y1 = self._clamp_img(*self.to_image(cx, cy))
            box = Box(self.current_class, x0, y0, x1, y1).normalized()
            if box.width < 1 or box.height < 1:
                return
            self._push_undo()
            self.boxes.append(box)
            self.selected = len(self.boxes) - 1
            self.update()
            self.changed.emit()
            self.selection_changed.emit()
            return
        if d.get("changed"):
            b = self.boxes[self.selected].normalized()
            if b.width < 1 or b.height < 1:  # kéo bẹp khung -> trả lại như cũ
                self.boxes[self.selected] = d["orig"]
                self._undo.pop()
            else:
                self.boxes[self.selected] = b
            self.update()
            self.changed.emit()
            self.selection_changed.emit()

    def _hover(self, cx: float, cy: float) -> None:
        if not self.has_image:
            return
        if self._space:
            self.setCursor(Qt.OpenHandCursor)
            return
        handle = self._hit_handle(cx, cy)
        if handle:
            self.setCursor(_HANDLE_CURSORS[handle])
        elif self.mode == "select" and self._hit_box(cx, cy) is not None:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self._update_cursor()

    def _update_cursor(self) -> None:
        if self._space:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor if self.mode == "draw" else Qt.ArrowCursor)

    def leaveEvent(self, _e) -> None:
        self._mouse = None
        self.update()

    def wheelEvent(self, e) -> None:
        dy = e.angleDelta().y()
        if dy:
            pos = e.position()
            self.zoom_at(1.2 if dy > 0 else 1 / 1.2, pos.x(), pos.y())

    def resizeEvent(self, _e) -> None:
        self.fit()

    # ================= Bàn phím (Tab, Space xử lý tại đây; phím khác ở cửa sổ chính) =================
    def set_space(self, pressed: bool) -> None:
        self._space = pressed
        if not (self._drag and self._drag["kind"] == "pan"):
            self._update_cursor()
        self.update()

    def event(self, e) -> bool:
        # Tab thường dùng để chuyển ô nhập; trên khung vẽ, Tab = chọn khung kế tiếp
        if e.type() == QEvent.KeyPress and e.key() == Qt.Key_Tab and not e.modifiers():
            self.select_next()
            return True
        return super().event(e)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Space:
            if not e.isAutoRepeat():
                self.set_space(True)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Space:
            if not e.isAutoRepeat():
                self.set_space(False)
            return
        super().keyReleaseEvent(e)

    def focusOutEvent(self, e) -> None:
        if self._space:
            self.set_space(False)
        super().focusOutEvent(e)
