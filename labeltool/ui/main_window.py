"""Cửa sổ chính: thanh lệnh kiểu File Explorer, thẻ "Gán nhãn" và thẻ "Huấn luyện"."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QKeyCombination, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QPushButton, QSlider, QStackedWidget, QVBoxLayout,
                               QWidget)

from ..archive import ArchiveError, human_size, import_project_zip, read_archive
from ..merge import merge_project_zips
from ..project import IMAGE_EXTS, PROJECT_FILE, Box, Project
from . import dialogs, glass, icons, theme, widgets
from .annotator import Annotator
from .common import load_settings, open_folder, save_settings
from .export_dialog import ExportDialog
from .thumb_list import ThumbList, pil_to_qimage
from .train_tab import TrainTab
from .zip_dialog import ZipExportDialog
from .widgets import Segmented, button, hsep, label, vsep

APP_TITLE = "Label Tool — Gán nhãn & Huấn luyện YOLO"
ABOUT_URL = "https://www.nhandream.id.vn"  # trang portfolio của tác giả
TAB_LABEL = 0
TAB_TRAIN = 1
AUTOSAVE_MS = 400

SHORTCUTS = [
    ("W", "vẽ khung"), ("V", "chọn / sửa"),
    ("A / ←", "ảnh trước"), ("D / →", "ảnh sau"),
    ("1–9", "chọn lớp (chế độ V: đổi lớp khung)"),
    ("Del", "xoá khung"), ("Tab", "chọn khung kế"),
    ("Ctrl+Z", "hoàn tác"), ("Esc", "bỏ chọn"),
    ("Lăn chuột", "phóng to / thu nhỏ"),
    ("F", "vừa khung"), ("H", "ẩn/hiện tên lớp"),
    ("Chuột phải", "kéo ảnh (hoặc Space + kéo)"),
    ("E", "xong ảnh (kể cả ảnh trống)"),
]


def _shortcut_html() -> str:
    rows = "".join(
        f"<tr><td style='padding:1px 10px 1px 0; color:{theme.ACCENT}; font-weight:600; white-space:nowrap'>{k}</td>"
        f"<td style='padding:1px 0; color:rgba(255,255,255,0.62)'>{v}</td></tr>" for k, v in SHORTCUTS)
    return f"<table cellspacing='0' cellpadding='0'>{rows}</table>"


def _swatch(color: str, size: int = 14) -> QPixmap:
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QColor(255, 255, 255, 60))
    p.setBrush(QColor(color))
    p.drawRoundedRect(1, 1, size * 2 - 2, size * 2 - 2, 7, 7)
    p.end()
    pm.setDevicePixelRatio(2)
    return pm


def _cmd(text: str, glyph: str | None, slot, tip: str | None = None) -> QPushButton:
    """Nút phẳng trên thanh lệnh (giống File Explorer)."""
    b = button(text, slot, obj="cmd", tip=tip)
    if glyph:
        b.setIcon(icons.icon(glyph))
        b.setIconSize(QSize(16, 16))
    return b


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 880)
        self.setMinimumSize(1100, 680)
        self.settings = load_settings()
        style = self.settings.get("glass", "acrylic")
        if style not in glass.STYLES.values():
            style = "acrylic"
        widgets.current_style = style
        try:
            opacity = float(self.settings.get("glass_opacity", theme.DEFAULT_OPACITY))
        except (TypeError, ValueError):
            opacity = theme.DEFAULT_OPACITY
        theme.set_opacity(QApplication.instance(), opacity)
        # Nền trong suốt để lớp kính của Windows hiện xuyên qua
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.project: Project | None = None
        self.images: list[Path] = []
        self.index: int | None = None
        self.image_size: tuple[int, int] = (0, 0)
        self.dirty = False
        self._busy = False
        self._glass_applied = False
        self._class_sig: tuple | None = None

        self._save_timer = QTimer(self, singleShot=True, interval=AUTOSAVE_MS, timeout=self.save_now)

        self.root = QWidget()
        self.root.setObjectName("root")
        self.setCentralWidget(self.root)
        outer = QVBoxLayout(self.root)
        outer.setContentsMargins(10, 6, 10, 4)
        outer.setSpacing(8)

        outer.addLayout(self._build_topbar())
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_label_tab())
        self.train_tab = TrainTab(self._default_runs_dir, self.set_status)
        self.stack.addWidget(self.train_tab)
        if self.settings.get("last_yaml") and Path(self.settings["last_yaml"]).is_file():
            self.train_tab.set_dataset(Path(self.settings["last_yaml"]))
        outer.addLayout(self._build_statusbar())

        self._bind_keys()
        self._set_root_glass(style != "off" and glass.supported())

        last = self.settings.get("last_project")
        if last and (Path(last) / PROJECT_FILE).exists():
            QTimer.singleShot(200, lambda: self.open_project(Path(last)))
        self._update_buttons()

    # ================= Dựng giao diện =================
    def _build_topbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(2)
        bar.addWidget(_cmd("Tạo dự án", icons.ADD, self.new_project))
        bar.addWidget(_cmd("Mở dự án", icons.FOLDER_OPEN, self.pick_project))
        bar.addWidget(vsep())
        self.import_btn = _cmd("Nhập ảnh", icons.PHOTO, self.import_files)
        bar.addWidget(self.import_btn)
        self.import_dir_btn = _cmd("Nhập cả thư mục", icons.FOLDER, self.import_folder)
        bar.addWidget(self.import_dir_btn)
        bar.addWidget(vsep())
        self.zip_btn = _cmd("Chuyển máy (.zip)", icons.EXPORT, self._show_zip_menu,
                            "Xuất dự án ra .zip / nhập dự án từ .zip để làm tiếp trên máy khác")
        bar.addWidget(self.zip_btn)
        self.folder_btn = _cmd("Mở thư mục", icons.OPEN_FILE,
                               lambda: self.project and open_folder(self.project.root), "Mở thư mục dự án")
        bar.addWidget(self.folder_btn)
        bar.addSpacing(12)
        self.project_label = label("Chưa mở dự án", "chip")
        bar.addWidget(self.project_label)
        bar.addStretch(1)
        self.tab_seg = Segmented(["Gán nhãn", "Huấn luyện"], self.show_tab)
        bar.addWidget(self.tab_seg)
        bar.addSpacing(10)
        self.export_btn = button("Hoàn thành / Xuất dataset", self.open_export, obj="primary")
        self.export_btn.setIcon(icons.icon(icons.CHECK, theme.ACCENT_TEXT))
        self.export_btn.setMinimumHeight(30)
        bar.addWidget(self.export_btn)
        return bar

    def _build_statusbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 0, 0)
        self.status = label("Sẵn sàng", "small")
        bar.addWidget(self.status, 1)
        bar.addWidget(label("Nền cửa sổ:", "small"))
        self.glass_box = QComboBox()
        self.glass_box.addItems(list(glass.STYLES))
        self.glass_box.setFocusPolicy(Qt.NoFocus)
        names = {v: k for k, v in glass.STYLES.items()}
        self.glass_box.setCurrentText(names[widgets.current_style])
        if not glass.supported():
            self.glass_box.setEnabled(False)
            self.glass_box.setToolTip("Hiệu ứng kính cần Windows 11 (bản 22H2 trở lên)")
        self.glass_box.currentTextChanged.connect(lambda t: self.set_glass(glass.STYLES[t]))
        bar.addWidget(self.glass_box)
        bar.addSpacing(10)
        self.opacity_title = label("Độ đục:", "small")
        bar.addWidget(self.opacity_title)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 90)
        self.opacity_slider.setFixedWidth(130)
        self.opacity_slider.setFocusPolicy(Qt.NoFocus)
        self.opacity_slider.setToolTip("Kéo sang trái để nền trong hơn, sang phải để chữ dễ đọc hơn")
        self.opacity_slider.setValue(round(theme.current_opacity * 100))
        self.opacity_value = label(f"{self.opacity_slider.value()}%", "small")
        self.opacity_value.setFixedWidth(34)
        # Áp dụng sau khi ngừng kéo một chút (đổi stylesheet toàn ứng dụng khá tốn)
        self._opacity_timer = QTimer(self, singleShot=True, interval=120, timeout=self._apply_opacity)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        bar.addWidget(self.opacity_slider)
        bar.addWidget(self.opacity_value)
        bar.addWidget(vsep())
        self.about_btn = _cmd("About me", icons.CONTACT, self.open_about, f"Mở trang giới thiệu tác giả: {ABOUT_URL}")
        self.about_btn.setFocusPolicy(Qt.NoFocus)
        bar.addWidget(self.about_btn)
        return bar

    def open_about(self) -> None:
        if not QDesktopServices.openUrl(QUrl(ABOUT_URL)):
            dialogs.info(self, "About me", f"Không mở được trình duyệt. Địa chỉ trang:\n{ABOUT_URL}")

    def _on_opacity(self, v: int) -> None:
        self.opacity_value.setText(f"{v}%")
        self._opacity_timer.start()

    def _apply_opacity(self) -> None:
        theme.set_opacity(QApplication.instance(), self.opacity_slider.value() / 100)
        self.canvas.update()
        self.settings["glass_opacity"] = round(theme.current_opacity, 2)
        save_settings(self.settings)

    def _build_label_tab(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.thumbs = ThumbList()
        self.thumbs.setFixedWidth(270)
        self.thumbs.image_clicked.connect(self.go_to)
        lay.addWidget(self.thumbs)

        center = QVBoxLayout()
        center.setSpacing(6)
        tools = QHBoxLayout()
        tools.setSpacing(4)
        prev_btn = _cmd("", icons.BACK, lambda: self.step(-1), "Ảnh trước (A / ←)")
        next_btn = _cmd("", icons.FORWARD, lambda: self.step(1), "Ảnh sau (D / →)")
        if prev_btn.icon().isNull():
            prev_btn.setText("◀")
            next_btn.setText("▶")
        tools.addWidget(prev_btn)
        tools.addWidget(next_btn)
        tools.addSpacing(6)
        self.mode_seg = Segmented(["Vẽ khung (W)", "Chọn / sửa (V)"],
                                  lambda i: self.canvas.set_mode("draw" if i == 0 else "select"))
        tools.addWidget(self.mode_seg)
        tools.addWidget(vsep())
        no_icons = icons.icon(icons.ZOOM_OUT).isNull()
        tools.addWidget(_cmd("−" if no_icons else "", icons.ZOOM_OUT,
                             lambda: self.canvas.zoom_center(1 / 1.25), "Thu nhỏ (−)"))
        tools.addWidget(_cmd("Vừa khung", icons.FIT, lambda: self.canvas.fit(), "Vừa khung (F)"))
        tools.addWidget(_cmd("+" if no_icons else "", icons.ZOOM_IN,
                             lambda: self.canvas.zoom_center(1.25), "Phóng to (+)"))
        tools.addStretch(1)
        self.image_label = label("", "muted")
        tools.addWidget(self.image_label)
        center.addLayout(tools)

        self.canvas = Annotator(self._class_color, self._class_name)
        self.canvas.before_draw = self._ensure_class
        self.canvas.changed.connect(self._on_boxes_changed)
        self.canvas.selection_changed.connect(self._refresh_side)
        center.addWidget(self.canvas, 1)
        lay.addLayout(center, 1)

        side = QFrame()
        side.setObjectName("panel")
        side.setFixedWidth(300)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(12, 12, 12, 12)
        sl.setSpacing(8)
        sl.addWidget(label("Lớp (class)", "title"))
        self.class_list = QListWidget()
        self.class_list.setFocusPolicy(Qt.NoFocus)
        self.class_list.setIconSize(QSize(14, 14))
        self.class_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.class_list.viewport().setAutoFillBackground(False)
        self.class_list.itemClicked.connect(lambda it: self.choose_class(self.class_list.row(it)))
        self.class_list.itemDoubleClicked.connect(lambda _it: self.rename_class())
        sl.addWidget(self.class_list, 1)
        self.class_empty = label("Chưa có lớp nào.\nBấm \"Thêm\" để tạo lớp đầu tiên (ví dụ: nguoi).",
                                 "muted", wrap=True)
        sl.addWidget(self.class_empty)
        cb = QHBoxLayout()
        cb.setSpacing(4)
        for text, glyph, slot in (("Thêm", icons.ADD, self.add_class), ("Đổi tên", icons.RENAME, self.rename_class),
                                  ("Đổi màu", icons.COLOR, lambda: self.recolor_class())):
            b = button(text, slot)
            b.setIcon(icons.icon(glyph))
            cb.addWidget(b)
        sl.addLayout(cb)
        sl.addWidget(hsep())

        self.info_label = label("", wrap=True)
        sl.addWidget(self.info_label)
        act = QGridLayout()
        act.setSpacing(4)
        self.del_box_btn = button("Xoá khung (Del)", lambda: self.canvas.delete_selected())
        act.addWidget(self.del_box_btn, 0, 0)
        done = button("Xong ảnh này (E)", self.mark_done)
        done.setIcon(icons.icon(icons.CHECK))
        act.addWidget(done, 0, 1)
        act.addWidget(button("Bỏ nhãn ảnh", self.clear_label), 1, 0)
        rm = button("Xoá ảnh", self.remove_image, obj="danger")
        rm.setIcon(icons.icon(icons.DELETE, theme.DANGER))
        act.addWidget(rm, 1, 1)
        sl.addLayout(act)
        sl.addWidget(hsep())
        sl.addWidget(label("Phím tắt", "h2"))
        keys = label(_shortcut_html(), "small")
        keys.setTextFormat(Qt.RichText)
        sl.addWidget(keys)
        lay.addWidget(side)
        return page

    # ================= Hiệu ứng kính =================
    def _set_root_glass(self, on: bool) -> None:
        self.root.setProperty("glass", "true" if on else "false")
        if hasattr(self, "opacity_slider"):
            for w in (self.opacity_title, self.opacity_slider, self.opacity_value):
                w.setEnabled(on)
        self.root.style().unpolish(self.root)
        self.root.style().polish(self.root)
        self.root.update()

    def set_glass(self, style: str) -> None:
        widgets.current_style = style
        try:
            opacity = float(self.settings.get("glass_opacity", theme.DEFAULT_OPACITY))
        except (TypeError, ValueError):
            opacity = theme.DEFAULT_OPACITY
        theme.set_opacity(QApplication.instance(), opacity)
        ok = glass.apply(self, style)
        self._set_root_glass(ok)
        self.settings["glass"] = style
        save_settings(self.settings)
        if style != "off" and not ok:
            self.set_status("Không bật được hiệu ứng kính trên máy này — dùng nền đặc")

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not self._glass_applied:
            self._glass_applied = True
            ok = glass.apply(self, widgets.current_style)
            self._set_root_glass(ok)

    # ================= Trạng thái chung =================
    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def _update_buttons(self) -> None:
        on = bool(self.project) and not self._busy
        for b in (self.import_btn, self.import_dir_btn, self.export_btn, self.folder_btn):
            b.setEnabled(on)
        self.zip_btn.setEnabled(not self._busy)

    def _default_runs_dir(self) -> Path:
        if self.project:
            return self.project.runs_dir
        return Path(__file__).resolve().parents[2] / "runs"

    def show_tab(self, i: int) -> None:
        if i == TAB_TRAIN:
            self.save_now()
        self.stack.setCurrentIndex(i)
        self.tab_seg.set_index(i)
        for sc in self._shortcuts:
            sc.setEnabled(i == TAB_LABEL)
        if i == TAB_LABEL:
            self.canvas.setFocus()

    # ================= Dự án =================
    def new_project(self) -> None:
        d = dialogs.ask_dir(self, "Chọn hoặc tạo một thư mục TRỐNG cho dự án mới")
        if not d:
            return
        root = Path(d)
        if (root / PROJECT_FILE).exists():
            self.open_project(root)
            return
        if any(root.iterdir()):
            if not dialogs.ask_yes_no(self, "Thư mục không trống",
                                      f"Thư mục {root} đã có tệp khác.\nVẫn tạo dự án ở đây?"):
                return
        try:
            Project.create(root)
        except OSError as e:
            dialogs.error(self, "Không tạo được dự án", str(e))
            return
        self.open_project(root)
        self.set_status("Đã tạo dự án. Bước tiếp theo: thêm lớp và nhập ảnh.")

    def pick_project(self) -> None:
        d = dialogs.ask_dir(self, "Chọn thư mục dự án (có tệp project.json)")
        if d:
            self.open_project(Path(d))

    def open_project(self, root: Path) -> None:
        self.save_now()
        try:
            self.project = Project.open(root)
        except Exception as e:
            dialogs.error(self, "Không mở được dự án", str(e))
            return
        self.settings["last_project"] = str(root)
        save_settings(self.settings)
        self.project_label.setText(f"Dự án: {self.project.name}")
        self.setWindowTitle(f"{self.project.name} — {APP_TITLE}")
        self.canvas.current_class = 0
        self._class_sig = None
        self._reload_images()
        self.train_tab.refresh_defaults()
        self._update_buttons()
        if self.images:
            first_unlabeled = next((i for i, p in enumerate(self.images) if not self.project.is_labeled(p)), 0)
            self.go_to(first_unlabeled)
        else:
            self.index = None
            self.canvas.load(None, [])
            self.image_label.setText("")
        self._refresh_side()
        self.set_status(f"Đã mở dự án {root}")

    def _status_of(self, path: Path) -> tuple[str, int]:
        lp = self.project.label_path(path)
        if not lp.exists():
            return ("none", 0)
        n = sum(1 for line in lp.read_text(encoding="utf-8").splitlines() if line.strip())
        return ("labeled", n) if n else ("empty", 0)

    def _reload_images(self) -> None:
        self.images = self.project.list_images() if self.project else []
        status = {p: self._status_of(p) for p in self.images}
        self.thumbs.set_items(self.images, status)

    # ================= Nhập ảnh =================
    def import_files(self) -> None:
        if not self.project:
            return
        pattern = " ".join("*" + e for e in IMAGE_EXTS)
        files = dialogs.ask_open_files(self, "Chọn ảnh để nhập (chọn được nhiều ảnh)",
                                       f"Ảnh ({pattern});;Tất cả (*.*)")
        if files:
            self._import([Path(f) for f in files])

    def import_folder(self) -> None:
        if not self.project:
            return
        d = dialogs.ask_dir(self, "Chọn thư mục ảnh (gồm cả thư mục con)")
        if not d:
            return
        if Path(d).resolve() == self.project.images_dir.resolve():
            dialogs.warning(self, "Không hợp lệ", "Đây là thư mục ảnh của chính dự án.")
            return
        files = sorted(p for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not files:
            dialogs.info(self, "Không có ảnh", "Thư mục không có ảnh jpg/png/webp/bmp nào.")
            return
        self._import(files)

    def _run_background(self, task, progress_text: str, done) -> None:
        """Chạy task(progress) ở luồng phụ, hiện tiến độ ở thanh trạng thái, rồi gọi done(("done", kết quả))
        hoặc done(("error", thông báo)) trên luồng giao diện."""
        self._busy = True
        self._update_buttons()
        events: queue.Queue = queue.Queue()

        def work():
            try:
                events.put(("done", task(lambda i, n, name: events.put(("p", i, n)))))
            except Exception as e:
                events.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()
        timer = QTimer(self, interval=80)

        def poll():
            try:
                while True:
                    ev = events.get_nowait()
                    if ev[0] == "p":
                        self.set_status(f"{progress_text} {ev[1]}/{ev[2]}")
                        continue
                    timer.stop()
                    timer.deleteLater()
                    self._busy = False
                    self._update_buttons()
                    done(ev)
                    return
            except queue.Empty:
                pass

        timer.timeout.connect(poll)
        timer.start()

    def _import(self, files: list[Path]) -> None:
        self.save_now()
        project = self.project
        self._run_background(lambda progress: project.import_images(files, progress=progress),
                             "Đang nhập ảnh…", self._import_done)

    def _import_done(self, ev) -> None:
        current = self.images[self.index] if self.index is not None else None
        self._reload_images()
        if ev[0] == "error":
            dialogs.error(self, "Lỗi nhập ảnh", ev[1])
            return
        ok, errors = ev[1]
        self.set_status(f"Đã nhập {ok} ảnh" + (f", {len(errors)} lỗi" if errors else ""))
        if errors:
            more = f"\n… và {len(errors) - 15} lỗi khác" if len(errors) > 15 else ""
            dialogs.warning(self, "Một số ảnh không nhập được", "\n".join(errors[:15]) + more)
        if current in self.images:
            self.index = self.images.index(current)
            self.thumbs.set_current(self.index)
        elif self.images:
            self.go_to(0)

    # ================= Chuyển dự án sang máy khác (.zip) =================
    def _show_zip_menu(self) -> None:
        menu = QMenu(self)
        exp = menu.addAction(icons.icon(icons.EXPORT), "Xuất dự án ra tệp .zip…", self.export_zip)
        exp.setEnabled(bool(self.project) and not self._busy)
        imp = menu.addAction(icons.icon(icons.FOLDER_OPEN), "Nhập dự án từ tệp .zip…", self.import_zip)
        imp.setEnabled(not self._busy)
        menu.addSeparator()
        mrg = menu.addAction(icons.icon(icons.FOLDER_OPEN), "Gộp nhiều tệp .zip thành một dự án…", self.merge_zips)
        mrg.setEnabled(not self._busy)
        menu.exec(self.zip_btn.mapToGlobal(self.zip_btn.rect().bottomLeft()))

    def export_zip(self) -> None:
        if not self.project:
            return
        self.save_now()
        dlg = ZipExportDialog(self, self.project, self.settings.get("last_zip_dir", ""))
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.finished.connect(lambda _r: self._remember_zip_dir(dlg.result_path))
        dlg.show()
        self._zip_dialog = dlg

    def _remember_zip_dir(self, path: Path | None) -> None:
        if path:
            self.settings["last_zip_dir"] = str(path.parent)
            save_settings(self.settings)

    def import_zip(self) -> None:
        f = dialogs.ask_open_file(self, "Chọn tệp .zip của dự án", "Tệp zip (*.zip)")
        if not f:
            return
        zip_path = Path(f)
        try:
            info = read_archive(zip_path)
        except (ArchiveError, OSError) as e:
            dialogs.error(self, "Không đọc được tệp .zip", str(e))
            return
        lines = [f"Dự án: {info['name']}",
                 f"{info['images']} ảnh" + (f" ({info['labeled']} đã gán nhãn)" if info.get("labeled") is not None else "")
                 + (f" • {info['boxes']} khung" if info.get("boxes") is not None else ""),
                 "Lớp: " + (", ".join(info["classes"]) or "(chưa có)"),
                 f"Dung lượng sau khi giải nén: {human_size(info['size'])}"]
        if info.get("created"):
            lines.append(f"Tạo lúc: {info['created'].replace('T', ' ')}")
        if not dialogs.ask_yes_no(self, "Nhập dự án từ .zip", "\n".join(lines) + "\n\nBấm \"Có\" rồi chọn thư mục "
                                  "để đặt dự án (sẽ tạo một thư mục con mới, không ghi đè dữ liệu có sẵn)."):
            return
        d = dialogs.ask_dir(self, "Chọn nơi đặt dự án", self.settings.get("last_import_dir", ""))
        if not d:
            return
        self.settings["last_import_dir"] = d
        self.save_now()
        self._run_background(lambda progress: import_project_zip(zip_path, Path(d), progress=progress),
                             "Đang giải nén dự án…", self._import_zip_done)

    def _import_zip_done(self, ev) -> None:
        if ev[0] == "error":
            dialogs.error(self, "Nhập dự án thất bại", ev[1])
            self.set_status("Nhập dự án thất bại")
            return
        target = ev[1]
        self.open_project(target)
        self.set_status(f"Đã nhập dự án vào {target}")

    # ================= Gộp công việc của nhiều người =================
    def merge_zips(self) -> None:
        files = dialogs.ask_open_files(self, "Chọn các tệp .zip cần gộp (chọn được nhiều tệp)", "Tệp zip (*.zip)")
        if not files:
            return
        zips = [Path(f) for f in files]
        lines = []
        for z in zips:
            try:
                info = read_archive(z)
            except (ArchiveError, OSError) as e:
                dialogs.error(self, "Không đọc được tệp .zip", f"{z.name}:\n{e}")
                return
            done = f", {info['labeled']} đã gán" if info.get("labeled") is not None else ""
            lines.append(f"• {z.name}: {info['images']} ảnh{done} — lớp: {', '.join(info['classes']) or '(chưa có)'}")
        text = ("\n".join(lines) + "\n\nLớp được ghép theo tên (không theo thứ tự). Ảnh trùng chỉ giữ một bản; "
                "nếu hai người gán khác nhau cho cùng một ảnh, nhãn đang có được giữ và ảnh đó được liệt kê để xem lại. "
                "Dữ liệu có sẵn không bị xoá.")
        options = ["Tạo dự án mới…"]
        if self.project:
            options.insert(0, f"Gộp vào dự án đang mở ({self.project.name})")
        choice = dialogs.ask_choice(self, f"Gộp {len(zips)} tệp .zip", text, options)
        if choice is None:
            return
        if options[choice] == "Tạo dự án mới…":
            d = dialogs.ask_dir(self, "Chọn hoặc tạo một thư mục TRỐNG cho dự án gộp")
            if not d:
                return
            root = Path(d)
            if not (root / PROJECT_FILE).exists() and any(root.iterdir()):
                dialogs.warning(self, "Thư mục không trống", f"Hãy chọn một thư mục trống.\n{root}")
                return
            try:
                Project.create(root)
            except OSError as e:
                dialogs.error(self, "Không tạo được dự án", str(e))
                return
        else:
            root = self.project.root
        self.save_now()

        def task(progress):
            # Dùng một bản Project riêng ở luồng phụ; giao diện mở lại dự án khi xong
            return merge_project_zips(Project.open(root), zips, progress=progress)

        self._run_background(task, "Đang gộp dự án…", lambda ev: self._merge_done(ev, root))

    def _merge_done(self, ev, root: Path) -> None:
        self.open_project(root)
        if ev[0] == "error":
            dialogs.error(self, "Gộp dự án thất bại", ev[1] + "\n\nCác ảnh đã gộp xong trước khi lỗi vẫn được giữ.")
            self.set_status("Gộp dự án thất bại")
            return
        rep = ev[1]
        self.last_merge_report = rep
        self.set_status(f"Đã gộp: thêm {rep.total_added} ảnh"
                        + (f", {rep.total_conflicts} ảnh nhãn khác nhau cần xem lại" if rep.total_conflicts else ""))
        show = dialogs.warning if rep.total_conflicts else dialogs.info
        show(self, "Đã gộp dự án", rep.to_text())

    # ================= Chuyển ảnh =================
    def go_to(self, index: int) -> None:
        if not self.project or not (0 <= index < len(self.images)):
            return
        self.save_now()
        path = self.images[index]
        try:
            with Image.open(path) as im:
                im.load()
                size = im.size
                qimg = pil_to_qimage(im)
        except Exception as e:
            dialogs.error(self, "Không mở được ảnh", f"{path.name}: {e}")
            return
        self.index = index
        self.image_size = size
        self.canvas.load(qimg, self.project.load_boxes(path, size))
        self.thumbs.set_current(index)
        self.image_label.setText(f"{path.name}   {size[0]}×{size[1]}   ({index + 1}/{len(self.images)})")
        self._refresh_side()
        if self.stack.currentIndex() == TAB_LABEL:
            self.canvas.setFocus()

    def step(self, delta: int) -> None:
        nxt = self.thumbs.neighbor(self.index, delta)
        if nxt is not None:
            self.go_to(nxt)
        else:
            self.set_status("Đã tới " + ("ảnh cuối" if delta > 0 else "ảnh đầu") + " của danh sách")

    # ================= Lưu nhãn =================
    def _on_boxes_changed(self) -> None:
        self.dirty = True
        self._save_timer.start()
        self._refresh_side()

    def save_now(self) -> None:
        self._save_timer.stop()
        if not self.dirty or not self.project or self.index is None:
            return
        path = self.images[self.index]
        try:
            self.project.save_boxes(path, self.image_size, self.canvas.boxes)
        except OSError as e:
            dialogs.error(self, "Không lưu được nhãn", str(e))
            return
        self.dirty = False
        self.thumbs.set_status(path, *self._status_of(path))
        self.set_status(f"Đã lưu nhãn {path.stem}.txt")

    def mark_done(self) -> None:
        """Ghi tệp nhãn ngay cả khi ảnh không có khung nào (ảnh nền)."""
        if self.index is None:
            return
        self.dirty = True
        self.save_now()
        self.step(1)

    def clear_label(self) -> None:
        if self.index is None or not self.project:
            return
        path = self.images[self.index]
        if self.canvas.boxes and not dialogs.ask_yes_no(
                self, "Bỏ nhãn ảnh", f"Xoá tất cả {len(self.canvas.boxes)} khung và đưa ảnh về \"chưa gán nhãn\"?"):
            return
        self._save_timer.stop()
        self.dirty = False
        self.project.clear_label(path)
        self.thumbs.set_status(path, "none", 0)
        self.go_to(self.index)

    def remove_image(self) -> None:
        if self.index is None or not self.project:
            return
        path = self.images[self.index]
        if not dialogs.ask_yes_no(self, "Xoá ảnh khỏi dự án",
                                  f"Xoá {path.name} và nhãn của nó khỏi dự án?\n"
                                  "(Chỉ xoá bản sao trong thư mục dự án, ảnh gốc bạn đã nhập không bị ảnh hưởng.)"):
            return
        self._save_timer.stop()
        self.dirty = False
        old = self.index
        self.project.remove_image(path)
        self._reload_images()
        if self.images:
            self.go_to(min(old, len(self.images) - 1))
        else:
            self.index = None
            self.canvas.load(None, [])
            self.image_label.setText("")
            self._refresh_side()

    # ================= Lớp =================
    def _class_color(self, i: int) -> str:
        return self.project.class_color(i) if self.project else "#999999"

    def _class_name(self, i: int) -> str:
        return self.project.class_name(i) if self.project else str(i)

    def _ensure_class(self) -> bool:
        if not self.project:
            return False
        if not self.project.classes:
            # Chưa có lớp: mở hộp thoại thêm lớp; người dùng vẽ lại sau khi tạo xong
            self.add_class()
            return False
        return True

    def _sync_classes(self) -> None:
        """Cập nhật danh sách lớp (tên, màu, số khung trên ảnh này, lớp đang chọn)."""
        classes = self.project.classes if self.project else []
        counts = [0] * len(classes)
        for b in self.canvas.boxes:
            if 0 <= b.cls < len(counts):
                counts[b.cls] += 1
        sig = tuple((c.name, c.color, n) for c, n in zip(classes, counts))
        if sig != self._class_sig:
            self._class_sig = sig
            self.class_list.clear()
            for i, (c, n) in enumerate(zip(classes, counts)):
                key = f"{i + 1}.  " if i < 9 else ""
                it = QListWidgetItem(_swatch(c.color), f"{key}{c.name}" + (f"   ·  {n}" if n else ""))
                it.setSizeHint(QSize(0, 32))
                it.setToolTip("Nhấp: chọn lớp  •  Nhấp đúp: đổi tên")
                self.class_list.addItem(it)
        self.class_empty.setVisible(bool(self.project) and not classes)
        cur = self.canvas.current_class if 0 <= self.canvas.current_class < len(classes) else -1
        self.class_list.setCurrentRow(cur)

    def choose_class(self, i: int) -> None:
        if not self.project or not (0 <= i < len(self.project.classes)):
            return
        self.canvas.current_class = i
        # Chỉ đổi lớp khung đang chọn ở chế độ Chọn/sửa; ở chế độ Vẽ, phím số chỉ chọn lớp cho khung kế tiếp
        if self.canvas.mode == "select":
            self.canvas.set_selected_class(i)
        self.canvas.update()
        self._refresh_side()

    def add_class(self) -> None:
        if not self.project:
            return
        name = dialogs.ask_text(self, "Thêm lớp", "Tên lớp mới (ví dụ: nguoi, dau, ghe):")
        if not name:
            return
        try:
            idx = self.project.add_class(name)
        except ValueError as e:
            dialogs.error(self, "Không thêm được lớp", str(e))
            return
        self.canvas.current_class = idx
        self._refresh_side()
        self.canvas.update()

    def rename_class(self) -> None:
        if not self.project or not self.project.classes:
            return
        i = self.canvas.current_class
        old = self.project.classes[i].name
        name = dialogs.ask_text(self, "Đổi tên lớp", f"Tên mới cho lớp \"{old}\":", old)
        if not name or name == old:
            return
        try:
            self.project.rename_class(i, name)
        except ValueError as e:
            dialogs.error(self, "Không đổi được tên", str(e))
            return
        self._refresh_side()
        self.canvas.update()

    def recolor_class(self, i: int | None = None) -> None:
        if not self.project or not self.project.classes:
            return
        i = self.canvas.current_class if i is None else i
        color = dialogs.ask_color(self, f"Màu cho lớp {self.project.classes[i].name}", self.project.classes[i].color)
        if color:
            self.project.set_class_color(i, color)
            self._refresh_side()
            self.canvas.update()

    # ================= Bảng bên phải =================
    def _refresh_side(self) -> None:
        self._sync_classes()
        if self.index is None or not self.project:
            self.info_label.setText("")
            self.del_box_btn.setEnabled(False)
            return
        b: Box | None = self.canvas.boxes[self.canvas.selected] if self.canvas.selected is not None else None
        lines = [f"<b>Ảnh này:</b> {len(self.canvas.boxes)} khung"]
        if b:
            lines.append(f"<b>Đang chọn:</b> {self._class_name(b.cls)}  ({b.width:.0f}×{b.height:.0f}px)")
            lines.append("<span style='color:rgba(255,255,255,0.6)'>Ở chế độ Chọn/sửa (V), bấm phím số hoặc "
                         "tên lớp để đổi lớp khung.</span>")
        else:
            lines.append("<span style='color:rgba(255,255,255,0.6)'>Chưa chọn khung nào.</span>")
        self.info_label.setText("<br>".join(lines))
        self.del_box_btn.setEnabled(b is not None)

    def set_mode(self, mode: str) -> None:
        self.mode_seg.set_index(0 if mode == "draw" else 1)
        self.canvas.set_mode(mode)

    # ================= Phím tắt =================
    def _bind_keys(self) -> None:
        """Phím tắt chỉ hoạt động ở thẻ Gán nhãn. Ô nhập chữ tự giữ các phím gõ chữ,
        hộp thoại là cửa sổ riêng nên không bị ảnh hưởng. Tab và Space do khung vẽ tự xử lý."""
        def zoom(f):
            return lambda: self.canvas.zoom_center(f)

        def toggle_labels():
            self.canvas.show_labels = not self.canvas.show_labels
            self.canvas.update()

        def save():
            self.dirty = True
            self.save_now()

        kp = Qt.KeypadModifier
        table = [
            (["A", Qt.Key_Left], lambda: self.step(-1)),
            (["D", Qt.Key_Right], lambda: self.step(1)),
            (["W"], lambda: self.set_mode("draw")),
            (["V"], lambda: self.set_mode("select")),
            ([Qt.Key_Delete, Qt.Key_Backspace], lambda: self.canvas.delete_selected()),
            ([Qt.Key_Escape], lambda: self.canvas.deselect()),
            (["F", "0"], lambda: self.canvas.fit()),
            ([Qt.Key_Plus, Qt.Key_Equal, QKeyCombination(kp, Qt.Key_Plus)], zoom(1.25)),
            ([Qt.Key_Minus, QKeyCombination(kp, Qt.Key_Minus)], zoom(1 / 1.25)),
            (["H"], toggle_labels),
            (["E"], self.mark_done),
            (["Ctrl+Z"], lambda: self.canvas.undo()),
            (["Ctrl+S"], save),
        ]
        for n in range(1, 10):
            key = getattr(Qt, f"Key_{n}")
            table.append(([str(n), QKeyCombination(kp, key)], lambda n=n: self.choose_class(n - 1)))

        self._shortcuts: list[QShortcut] = []
        for keys, fn in table:
            for k in keys:
                seq = QKeySequence(k)
                sc = QShortcut(seq, self)
                sc.setContext(Qt.WindowShortcut)
                sc.setAutoRepeat(True)
                sc.activated.connect(lambda fn=fn: self._run_key(fn))
                self._shortcuts.append(sc)

    def _run_key(self, fn) -> None:
        if self.project is None or self.stack.currentIndex() != TAB_LABEL:
            return
        fn()

    # ================= Xuất / đóng =================
    def open_export(self) -> None:
        if not self.project:
            return
        self.save_now()
        dlg = ExportDialog(self, self.project, on_use_for_training=self._use_dataset)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.show()
        self._export_dialog = dlg

    def _use_dataset(self, yaml_path: Path) -> None:
        self.train_tab.set_dataset(yaml_path)
        self.settings["last_yaml"] = str(yaml_path)
        save_settings(self.settings)
        self.show_tab(TAB_TRAIN)

    def closeEvent(self, e) -> None:
        self.save_now()
        if self.train_tab.running():
            if not dialogs.ask_yes_no(self, "Đang huấn luyện", "Quá trình huấn luyện đang chạy. Dừng và thoát?"):
                e.ignore()
                return
            self.train_tab.worker.stop()
        data = self.train_tab.e_data.text().strip()
        if data:
            self.settings["last_yaml"] = data
        save_settings(self.settings)
        e.accept()
