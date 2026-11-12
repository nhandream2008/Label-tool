"""Thẻ "Huấn luyện": chọn dataset, mô hình gốc, tham số; chạy ultralytics ở tiến trình riêng."""

from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QPlainTextEdit,
                               QProgressBar, QScrollArea, QVBoxLayout, QWidget)

from ..train_runner import WorkerProcess, cuda_available, missing_packages
from . import dialogs
from .common import open_folder
from .widgets import button, hsep, label

BASE_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
CUSTOM_MODEL = "Tệp .pt tự chọn…"
DEVICES = {"Tự động": "auto", "CPU": "cpu", "GPU (CUDA)": "0"}
MAX_LOG_LINES = 3000

MISSING_MSG = (
    "Chưa cài thư viện huấn luyện: {pkgs}.\n\n"
    "Mở cửa sổ lệnh trong thư mục label-tool rồi chạy:\n"
    "    .venv\\Scripts\\pip install -r requirements.txt\n\n"
    "Muốn dùng GPU NVIDIA, xem hướng dẫn cài torch bản CUDA trong README.md."
)


class TrainTab(QWidget):
    def __init__(self, get_default_runs_dir: Callable[[], Path], set_status: Callable[[str], None], parent=None):
        super().__init__(parent)
        self.get_default_runs_dir = get_default_runs_dir
        self.set_status = set_status
        self.worker: WorkerProcess | None = None
        self.worker_kind = ""
        self.best_path: Path | None = None
        self.save_dir: Path | None = None
        self._custom_model = ""
        self._cuda: bool | None = None   # None = chưa biết
        self._cuda_result: queue.Queue = queue.Queue()
        self._error = ""

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # ---------- Cột trái: tham số ----------
        left = QFrame()
        left.setObjectName("panel")
        left.setFixedWidth(380)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("scrollbody")
        scroll.setWidget(body)
        scroll.viewport().setAutoFillBackground(False)
        body.setAutoFillBackground(False)
        ll.addWidget(scroll)
        form = QVBoxLayout(body)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(4)

        def section(text):
            form.addSpacing(8)
            form.addWidget(label(text))

        form.addWidget(label("Cấu hình huấn luyện", "title"))
        section("Dataset (data.yaml)")
        row = QHBoxLayout()
        self.e_data = QLineEdit()
        self.e_data.setPlaceholderText("đường dẫn tới data.yaml")
        row.addWidget(self.e_data, 1)
        row.addWidget(button("Chọn…", self._pick_yaml))
        form.addLayout(row)

        section("Mô hình gốc")
        self.model_menu = QComboBox()
        self.model_menu.addItems(BASE_MODELS + [CUSTOM_MODEL])
        self.model_menu.activated.connect(self._on_model)
        form.addWidget(self.model_menu)
        form.addWidget(label("n: nhanh, nhẹ  •  s: cân bằng  •  m: chính xác hơn, chậm hơn", "small", wrap=True))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self.e_epochs = self._field(grid, "Số epoch", "50", 0, 0)
        self.e_imgsz = self._field(grid, "Kích thước ảnh (imgsz)", "640", 0, 1)
        self.e_batch = self._field(grid, "Batch (-1 = tự động)", "16", 2, 0)
        self.e_patience = self._field(grid, "Dừng sớm sau (epoch)", "50", 2, 1)
        form.addSpacing(8)
        form.addLayout(grid)

        section("Thiết bị")
        self.device_menu = QComboBox()
        self.device_menu.addItems(list(DEVICES))
        form.addWidget(self.device_menu)
        self.gpu_label = label("Đang kiểm tra GPU…", "small", wrap=True)
        form.addWidget(self.gpu_label)

        section("Thư mục lưu kết quả")
        row = QHBoxLayout()
        self.e_project = QLineEdit()
        row.addWidget(self.e_project, 1)
        row.addWidget(button("Chọn…", self._pick_runs))
        form.addLayout(row)

        section("Tên lần chạy")
        self.e_name = QLineEdit()
        form.addWidget(self.e_name)

        form.addSpacing(16)
        btns = QHBoxLayout()
        self.start_btn = button("▶  Bắt đầu huấn luyện", self.start, obj="primary")
        self.start_btn.setMinimumHeight(34)
        btns.addWidget(self.start_btn, 1)
        self.stop_btn = button("■  Dừng", self.stop, obj="danger")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.stop_btn)
        form.addLayout(btns)
        form.addStretch()
        lay.addWidget(left)

        # ---------- Cột phải: tiến độ, log, kết quả ----------
        right = QFrame()
        right.setObjectName("panel")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 12, 16, 14)
        rl.setSpacing(6)
        self.missing_banner = label("", "warn", wrap=True)
        self.missing_banner.hide()
        rl.addWidget(self.missing_banner)
        self.progress_label = label("Chưa chạy", "big")
        rl.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        rl.addWidget(self.progress)
        self.metrics_label = label("", "muted", wrap=True)
        rl.addWidget(self.metrics_label)

        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log.setMaximumBlockCount(MAX_LOG_LINES)
        rl.addWidget(self.log, 1)

        rl.addWidget(hsep())
        res = QHBoxLayout()
        self.result_label = label("Kết quả sẽ hiện ở đây khi huấn luyện xong.", wrap=True)
        self.result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        res.addWidget(self.result_label, 1)
        self.open_btn = button("Mở thư mục kết quả", lambda: self.save_dir and open_folder(self.save_dir))
        self.open_btn.setEnabled(False)
        res.addWidget(self.open_btn, 0, Qt.AlignTop)
        self.onnx_btn = button("Xuất best.pt → ONNX", self.export_onnx)
        self.onnx_btn.setEnabled(False)
        res.addWidget(self.onnx_btn, 0, Qt.AlignTop)
        rl.addLayout(res)
        lay.addWidget(right, 1)

        self.refresh_defaults()
        self._check_env()
        self._timer = QTimer(self, interval=100, timeout=self._poll)
        self._timer.start()

    # ---------- Tiện ích dựng form ----------
    @staticmethod
    def _field(grid: QGridLayout, text: str, value: str, row: int, col: int) -> QLineEdit:
        grid.addWidget(label(text, "small"), row, col)
        e = QLineEdit(value)
        grid.addWidget(e, row + 1, col)
        return e

    def refresh_defaults(self) -> None:
        """Điền thư mục lưu và tên lần chạy mặc định (gọi khi mở dự án khác)."""
        if not self.running():
            self.e_project.setText(str(self.get_default_runs_dir()))
            self.e_name.setText("train-" + datetime.now().strftime("%Y%m%d-%H%M"))

    def set_dataset(self, yaml_path: Path) -> None:
        self.e_data.setText(str(yaml_path))

    def _pick_yaml(self) -> None:
        f = dialogs.ask_open_file(self, "Chọn data.yaml", "YAML (*.yaml *.yml);;Tất cả (*.*)")
        if f:
            self.set_dataset(Path(f))

    def _pick_runs(self) -> None:
        d = dialogs.ask_dir(self, "Chọn thư mục lưu kết quả huấn luyện")
        if d:
            self.e_project.setText(d)

    def _on_model(self, _index: int) -> None:
        value = self.model_menu.currentText()
        if value != CUSTOM_MODEL:
            if value in BASE_MODELS:
                self._custom_model = ""
            return
        f = dialogs.ask_open_file(self, "Chọn tệp trọng số .pt", "PyTorch (*.pt);;Tất cả (*.*)")
        # Bỏ mục tệp tự chọn cũ (nếu có) rồi thêm mục mới
        while self.model_menu.count() > len(BASE_MODELS) + 1:
            self.model_menu.removeItem(self.model_menu.count() - 1)
        if f:
            self._custom_model = f
            self.model_menu.addItem(Path(f).name)
            self.model_menu.setCurrentIndex(self.model_menu.count() - 1)
        else:
            self._custom_model = ""
            self.model_menu.setCurrentIndex(0)

    def _check_env(self) -> None:
        missing = missing_packages()
        if missing:
            self.missing_banner.setText("⚠ " + MISSING_MSG.format(pkgs=", ".join(missing)).split("\n\n")[0]
                                        + " Xem README.md để cài đặt.")
            self.missing_banner.show()
            self.gpu_label.setText("Chưa cài torch")
            return
        # Import torch mất vài giây, nên kiểm tra GPU ở luồng nền
        threading.Thread(target=lambda: self._cuda_result.put(cuda_available()), daemon=True).start()

    # ---------- Log ----------
    def append_log(self, text: str) -> None:
        bar = self.log.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 4
        self.log.appendPlainText(text)
        if at_end:
            bar.setValue(bar.maximum())

    # ---------- Chạy ----------
    def running(self) -> bool:
        return self.worker is not None and self.worker.running()

    def _read_config(self) -> dict:
        data = self.e_data.text().strip()
        if not data or not Path(data).is_file():
            raise ValueError("Hãy chọn tệp data.yaml của dataset (xuất ở bước \"Hoàn thành / Xuất dataset\").")

        def as_int(entry, name, lo, hi):
            try:
                v = int(entry.text().strip())
            except ValueError:
                raise ValueError(f"\"{name}\" phải là số nguyên")
            if not lo <= v <= hi:
                raise ValueError(f"\"{name}\" phải trong khoảng {lo}–{hi}")
            return v

        epochs = as_int(self.e_epochs, "Số epoch", 1, 10000)
        imgsz = as_int(self.e_imgsz, "Kích thước ảnh", 32, 4096)
        if imgsz % 32:
            raise ValueError("Kích thước ảnh nên là bội số của 32 (ví dụ 320, 640).")
        batch = as_int(self.e_batch, "Batch", -1, 1024)
        if batch == 0:
            raise ValueError("Batch phải khác 0")
        patience = as_int(self.e_patience, "Dừng sớm", 1, 10000)
        model = self._custom_model or self.model_menu.currentText()
        if model == CUSTOM_MODEL:
            raise ValueError("Hãy chọn tệp mô hình .pt")
        if self._custom_model and not Path(self._custom_model).is_file():
            raise ValueError("Không tìm thấy tệp mô hình đã chọn")
        project = self.e_project.text().strip()
        name = self.e_name.text().strip()
        if not project or not name:
            raise ValueError("Hãy nhập thư mục lưu và tên lần chạy")
        if (Path(project) / name).exists():
            raise ValueError(f"Thư mục \"{Path(project) / name}\" đã tồn tại. Hãy đặt tên lần chạy khác.")
        device = DEVICES[self.device_menu.currentText()]
        if device == "0" and self._cuda is False:
            raise ValueError("Máy không có GPU CUDA hoặc torch đang là bản CPU. Hãy chọn \"CPU\" hoặc \"Tự động\".")
        if batch == -1 and device != "0" and not self._cuda:
            raise ValueError("Batch tự động (-1) chỉ dùng được với GPU. Hãy nhập số cụ thể, ví dụ 8 hoặc 16.")
        return {"mode": "train", "model": model, "data": str(Path(data).resolve()), "epochs": epochs,
                "imgsz": imgsz, "batch": batch, "device": device, "patience": patience,
                "project": str(Path(project).resolve()), "name": name, "workers": 2}

    def start(self) -> None:
        if self.running():
            return
        missing = missing_packages()
        if missing:
            dialogs.error(self, "Thiếu thư viện", MISSING_MSG.format(pkgs=", ".join(missing)))
            return
        try:
            cfg = self._read_config()
        except ValueError as e:
            dialogs.error(self, "Cấu hình chưa hợp lệ", str(e))
            return
        self.best_path = None
        self.save_dir = None
        self.open_btn.setEnabled(False)
        self.onnx_btn.setEnabled(False)
        self.result_label.setText("Đang huấn luyện…")
        self.metrics_label.setText("")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.log.clear()
        if cfg["model"] in BASE_MODELS:
            self.append_log(f"Lưu ý: lần đầu dùng {cfg['model']}, ultralytics sẽ tự tải trọng số gốc (cần Internet).")
        if self._run(cfg, "train"):
            self.progress_label.setText("Đang khởi động… (nạp torch và dữ liệu)")

    def _run(self, cfg: dict, kind: str) -> bool:
        self.worker = WorkerProcess(cfg)
        self.worker_kind = kind
        try:
            self.worker.start()
        except OSError as e:
            dialogs.error(self, "Không chạy được", f"Không khởi động được tiến trình: {e}")
            self.worker = None
            return False
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.set_status("Đang huấn luyện…" if kind == "train" else "Đang xuất ONNX…")
        return True

    def stop(self) -> None:
        if not self.running():
            return
        if dialogs.ask_yes_no(self, "Dừng huấn luyện",
                              "Dừng ngay bây giờ? Trọng số của epoch gần nhất (last.pt) vẫn được giữ lại."):
            self.worker.stop()
            self.append_log("— Đã yêu cầu dừng —")

    def export_onnx(self) -> None:
        if self.running() or not self.best_path:
            return
        missing = missing_packages()
        if missing:
            dialogs.error(self, "Thiếu thư viện", MISSING_MSG.format(pkgs=", ".join(missing)))
            return
        try:
            imgsz = int(self.e_imgsz.text())
        except ValueError:
            imgsz = 640
        if not dialogs.ask_yes_no(self, "Xuất ONNX",
                                  f"Xuất {self.best_path.name} sang ONNX với kích thước ảnh {imgsz}?\n"
                                  f"Tệp .onnx sẽ nằm cạnh best.pt."):
            return
        self.append_log(f"=== Xuất ONNX: {self.best_path} (imgsz={imgsz}) ===")
        if self._run({"mode": "export", "weights": str(self.best_path), "imgsz": imgsz}, "export"):
            self.progress_label.setText("Đang xuất ONNX…")
            self.progress.setRange(0, 0)  # thanh chạy liên tục (không biết trước thời gian)

    # ---------- Nhận sự kiện từ tiến trình con ----------
    def _poll(self) -> None:
        if self._cuda is None and not self._cuda_result.empty():
            ok = self._cuda_result.get_nowait()
            self._cuda = ok
            self.gpu_label.setText("Có GPU CUDA — nên chọn \"Tự động\" hoặc \"GPU\"" if ok else
                                   "Không thấy GPU CUDA — sẽ huấn luyện bằng CPU (chậm hơn)")
        w = self.worker
        if w is None:
            return
        n = 0
        try:
            while n < 300:  # giới hạn mỗi lượt để giao diện luôn mượt
                kind, data = w.events.get_nowait()
                n += 1
                self._handle(kind, data)
        except queue.Empty:
            pass

    def _handle(self, kind: str, data) -> None:
        if kind == "log":
            self.append_log(data)
        elif kind == "BATCH":
            epochs = max(1, data["epochs"])
            nb = max(1, data["batches"])
            frac = (data["epoch"] + data["batch"] / nb) / epochs
            self.progress.setValue(int(min(frac, 1.0) * 1000))
            self.progress_label.setText(
                f"Epoch {data['epoch'] + 1}/{epochs}  •  lô {data['batch']}/{nb}  •  {frac * 100:.0f}%")
        elif kind == "EPOCH":
            m = data.get("metrics", {})
            parts = []
            for key, name in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
                              ("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")):
                if key in m:
                    parts.append(f"{name}: {m[key]:.3f}")
            prefix = "Đánh giá best.pt" if data.get("final") else f"Sau epoch {data['epoch']}/{data['epochs']}"
            self.metrics_label.setText(prefix + " — " + "  •  ".join(parts))
            if not data.get("final"):
                self.progress.setValue(int(data["epoch"] / max(1, data["epochs"]) * 1000))
        elif kind == "DONE":
            self.save_dir = Path(data["save_dir"]) if data.get("save_dir") else None
            self.best_path = Path(data["best"]) if data.get("best") else None
        elif kind == "EXPORTED":
            self.append_log(f"Đã xuất ONNX: {data['path']}")
            self.result_label.setText(self.result_label.text() + f"\nONNX: {data['path']}")
        elif kind == "ERROR":
            self.append_log("LỖI: " + data.get("message", ""))
            self._error = data.get("message", "")
        elif kind == "exit":
            self._finished(data)

    def _finished(self, code: int) -> None:
        w = self.worker
        kind = self.worker_kind
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setRange(0, 1000)
        error = self._error
        self._error = ""

        if kind == "export":
            self.progress.setValue(1000 if code == 0 else 0)
            self.progress_label.setText("Xuất ONNX xong" if code == 0 else "Xuất ONNX thất bại")
            self.set_status("Xuất ONNX xong" if code == 0 else "Xuất ONNX thất bại")
            if code != 0:
                dialogs.error(self, "Xuất ONNX thất bại", error or f"Mã lỗi {code}. Xem log để biết chi tiết.")
            self.onnx_btn.setEnabled(bool(self.best_path))
            self.open_btn.setEnabled(bool(self.save_dir))
            return

        if w and w.stopped_by_user:
            # Dừng giữa chừng: vẫn tìm best.pt đã lưu
            run_dir = Path(self.e_project.text()) / self.e_name.text()
            best = run_dir / "weights" / "best.pt"
            self.save_dir = run_dir if run_dir.exists() else None
            self.best_path = best if best.exists() else None
            self.progress_label.setText("Đã dừng theo yêu cầu")
            self.set_status("Đã dừng huấn luyện")
        elif code == 0 and self.best_path:
            self.progress.setValue(1000)
            self.progress_label.setText("Huấn luyện xong ✓")
            self.set_status("Huấn luyện xong")
        else:
            self.progress_label.setText("Huấn luyện thất bại")
            self.set_status("Huấn luyện thất bại")
            dialogs.error(self, "Huấn luyện thất bại", error or f"Mã lỗi {code}. Xem log để biết chi tiết.")

        lines = []
        if self.save_dir:
            lines.append(f"Thư mục kết quả: {self.save_dir}")
        if self.best_path:
            lines.append(f"best.pt: {self.best_path}")
        self.result_label.setText("\n".join(lines) or "Không có kết quả.")
        self.open_btn.setEnabled(bool(self.save_dir))
        self.onnx_btn.setEnabled(bool(self.best_path))
        self.refresh_name_only()

    def refresh_name_only(self) -> None:
        """Gợi ý tên lần chạy mới để lần sau không trùng thư mục."""
        self.e_name.setText("train-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
