"""Tiến trình con chạy huấn luyện / xuất ONNX bằng thư viện ``ultralytics``.

Giao diện chạy tệp này trong một tiến trình riêng (subprocess) để:
- cửa sổ không bao giờ bị đơ khi huấn luyện;
- nút "Dừng" chỉ cần kết thúc tiến trình con là xong;
- lỗi của torch/CUDA không làm sập giao diện.

Cách gọi:  python train_worker.py <tệp-cấu-hình.json>

Tiến trình con in ra các dòng log thông thường và các dòng đặc biệt bắt đầu bằng
``@@`` (kèm JSON) để giao diện cập nhật thanh tiến độ:

    @@BATCH  {"epoch": 0, "epochs": 10, "batch": 3, "batches": 20}
    @@EPOCH  {"epoch": 1, "epochs": 10, "metrics": {...}}
    @@DONE   {"save_dir": "...", "best": "...", "last": "..."}
    @@EXPORTED {"path": "..."}
    @@ERROR  {"message": "..."}
"""

from __future__ import annotations

import json
import sys
import traceback


def emit(tag: str, payload: dict) -> None:
    print(f"@@{tag} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def check_imports() -> bool:
    """Kiểm tra torch và ultralytics; báo lỗi tiếng Việt nếu thiếu."""
    missing = []
    for mod in ("torch", "ultralytics"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        emit("ERROR", {"message": (
            "Chưa cài thư viện: " + ", ".join(missing) + ".\n"
            "Hãy chạy:  pip install -r label-tool/requirements.txt\n"
            "(hoặc: pip install ultralytics)")})
        return False
    return True


def to_float_dict(d) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[str(k)] = round(float(v), 5)
        except (TypeError, ValueError):
            pass
    return out


def run_train(cfg: dict) -> None:
    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    last_batch_report = {"t": 0.0}

    def on_train_batch_end(trainer):
        # Báo tiến độ theo lô, tối đa ~4 lần/giây để không làm ngập log
        import time
        now = time.time()
        nb = len(trainer.train_loader) if getattr(trainer, "train_loader", None) else 0
        i = getattr(trainer, "_lt_batch", 0) + 1
        trainer._lt_batch = i
        if now - last_batch_report["t"] >= 0.25 or i >= nb:
            last_batch_report["t"] = now
            emit("BATCH", {"epoch": trainer.epoch, "epochs": trainer.epochs, "batch": i, "batches": nb})

    def on_train_epoch_start(trainer):
        trainer._lt_batch = 0

    def on_fit_epoch_end(trainer):
        # Gọi sau khi train + validate xong một epoch. Ultralytics gọi thêm một lần
        # khi đánh giá best.pt lúc kết thúc (khi đó epoch vượt quá tổng số epoch).
        n = trainer.epoch + 1
        emit("EPOCH", {"epoch": min(n, trainer.epochs), "epochs": trainer.epochs,
                       "final": n > trainer.epochs,
                       "metrics": to_float_dict(trainer.metrics)})

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    args = {
        "data": cfg["data"],
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": cfg["batch"],
        "project": cfg["project"],
        "name": cfg["name"],
        "exist_ok": False,
        "workers": int(cfg.get("workers", 2)),
        "patience": int(cfg.get("patience", 100)),
    }
    if cfg.get("device") not in (None, "", "auto"):
        args["device"] = cfg["device"]

    print("Tham số huấn luyện: " + json.dumps(args, ensure_ascii=False), flush=True)
    model.train(**args)

    trainer = model.trainer
    best = trainer.best if trainer.best.exists() else None
    emit("DONE", {
        "save_dir": str(trainer.save_dir),
        "best": str(best) if best else "",
        "last": str(trainer.last) if trainer.last.exists() else "",
    })


def run_export(cfg: dict) -> None:
    from ultralytics import YOLO

    model = YOLO(cfg["weights"])
    kwargs = {"format": "onnx", "imgsz": int(cfg.get("imgsz", 640))}
    if cfg.get("opset"):
        kwargs["opset"] = int(cfg["opset"])
    path = model.export(**kwargs)
    emit("EXPORTED", {"path": str(path)})


def main() -> int:
    # Đảm bảo in được tiếng Việt khi stdout là ống dẫn (pipe) trên Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if len(sys.argv) != 2:
        emit("ERROR", {"message": "Thiếu tệp cấu hình"})
        return 2
    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)

    if not check_imports():
        return 3
    try:
        if cfg.get("mode") == "export":
            run_export(cfg)
        else:
            run_train(cfg)
        return 0
    except Exception as e:  # báo lỗi về giao diện thay vì im lặng thoát
        traceback.print_exc()
        emit("ERROR", {"message": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    # Bảo vệ bằng __main__ là bắt buộc trên Windows: DataLoader dùng nhiều tiến trình
    # sẽ import lại tệp này.
    sys.exit(main())
