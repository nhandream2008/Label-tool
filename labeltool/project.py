"""Quản lý dự án gán nhãn: thư mục ảnh, danh sách lớp và tệp nhãn YOLO.

Cấu trúc một thư mục dự án:

    <du-an>/
        project.json   thông tin dự án và danh sách lớp (tên + màu)
        images/        ảnh đã nhập (được sao chép vào đây)
        labels/        nhãn YOLO, mỗi ảnh một tệp <ten-anh>.txt
        exports/       các dataset đã xuất
        runs/          kết quả huấn luyện

Mỗi dòng trong tệp nhãn có dạng ``class x_center y_center width height``,
toạ độ chuẩn hoá về [0, 1] theo kích thước ảnh (định dạng YOLO của Ultralytics).
Ảnh có tệp nhãn (kể cả tệp rỗng) được coi là "đã gán nhãn"; tệp rỗng nghĩa là
ảnh nền không có đối tượng nào.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

# Các định dạng ảnh được hỗ trợ khi nhập
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

PROJECT_FILE = "project.json"

# Bảng màu mặc định cho lớp mới (dễ phân biệt trên nền ảnh)
DEFAULT_COLORS = [
    "#ff3b30", "#34c759", "#0a84ff", "#ffcc00", "#af52de",
    "#ff9500", "#5ac8fa", "#ff2d55", "#a2845e", "#30d158",
]

# Mã EXIF cho hướng xoay ảnh
EXIF_ORIENTATION = 0x0112


@dataclass
class LabelClass:
    name: str
    color: str


@dataclass
class Box:
    """Một khung bao, toạ độ theo pixel của ảnh (góc trên-trái, góc dưới-phải)."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "Box":
        """Đảm bảo x1 < x2 và y1 < y2."""
        return Box(self.cls, min(self.x1, self.x2), min(self.y1, self.y2),
                   max(self.x1, self.x2), max(self.y1, self.y2))

    def clamp(self, w: int, h: int) -> "Box":
        b = self.normalized()
        return Box(b.cls, max(0.0, min(b.x1, w)), max(0.0, min(b.y1, h)),
                   max(0.0, min(b.x2, w)), max(0.0, min(b.y2, h)))

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    def copy(self) -> "Box":
        return Box(self.cls, self.x1, self.y1, self.x2, self.y2)


def boxes_to_yolo_lines(boxes: list[Box], w: int, h: int) -> list[str]:
    """Đổi khung pixel sang các dòng YOLO (đã chuẩn hoá)."""
    lines = []
    for b in boxes:
        b = b.clamp(w, h)
        if b.width < 1 or b.height < 1:
            continue
        xc = (b.x1 + b.x2) / 2 / w
        yc = (b.y1 + b.y2) / 2 / h
        bw = b.width / w
        bh = b.height / h
        lines.append(f"{b.cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def parse_yolo_lines(text: str, w: int, h: int) -> list[Box]:
    """Đọc các dòng YOLO và đổi sang khung pixel. Bỏ qua dòng hỏng."""
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = (float(p) for p in parts[1:])
        except ValueError:
            continue
        boxes.append(Box(cls, (xc - bw / 2) * w, (yc - bh / 2) * h,
                         (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def read_image_size(path: Path) -> tuple[int, int]:
    """Đọc kích thước ảnh (không giải mã toàn bộ ảnh)."""
    with Image.open(path) as im:
        return im.size


def write_text_atomic(path: Path, text: str) -> None:
    """Ghi tệp an toàn: ghi ra tệp tạm rồi đổi tên, tránh hỏng tệp khi mất điện."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Project:
    root: Path
    name: str = ""
    classes: list[LabelClass] = field(default_factory=list)

    # ---------- Đường dẫn ----------
    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def label_path(self, image_path: Path) -> Path:
        return self.labels_dir / (image_path.stem + ".txt")

    # ---------- Tạo / mở / lưu ----------
    @classmethod
    def create(cls, root: Path, name: str | None = None) -> "Project":
        root = Path(root)
        if (root / PROJECT_FILE).exists():
            return cls.open(root)
        root.mkdir(parents=True, exist_ok=True)
        p = cls(root=root, name=name or root.name)
        p._ensure_dirs()
        p.save()
        return p

    @classmethod
    def open(cls, root: Path) -> "Project":
        root = Path(root)
        meta_path = root / PROJECT_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {PROJECT_FILE} trong {root}")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = [LabelClass(c["name"], c.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))
                   for i, c in enumerate(data.get("classes", []))]
        p = cls(root=root, name=data.get("name", root.name), classes=classes)
        p._ensure_dirs()
        return p

    def _ensure_dirs(self) -> None:
        for d in (self.images_dir, self.labels_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        data = {
            "version": 1,
            "name": self.name,
            "classes": [{"name": c.name, "color": c.color} for c in self.classes],
        }
        write_text_atomic(self.root / PROJECT_FILE,
                          json.dumps(data, ensure_ascii=False, indent=2))

    # ---------- Lớp ----------
    def add_class(self, name: str, color: str | None = None) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Tên lớp không được để trống")
        if any(c.name == name for c in self.classes):
            raise ValueError(f"Lớp '{name}' đã tồn tại")
        color = color or DEFAULT_COLORS[len(self.classes) % len(DEFAULT_COLORS)]
        self.classes.append(LabelClass(name, color))
        self.save()
        return len(self.classes) - 1

    def rename_class(self, index: int, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Tên lớp không được để trống")
        if any(c.name == new_name for i, c in enumerate(self.classes) if i != index):
            raise ValueError(f"Lớp '{new_name}' đã tồn tại")
        self.classes[index].name = new_name
        self.save()

    def set_class_color(self, index: int, color: str) -> None:
        self.classes[index].color = color
        self.save()

    def class_color(self, index: int) -> str:
        if 0 <= index < len(self.classes):
            return self.classes[index].color
        return "#999999"

    def class_name(self, index: int) -> str:
        if 0 <= index < len(self.classes):
            return self.classes[index].name
        return f"lớp {index}?"

    # ---------- Ảnh ----------
    def list_images(self) -> list[Path]:
        if not self.images_dir.exists():
            return []
        files = [p for p in self.images_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        files.sort(key=lambda p: p.name.lower())
        return files

    def is_labeled(self, image_path: Path) -> bool:
        return self.label_path(image_path).exists()

    def _unique_stem(self, stem: str, taken: set[str]) -> str:
        """Tên (không đuôi) chưa dùng — vì nhãn được gắn theo tên ảnh."""
        candidate = stem
        i = 1
        while candidate.lower() in taken:
            candidate = f"{stem}_{i}"
            i += 1
        taken.add(candidate.lower())
        return candidate

    def import_images(self, files: list[Path], progress=None) -> tuple[int, list[str]]:
        """Sao chép ảnh vào dự án. Trả về (số ảnh nhập được, danh sách lỗi).

        Ảnh có thông tin xoay EXIF sẽ được xoay đúng chiều trước khi lưu,
        để toạ độ nhãn khớp ở mọi nơi (trình xem ảnh, OpenCV, trình duyệt).
        """
        self._ensure_dirs()
        taken = {p.stem.lower() for p in self.list_images()}
        ok = 0
        errors: list[str] = []
        for i, src in enumerate(files):
            src = Path(src)
            if progress:
                progress(i, len(files), src.name)
            if src.suffix.lower() not in IMAGE_EXTS:
                errors.append(f"{src.name}: định dạng không hỗ trợ")
                continue
            try:
                with Image.open(src) as im:
                    im.verify()  # kiểm tra ảnh không hỏng
                stem = self._unique_stem(src.stem, taken)
                dst = self.images_dir / (stem + src.suffix.lower())
                with Image.open(src) as im:
                    orientation = im.getexif().get(EXIF_ORIENTATION, 1)
                    if orientation not in (1, None):
                        fixed = ImageOps.exif_transpose(im)
                        save_kwargs = {"quality": 95} if dst.suffix in (".jpg", ".jpeg", ".webp") else {}
                        fixed.save(dst, **save_kwargs)
                    else:
                        shutil.copy2(src, dst)
                ok += 1
            except Exception as e:  # ảnh hỏng hoặc không đọc được
                errors.append(f"{src.name}: {e}")
        if progress:
            progress(len(files), len(files), "")
        return ok, errors

    def remove_image(self, image_path: Path) -> None:
        """Xoá ảnh và nhãn của nó khỏi dự án."""
        lp = self.label_path(image_path)
        if lp.exists():
            lp.unlink()
        if image_path.exists():
            image_path.unlink()

    # ---------- Nhãn ----------
    def load_boxes(self, image_path: Path, size: tuple[int, int]) -> list[Box]:
        lp = self.label_path(image_path)
        if not lp.exists():
            return []
        return parse_yolo_lines(lp.read_text(encoding="utf-8"), *size)

    def save_boxes(self, image_path: Path, size: tuple[int, int], boxes: list[Box]) -> None:
        lines = boxes_to_yolo_lines(boxes, *size)
        write_text_atomic(self.label_path(image_path), "\n".join(lines) + ("\n" if lines else ""))

    def clear_label(self, image_path: Path) -> None:
        """Đưa ảnh về trạng thái "chưa gán nhãn" (xoá tệp nhãn)."""
        lp = self.label_path(image_path)
        if lp.exists():
            lp.unlink()

    def stats(self) -> dict:
        """Thống kê nhanh: số ảnh, số ảnh đã gán, số khung theo lớp."""
        images = self.list_images()
        labeled = 0
        per_class = [0] * len(self.classes)
        total_boxes = 0
        for img in images:
            lp = self.label_path(img)
            if not lp.exists():
                continue
            labeled += 1
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                total_boxes += 1
                try:
                    c = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= c < len(per_class):
                    per_class[c] += 1
        return {"images": len(images), "labeled": labeled,
                "boxes": total_boxes, "per_class": per_class}
