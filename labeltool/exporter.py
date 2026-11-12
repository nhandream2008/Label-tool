"""Xuất dataset theo kiểu Roboflow "YOLO PyTorch TXT".

Cấu trúc thư mục xuất ra:

    <dataset>/
        data.yaml
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels      (chỉ khi tỉ lệ test > 0)
        <split>/_annotations.coco.json (khi bật xuất COCO)

``data.yaml`` dùng đường dẫn tương đối với chính tệp yaml (Ultralytics lấy thư mục
chứa yaml làm gốc khi không có khoá ``path``), nên có thể chuyển cả thư mục
dataset sang máy khác mà không phải sửa.

Tăng cường dữ liệu (lật, độ sáng) chỉ áp dụng cho tập train, sau khi đã chia
tập, để ảnh valid/test không bị "rò rỉ" sang train.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance, ImageOps

from .project import Box, Project, boxes_to_yolo_lines, parse_yolo_lines

SPLITS = ("train", "valid", "test")


@dataclass
class ExportOptions:
    out_dir: Path
    train: float = 0.7
    valid: float = 0.2
    test: float = 0.1
    seed: int | None = None
    # Đổi kích thước: "none" | "stretch" (kéo giãn về W×H) | "fit" (thu nhỏ giữ tỉ lệ, cạnh dài = W)
    resize_mode: str = "none"
    resize_w: int = 640
    resize_h: int = 640
    # Tăng cường cho tập train
    aug_flip: bool = False
    aug_brightness: float = 0.0  # ví dụ 0.25 = ngẫu nhiên trong khoảng ±25%
    aug_copies: int = 1          # số bản tăng cường thêm cho mỗi ảnh train
    include_unlabeled: bool = False  # đưa cả ảnh chưa gán nhãn vào (làm ảnh nền)
    coco: bool = False


@dataclass
class ExportSummary:
    out_dir: Path
    yaml_path: Path
    class_names: list[str]
    # split -> số ảnh / số khung / số khung theo lớp
    images: dict[str, int] = field(default_factory=dict)
    boxes: dict[str, int] = field(default_factory=dict)
    per_class: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"Thư mục: {self.out_dir}", f"data.yaml: {self.yaml_path}", ""]
        header = f"{'Tập':<8}{'Ảnh':>8}{'Khung':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for s in SPLITS:
            if s in self.images:
                lines.append(f"{s:<8}{self.images[s]:>8}{self.boxes[s]:>8}")
        lines.append(f"{'Tổng':<8}{sum(self.images.values()):>8}{sum(self.boxes.values()):>8}")
        lines.append("")
        lines.append("Số khung theo lớp:")
        used = [s for s in SPLITS if s in self.per_class]
        lines.append(f"  {'Lớp':<20}" + "".join(f"{s:>8}" for s in used) + f"{'Tổng':>8}")
        for ci, name in enumerate(self.class_names):
            counts = [self.per_class[s][ci] for s in used]
            lines.append(f"  {name[:20]:<20}" + "".join(f"{c:>8}" for c in counts) + f"{sum(counts):>8}")
        if self.warnings:
            lines.append("")
            lines.append("Lưu ý:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def split_counts(n: int, train: float, valid: float, test: float) -> tuple[int, int, int]:
    """Tính số ảnh mỗi tập. Đảm bảo có ít nhất 1 ảnh train và 1 ảnh valid khi n >= 2
    (Ultralytics bắt buộc phải có tập val)."""
    total = train + valid + test
    if total <= 0:
        raise ValueError("Tổng tỉ lệ phải lớn hơn 0")
    train, valid, test = train / total, valid / total, test / total
    n_valid = round(n * valid)
    n_test = round(n * test)
    if n >= 2:
        n_valid = max(1, n_valid)
    if test > 0 and n >= 3:
        n_test = max(1, n_test)
    if test == 0:
        n_test = 0
    n_train = n - n_valid - n_test
    while n_train < 1 and n >= 2:
        # Lấy bớt từ test rồi tới valid để train có ít nhất 1 ảnh
        if n_test > 0:
            n_test -= 1
        elif n_valid > 1:
            n_valid -= 1
        else:
            break
        n_train = n - n_valid - n_test
    return n_train, n_valid, n_test


def _normalize_mode(im: Image.Image) -> Image.Image:
    """Ảnh bảng màu (P), CMYK... được đổi sang RGB để resize/chỉnh sáng chính xác."""
    if im.mode in ("RGB", "RGBA", "L"):
        return im
    return im.convert("RGBA" if "transparency" in im.info else "RGB")


def _resize(im: Image.Image, boxes: list[Box], opts: ExportOptions) -> tuple[Image.Image, list[Box]]:
    """Đổi kích thước ảnh và co giãn toạ độ khung tương ứng."""
    im = _normalize_mode(im)
    if opts.resize_mode == "none":
        return im, boxes
    w, h = im.size
    if opts.resize_mode == "stretch":
        nw, nh = opts.resize_w, opts.resize_h
    else:  # fit: cạnh dài nhất = resize_w, giữ tỉ lệ, không phóng to ảnh nhỏ
        scale = min(1.0, opts.resize_w / max(w, h))
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) == (w, h):
        return im, boxes
    sx, sy = nw / w, nh / h
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    boxes = [Box(b.cls, b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy) for b in boxes]
    return im, boxes


def _augment(im: Image.Image, boxes: list[Box], rng: random.Random, opts: ExportOptions,
             ) -> tuple[Image.Image, list[Box], str]:
    """Tạo một bản tăng cường. Trả về (ảnh, khung, hậu tố mô tả)."""
    tags = []
    w, _ = im.size
    flip = False
    if opts.aug_flip:
        # Nếu chỉ bật lật thì luôn lật (tránh tạo bản sao giống hệt ảnh gốc)
        flip = True if opts.aug_brightness <= 0 else rng.random() < 0.5
    if flip:
        im = ImageOps.mirror(im)
        boxes = [Box(b.cls, w - b.x2, b.y1, w - b.x1, b.y2) for b in boxes]
        tags.append("flip")
    if opts.aug_brightness > 0:
        factor = 1.0 + rng.uniform(-opts.aug_brightness, opts.aug_brightness)
        im = ImageEnhance.Brightness(im).enhance(factor)
        tags.append(f"b{round(factor * 100)}")
    return im, boxes, "_".join(tags) or "aug"


def _save_image(im: Image.Image, dst: Path) -> None:
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        im.convert("RGB").save(dst, quality=95)
    elif dst.suffix.lower() == ".webp":
        im.save(dst, quality=95)
    else:
        im.save(dst)


class _CocoWriter:
    """Gom dữ liệu COCO cho một tập (train/valid/test)."""

    def __init__(self, class_names: list[str]):
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        # COCO quy ước id lớp bắt đầu từ 1
        self.categories = [{"id": i + 1, "name": n, "supercategory": "none"}
                           for i, n in enumerate(class_names)]

    def add(self, file_name: str, w: int, h: int, boxes: list[Box]) -> None:
        image_id = len(self.images) + 1
        self.images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})
        for b in boxes:
            b = b.clamp(w, h)
            if b.width < 1 or b.height < 1:
                continue
            self.annotations.append({
                "id": len(self.annotations) + 1,
                "image_id": image_id,
                "category_id": b.cls + 1,
                "bbox": [round(b.x1, 2), round(b.y1, 2), round(b.width, 2), round(b.height, 2)],
                "area": round(b.width * b.height, 2),
                "iscrowd": 0,
            })

    def write(self, path: Path) -> None:
        data = {
            "info": {"description": "Xuất từ Label Tool", "date_created": datetime.now().isoformat(timespec="seconds")},
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def export_dataset(project: Project, opts: ExportOptions, progress=None) -> ExportSummary:
    """Xuất dataset. ``progress(i, n, text)`` được gọi để báo tiến độ (có thể None)."""
    if not project.classes:
        raise ValueError("Dự án chưa có lớp nào. Hãy thêm lớp và gán nhãn trước khi xuất.")

    images = project.list_images()
    chosen = [p for p in images if opts.include_unlabeled or project.is_labeled(p)]
    if len(chosen) < 2:
        raise ValueError("Cần ít nhất 2 ảnh đã gán nhãn để chia tập train/valid.")

    out = Path(opts.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Thư mục xuất đã có dữ liệu: {out}\nHãy chọn thư mục trống.")

    rng = random.Random(opts.seed)
    order = chosen[:]
    rng.shuffle(order)
    n_train, n_valid, n_test = split_counts(len(order), opts.train, opts.valid, opts.test)
    assignment = {
        "train": order[:n_train],
        "valid": order[n_train:n_train + n_valid],
        "test": order[n_train + n_valid:n_train + n_valid + n_test],
    }

    names = [c.name for c in project.classes]
    nc = len(names)
    summary = ExportSummary(out_dir=out, yaml_path=out / "data.yaml", class_names=names)
    used_splits = [s for s in SPLITS if assignment[s]]

    for s in used_splits:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
        summary.images[s] = 0
        summary.boxes[s] = 0
        summary.per_class[s] = [0] * nc

    bad_class = 0
    total_steps = len(chosen)
    step = 0
    for s in used_splits:
        coco = _CocoWriter(names) if opts.coco else None
        for img_path in assignment[s]:
            step += 1
            if progress:
                progress(step, total_steps, img_path.name)
            with Image.open(img_path) as raw:
                raw.load()
                size = raw.size
                boxes = project.load_boxes(img_path, size)
                # Bỏ khung có id lớp không hợp lệ (ví dụ tệp nhãn chép từ nơi khác)
                valid_boxes = [b for b in boxes if 0 <= b.cls < nc]
                bad_class += len(boxes) - len(valid_boxes)

                variants: list[tuple[str, Image.Image | None, list[Box]]] = []
                needs_reencode = opts.resize_mode != "none"
                if needs_reencode:
                    im, bx = _resize(raw.copy(), valid_boxes, opts)
                    variants.append((img_path.stem, im, bx))
                else:
                    variants.append((img_path.stem, None, valid_boxes))  # None = chép nguyên tệp

                if s == "train" and (opts.aug_flip or opts.aug_brightness > 0):
                    base_im, base_bx = _resize(raw.copy(), valid_boxes, opts)
                    for k in range(max(1, opts.aug_copies)):
                        im2, bx2, tag = _augment(base_im, base_bx, rng, opts)
                        variants.append((f"{img_path.stem}_aug{k + 1}_{tag}", im2, bx2))

            for stem, im, bx in variants:
                dst = out / s / "images" / (stem + img_path.suffix.lower())
                if im is None:
                    shutil.copy2(img_path, dst)
                    w, h = size
                else:
                    _save_image(im, dst)
                    w, h = im.size
                lines = boxes_to_yolo_lines(bx, w, h)
                (out / s / "labels" / (stem + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                # Đếm theo đúng những gì đã ghi ra tệp
                written = parse_yolo_lines("\n".join(lines), w, h)
                summary.images[s] += 1
                summary.boxes[s] += len(written)
                for b in written:
                    summary.per_class[s][b.cls] += 1
                if coco:
                    coco.add(dst.name, w, h, written)
        if coco:
            coco.write(out / s / "_annotations.coco.json")

    data = {
        "train": "train/images",
        "val": "valid/images",
    }
    if assignment["test"]:
        data["test"] = "test/images"
    data["nc"] = nc
    data["names"] = names
    header = (
        "# Dataset xuất từ Label Tool (cấu trúc kiểu Roboflow YOLO).\n"
        "# Đường dẫn tính tương đối với thư mục chứa tệp này.\n"
    )
    summary.yaml_path.write_text(
        header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # Lưu lại thông số xuất để tra cứu sau này
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_project": str(project.root),
        "ratios": [opts.train, opts.valid, opts.test],
        "seed": opts.seed,
        "resize": [opts.resize_mode, opts.resize_w, opts.resize_h],
        "augment": {"flip": opts.aug_flip, "brightness": opts.aug_brightness, "copies": opts.aug_copies},
        "images": summary.images,
        "boxes": summary.boxes,
    }
    (out / "export_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if bad_class:
        summary.warnings.append(f"Bỏ qua {bad_class} khung có số lớp không hợp lệ.")
    if not assignment["test"] and opts.test > 0:
        summary.warnings.append("Quá ít ảnh nên không tạo được tập test.")
    empty_classes = [names[i] for i in range(nc)
                     if sum(summary.per_class[s][i] for s in used_splits) == 0]
    if empty_classes:
        summary.warnings.append("Lớp chưa có khung nào: " + ", ".join(empty_classes))
    return summary


def default_export_dir(project: Project) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return project.exports_dir / f"{project.root.name}-{stamp}"
