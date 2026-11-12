"""Tạo ảnh tổng hợp (hình tròn, hình vuông trên nền nhiễu) để thử công cụ.

Không dùng ảnh người thật. Tuỳ chọn ``--labels`` ghi luôn nhãn YOLO đúng để thử
nhanh phần xuất dataset và huấn luyện.

Ví dụ:
    python tools/make_synthetic.py D:/thu-nghiem/anh --count 20
    python tools/make_synthetic.py D:/thu-nghiem/du-an --count 20 --project
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASSES = ["hinh_tron", "hinh_vuong"]


def make_image(rng: random.Random, w: int = 480, h: int = 360):
    """Trả về (ảnh, danh sách khung (lớp, x1, y1, x2, y2))."""
    bg = Image.effect_noise((w, h), rng.randint(20, 60)).convert("RGB")
    tint = Image.new("RGB", (w, h), tuple(rng.randint(40, 200) for _ in range(3)))
    im = Image.blend(bg, tint, 0.6).filter(ImageFilter.GaussianBlur(1))
    draw = ImageDraw.Draw(im)
    boxes = []
    for _ in range(rng.randint(1, 5)):
        cls = rng.randint(0, 1)
        s = rng.randint(30, 90)
        x1 = rng.randint(0, w - s)
        y1 = rng.randint(0, h - s)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if cls == 0:
            draw.ellipse((x1, y1, x1 + s, y1 + s), fill=color, outline=(0, 0, 0), width=2)
        else:
            draw.rectangle((x1, y1, x1 + s, y1 + s), fill=color, outline=(255, 255, 255), width=2)
        boxes.append((cls, x1, y1, x1 + s, y1 + s))
    return im, boxes


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo ảnh tổng hợp để thử công cụ gán nhãn")
    ap.add_argument("out", type=Path, help="thư mục đích")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", action="store_true",
                    help="tạo luôn một dự án đã gán nhãn sẵn (để thử xuất và huấn luyện)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.project:
        from labeltool.project import Box, Project
        proj = Project.create(args.out)
        for name in CLASSES:
            if name not in [c.name for c in proj.classes]:
                proj.add_class(name)
        for i in range(args.count):
            im, boxes = make_image(rng)
            path = proj.images_dir / f"tonghop_{i:03d}.jpg"
            im.save(path, quality=92)
            proj.save_boxes(path, im.size, [Box(c, x1, y1, x2, y2) for c, x1, y1, x2, y2 in boxes])
        print(f"Đã tạo dự án {args.out} với {args.count} ảnh đã gán nhãn")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for i in range(args.count):
            im, _ = make_image(rng)
            im.save(args.out / f"tonghop_{i:03d}.jpg", quality=92)
        print(f"Đã tạo {args.count} ảnh trong {args.out}")


if __name__ == "__main__":
    main()
