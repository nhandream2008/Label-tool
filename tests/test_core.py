"""Kiểm thử phần lõi (không cần giao diện): nhãn YOLO, nhập ảnh, chia tập, xuất dataset.

Chạy:  .venv/Scripts/python -m pytest tests -q     (hoặc python tests/test_core.py)
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from labeltool.exporter import ExportOptions, export_dataset, split_counts  # noqa: E402
from labeltool.project import Box, Project, boxes_to_yolo_lines, parse_yolo_lines  # noqa: E402
from make_synthetic import make_image  # noqa: E402


def test_yolo_roundtrip():
    boxes = [Box(0, 10, 20, 110, 70), Box(1, 300, 200, 400, 300)]
    lines = boxes_to_yolo_lines(boxes, 400, 300)
    assert lines[0] == "0 0.150000 0.150000 0.250000 0.166667"
    back = parse_yolo_lines("\n".join(lines), 400, 300)
    for a, b in zip(boxes, back):
        assert a.cls == b.cls
        assert abs(a.x1 - b.x1) < 0.01 and abs(a.y2 - b.y2) < 0.01


def test_clamp_and_tiny_boxes_dropped():
    lines = boxes_to_yolo_lines([Box(0, -50, -50, 50, 50), Box(0, 5, 5, 5.5, 5.5)], 100, 100)
    assert len(lines) == 1
    assert lines[0] == "0 0.250000 0.250000 0.500000 0.500000"


def test_split_counts():
    assert split_counts(10, 0.7, 0.2, 0.1) == (7, 2, 1)
    assert split_counts(2, 0.7, 0.2, 0.1) == (1, 1, 0)
    assert split_counts(3, 0.7, 0.2, 0.1) == (1, 1, 1)
    assert split_counts(5, 0.8, 0.2, 0.0) == (4, 1, 0)
    assert sum(split_counts(97, 70, 20, 10)) == 97


def _make_project(tmp: Path, n: int = 12) -> Project:
    src = tmp / "nguon"
    src.mkdir()
    rng = random.Random(1)
    truth = {}
    for i in range(n):
        im, boxes = make_image(rng)
        p = src / f"anh_{i}.png"
        im.save(p)
        truth[p.stem] = boxes
    # Trùng tên khác đuôi -> phải được đổi tên để nhãn không ghi đè nhau
    Image.new("RGB", (64, 48), "red").save(src / "anh_0.jpg")

    proj = Project.create(tmp / "du-an")
    proj.add_class("tron")
    proj.add_class("vuong")
    ok, errors = proj.import_images(sorted(src.iterdir()))
    assert ok == n + 1 and not errors
    stems = {p.stem for p in proj.list_images()}
    assert "anh_0" in stems and "anh_0_1" in stems
    for img in proj.list_images():
        if img.stem in truth:
            proj.save_boxes(img, Image.open(img).size,
                            [Box(c, *xy) for c, *xy in truth[img.stem]])
    return proj


def test_export_dataset_layout_and_coco():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        proj = _make_project(tmp)
        st = proj.stats()
        assert st["labeled"] == 12 and st["images"] == 13

        opts = ExportOptions(out_dir=tmp / "ds", seed=42, coco=True,
                             aug_flip=True, aug_brightness=0.2, aug_copies=2,
                             resize_mode="fit", resize_w=320)
        summary = export_dataset(proj, opts)
        out = tmp / "ds"
        data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
        assert data["nc"] == 2 and data["names"] == ["tron", "vuong"]
        assert data["train"] == "train/images" and data["val"] == "valid/images"
        assert data["test"] == "test/images"
        # 12 ảnh có nhãn: 8/2/2 (làm tròn 8.4/2.4/1.2 -> valid 2, test 1, train 9)
        n_tr, n_va, n_te = split_counts(12, 0.7, 0.2, 0.1)
        assert summary.images["train"] == n_tr * 3  # gốc + 2 bản tăng cường
        assert summary.images["valid"] == n_va
        assert summary.images["test"] == n_te
        for s in ("train", "valid", "test"):
            imgs = sorted((out / s / "images").iterdir())
            labels = sorted((out / s / "labels").iterdir())
            assert [p.stem for p in imgs] == [p.stem for p in labels]
            for p in imgs:
                assert max(Image.open(p).size) <= 320
            coco = json.loads((out / s / "_annotations.coco.json").read_text(encoding="utf-8"))
            assert len(coco["images"]) == len(imgs)
            assert len(coco["annotations"]) == summary.boxes[s]
            assert coco["categories"][0] == {"id": 1, "name": "tron", "supercategory": "none"}
        # Các tập không được trùng ảnh gốc
        def originals(s):
            return {p.stem.split("_aug")[0] for p in (out / s / "images").iterdir()}
        assert not (originals("train") & originals("valid"))
        assert not (originals("valid") & originals("test"))
        assert "Số khung theo lớp" in summary.to_text()


def test_flip_moves_box():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        proj = Project.create(tmp / "p")
        proj.add_class("a")
        for i in range(2):
            p = proj.images_dir / f"x{i}.png"
            Image.new("RGB", (100, 50), "white").save(p)
            proj.save_boxes(p, (100, 50), [Box(0, 0, 0, 20, 10)])
        export_dataset(proj, ExportOptions(out_dir=tmp / "ds", train=1, valid=1, test=0,
                                           seed=0, aug_flip=True))
        flipped = [p for p in (tmp / "ds" / "train" / "labels").iterdir() if "flip" in p.stem]
        assert len(flipped) == 1
        assert flipped[0].read_text().strip() == "0 0.900000 0.100000 0.200000 0.200000"


def test_project_zip_roundtrip():
    import zipfile

    from labeltool.archive import ArchiveError, export_project_zip, import_project_zip, read_archive
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        proj = _make_project(tmp)
        proj.rename_class(0, "người")  # tên có dấu phải giữ nguyên
        export_dataset(proj, ExportOptions(out_dir=proj.exports_dir / "ds1", seed=1))
        z = export_project_zip(proj, tmp / "out" / "goi", include_exports=True)
        assert z.name == "goi.zip" and z.exists()
        info = read_archive(z)
        assert info["images"] == len(proj.list_images()) and info["classes"][0] == "người"
        dest = tmp / "may-khac"
        a = import_project_zip(z, dest)
        b = import_project_zip(z, dest)  # nhập lần 2: thư mục mới, không ghi đè
        assert a != b and a.exists() and b.exists()
        p2 = Project.open(a)
        assert [c.name for c in p2.classes] == [c.name for c in proj.classes]
        assert [c.color for c in p2.classes] == [c.color for c in proj.classes]
        assert p2.stats() == proj.stats()
        for img in proj.list_images():
            assert (a / "images" / img.name).read_bytes() == img.read_bytes()
            lp = proj.label_path(img)
            if lp.exists():
                assert (a / "labels" / lp.name).read_text() == lp.read_text()
        assert (a / "exports" / "ds1" / "data.yaml").exists()
        assert not list(dest.glob(".*"))  # không còn thư mục tạm

        # Không lưu zip vào trong chính thư mục dự án
        try:
            export_project_zip(proj, proj.root / "x.zip")
            raise AssertionError("phải báo lỗi")
        except ArchiveError:
            pass
        # Zip độc hại (đường dẫn "..") bị từ chối, không ghi gì ra ngoài
        bad = tmp / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("p/project.json", '{"name": "p", "classes": []}')
            zf.writestr("p/../../evil.txt", "x")
        for fn in (lambda: read_archive(bad), lambda: import_project_zip(bad, tmp / "dich")):
            try:
                fn()
                raise AssertionError("phải báo lỗi")
            except ArchiveError:
                pass
        assert not (tmp / "evil.txt").exists() and not (tmp.parent / "evil.txt").exists()
        # Zip không phải dự án
        other = tmp / "other.zip"
        with zipfile.ZipFile(other, "w") as zf:
            zf.writestr("readme.txt", "hi")
        try:
            read_archive(other)
            raise AssertionError("phải báo lỗi")
        except ArchiveError:
            pass


def test_merge_project_zips():
    """Chia việc cho 2 người (thứ tự lớp khác nhau, có ảnh trùng), rồi gộp lại."""
    import shutil

    from labeltool.archive import export_project_zip
    from labeltool.merge import merge_project_zips

    def named(proj, img):  # nhãn của ảnh theo TÊN lớp, để so sánh giữa các dự án
        lp = proj.label_path(img)
        if not lp.exists():
            return None
        out = []
        for line in lp.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                c, *rest = line.split()
                out.append((proj.class_name(int(c)).casefold(), *rest))
        return sorted(out)

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        src = _make_project(tmp)
        imgs = src.list_images()
        labeled = [p for p in imgs if src.is_labeled(p)]

        def worker(name, classes, part, swap):
            p = Project.create(tmp / name)
            for c in classes:
                p.add_class(c)
            for img in part:
                shutil.copy2(img, p.images_dir / img.name)
                lp = src.label_path(img)
                if lp.exists():
                    lines = []
                    for line in lp.read_text().splitlines():
                        c, *rest = line.split()
                        lines.append(" ".join([str(1 - int(c)) if swap else c] + rest))
                    (p.labels_dir / lp.name).write_text("\n".join(lines) + "\n")
            return p

        # Người A: lớp xếp ngược thứ tự và viết hoa; ảnh 0..6
        a = worker("A", ["VUONG", "tron"], labeled[:7], swap=True)
        a.clear_label(a.images_dir / labeled[6].name)  # A chưa gán ảnh 6
        # Người B: ảnh 5..11 (trùng 5, 6 với A) + lớp mới; ảnh 5 gán khác A -> xung đột
        b = worker("B", ["tron", "vuong", "người"], labeled[5:], swap=False)
        with open(b.label_path(b.images_dir / labeled[5].name), "a") as f:
            f.write("2 0.5 0.5 0.1 0.1\n")
        # B có ảnh khác nội dung nhưng trùng tên với ảnh của A
        Image.new("RGB", (50, 40), "blue").save(b.images_dir / labeled[0].name)
        (b.labels_dir / (labeled[0].stem + ".txt")).write_text("2 0.5 0.5 0.2 0.2\n")

        za = export_project_zip(a, tmp / "zip" / "a.zip")
        zb = export_project_zip(b, tmp / "zip" / "b.zip")
        tong = Project.create(tmp / "tong")
        rep = merge_project_zips(tong, [za, zb])

        assert [c.name for c in tong.classes] == ["VUONG", "tron", "người"]
        ra, rb = rep.sources
        assert (ra.added, ra.added_labeled) == (7, 6)
        assert rb.filled == 1 and rb.conflicts == [labeled[5].name] and rb.renamed == 1
        assert rb.added == 12 - 7 + 1  # ảnh 7..11 + ảnh xanh bị đổi tên
        assert len(tong.list_images()) == 13
        assert "nhãn khác nhau" in rep.to_text()

        # Mọi ảnh gốc có mặt đúng một lần, nhãn (theo tên lớp) khớp bản gốc — trừ ảnh xung đột giữ nhãn của A
        by_bytes = {p.read_bytes(): p for p in tong.list_images()}
        for img in labeled:
            t = by_bytes[img.read_bytes()]
            assert named(tong, t) == named(src, img), img.name
        renamed = [p for p in tong.list_images() if p.stem.startswith(labeled[0].stem + "_")]
        assert len(renamed) == 1 and named(tong, renamed[0]) == [("người", "0.5", "0.5", "0.2", "0.2")]

        # Gộp lại lần nữa: không thêm gì
        again = merge_project_zips(tong, [za])
        assert again.total_added == 0 and len(tong.list_images()) == 13
        # Dự án gộp xuất dataset được
        export_dataset(Project.open(tong.root), ExportOptions(out_dir=tmp / "ds", seed=1))
        data = yaml.safe_load((tmp / "ds" / "data.yaml").read_text(encoding="utf-8"))
        assert data["names"] == ["VUONG", "tron", "người"]
        assert not list(tong.root.rglob("*.tmp"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
