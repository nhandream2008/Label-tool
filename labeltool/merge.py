"""Gộp nhiều gói dự án (.zip) thành một dự án lớn — dùng khi chia việc gán nhãn cho nhiều người.

Cách làm:
- Mỗi người nhận một phần ảnh, gán nhãn trên máy mình rồi "Xuất dự án ra tệp .zip".
- Người tổng hợp mở (hoặc tạo) dự án đích rồi gộp tất cả các tệp .zip vào.

Quy tắc gộp:
- Lớp được ghép theo TÊN (không phân biệt hoa/thường, bỏ khoảng trắng hai đầu), không theo số thứ tự,
  nên mỗi người xếp lớp theo thứ tự khác nhau vẫn gộp đúng. Lớp chưa có trong dự án đích được thêm mới
  (giữ màu của gói nếu chưa trùng màu).
- Ảnh trùng NỘI DUNG (cùng tệp ảnh, kể cả khác tên) chỉ giữ một bản:
    * dự án đích chưa gán nhãn ảnh đó -> lấy nhãn từ gói;
    * cả hai đều đã gán và giống nhau -> bỏ qua;
    * cả hai đều đã gán nhưng khác nhau -> GIỮ nhãn đang có, ghi tên ảnh vào danh sách "nhãn khác nhau"
      để người tổng hợp xem lại.
- Ảnh khác nội dung nhưng trùng tên -> đổi tên (thêm _1, _2…) để nhãn không ghi đè nhau.
- Không bao giờ xoá hay sửa ảnh/nhãn đã có trong dự án đích (trừ việc điền nhãn cho ảnh chưa gán).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .archive import ArchiveError, Progress, _safe_parts, human_size, read_archive
from .project import DEFAULT_COLORS, IMAGE_EXTS, PROJECT_FILE, Project, write_text_atomic


def _key(name: str) -> str:
    return name.strip().casefold()


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _norm_label(text: str) -> str:
    """Chuẩn hoá nội dung nhãn để so sánh (bỏ dòng trống, khoảng trắng thừa)."""
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


@dataclass
class SourceResult:
    name: str
    file: str
    added: int = 0            # ảnh mới thêm vào
    added_labeled: int = 0    # trong đó đã có nhãn
    filled: int = 0           # ảnh trùng, dự án đích chưa gán -> lấy nhãn từ gói
    duplicates: int = 0       # ảnh trùng, bỏ qua
    renamed: int = 0          # ảnh trùng tên khác nội dung -> đã đổi tên
    conflicts: list[str] = field(default_factory=list)   # ảnh trùng nhưng nhãn khác nhau
    dropped_lines: int = 0    # dòng nhãn hỏng / lớp không xác định


@dataclass
class MergeReport:
    project: Path
    sources: list[SourceResult] = field(default_factory=list)
    new_classes: list[str] = field(default_factory=list)

    def to_text(self, max_names: int = 15) -> str:
        lines = []
        for s in self.sources:
            lines.append(f"• {s.name} ({s.file}): thêm {s.added} ảnh ({s.added_labeled} đã gán nhãn)")
            extra = []
            if s.filled:
                extra.append(f"điền nhãn cho {s.filled} ảnh chưa gán")
            if s.duplicates:
                extra.append(f"bỏ qua {s.duplicates} ảnh trùng")
            if s.renamed:
                extra.append(f"đổi tên {s.renamed} ảnh trùng tên")
            if s.dropped_lines:
                extra.append(f"bỏ {s.dropped_lines} dòng nhãn hỏng")
            if extra:
                lines.append("    " + ", ".join(extra))
            if s.conflicts:
                names = ", ".join(s.conflicts[:max_names])
                more = f" … và {len(s.conflicts) - max_names} ảnh khác" if len(s.conflicts) > max_names else ""
                lines.append(f"    ⚠ {len(s.conflicts)} ảnh trùng nhưng nhãn khác nhau (giữ nhãn đang có): {names}{more}")
        if self.new_classes:
            lines.append("Lớp mới được thêm: " + ", ".join(self.new_classes))
        return "\n".join(lines)

    @property
    def total_added(self) -> int:
        return sum(s.added for s in self.sources)

    @property
    def total_conflicts(self) -> int:
        return sum(len(s.conflicts) for s in self.sources)


def _image_members(zf: zipfile.ZipFile, prefix: str) -> tuple[list[zipfile.ZipInfo], dict[str, zipfile.ZipInfo]]:
    """Ảnh nằm trực tiếp trong images/ và nhãn trong labels/ (theo tên không đuôi, chữ thường)."""
    images, labels = [], {}
    for info in zf.infolist():
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        parts = _safe_parts(info.filename[len(prefix):])
        if not parts or len(parts) != 2:
            continue
        folder, name = parts
        suffix = Path(name).suffix.lower()
        if folder == "images" and suffix in IMAGE_EXTS:
            images.append(info)
        elif folder == "labels" and suffix == ".txt":
            labels[Path(name).stem.lower()] = info
    images.sort(key=lambda i: i.filename.lower())
    return images, labels


def _remap_label(text: str, class_map: dict[int, int]) -> tuple[str, int]:
    """Đổi chỉ số lớp theo class_map. Trả về (nội dung mới, số dòng bị bỏ)."""
    out, dropped = [], 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            dropped += 1
            continue
        try:
            cls = int(float(parts[0]))
            [float(p) for p in parts[1:]]
        except ValueError:
            dropped += 1
            continue
        if cls not in class_map:
            dropped += 1
            continue
        out.append(" ".join([str(class_map[cls])] + parts[1:]))
    return "\n".join(out) + ("\n" if out else ""), dropped


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def merge_project_zips(project: Project, zip_paths: list[Path], progress: Progress | None = None) -> MergeReport:
    """Gộp các gói dự án .zip vào `project` (chỉ thêm, không xoá dữ liệu có sẵn)."""
    zip_paths = [Path(z) for z in zip_paths]
    infos = [read_archive(z) for z in zip_paths]  # kiểm tra an toàn trước khi ghi bất cứ thứ gì
    need = sum(i["size"] for i in infos)
    free = shutil.disk_usage(project.root).free
    if need > free - 50 * 1024 * 1024:
        raise ArchiveError(f"Ổ đĩa không đủ chỗ: cần khoảng {human_size(need)}, còn trống {human_size(free)}.")

    report = MergeReport(project=project.root)
    # Dấu vân tay nội dung các ảnh đang có -> phát hiện ảnh trùng (kể cả khác tên)
    by_hash: dict[str, Path] = {}
    existing = project.list_images()
    total = len(existing) + sum(i["images"] for i in infos)
    done = 0
    for img in existing:
        by_hash.setdefault(_sha1(img.read_bytes()), img)
        done += 1
        if progress:
            progress(done, total, img.name)
    taken = {p.stem.lower() for p in existing}

    for zp, info in zip(zip_paths, infos):
        res = SourceResult(name=info["name"], file=zp.name)
        report.sources.append(res)
        prefix = info["prefix"]
        with zipfile.ZipFile(zp) as zf:
            meta = json.loads(zf.read(prefix + PROJECT_FILE).decode("utf-8"))
            # Ghép lớp theo tên
            class_map: dict[int, int] = {}
            for i, c in enumerate(meta.get("classes", [])):
                if not isinstance(c, dict) or not str(c.get("name", "")).strip():
                    continue
                name = str(c["name"]).strip()
                idx = next((j for j, t in enumerate(project.classes) if _key(t.name) == _key(name)), None)
                if idx is None:
                    color = str(c.get("color") or "")
                    used = {t.color.lower() for t in project.classes}
                    if not color.startswith("#") or color.lower() in used:
                        color = DEFAULT_COLORS[len(project.classes) % len(DEFAULT_COLORS)]
                    idx = project.add_class(name, color)
                    report.new_classes.append(name)
                class_map[i] = idx

            images, labels = _image_members(zf, prefix)
            for m in images:
                name = _safe_parts(m.filename[len(prefix):])[-1]
                stem, suffix = Path(name).stem, Path(name).suffix.lower()
                data = zf.read(m)
                label_info = labels.get(stem.lower())
                label_text = None
                if label_info is not None:
                    raw = zf.read(label_info).decode("utf-8", errors="replace")
                    label_text, dropped = _remap_label(raw, class_map)
                    res.dropped_lines += dropped

                h = _sha1(data)
                same = by_hash.get(h)
                if same is not None:
                    lp = project.label_path(same)
                    if label_text is None:
                        res.duplicates += 1
                    elif not lp.exists():
                        write_text_atomic(lp, label_text)
                        res.filled += 1
                    elif _norm_label(lp.read_text(encoding="utf-8")) == _norm_label(label_text):
                        res.duplicates += 1
                    else:
                        res.conflicts.append(same.name)
                else:
                    new_stem = project._unique_stem(stem, taken)
                    if new_stem != stem:
                        res.renamed += 1
                    dst = project.images_dir / (new_stem + suffix)
                    _write_bytes_atomic(dst, data)
                    if label_text is not None:
                        write_text_atomic(project.label_path(dst), label_text)
                        res.added_labeled += 1
                    by_hash[h] = dst
                    res.added += 1
                done += 1
                if progress:
                    progress(done, total, name)
    return report
