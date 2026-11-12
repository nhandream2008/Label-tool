"""Đóng gói dự án thành tệp .zip để chuyển sang máy khác, và nhập lại từ tệp .zip.

Tệp zip có dạng:

    <ten-du-an>/
        labeltool-archive.json   thông tin gói (phiên bản, số ảnh, lớp…)
        project.json
        images/ ...
        labels/ ...
        exports/ ... (tuỳ chọn)
        runs/ ...    (tuỳ chọn)

project.json chỉ lưu tên dự án và danh sách lớp (không có đường dẫn tuyệt đối), còn data.yaml
trong exports/ không ghi "path", nên giải nén ở đâu cũng dùng được ngay.

Khi nhập, mọi đường dẫn trong zip được kiểm tra (không cho "..", đường dẫn tuyệt đối,
liên kết tượng trưng) để tệp zip lạ không thể ghi ra ngoài thư mục đích. Dữ liệu được giải
nén vào thư mục tạm rồi mới đổi tên, và không bao giờ ghi đè lên thư mục đã có.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable

from .project import PROJECT_FILE, Project

MANIFEST = "labeltool-archive.json"
FORMAT = "labeltool-project"
VERSION = 1

# Tệp đã nén sẵn: lưu thẳng (không nén lại) cho nhanh
_STORED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".pt", ".onnx", ".zip"}
# Ký tự không hợp lệ trong tên tệp Windows
_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

Progress = Callable[[int, int, str], None]


class ArchiveError(Exception):
    """Lỗi khi đọc/nhập tệp zip dự án (thông báo bằng tiếng Việt)."""


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def part_sizes(project: Project) -> dict[str, int]:
    """Dung lượng (byte) từng phần: dữ liệu chính, dataset đã xuất, kết quả huấn luyện."""
    return {
        "data": _dir_size(project.images_dir) + _dir_size(project.labels_dir),
        "exports": _dir_size(project.exports_dir),
        "runs": _dir_size(project.runs_dir),
    }


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _project_files(project: Project, include_exports: bool, include_runs: bool) -> list[Path]:
    files = [project.root / PROJECT_FILE]
    dirs = [project.images_dir, project.labels_dir]
    if include_exports:
        dirs.append(project.exports_dir)
    if include_runs:
        dirs.append(project.runs_dir)
    for d in dirs:
        if d.is_dir():
            files.extend(sorted(p for p in d.rglob("*") if p.is_file() and not p.name.endswith(".tmp")))
    return files


def default_zip_name(project: Project) -> str:
    return f"{project.root.name}-{datetime.now().strftime('%Y%m%d-%H%M')}.zip"


def export_project_zip(project: Project, out_zip: Path, include_exports: bool = False,
                       include_runs: bool = False, progress: Progress | None = None) -> Path:
    """Ghi toàn bộ dự án ra một tệp .zip. Trả về đường dẫn tệp zip."""
    out_zip = Path(out_zip)
    if out_zip.suffix.lower() != ".zip":
        out_zip = out_zip.with_name(out_zip.name + ".zip")
    root = project.root.resolve()
    if out_zip.resolve().is_relative_to(root):
        raise ArchiveError("Hãy lưu tệp .zip ra ngoài thư mục dự án.")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    files = _project_files(project, include_exports, include_runs)
    st = project.stats()
    top = project.root.name
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "name": project.name,
        "folder": top,
        "created": datetime.now().isoformat(timespec="seconds"),
        "images": st["images"],
        "labeled": st["labeled"],
        "boxes": st["boxes"],
        "classes": [c.name for c in project.classes],
        "includes": ["images", "labels"] + (["exports"] if include_exports else []) + (["runs"] if include_runs else []),
    }

    # Ghi ra tệp tạm rồi đổi tên: không để lại tệp zip hỏng nếu bị lỗi giữa chừng
    part = out_zip.with_name(out_zip.name + ".part")
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr(f"{top}/{MANIFEST}", json.dumps(manifest, ensure_ascii=False, indent=2))
            n = len(files)
            for i, f in enumerate(files, 1):
                arc = f"{top}/{f.relative_to(project.root).as_posix()}"
                ctype = zipfile.ZIP_STORED if f.suffix.lower() in _STORED_EXTS else zipfile.ZIP_DEFLATED
                zf.write(f, arc, compress_type=ctype)
                if progress:
                    progress(i, n, f.name)
        os.replace(part, out_zip)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return out_zip


def _safe_parts(name: str) -> list[str] | None:
    """Tách tên trong zip thành các phần an toàn; None nếu tên nguy hiểm."""
    if not name or name.startswith(("/", "\\")) or "\\" in name:
        return None
    parts = [p for p in PurePosixPath(name).parts if p not in ("", ".")]
    for p in parts:
        if p == ".." or ":" in p or _BAD_CHARS.search(p) or p.rstrip(" .") != p:
            return None
        if p.split(".")[0].upper() in _RESERVED:
            return None
    return parts


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def read_archive(zip_path: Path) -> dict:
    """Kiểm tra tệp zip và đọc thông tin dự án bên trong (không giải nén)."""
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise ArchiveError("Đây không phải tệp .zip hợp lệ.")
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        for info in infos:
            if _safe_parts(info.filename) is None or _is_symlink(info):
                raise ArchiveError(f"Tệp zip chứa đường dẫn không an toàn: {info.filename}")
        # project.json phải nằm ở gốc zip hoặc trong đúng một thư mục cấp đầu
        names = {i.filename.rstrip("/") for i in infos}
        if PROJECT_FILE in names:
            prefix = ""
        else:
            tops = {_safe_parts(n)[0] for n in names if _safe_parts(n)}
            hits = [t for t in tops if f"{t}/{PROJECT_FILE}" in names]
            if len(hits) != 1:
                raise ArchiveError("Không tìm thấy project.json trong tệp zip — đây không phải gói dự án Label Tool.")
            prefix = hits[0] + "/"
        try:
            meta = json.loads(zf.read(prefix + PROJECT_FILE).decode("utf-8"))
        except (ValueError, KeyError) as e:
            raise ArchiveError(f"project.json trong tệp zip bị hỏng: {e}")
        manifest = {}
        if prefix + MANIFEST in names:
            try:
                manifest = json.loads(zf.read(prefix + MANIFEST).decode("utf-8"))
            except ValueError:
                manifest = {}
        members = [i for i in infos if i.filename.startswith(prefix) and not i.is_dir()
                   and i.filename != prefix + MANIFEST]
        images = sum(1 for i in members if i.filename.startswith(prefix + "images/"))
        return {
            "prefix": prefix,
            "folder": prefix.rstrip("/") or zip_path.stem,
            "name": meta.get("name") or manifest.get("name") or zip_path.stem,
            "classes": [c.get("name", "?") for c in meta.get("classes", []) if isinstance(c, dict)],
            "images": images,
            "labeled": manifest.get("labeled"),
            "boxes": manifest.get("boxes"),
            "created": manifest.get("created"),
            "includes": manifest.get("includes"),
            "files": len(members),
            "size": sum(i.file_size for i in members),
        }


def _free_target(parent: Path, folder: str) -> Path:
    base = _BAD_CHARS.sub("_", folder).rstrip(" .") or "du-an"
    target = parent / base
    n = 2
    while target.exists():
        target = parent / f"{base} ({n})"
        n += 1
    return target


def import_project_zip(zip_path: Path, dest_parent: Path, progress: Progress | None = None) -> Path:
    """Giải nén gói dự án vào một thư mục MỚI bên trong dest_parent. Trả về thư mục dự án."""
    info = read_archive(zip_path)
    dest_parent = Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_parent).free
    if info["size"] > free - 50 * 1024 * 1024:
        raise ArchiveError(f"Ổ đĩa không đủ chỗ: cần khoảng {human_size(info['size'])}, "
                           f"còn trống {human_size(free)}.")
    target = _free_target(dest_parent, info["folder"])
    tmp = dest_parent / f".{target.name}.dang-nhap"
    shutil.rmtree(tmp, ignore_errors=True)
    prefix = info["prefix"]
    try:
        tmp.mkdir()
        tmp_root = tmp.resolve()
        with zipfile.ZipFile(zip_path) as zf:
            members = [i for i in zf.infolist() if i.filename.startswith(prefix) and not i.is_dir()
                       and i.filename != prefix + MANIFEST]
            n = len(members)
            for k, m in enumerate(members, 1):
                parts = _safe_parts(m.filename[len(prefix):])
                if not parts:
                    continue
                dest = tmp.joinpath(*parts)
                if not dest.resolve().is_relative_to(tmp_root):
                    raise ArchiveError(f"Tệp zip chứa đường dẫn không an toàn: {m.filename}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
                if progress:
                    progress(k, n, parts[-1])
        for sub in ("images", "labels"):
            (tmp / sub).mkdir(exist_ok=True)
        Project.open(tmp)  # kiểm tra đọc được trước khi đưa vào chỗ thật
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target
