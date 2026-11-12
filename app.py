"""Label Tool — công cụ gán nhãn ảnh và huấn luyện YOLO trên máy tính.

Chạy:  python label-tool/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Cho phép chạy từ bất kỳ thư mục nào
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    try:
        import PIL  # noqa: F401
        import yaml  # noqa: F401
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        print(f"Thiếu thư viện: {e.name}\n"
              "Hãy cài:  pip install -r label-tool/requirements.txt", file=sys.stderr)
        sys.exit(1)

    from labeltool.ui import theme
    from labeltool.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Label Tool")
    theme.setup_app(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
