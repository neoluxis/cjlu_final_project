"""Application entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app_window import AppController


def main() -> int:
    app = QApplication(sys.argv)
    controller = AppController(Path(__file__).with_name('main_window.ui'))
    controller.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())

