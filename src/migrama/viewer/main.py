"""Main entry point for the zarr sequence viewer."""

import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main():
    """Launch the zarr sequence viewer application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
