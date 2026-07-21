"""
Запуск:
    python main.py
"""

import tkinter as tk

# Настроить логирование до импорта остальных модулей
from logger import setup_logging
setup_logging()

from gui import App


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
