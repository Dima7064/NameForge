"""Пути для запуска из исходников и из .exe (PyInstaller frozen).

- get_install_dir() — read-only данные, вшитые в .exe (models, data/train.json,
  assets, config.json). В .exe это _MEIPASS, в исходниках — корень проекта.
- get_user_data_dir() — writable данные пользователя (favorites, feedback,
  exports, отчёты). В .exe — папка user_data/ рядом с .exe (НЕ внутри
  temp _MEIPASS, иначе данные теряются при закрытии). В исходниках — data/.
"""
import sys
from pathlib import Path


def get_install_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def get_user_data_dir() -> Path:
    if getattr(sys, "frozen", False):
        d = Path(sys.executable).parent / "user_data"
    else:
        d = Path(__file__).resolve().parent.parent / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
