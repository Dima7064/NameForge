"""Настройка pytest: добавляет корень проекта в sys.path,
чтобы работали импорты вида `from ml.preprocess import ...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
