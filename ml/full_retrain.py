
"""Полный переобучающий конвейер.

Запуск (из корня проекта):
    python -m ml.full_retrain

Делает всё: препроцессинг, сплит, обучение LSTM / Transformer / Scorers,
обучение модели городов, генерация примеров в docs/.

ВАЖНО: чтобы добавить новые имена или города — просто допиши их в
data/seed_names.json или data/cities.json и запусти эту команду.
Никаких правок кода не нужно.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

MODULES = [
    "ml.preprocess",
    "ml.split",
    "ml.report",
    "ml.train_lstm",
    "ml.train_transformer",
    "ml.train_scorers",
    "ml.train_city_lstm",
    "ml.run_lstm",
    "ml.run_transformer",
    "ml.run_scorers",
]


def run_module(name: str) -> None:
    print(f"\n{'=' * 60}\n=== {name}\n{'=' * 60}")
    subprocess.run([sys.executable, "-m", name], check=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    # Автоматически размечаем gender для новых имён без gender
    names_path = root / "data" / "seed_names.json"
    if names_path.exists():
        with names_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        missing = sum(1 for d in data if "gender" not in d)
        if missing:
            print(f"⚠️ {missing} имён без gender — запускаю ml.tag_gender")
            run_module("ml.tag_gender")

    for i, mod in enumerate(MODULES, start=1):
        print(f"\n📦 Шаг {i}/{len(MODULES)}: {mod}")
        run_module(mod)

    print("\n✅ Готово. Проверь docs/*.md")


if __name__ == "__main__":
    main()