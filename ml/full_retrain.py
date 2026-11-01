"""Полный переобучающий конвейер: расширение → препроцессинг →
сплит → отчёт → обучение LSTM / Transformer / Scorers → генерация примеров.

Запуск:
    python -m ml.full_retrain
"""
import subprocess
import sys

from ml.expand_dataset import expand

MODULES = [
    "ml.preprocess",
    "ml.split",
    "ml.report",
    "ml.train_lstm",
    "ml.train_transformer",
    "ml.train_scorers",
    "ml.run_lstm",
    "ml.run_transformer",
    "ml.run_scorers",
]


def run_module(name: str) -> None:
    print(f"\n{'=' * 60}\n=== {name}\n{'=' * 60}")
    subprocess.run([sys.executable, "-m", name], check=True)


def main() -> None:
    print("📦 Шаг 1/10: расширение датасета")
    expand()

    for i, mod in enumerate(MODULES, start=2):
        print(f"\n📦 Шаг {i}/{len(MODULES) + 1}: {mod}")
        run_module(mod)

    print("\n✅ Готово. Проверь:")
    print("   docs/data_report.md       — распределение по стилям")
    print("   docs/scorers_report.md    — accuracy classifier")
    print("   docs/lstm_examples.md     — новые сгенерированные имена")
    print("   models/lstm_history.json  — loss/PPL")


if __name__ == "__main__":
    main()
