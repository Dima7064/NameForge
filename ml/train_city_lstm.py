"""Обучение LSTM для генерации городов.

Использует ту же архитектуру LSTMNameGen, но на датасете data/cities.json.
Чекпоинт сохраняется отдельно: models/city_lstm.pt.

Запуск:
    python -m ml.train_city_lstm
"""
import json
import random
from pathlib import Path

from ml.lstm_gen import save_lstm, train_lstm


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    with (root / "data" / "cities.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Перемешиваем и делим 85/15
    random.seed(42)
    random.shuffle(data)
    n = len(data)
    split_idx = int(n * 0.85)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    model, itos, stoi, history = train_lstm(
        train_data, val_data,
        vocab_data=data,
        epochs=200, lr=1e-3, batch_size=16,
        weight_decay=1e-5,
        patience=30, seed=42, verbose=True,
    )

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / "city_lstm.pt"
    save_lstm(model, itos, stoi, history, ckpt_path)

    print(f"\nМодель: {ckpt_path}")
    print(f"Итог: train_ppl={history['train_ppl'][-1]:.2f}, "
          f"val_ppl={history['val_ppl'][-1]:.2f}")


if __name__ == "__main__":
    main()
