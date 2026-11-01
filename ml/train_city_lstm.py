
"""Обучение LSTM для городов. Меньшая модель против переобучения.

Было: hidden=128, layers=2, dropout=0.2 → val_ppl=14.23 (переобучение).
Стало: hidden=64, layers=1, dropout=0.3, weight_decay=1e-3.
"""
import json
import random
from pathlib import Path

from ml.lstm_gen import save_lstm, train_lstm


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    with (root / "data" / "cities.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Города — без гендера: ставим "male" (модель всё равно не учится на gender)
    for d in data:
        d["gender"] = "male"

    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.85)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    model, itos, stoi, history = train_lstm(
        train_data, val_data,
        vocab_data=data,
        hidden_size=64, num_layers=1, dropout=0.3,
        epochs=100, lr=1e-3, batch_size=16,
        weight_decay=1e-3,
        patience=15, seed=42, verbose=True,
    )

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    save_lstm(model, itos, stoi, history, models_dir / "city_lstm.pt")

    print(f"\nИтог: train_ppl={history['train_ppl'][-1]:.2f}, "
          f"val_ppl={history['val_ppl'][-1]:.2f}")


if __name__ == "__main__":
    main()