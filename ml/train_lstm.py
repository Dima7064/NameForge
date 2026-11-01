
"""Обучение LSTM-генератора из data/train.json.

Запуск (из корня проекта):
    python -m ml.train_lstm
"""
import json
from pathlib import Path

from ml.lstm_gen import save_lstm, train_lstm


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    with (root / "data" / "train.json").open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    with (root / "data" / "val.json").open("r", encoding="utf-8") as f:
        val_data = json.load(f)
    with (root / "data" / "test.json").open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    if len(train_data) < 50:
        print("⚠️ Очень мало данных для обучения. Имена будут плохие — это ожидаемо.")

    model, itos, stoi, history = train_lstm(
        train_data, val_data,
        vocab_data=train_data + val_data + test_data,
        epochs=150, lr=1e-3, batch_size=16,
        patience=15, seed=42, verbose=True,
    )

    root_models = root / "models"
    root_models.mkdir(exist_ok=True)

    ckpt_path = root_models / "lstm.pt"
    save_lstm(model, itos, stoi, history, ckpt_path)
    print(f"Модель: {ckpt_path}")

    hist_path = root_models / "lstm_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"История: {hist_path}")

    print(f"Итог: train_ppl={history['train_ppl'][-1]:.2f}, "
          f"val_ppl={history['val_ppl'][-1]:.2f}, "
          f"эпох: {len(history['train_loss'])}")


if __name__ == "__main__":
    main()