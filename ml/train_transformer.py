"""Обучение Transformer-генератора из data/train.json.

Запуск (из корня проекта):
    python -m ml.train_transformer
"""
import json
from pathlib import Path

from ml.transformer_gen import save_transformer, train_transformer


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    with (root / "data" / "train.json").open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    with (root / "data" / "val.json").open("r", encoding="utf-8") as f:
        val_data = json.load(f)
    with (root / "data" / "test.json").open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    if len(train_data) < 200:
        print("⚠️ Мало данных для Transformer. Ожидаемо хуже LSTM — это нормально.")

    model, itos, stoi, history = train_transformer(
        train_data, val_data,
        vocab_data=train_data + val_data + test_data,
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
        dropout=0.2,
        epochs=100, lr=1e-3, batch_size=16,
        patience=15, seed=42, verbose=True,
    )

    root_models = root / "models"
    root_models.mkdir(exist_ok=True)

    ckpt_path = root_models / "transformer.pt"
    save_transformer(model, itos, stoi, history, ckpt_path)
    print(f"Модель: {ckpt_path}")

    hist_path = root_models / "transformer_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"История: {hist_path}")

    print(f"Итог: train_ppl={history['train_ppl'][-1]:.2f}, "
          f"val_ppl={history['val_ppl'][-1]:.2f}, "
          f"эпох: {len(history['train_loss'])}")


if __name__ == "__main__":
    main()
