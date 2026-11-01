"""Стратифицированный сплит датасета на train/val/test.

Стратификация: для каждого стиля отдельно режем 80/10/10, чтобы
в каждом сплите были представлены все 5 стилей в правильных пропорциях.
"""
import json
import random
from collections import defaultdict
from pathlib import Path


def split(data: list[dict], train: float = 0.8, val: float = 0.1,
          test: float = 0.1, seed: int = 42) -> dict[str, list[dict]]:
    assert abs(train + val + test - 1.0) < 1e-9, "Сумма долей должна быть 1.0"
    random.seed(seed)

    by_style: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        by_style[entry["style"]].append(entry)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for style, items in by_style.items():
        shuffled = list(items)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * val)
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])
    return splits


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    with (root / "data" / "names.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    splits = split(data)

    for name, items in splits.items():
        path = root / "data" / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"{name}: {len(items)} → {path}")

    # Проверка отсутствия утечки
    train_names = {e["name"].lower() for e in splits["train"]}
    val_names = {e["name"].lower() for e in splits["val"]}
    test_names = {e["name"].lower() for e in splits["test"]}
    assert not (train_names & val_names), "Утечка val в train!"
    assert not (train_names & test_names), "Утечка test в train!"
    assert not (val_names & test_names), "Утечка test в val!"
    print("Утечек между сплитами нет.")
