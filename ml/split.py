
"""Стратифицированный сплит датасета на train/val/test.

Стратификация: для каждого стиля отдельно режем 80/10/10, чтобы
в каждом сплите были представлены все 5 стилей в правильных пропорциях.

Важно: одно и то же имя может быть в разных стилях (например, «Эрик» —
северное и орочье). Это НЕ утечка: модель должна научиться различать
стили. Утечка = пересечение по (name, style) паре.
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


def _keys(items: list[dict]) -> set[tuple[str, str]]:
    """Ключ для проверки утечки: (имя в нижнем регистре, стиль)."""
    return {(e["name"].lower(), e["style"]) for e in items}


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

    # Проверка отсутствия утечки — по (name, style), а не только по name.
    # Одно и то же имя может быть в разных стилях — это нормально.
    train_keys = _keys(splits["train"])
    val_keys = _keys(splits["val"])
    test_keys = _keys(splits["test"])
    assert not (train_keys & val_keys), \
        f"Утечка val в train: {(train_keys & val_keys)[:5]}"
    assert not (train_keys & test_keys), \
        f"Утечка test в train: {(train_keys & test_keys)[:5]}"
    assert not (val_keys & test_keys), \
        f"Утечка test в val: {(val_keys & test_keys)[:5]}"
    print("Утечек (по (name, style)) между сплитами нет.")