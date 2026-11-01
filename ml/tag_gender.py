"""Эвристическая разметка пола для существующих имён.

Правила:
- Окончание на "а", "я", "ия", "иэль", "эль", "ель" → female
- Окончание на согласную → male
- Иначе → male (по умолчанию для фэнтези-имён на согласную)

Идемпотентен: если gender уже стоит — не трогает.

Запуск:
    python -m ml.tag_gender
"""
import json
from pathlib import Path

FEMALE_ENDINGS = (
    "а", "я", "ия", "уа", "ина", "иэль", "эль", "ель", "иль", "эла",
)


def guess_gender(name: str) -> str:
    lower = name.lower()
    for ending in FEMALE_ENDINGS:
        if lower.endswith(ending):
            return "female"
    return "male"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "seed_names.json"

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tagged = 0
    for entry in data:
        if "gender" not in entry or entry["gender"] not in ("male", "female"):
            entry["gender"] = guess_gender(entry["name"])
            tagged += 1

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    from collections import Counter
    c = Counter(e["gender"] for e in data)
    print(f"Размечено: {tagged} из {len(data)}")
    print(f"Распределение: {dict(c)}")


if __name__ == "__main__":
    main()
