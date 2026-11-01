
"""Очистка и нормализация имён с поддержкой gender."""
import json
import re
from collections import Counter
from pathlib import Path

VALID_CHARS = re.compile(r"^[А-ЯЁа-яё'\-]+$")
MIN_LEN = 2
VALID_STYLES = {"elven", "human", "orcish", "northern", "southern"}
VALID_GENDERS = {"male", "female", "unisex"}


def normalize(name: str) -> str:
    name = name.strip().replace("’", "'").replace("‘", "'")
    if not VALID_CHARS.match(name):
        raise ValueError(f"Недопустимые символы: {name!r}")
    if len(name) < MIN_LEN:
        raise ValueError(f"Длина {len(name)} < {MIN_LEN}: {name!r}")
    return name[0].upper() + name[1:].lower()


def preprocess(raw: list[dict]) -> list[dict]:
    """Дедуп по (name, style). При конфликте gender — specific > unisex."""
    result: dict[tuple[str, str], dict] = {}
    for entry in raw:
        name = entry.get("name", "")
        style = entry.get("style", "")
        source = entry.get("source", "unknown")
        gender = entry.get("gender", "unisex")
        if style not in VALID_STYLES:
            continue
        if gender not in VALID_GENDERS:
            gender = "unisex"
        try:
            clean = normalize(name)
        except ValueError:
            continue
        key = (clean.lower(), style)
        if key in result:
            old = result[key]
            if old["gender"] == "unisex" and gender != "unisex":
                result[key] = {"name": clean, "style": style,
                               "source": source, "gender": gender}
        else:
            result[key] = {"name": clean, "style": style,
                           "source": source, "gender": gender}
    return list(result.values())


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "seed_names.json"
    out_path = root / "data" / "names.json"

    with raw_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = preprocess(raw)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Обработано {len(raw)} → {len(cleaned)} имён")
    print("Стили:", dict(Counter(e["style"] for e in cleaned)))
    print("Пол:", dict(Counter(e["gender"] for e in cleaned)))