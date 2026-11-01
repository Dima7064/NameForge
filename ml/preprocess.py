
"""Очистка и нормализация имён для NameForge.

Правила:
- Разрешены только кириллица (А-Я, а-я, Ё, ё), дефис, апостроф.
- Первая буква заглавная, остальные строчные.
- Минимальная длина 2 (пустые и однобуквенные отсеиваются).
- Верхней границы по длине НЕТ — только по слогам (в пайплайне).
- Дубликаты (без учёта регистра + стиль) удаляются.
- Стиль должен быть одним из 5 допустимых.
"""
import json
import re
from collections import Counter
from pathlib import Path

VALID_CHARS = re.compile(r"^[А-ЯЁа-яё'\-]+$")
MIN_LEN = 2
VALID_STYLES = {"elven", "human", "orcish", "northern", "southern"}


def normalize(name: str) -> str:
    name = name.strip()
    name = name.replace("’", "'").replace("‘", "'")
    if not VALID_CHARS.match(name):
        raise ValueError(f"Недопустимые символы: {name!r}")
    if len(name) < MIN_LEN:
        raise ValueError(f"Длина {len(name)} < {MIN_LEN}: {name!r}")
    return name[0].upper() + name[1:].lower()


def preprocess(raw: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for entry in raw:
        name = entry.get("name", "")
        style = entry.get("style", "")
        source = entry.get("source", "unknown")
        if style not in VALID_STYLES:
            continue
        try:
            clean = normalize(name)
        except ValueError:
            continue
        key = (clean.lower(), style)
        if key in seen:
            continue
        seen.add(key)
        result.append({"name": clean, "style": style, "source": source})
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "seed_names.json"
    out_path = root / "data" / "names.json"

    with raw_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = preprocess(raw)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Обработано {len(raw)} → {len(cleaned)} имён → {out_path}")
    print("Распределение по стилям:", dict(Counter(e["style"] for e in cleaned)))