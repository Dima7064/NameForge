"""Удаление коллизий: имён, которые встречаются в ≥2 стилях.

Причина: транслитератор превращает разные латинские имена
(Arcul, Arcúl, Arkul) в одинаковые кириллические («Аркул»).
Это создаёт style leakage: модель учит «Аркул = human+northern+southern».

Решение: для каждой группы одинаковых имён оставляем ТОЛЬКО ОДИН стиль
(тот, где имя встречается чаще всего; при равенстве — первый по алфавиту).

Запуск:
    python -m ml.clean_collisions
"""
import json
from collections import defaultdict
from pathlib import Path


def clean(data: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Удаляет коллизии. Возвращает (cleaned, stats)."""
    # Группируем по имени (в нижнем регистре)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for e in data:
        by_name[e["name"].lower()].append(e)

    cleaned: list[dict] = []
    collisions = 0
    styles_removed = defaultdict(int)

    for name, entries in by_name.items():
        if len(entries) == 1:
            cleaned.append(entries[0])
            continue

        # Коллизия: одно имя в нескольких стилях
        # Считаем, сколько раз каждый стиль встречается с этим именем
        style_counts = defaultdict(int)
        for e in entries:
            style_counts[e["style"]] += 1

        # Оставляем стиль с максимумом; при равенстве — первый по алфавиту
        best_style = sorted(style_counts.items(),
                            key=lambda x: (-x[1], x[0]))[0][0]

        # Берём первую запись с этим стилем
        kept = next(e for e in entries if e["style"] == best_style)
        cleaned.append(kept)

        collisions += 1
        for e in entries:
            if e["style"] != best_style:
                styles_removed[e["style"]] += 1

    stats = {
        "Всего на входе": len(data),
        "Всего на выходе": len(cleaned),
        "Коллизий": collisions,
        "Удалено по стилям": dict(styles_removed),
    }
    return cleaned, stats


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "seed_names.json"

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned, stats = clean(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("📊 Статистика очистки:")
    for k, v in stats.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
