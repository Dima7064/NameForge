
"""Генерация отчёта о датасете NameForge.

Пишет docs/data_report.md с таблицами: распределение по стилям,
средняя длина / слоги, паттерн CV, топ-5 длинных имён, чек-лист.
"""
import json
from collections import Counter
from pathlib import Path

VOWELS = set("аеёиоуыэюя")
CONSONANTS = set("бвгджзйклмнпрстфхцчшщ")


def count_syllables(name: str) -> int:
    """Число слогов = число гласных букв (русский принцип слогоделения).

    В русском языке каждая гласная — отдельный слог, дифтонгов почти нет.
    Поэтому «аэ» — это два слога, а не один.

    Примеры:
        Роланд       → 2 (о, а)
        Аэлиониэль   → 6 (а, э, и, о, и, э)
        Торгрим      → 2 (о, и)
        Тхарг        → 1 (а)
    """
    return sum(1 for ch in name.lower() if ch in VOWELS)


def cv_pattern(name: str) -> str:
    """Паттерн C/V. ь, ъ, дефисы и апострофы игнорируются.

    ь и ъ — модификаторы предыдущей согласной, они не несут
    собственного звука, поэтому не считаются ни C, ни V.
    """
    result = []
    for ch in name.lower():
        if ch in VOWELS:
            result.append("V")
        elif ch in CONSONANTS:
            result.append("C")
    return "-".join(result)


def generate(data: list[dict], out_path: Path) -> None:
    by_style = Counter(e["style"] for e in data)
    total = len(data)

    lines = ["# Отчёт по датасету NameForge", ""]
    lines.append(f"**Всего имён:** {total}")
    lines.append("")
    lines.append("## Распределение по стилям")
    lines.append("")
    lines.append("| Стиль | Кол-во | Доля |")
    lines.append("|-------|--------|------|")
    for style in sorted(by_style):
        count = by_style[style]
        lines.append(f"| {style} | {count} | {count / total:.1%} |")
    lines.append("")

    lines.append("## Длина и слоги по стилям")
    lines.append("")
    lines.append("| Стиль | Ср. длина | Мин–макс длина | Ср. слоги | Мин–макс слоги |")
    lines.append("|-------|-----------|----------------|-----------|----------------|")
    for style in sorted(by_style):
        items = [e["name"] for e in data if e["style"] == style]
        lengths = [len(n) for n in items]
        syllables = [count_syllables(n) for n in items]
        lines.append(
            f"| {style} | {sum(lengths) / len(lengths):.1f} | "
            f"{min(lengths)}–{max(lengths)} | "
            f"{sum(syllables) / len(syllables):.1f} | "
            f"{min(syllables)}–{max(syllables)} |"
        )
    lines.append("")
    lines.append(
        "> Замечание: слоги считаются как число гласных букв. "
        "Некоторые имена имеют 1 слог (например, орочье «Тхарг») "
        "или 6 слогов («Аэлиониэль») — это влияет на диапазон ползунка в UI."
    )
    lines.append("")

    lines.append("## Паттерн CV — примеры")
    lines.append("")
    for style in sorted(by_style):
        sample = [e["name"] for e in data if e["style"] == style][:3]
        examples = ", ".join(f"`{n}` ({cv_pattern(n)})" for n in sample)
        lines.append(f"- **{style}**: {examples}")
    lines.append("")

    lines.append("## Топ-5 самых длинных имён")
    lines.append("")
    top = sorted(data, key=lambda e: -len(e["name"]))[:5]
    for e in top:
        lines.append(f"- `{e['name']}` — {e['style']}, {len(e['name'])} символов")
    lines.append("")

    lines.append("## Чек-лист")
    lines.append("")
    lines.append(f"- [x] Всего ≥ 50 имён: **{total >= 50}** ({total})")
    lines.append(f"- [x] Все 5 стилей представлены: **{len(by_style) == 5}**")
    lines.append(
        f"- [x] Минимум 10 имён на стиль: "
        f"**{all(c >= 10 for c in by_style.values())}**"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт записан: {out_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    with (root / "data" / "names.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    generate(data, root / "docs" / "data_report.md")