"""Генерация примеров из бейзлайнов для отчёта.

Запуск (из корня проекта):
    python -m ml.run_baselines
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from ml.markov import MarkovModel
from ml.ngram import KneserNey
from ml.report import count_syllables

SEED = 42
N_PER_STYLE = 10
SYLLABLE_RANGE = (2, 6)
MAX_ATTEMPTS_PER_NAME = 200


def load_by_style(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_style: dict[str, list[str]] = defaultdict(list)
    for e in data:
        by_style[e["style"]].append(e["name"])
    return dict(by_style)


def gen_filtered(model, style: str, n: int, syllable_range: tuple[int, int],
                 max_attempts_per_name: int = MAX_ATTEMPTS_PER_NAME) -> list[str]:
    """Генерирует имена, отбрасывая те, что не попадают в диапазон слогов."""
    rng = random.Random(f"{SEED}_{style}")
    result = []
    for _ in range(n * max_attempts_per_name):
        if len(result) >= n:
            break
        name = model.sample(rng)
        if not name:
            continue
        name = name[0].upper() + name[1:].lower()
        syl = count_syllables(name)
        if syllable_range[0] <= syl <= syllable_range[1] and name not in result:
            result.append(name)
    return result


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    by_style = load_by_style(root / "data" / "train.json")

    lines = [
        "# Примеры имён от бейзлайнов",
        "",
        f"Seed: `{SEED}`. Диапазон слогов: {SYLLABLE_RANGE[0]}–{SYLLABLE_RANGE[1]}. "
        f"По {N_PER_STYLE} имён на модель.",
        "",
        "> Сгенерировано на train-датасете (20 имён на стиль). "
        "При расширении датасета качество вырастет.",
        "",
    ]

    for style in sorted(by_style):
        names = by_style[style]
        lines.append(f"## {style} ({len(names)} имён в train)")
        lines.append("")

        configs = [
            ("Марков (order=2)", MarkovModel(order=2)),
            ("Марков (order=3)", MarkovModel(order=3)),
            ("Kneser-Ney (order=3)", KneserNey(order=3, discount=0.75)),
            ("Kneser-Ney (order=4)", KneserNey(order=4, discount=0.75)),
        ]
        for label, model in configs:
            model.train(names)
            generated = gen_filtered(model, style, N_PER_STYLE, SYLLABLE_RANGE)
            if generated:
                lines.append(f"**{label}:** {', '.join(generated)}")
            else:
                lines.append(f"**{label}:** _(ни одного имени не попало в диапазон слогов)_")
        lines.append("")

    out = root / "docs" / "baselines_examples.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Записано: {out}")


if __name__ == "__main__":
    main()
