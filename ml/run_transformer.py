"""Генерация примеров из Transformer для отчёта (docs/transformer_examples.md).

Запуск (из корня проекта):
    python -m ml.run_transformer
"""
from pathlib import Path

from ml.report import count_syllables
from ml.transformer_gen import STYLES, load_transformer, sample

SEED = 42
N_PER_STYLE = 10
SYLLABLE_RANGE = (2, 6)
MAX_ATTEMPTS = 300


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ckpt = root / "models" / "transformer.pt"
    if not ckpt.exists():
        print(f"Модель не найдена: {ckpt}. Сначала запусти python -m ml.train_transformer")
        return

    model, itos, stoi = load_transformer(ckpt)

    lines = [
        "# Примеры имён от Transformer",
        "",
        f"Seed: `{SEED}`. Диапазон слогов: {SYLLABLE_RANGE[0]}–{SYLLABLE_RANGE[1]}. "
        f"По {N_PER_STYLE} имён на стиль.",
        "",
        "Параметры сэмплирования: temperature=0.8, top-k=40, top-p=0.9.",
        "",
    ]
    for style in STYLES:
        generated: list[str] = []
        for attempt in range(N_PER_STYLE * MAX_ATTEMPTS):
            if len(generated) >= N_PER_STYLE:
                break
            name = sample(
                model, itos, stoi, style=style,
                temperature=0.8, top_k=40, top_p=0.9,
                seed=SEED + attempt,
            )
            if not name:
                continue
            name = name[0].upper() + name[1:].lower()
            syl = count_syllables(name)
            if SYLLABLE_RANGE[0] <= syl <= SYLLABLE_RANGE[1] and name not in generated:
                generated.append(name)
        if generated:
            lines.append(f"**{style}:** {', '.join(generated)}")
        else:
            lines.append(f"**{style}:** _(ни одного имени не попало в диапазон слогов)_")
        lines.append("")

    out = root / "docs" / "transformer_examples.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Записано: {out}")


if __name__ == "__main__":
    main()
