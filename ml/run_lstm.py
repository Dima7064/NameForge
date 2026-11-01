
"""Примеры LSTM по стилям и полам → docs/lstm_examples.md."""
from pathlib import Path

from ml.lstm_gen import STYLES, load_lstm, sample
from ml.report import count_syllables

SEED = 42
N_PER = 5
SYLLABLE_RANGE = (2, 6)
MAX_ATTEMPTS = 300


def gen_one_gender(model, itos, stoi, style, gender):
    out = []
    for attempt in range(N_PER * MAX_ATTEMPTS):
        if len(out) >= N_PER:
            break
        name = sample(model, itos, stoi, style=style, gender=gender,
                      seed=SEED + attempt)
        if not name:
            continue
        name = name[0].upper() + name[1:].lower()
        syl = count_syllables(name)
        if SYLLABLE_RANGE[0] <= syl <= SYLLABLE_RANGE[1] and name not in out:
            out.append(name)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ckpt = root / "models" / "lstm.pt"
    if not ckpt.exists():
        print(f"Нет {ckpt}. Запусти python -m ml.train_lstm")
        return

    model, itos, stoi = load_lstm(ckpt)

    lines = [
        "# Примеры LSTM (муж / жен)", "",
        f"Seed: `{SEED}`. По {N_PER} на пол × стиль.", "",
    ]
    for style in STYLES:
        for gender in ("male", "female"):
            names = gen_one_gender(model, itos, stoi, style, gender)
            tag = "♂" if gender == "male" else "♀"
            if names:
                lines.append(f"**{style} {tag}:** {', '.join(names)}")
            else:
                lines.append(f"**{style} {tag}:** _(нет в диапазоне)_")
        lines.append("")

    out = root / "docs" / "lstm_examples.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Записано: {out}")


if __name__ == "__main__":
    main()