"""Примеры оценок скореров на сгенерированных именах.

Запуск (из корня проекта):
    python -m ml.run_scorers

Результат: docs/scorers_examples.md
"""
import json
from pathlib import Path

import torch

from ml.classifier import load_classifier, predict_proba
from ml.lstm_gen import STYLES, load_lstm, sample
from ml.novelty import NoveltyScorer
from ml.plausibility import BigramPlausibility
from ml.report import count_syllables

SEED = 42
N_PER_STYLE = 5
SYLLABLE_RANGE = (2, 6)
MAX_ATTEMPTS = 200


def _load_scorers(root: Path):
    """Загружает classifier и восстанавливает plausibility/novelty из scorers.pt."""
    classifier, stoi = load_classifier(root / "models" / "classifier.pt")
    ckpt = torch.load(root / "models" / "scorers.pt",
                      map_location="cpu", weights_only=False)

    p_data = ckpt["plausibility"]
    plausibility = BigramPlausibility(smoothing=p_data["smoothing"])
    plausibility.counts = p_data["counts"]
    plausibility.context_counts = p_data["context_counts"]
    plausibility.vocab = set(p_data["vocab"])
    plausibility.mean_lp = p_data["mean_lp"]
    plausibility.std_lp = p_data["std_lp"]

    n_data = ckpt["novelty"]
    novelty = NoveltyScorer(ngram_sizes=n_data["ngram_sizes"])
    novelty.train_ngrams = n_data["train_ngrams"]

    return classifier, stoi, plausibility, novelty


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    lstm_path = root / "models" / "lstm.pt"
    classifier_path = root / "models" / "classifier.pt"
    scorers_path = root / "models" / "scorers.pt"

    if not all(p.exists() for p in [lstm_path, classifier_path, scorers_path]):
        print("Сначала запусти: python -m ml.train_lstm и python -m ml.train_scorers")
        return

    lstm, itos, stoi_lstm = load_lstm(lstm_path)
    classifier, stoi_cls, plausibility, novelty = _load_scorers(root)

    lines = [
        "# Оценки скореров на примерах LSTM",
        "",
        f"Seed: `{SEED}`. По {N_PER_STYLE} имён на стиль. "
        "Каждое имя — с тремя скорами.",
        "",
        "Формула итогового скора: `0.5·style + 0.25·novelty + 0.25·plausibility`.",
        "",
    ]

    for style in STYLES:
        lines.append(f"## {style}")
        lines.append("")
        lines.append("| Имя | Слоги | Style (target) | Novelty | Plaus | Total |")
        lines.append("|---|---|---|---|---|---|")

        generated: list[str] = []
        for attempt in range(N_PER_STYLE * MAX_ATTEMPTS):
            if len(generated) >= N_PER_STYLE:
                break
            name = sample(lstm, itos, stoi_lstm, style=style, seed=SEED + attempt)
            if not name:
                continue
            name = name[0].upper() + name[1:].lower()
            syl = count_syllables(name)
            if SYLLABLE_RANGE[0] <= syl <= SYLLABLE_RANGE[1] and name not in generated:
                generated.append(name)

        if not generated:
            lines.append("| _(нет имён в диапазоне слогов)_ | — | — | — | — | — |")
            lines.append("")
            continue

        for name in generated:
            probs = predict_proba(classifier, stoi_cls, name)
            style_score = probs[style]
            nov = novelty.score(name)
            pl = plausibility.score(name)
            total = 0.5 * style_score + 0.25 * nov + 0.25 * pl
            lines.append(
                f"| `{name}` | {count_syllables(name)} | "
                f"{style_score:.2f} | {nov:.2f} | {pl:.2f} | **{total:.2f}** |"
            )
        lines.append("")

    out = root / "docs" / "scorers_examples.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Записано: {out}")


if __name__ == "__main__":
    main()
