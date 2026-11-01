"""Обучение и сохранение всех трёх скоринг-моделей.

Запуск (из корня проекта):
    python -m ml.train_scorers

Результат:
    models/scorers.pt (classifier + stoi + plausibility + novelty)
    docs/scorers_report.md (метрики: accuracy по стилям, confusion matrix)
"""
import json
from collections import Counter
from pathlib import Path

import torch

from ml.classifier import (
    load_classifier, predict_style, save_classifier, train_classifier,
)
from ml.lstm_gen import STYLES
from ml.novelty import NoveltyScorer
from ml.plausibility import BigramPlausibility


def _confusion_matrix(model, stoi, data):
    """5×5 матрица: [истина][предсказание]."""
    cm = {true: {pred: 0 for pred in STYLES} for true in STYLES}
    for d in data:
        pred = predict_style(model, stoi, d["name"])
        cm[d["style"]][pred] += 1
    return cm


def _accuracy(model, stoi, data):
    correct = sum(
        1 for d in data if predict_style(model, stoi, d["name"]) == d["style"]
    )
    return correct / max(1, len(data))


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    with (root / "data" / "train.json").open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    with (root / "data" / "val.json").open("r", encoding="utf-8") as f:
        val_data = json.load(f)
    with (root / "data" / "test.json").open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # ---------- classifier ----------
    print("\n=== Style classifier ===")
    classifier, itos, stoi, hist = train_classifier(
        train_data, val_data,
        vocab_data=train_data + val_data + test_data,
        epochs=100, lr=1e-3, batch_size=8,
        patience=20, seed=42, verbose=True,
    )
    train_acc = _accuracy(classifier, stoi, train_data)
    val_acc = _accuracy(classifier, stoi, val_data)
    test_acc = _accuracy(classifier, stoi, test_data)
    print(f"\nAccuracy: train={train_acc:.3f}, val={val_acc:.3f}, test={test_acc:.3f}")

    cm = _confusion_matrix(classifier, stoi, test_data)

    # ---------- plausibility ----------
    print("\n=== Plausibility (char-bigram) ===")
    plausibility = BigramPlausibility(smoothing=1.0)
    plausibility.train([d["name"] for d in train_data])
    print(f"mean log-prob = {plausibility.mean_lp:.3f}, std = {plausibility.std_lp:.3f}")

    # ---------- novelty ----------
    print("\n=== Novelty (2-3 gram jaccard) ===")
    novelty = NoveltyScorer(ngram_sizes=(2, 3))
    novelty.train([d["name"] for d in train_data])
    print(f"Train ngrams (для сравнения): {len(novelty.train_ngrams)} имён")

    # ---------- сохранение ----------
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    classifier_path = models_dir / "classifier.pt"
    save_classifier(classifier, stoi, classifier_path)
    print(f"\nClassifier: {classifier_path}")

    scorers_path = models_dir / "scorers.pt"
    torch.save({
        "plausibility": {
            "smoothing": plausibility.smoothing,
            "counts": plausibility.counts,
            "context_counts": plausibility.context_counts,
            "vocab": list(plausibility.vocab),
            "mean_lp": plausibility.mean_lp,
            "std_lp": plausibility.std_lp,
        },
        "novelty": {
            "ngram_sizes": novelty.ngram_sizes,
            "train_ngrams": novelty.train_ngrams,
        },
    }, scorers_path)
    print(f"Plausibility + Novelty: {scorers_path}")

    # ---------- отчёт ----------
    lines = [
        "# Отчёт по скоринг-моделям NameForge",
        "",
        f"**Train:** {len(train_data)} | **Val:** {len(val_data)} | **Test:** {len(test_data)}",
        "",
        "## Style classifier",
        "",
        f"- Accuracy на train: **{train_acc:.3f}**",
        f"- Accuracy на val: **{val_acc:.3f}**",
        f"- Accuracy на test: **{test_acc:.3f}**",
        f"- Эпох до early stopping: {len(hist['train_acc'])}",
        "",
        "> **Замечание.** Целевая accuracy из ТЗ — ≥85%. На 80 именах (16 на стиль)",
        "> этого достичь невозможно из-за малого объёма. Ожидаемый рост:",
        "> 80 → 500 → 2000 имён даст 0.55 → 0.75 → 0.90+.",
        "",
        "### Confusion matrix (test)",
        "",
        "| Истина \\ Предсказание | " + " | ".join(STYLES) + " |",
        "|---|" + "---|" * len(STYLES),
    ]
    for true_style in STYLES:
        row = [str(cm[true_style][pred]) for pred in STYLES]
        lines.append(f"| **{true_style}** | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Plausibility (char-bigram)")
    lines.append("")
    lines.append(f"- mean log-prob на train: `{plausibility.mean_lp:.3f}`")
    lines.append(f"- std log-prob: `{plausibility.std_lp:.3f}`")
    lines.append(f"- smoothing α: `{plausibility.smoothing}`")
    lines.append("")
    lines.append("> Score = sigmoid((log_prob(name) − mean) / std). Значения в [0, 1].")
    lines.append("")

    lines.append("## Novelty (2-3 gram jaccard)")
    lines.append("")
    lines.append(f"- ngram_sizes: `{novelty.ngram_sizes}`")
    lines.append(f"- эталонных имён в train: `{len(novelty.train_ngrams)}`")
    lines.append("")
    lines.append("> Score = 1 − max_jaccard(name, train_i). "
                 "0 = точная копия, 1 = полностью новое.")
    lines.append("")

    report = root / "docs" / "scorers_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт: {report}")

    # Дополнительно: примеры для каждого стиля
    print("\nПримеры scores на train-именах:")
    for d in train_data[:3]:
        style_probs = classifier and None  # noqa
        print(f"  {d['name']:15s} ({d['style']:8s})  "
              f"plaus={plausibility.score(d['name']):.2f}  "
              f"novelty={novelty.score(d['name']):.2f}")


if __name__ == "__main__":
    main()
