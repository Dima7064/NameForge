
"""Бенчмарк 4 моделей × 5 метрик для отчёта на защите.

Модели загружаются один раз и кэшируются в _CACHE.
"""
import json
import math
from collections import Counter
from pathlib import Path

from ml.classifier import load_classifier, predict_proba
from ml.lstm_gen import STYLES, load_lstm
from ml.lstm_gen import sample as lstm_sample
from ml.markov import MarkovModel
from ml.ngram import KneserNey
from ml.report import count_syllables
from ml.transformer_gen import load_transformer
from ml.transformer_gen import sample as transformer_sample

_CACHE: dict = {}


def _get_models(models_dir: Path, data_dir: Path):
    key = (str(models_dir), str(data_dir))
    if key in _CACHE:
        return _CACHE[key]
    with (data_dir / "train.json").open(encoding="utf-8") as f:
        train = json.load(f)
    train_by_style: dict[str, list[str]] = {}
    for d in train:
        train_by_style.setdefault(d["style"], []).append(d["name"])
    train_names = {d["name"].lower() for d in train}

    lstm_model, itos_l, stoi_l = load_lstm(models_dir / "lstm.pt")
    trans_model, itos_t, stoi_t = load_transformer(models_dir / "transformer.pt")
    classifier, stoi_c = load_classifier(models_dir / "classifier.pt")

    with (models_dir / "lstm_history.json").open(encoding="utf-8") as f:
        lstm_h = json.load(f)
    with (models_dir / "transformer_history.json").open(encoding="utf-8") as f:
        trans_h = json.load(f)

    ppl_map = {
        "LSTM": lstm_h["val_ppl"][-1],
        "Transformer": trans_h["val_ppl"][-1],
    }
    payload = {
        "train_names": train_names,
        "train_by_style": train_by_style,
        "lstm": (lstm_model, itos_l, stoi_l),
        "trans": (trans_model, itos_t, stoi_t),
        "classifier": (classifier, stoi_c),
        "ppl": ppl_map,
    }
    _CACHE[key] = payload
    return payload


def _entropy(names: list[str]) -> float:
    counter = Counter(c for n in names for c in n.lower())
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def benchmark(models_dir: Path, data_dir: Path,
              n_per_style: int = 5, seed: int = 42) -> dict[str, dict]:
    ctx = _get_models(models_dir, data_dir)
    train_names = ctx["train_names"]
    train_by_style = ctx["train_by_style"]
    lstm_model, itos_l, stoi_l = ctx["lstm"]
    trans_model, itos_t, stoi_t = ctx["trans"]
    classifier, stoi_c = ctx["classifier"]
    ppl_map = ctx["ppl"]

    import random

    def gen_markov(style: str, s: int) -> str:
        m = MarkovModel(order=2)
        m.train(train_by_style[style])
        return m.sample(random.Random(s))

    def gen_kn(style: str, s: int) -> str:
        kn = KneserNey(order=3)
        kn.train(train_by_style[style])
        return kn.sample(random.Random(s))

    def gen_lstm(style: str, s: int) -> str:
        return lstm_sample(lstm_model, itos_l, stoi_l, style, seed=s)

    def gen_trans(style: str, s: int) -> str:
        return transformer_sample(trans_model, itos_t, stoi_t, style, seed=s)

    configs = [
        ("Марков (order=2)", gen_markov),
        ("Kneser-Ney (order=3)", gen_kn),
        ("LSTM", gen_lstm),
        ("Transformer", gen_trans),
    ]

    results: dict[str, dict] = {}
    for model_name, gen_fn in configs:
        pairs: list[tuple[str, str]] = []
        for style in STYLES:
            generated: set[str] = set()
            attempts = 0
            while len(generated) < n_per_style and attempts < n_per_style * 20:
                try:
                    nm = gen_fn(style, seed + attempts)
                except Exception:
                    break
                attempts += 1
                if not nm:
                    continue
                nm = nm[0].upper() + nm[1:].lower()
                if not (1 <= count_syllables(nm) <= 6):
                    continue
                if nm.lower() in generated:
                    continue
                generated.add(nm.lower())
                pairs.append((nm, style))

        if not pairs:
            results[model_name] = {
                "perplexity": None, "novelty": 0.0,
                "diversity": 0.0, "style": 0.0, "count": 0,
            }
            continue

        novelty = sum(
            1 for n, _ in pairs if n.lower() not in train_names
        ) / len(pairs)
        diversity = _entropy([n for n, _ in pairs])
        style_score = sum(
            predict_proba(classifier, stoi_c, n).get(st, 0.0)
            for n, st in pairs
        ) / len(pairs)
        results[model_name] = {
            "perplexity": ppl_map.get(model_name),
            "novelty": novelty,
            "diversity": diversity,
            "style": style_score,
            "count": len(pairs),
        }
    return results


def to_markdown(results: dict) -> str:
    lines = [
        "| Модель | Perplexity ↓ | Novelty ↑ | Diversity ↑ | Style ↑ | Human |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        ppl = f"{m['perplexity']:.2f}" if m["perplexity"] else "—"
        lines.append(
            f"| **{name}** | {ppl} | {m['novelty']:.0%} | "
            f"{m['diversity']:.2f} bit | {m['style']:.0%} | — |"
        )
    lines.append("")
    lines.append(
        "_PPL — из истории обучения (NN) или отсутствует (baselines). "
        "Novelty — % имён вне train. Diversity — Shannon entropy символов. "
        "Style — средняя вероятность целевого стиля от classifier._"
    )
    return "\n".join(lines)