"""Тесты для пайплайна (generate → filter → score → rank).

Все тесты используют крошечные модели, обученные в фикстуре, чтобы
не зависеть от реальных обученных чекпоинтов. Это делает тесты
изолированными и быстрыми.
"""
import pytest
import torch

from app.pipeline import Pipeline, ScoredName
from ml.classifier import save_classifier, train_classifier
from ml.lstm_gen import STYLES, save_lstm, train_lstm
from ml.novelty import NoveltyScorer
from ml.plausibility import BigramPlausibility
from ml.transformer_gen import save_transformer, train_transformer


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    """Обучает крошечные модели на мини-датасете и сохраняет их в tmp."""
    tmp = tmp_path_factory.mktemp("models")

    data = [
        {"name": "Аэлион", "style": "elven"},
        {"name": "Сильван", "style": "elven"},
        {"name": "Элен", "style": "elven"},
        {"name": "Роланд", "style": "human"},
        {"name": "Роберт", "style": "human"},
        {"name": "Эдвин", "style": "human"},
        {"name": "Грохар", "style": "orcish"},
        {"name": "Узгаш", "style": "orcish"},
        {"name": "Могра", "style": "orcish"},
        {"name": "Торгрим", "style": "northern"},
        {"name": "Бьорнар", "style": "northern"},
        {"name": "Ивар", "style": "northern"},
        {"name": "Санчо", "style": "southern"},
        {"name": "Лукано", "style": "southern"},
        {"name": "Беллина", "style": "southern"},
    ]

    # LSTM
    lstm, itos, stoi, hist = train_lstm(
        data, data, vocab_data=data, epochs=3, verbose=False,
    )
    save_lstm(lstm, itos, stoi, hist, tmp / "lstm.pt")

    # Transformer
    trans, itos_t, stoi_t, hist_t = train_transformer(
        data, data, vocab_data=data, epochs=3, verbose=False,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )
    save_transformer(trans, itos_t, stoi_t, hist_t, tmp / "transformer.pt")

    # Classifier
    cls, _, stoi_c, _ = train_classifier(
        data, data, vocab_data=data, epochs=3, verbose=False,
    )
    save_classifier(cls, stoi_c, tmp / "classifier.pt")

    # Scorers
    plaus = BigramPlausibility()
    plaus.train([d["name"] for d in data])
    nov = NoveltyScorer()
    nov.train([d["name"] for d in data])

    torch.save({
        "plausibility": {
            "smoothing": plaus.smoothing,
            "counts": plaus.counts,
            "context_counts": plaus.context_counts,
            "vocab": list(plaus.vocab),
            "mean_lp": plaus.mean_lp,
            "std_lp": plaus.std_lp,
        },
        "novelty": {
            "ngram_sizes": nov.ngram_sizes,
            "train_ngrams": nov.train_ngrams,
        },
    }, tmp / "scorers.pt")

    return tmp


# ---------- инициализация ----------

def test_pipeline_loads_lstm(models_dir):
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    assert p.model_type == "lstm"


def test_pipeline_loads_transformer(models_dir):
    p = Pipeline(model_type="transformer", models_dir=models_dir)
    assert p.model_type == "transformer"


def test_pipeline_rejects_bad_model_type(models_dir):
    with pytest.raises(ValueError):
        Pipeline(model_type="banana", models_dir=models_dir)


def test_pipeline_fails_on_missing_models(tmp_path):
    with pytest.raises(FileNotFoundError):
        Pipeline(model_type="lstm", models_dir=tmp_path)


# ---------- валидация параметров ----------

def test_generate_rejects_bad_style(models_dir):
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    with pytest.raises(ValueError):
        p.generate(style="unknown", syllables=2, count=3, seed=0)


def test_generate_rejects_bad_syllables(models_dir):
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    with pytest.raises(ValueError):
        p.generate(style="elven", syllables=0, count=3, seed=0)
    with pytest.raises(ValueError):
        p.generate(style="elven", syllables=11, count=3, seed=0)


def test_generate_rejects_bad_count(models_dir):
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    with pytest.raises(ValueError):
        p.generate(style="elven", syllables=2, count=0, seed=0)
    with pytest.raises(ValueError):
        p.generate(style="elven", syllables=2, count=51, seed=0)


def test_generate_rejects_bad_temperature(models_dir):
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    with pytest.raises(ValueError):
        p.generate(style="elven", syllables=2, count=3, seed=0,
                   temperature=0.0)
    with pytest.raises(ValueError):
        p.generate(style="elven", syllables=2, count=3, seed=0,
                   temperature=2.5)


# ---------- поведение generate ----------

def test_generate_returns_list(models_dir):
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=3, seed=42)
    assert isinstance(results, list)
    assert all(isinstance(r, ScoredName) for r in results)
    assert len(results) <= 3


def test_generate_respects_syllables(models_dir):
    """Все сгенерированные имена должны иметь ровно запрошенное число слогов."""
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=5, seed=42)
    for r in results:
        assert r.breakdown["syllables"] == 2, (
            f"{r.name}: {r.breakdown['syllables']} слогов, ожидали 2"
        )


def test_generate_scores_in_range(models_dir):
    """Все скоры должны быть в [0, 1]."""
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=5, seed=42)
    for r in results:
        for key in ("style", "novelty", "plausibility", "total"):
            assert 0.0 <= r.scores[key] <= 1.0, (
                f"{r.name} {key}={r.scores[key]}"
            )


def test_generate_deterministic_with_seed(models_dir):
    """С одинаковым seed — идентичный результат."""
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    a = p.generate(style="elven", syllables=2, count=3, seed=123)
    b = p.generate(style="elven", syllables=2, count=3, seed=123)
    assert [r.name for r in a] == [r.name for r in b]
    for ra, rb in zip(a, b):
        assert ra.scores == rb.scores


def test_generate_differs_with_different_seed(models_dir):
    """С разными seed — имена различаются (хотя бы одно)."""
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    a = p.generate(style="elven", syllables=2, count=3, seed=1)
    b = p.generate(style="elven", syllables=2, count=3, seed=999)
    # Не требуем полного различия (на 15 именах может совпасть 1-2),
    # но хотя бы одно должно отличаться.
    names_a = {r.name for r in a}
    names_b = {r.name for r in b}
    assert names_a != names_b


def test_generate_sorted_by_total(models_dir):
    """Результат отсортирован по total (убыв.)."""
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=5, seed=42)
    totals = [r.scores["total"] for r in results]
    assert totals == sorted(totals, reverse=True)


def test_scored_name_has_breakdown(models_dir):
    """У ScoredName заполнены все поля breakdown."""
    p = Pipeline(model_type="lstm", models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=1, seed=42)
    if results:
        r = results[0]
        assert "syllables" in r.breakdown
        assert "pattern" in r.breakdown
        assert "style_probs" in r.breakdown
        assert set(r.breakdown["style_probs"].keys()) == set(STYLES)
