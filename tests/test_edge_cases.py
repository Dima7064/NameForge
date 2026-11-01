"""Краевые случаи: extremes параметров, невалидные входы."""
import pytest
import torch

from app.pipeline import Pipeline
from ml.classifier import save_classifier, train_classifier
from ml.lstm_gen import save_lstm, train_lstm
from ml.novelty import NoveltyScorer
from ml.plausibility import BigramPlausibility
from ml.transformer_gen import save_transformer, train_transformer


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
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
    lstm, itos, stoi, hist = train_lstm(data, data, vocab_data=data,
                                         epochs=3, verbose=False)
    save_lstm(lstm, itos, stoi, hist, tmp / "lstm.pt")

    trans, itos_t, stoi_t, hist_t = train_transformer(
        data, data, vocab_data=data, epochs=3, verbose=False,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )
    save_transformer(trans, itos_t, stoi_t, hist_t, tmp / "transformer.pt")

    cls, _, stoi_c, _ = train_classifier(data, data, vocab_data=data,
                                         epochs=3, verbose=False)
    save_classifier(cls, stoi_c, tmp / "classifier.pt")

    plaus = BigramPlausibility()
    plaus.train([d["name"] for d in data])
    nov = NoveltyScorer()
    nov.train([d["name"] for d in data])
    torch.save({
        "plausibility": {
            "smoothing": plaus.smoothing, "counts": plaus.counts,
            "context_counts": plaus.context_counts,
            "vocab": list(plaus.vocab),
            "mean_lp": plaus.mean_lp, "std_lp": plaus.std_lp,
        },
        "novelty": {
            "ngram_sizes": nov.ngram_sizes,
            "train_ngrams": nov.train_ngrams,
        },
    }, tmp / "scorers.pt")
    return tmp


def test_generate_min_count(models_dir):
    """count=1 — минимум."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=1, seed=42)
    assert len(results) <= 1


def test_generate_max_count(models_dir):
    """count=50 — максимум по ТЗ."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=50, seed=42)
    # Может быть меньше, если кандидатов не хватило — это не ошибка
    assert len(results) <= 50


def test_generate_syllables_1(models_dir):
    """1 слог — граница снизу."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="orcish", syllables=1, count=5, seed=42)
    for r in results:
        assert r.breakdown["syllables"] == 1


def test_generate_syllables_6(models_dir):
    """6 слогов — граница сверху."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=6, count=5, seed=42)
    # Может быть пусто на маленьком датасете — не падаем
    for r in results:
        assert r.breakdown["syllables"] == 6


def test_generate_min_temperature(models_dir):
    """temperature=0.1 — минимум."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=3, seed=42,
                         temperature=0.1)
    assert isinstance(results, list)


def test_generate_max_temperature(models_dir):
    """temperature=2.0 — максимум."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=3, seed=42,
                         temperature=2.0)
    assert isinstance(results, list)


def test_generate_top_k_off(models_dir):
    """top_k=0 — отключение top-k."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=3, seed=42,
                         top_k=0)
    assert isinstance(results, list)


def test_generate_top_p_off(models_dir):
    """top_p=1.0 — отключение nucleus."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=3, seed=42,
                         top_p=1.0)
    assert isinstance(results, list)


def test_generate_weights_zero_allowed(models_dir):
    """Веса могут быть нулевыми — тогда total считается без них."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=3, seed=42,
                         w_style=0.0, w_novelty=0.0, w_plaus=0.0)
    for r in results:
        assert r.scores["total"] == 0.0


def test_explain_empty_name(models_dir):
    """explain("") — не должно падать."""
    p = Pipeline(models_dir=models_dir)
    exp = p.explain("", "elven")
    assert exp["name"] == ""
    assert exp["syllable_count"] == 0


def test_explain_single_char(models_dir):
    """explain односимвольного имени."""
    p = Pipeline(models_dir=models_dir)
    exp = p.explain("А", "elven")
    assert exp["name"] == "А"


def test_explain_unknown_style(models_dir):
    """explain со странным стилем — все пробы, но style_score=0."""
    p = Pipeline(models_dir=models_dir)
    exp = p.explain("Аэлион", "unknown_style")
    assert exp["scores"]["style"] == 0.0


def test_generate_with_stats_counts(models_dir):
    """Воронка: все этапы логически упорядочены."""
    p = Pipeline(models_dir=models_dir)
    _, stats = p.generate_with_stats(style="elven", syllables=2, count=5, seed=42)
    assert stats["Всего сгенерировано"] >= stats["Прошли фильтр слогов"]
    assert stats["Прошли фильтр слогов"] >= stats["Уникальных"]
    assert stats["Уникальных"] >= stats["В финальный топ"]
    assert stats["В финальный топ"] <= 5


def test_generate_n_candidates_min(models_dir):
    """n_candidates=1 — минимум."""
    p = Pipeline(models_dir=models_dir)
    results = p.generate(style="elven", syllables=2, count=1, seed=42,
                         n_candidates=1)
    assert isinstance(results, list)


def test_generate_max_iterations_1(models_dir):
    """max_iterations=1 — только одна порция."""
    p = Pipeline(models_dir=models_dir)
    _, stats = p.generate_with_stats(
        style="elven", syllables=2, count=10, seed=42,
        n_candidates=10, max_iterations=1,
    )
    assert stats["Всего сгенерировано"] == 10
