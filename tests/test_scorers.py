
"""Тесты для скоринг-моделей: classifier, plausibility, novelty."""
import torch

from ml.classifier import (
    StyleClassifier, build_char_vocab, encode_for_classifier,
    predict_proba, predict_style, train_classifier,
)
from ml.lstm_gen import STYLES
from ml.novelty import NoveltyScorer
from ml.plausibility import BigramPlausibility


# ---------- classifier: словарь ----------

def test_build_char_vocab():
    itos, stoi = build_char_vocab(["Аб", "Ва"])
    assert itos[0] == "<PAD>"
    for ch in "АбВа":
        assert ch in stoi


def test_encode_for_classifier():
    _, stoi = build_char_vocab(["Аб"])
    encoded = encode_for_classifier("Аб", stoi)
    assert len(encoded) == 2
    assert all(isinstance(i, int) for i in encoded)


def test_encode_oov_becomes_pad():
    _, stoi = build_char_vocab(["Аб"])
    encoded = encode_for_classifier("Аз", stoi)  # 'з' вне словаря
    assert encoded[1] == 0  # PAD


# ---------- classifier: forward ----------

def test_classifier_forward_shape():
    # ВАЖНО: индексы в x должны быть < vocab_size. Строим словарь
    # и берём индексы из него, а не «магические» числа — иначе IndexError.
    itos, stoi = build_char_vocab(["Абв"])
    model = StyleClassifier(vocab_size=len(itos))
    # encode_for_classifier сам подставит корректные индексы; PAD в конце
    # — имитация батча с паддингом.
    encoded = encode_for_classifier("Абв", stoi)
    x = torch.tensor([encoded + [0]])
    logits = model(x)
    assert logits.shape == (1, 5)  # 5 стилей


# ---------- classifier: обучение ----------

def _tiny_data():
    return [
        {"name": "Аэлион", "style": "elven"},
        {"name": "Сильван", "style": "elven"},
        {"name": "Роланд", "style": "human"},
        {"name": "Роберт", "style": "human"},
        {"name": "Грохар", "style": "orcish"},
        {"name": "Узгаш", "style": "orcish"},
        {"name": "Торгрим", "style": "northern"},
        {"name": "Бьорнар", "style": "northern"},
        {"name": "Санчо", "style": "southern"},
        {"name": "Лукано", "style": "southern"},
    ]


def test_classifier_train_runs():
    data = _tiny_data()
    model, itos, stoi, hist = train_classifier(
        data, data, epochs=5, lr=1e-2, batch_size=4, patience=100,
        verbose=False,
    )
    assert len(hist["train_acc"]) <= 5
    assert all(0 <= acc <= 1 for acc in hist["train_acc"])


def test_classifier_predict_proba():
    data = _tiny_data()
    model, itos, stoi, _ = train_classifier(
        data, data, epochs=5, lr=1e-2, batch_size=4, patience=100,
        verbose=False,
    )
    probs = predict_proba(model, stoi, "Аэлион")
    assert set(probs.keys()) == set(STYLES)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_classifier_predict_style():
    data = _tiny_data()
    model, itos, stoi, _ = train_classifier(
        data, data, epochs=5, lr=1e-2, batch_size=4, patience=100,
        verbose=False,
    )
    style = predict_style(model, stoi, "Аэлион")
    assert style in STYLES


# ---------- plausibility ----------

def test_plausibility_real_higher_than_garbage():
    """Реальное имя должно получить больший score, чем случайный набор."""
    p = BigramPlausibility()
    p.train(["Аэлион", "Сильван", "Эленвэн", "Роланд", "Роберт"])
    real = p.score("Аэлион")
    garbage = p.score("Щъъщщъ")
    assert real > garbage, f"real={real:.2f}, garbage={garbage:.2f}"


def test_plausibility_score_in_range():
    p = BigramPlausibility()
    p.train(["Аэлион", "Сильван", "Эленвэн"])
    for name in ["Аэлион", "Роланд", "Щъъ", ""]:
        s = p.score(name)
        assert 0.0 <= s <= 1.0


def test_plausibility_empty_returns_zero():
    p = BigramPlausibility()
    p.train(["Аэлион"])
    assert p.score("") == 0.0


# ---------- novelty ----------

def test_novelty_exact_copy_is_zero():
    n = NoveltyScorer()
    n.train(["Аэлион", "Сильван", "Роланд"])
    assert n.score("Аэлион") == 0.0


def test_novelty_garbage_is_high():
    n = NoveltyScorer()
    n.train(["Аэлион", "Сильван", "Роланд"])
    s = n.score("Щъыъй")
    assert s > 0.8, f"Ожидали >0.8, получили {s:.2f}"


def test_novelty_in_range():
    n = NoveltyScorer()
    n.train(["Аэлион", "Сильван"])
    for name in ["Аэлион", "Сильван", "Совсемновое", "Аэл"]:
        s = n.score(name)
        assert 0.0 <= s <= 1.0


def test_novelty_empty_train_returns_one():
    n = NoveltyScorer()
    assert n.score("Всё") == 1.0