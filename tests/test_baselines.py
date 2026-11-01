
"""Тесты для бейзлайнов: марков и Kneser-Ney."""
import random

import pytest

from ml.markov import MarkovModel
from ml.ngram import KneserNey


# ---------- марков ----------

def test_markov_rejects_bad_order():
    with pytest.raises(ValueError):
        MarkovModel(order=0)


def test_markov_generates_from_train_alphabet():
    names = ["Роланд", "Роберт", "Ричард"]
    m = MarkovModel(order=2)
    m.train(names)
    # allowed строится ИЗ ДАННЫХ, а не вручную: так тест не сломается,
    # если поменять список имён выше.
    allowed = set("".join(names))
    rng = random.Random(42)
    for _ in range(20):
        name = m.sample(rng)
        assert all(ch in allowed for ch in name), f"Символ не из train: {name!r}"


def test_markov_deterministic_with_seed():
    names = ["Аэлион", "Сильван", "Элен", "Лориэн", "Наэлир"]
    m = MarkovModel(order=3)
    m.train(names)
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    for _ in range(10):
        assert m.sample(rng1) == m.sample(rng2)


def test_markov_empty_for_unknown_context():
    # Обучен на одном имени — другой старт должен дать пустоту или backoff.
    m = MarkovModel(order=3)
    m.train(["Аб"])
    # Генерация не должна падать, даже если контекста не хватает
    rng = random.Random(0)
    result = m.sample(rng)
    assert isinstance(result, str)


# ---------- Kneser-Ney ----------

def test_kn_rejects_bad_order():
    with pytest.raises(ValueError):
        KneserNey(order=1)


def test_kn_unigram_sums_to_one():
    kn = KneserNey(order=3, discount=0.5)
    kn.train(["Аэлион", "Сильван", "Элен"])
    dist = kn.distribution("")  # униграмма
    total = sum(p for _, p in dist)
    assert abs(total - 1.0) < 1e-6


def test_kn_distribution_sums_to_one():
    kn = KneserNey(order=3, discount=0.5)
    kn.train(["Аэлион", "Сильван", "Элен"])
    for ctx in ["^^", "^А", "Аэ", "эл"]:
        dist = kn.distribution(ctx)
        total = sum(p for _, p in dist)
        assert abs(total - 1.0) < 1e-6, f"Сумма ≠ 1 для ctx={ctx!r}"


def test_kn_common_transition_higher_than_rare():
    # '^А' всегда идёт 'б'; 'в' не встречается вовсе.
    kn = KneserNey(order=3, discount=0.5)
    kn.train(["Аб", "Аб", "Аб"])
    p_ab = kn.prob("^А", "б")
    p_av = kn.prob("^А", "в")
    assert p_ab > p_av
    assert p_ab > 0.3  # раз 'б' всегда следует — вероятность должна быть высокой


def test_kn_deterministic_with_seed():
    names = ["Аэлион", "Сильван", "Элен", "Лориэн", "Наэлир"]
    kn = KneserNey(order=3)
    kn.train(names)
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    for _ in range(10):
        assert kn.sample(rng1) == kn.sample(rng2)


def test_kn_generates_from_train_alphabet():
    names = ["Роланд", "Роберт", "Ричард"]
    kn = KneserNey(order=3)
    kn.train(names)
    allowed = set("".join(names))
    rng = random.Random(42)
    for _ in range(20):
        name = kn.sample(rng)
        assert all(ch in allowed for ch in name), f"Символ не из train: {name!r}"