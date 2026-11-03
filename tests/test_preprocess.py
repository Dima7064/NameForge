
"""Базовые тесты для препроцессинга, сплита, слогов и транслитерации."""
import pytest

from ml.preprocess import normalize, preprocess
from ml.report import count_syllables, cv_pattern
from ml.split import split
from ml.transliterate import transliterate


# ---------- normalize ----------
#
# MIN_LEN=2, верхнего лимита нет.

def test_normalize_basic():
    assert normalize("роланд") == "Роланд"
    assert normalize("РОЛАНД") == "Роланд"
    assert normalize("Роланд") == "Роланд"


def test_normalize_strips_whitespace():
    assert normalize("  Роланд  ") == "Роланд"


def test_normalize_rejects_latin():
    with pytest.raises(ValueError):
        normalize("Roland")


def test_normalize_rejects_too_short():
    with pytest.raises(ValueError):
        normalize("А")


def test_normalize_accepts_2_chars():
    assert normalize("Аб") == "Аб"


def test_normalize_accepts_long_names():
    """Верхнего лимита нет — длинные имена допустимы.

    Функция делает первую букву заглавной, остальные строчными.
    Поэтому ожидаемый результат — 'А' + 'а'*49, а не исходная строка.
    """
    long_name = "А" * 50
    result = normalize(long_name)
    assert len(result) == 50
    assert result[0] == "А"
    assert result[1:] == "а" * 49


# ---------- preprocess ----------

def test_preprocess_deduplicates():
    raw = [
        {"name": "Роланд", "style": "human", "source": "x"},
        {"name": "роланд", "style": "human", "source": "y"},
        {"name": "РОЛАНД", "style": "human", "source": "z"},
    ]
    result = preprocess(raw)
    assert len(result) == 1
    assert result[0]["name"] == "Роланд"


def test_preprocess_skips_invalid_style():
    raw = [{"name": "Роланд", "style": "unknown", "source": "x"}]
    assert preprocess(raw) == []


# ---------- split ----------

def test_split_no_overlap():
    names = [
        "Альфа", "Бета", "Гамма", "Дельта", "Эпсилон",
        "Дзета", "Тета", "Йота", "Каппа", "Лямбда",
        "Омикрон", "Сигма", "Тау", "Омега", "Феникс",
        "Грифон", "Дракон", "Единорог", "Кентавр", "Пегас",
    ]
    data = [{"name": n, "style": "human", "source": "t"} for n in names]
    splits = split(data)
    train_names = {e["name"] for e in splits["train"]}
    val_names = {e["name"] for e in splits["val"]}
    test_names = {e["name"] for e in splits["test"]}
    assert not (train_names & test_names)
    assert not (train_names & val_names)
    assert not (val_names & test_names)


def test_split_sizes():
    names = [f"Имя{chr(0x0410 + i)}" for i in range(20)]
    data = [{"name": n, "style": "human", "source": "t"} for n in names]
    splits = split(data)
    assert len(splits["train"]) == 16
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2


# ---------- слоги ----------

def test_count_syllables():
    assert count_syllables("Роланд") == 2
    assert count_syllables("Аэлиониэль") == 6
    assert count_syllables("Торгрим") == 2
    assert count_syllables("Тхарг") == 1


def test_cv_pattern():
    assert cv_pattern("Роланд") == "C-V-C-V-C-C"
    assert cv_pattern("Аэлиониэль") == "V-V-C-V-V-C-V-V-C"


# ---------- транслитерация ----------

def test_transliterate_elven():
    assert transliterate("Aelioniel") == "Аэлиониэль"
    assert transliterate("Silvanor") == "Сильванор"
    assert transliterate("Alfred") == "Альфред"


def test_transliterate_simple():
    assert transliterate("Thorgrim") == "Торгрим"
    assert transliterate("Grokhar") == "Грохар"


def test_transliterate_l_before_vowel_stays_soft_via_i():
    assert transliterate("Aelion") == "Аэлион"