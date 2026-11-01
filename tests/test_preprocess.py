
"""Базовые тесты для препроцессинга, сплита, слогов и транслитерации."""
import pytest

from ml.preprocess import normalize, preprocess
from ml.report import count_syllables, cv_pattern
from ml.split import split
from ml.transliterate import transliterate


# ---------- normalize ----------

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
        normalize("Аб")


def test_normalize_rejects_too_long():
    with pytest.raises(ValueError):
        normalize("А" * 25)


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
#
# ВАЖНО: split() рассчитан на УЖЕ очищенные данные (дубликаты удалены
# на шаге preprocess). Поэтому тест подаёт уникальные имена.

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
    assert count_syllables("Роланд") == 2        # о, а
    assert count_syllables("Аэлиониэль") == 6    # а, э, и, о, и, э
    assert count_syllables("Торгрим") == 2       # о, и
    assert count_syllables("Тхарг") == 1         # а


def test_cv_pattern():
    assert cv_pattern("Роланд") == "C-V-C-V-C-C"
    # ь не считается ни C, ни V — она модифицирует предыдущую согласную.
    assert cv_pattern("Аэлиониэль") == "V-V-C-V-V-C-V-V-C"


# ---------- транслитерация ----------

def test_transliterate_elven():
    # -iel в финале → -иэль
    assert transliterate("Aelioniel") == "Аэлиониэль"
    # l перед согласной → ль
    assert transliterate("Silvanor") == "Сильванор"
    assert transliterate("Alfred") == "Альфред"


def test_transliterate_simple():
    assert transliterate("Thorgrim") == "Торгрим"
    assert transliterate("Grokhar") == "Грохар"


def test_transliterate_l_before_vowel_stays_soft_via_i():
    # 'li' перед гласной: 'l' не смягчается правилом перед согласной,
    # мягкость даёт следующая 'и'.
    assert transliterate("Aelion") == "Аэлион"