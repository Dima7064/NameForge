"""Тест на корректность проверки утечки в split().

Одно и то же имя в разных стилях — норма, а не утечка.
Утечка = пересечение по (name, style) паре.
"""
from ml.split import _keys, split


def test_same_name_different_style_is_not_leak():
    """«Эрик» в northern и в orcish — это два разных примера, не утечка."""
    data = [
        {"name": "Эрик", "style": "northern", "source": "t"},
        {"name": "Эрик", "style": "orcish", "source": "t"},
        {"name": "Торгрим", "style": "northern", "source": "t"},
        {"name": "Узгаш", "style": "orcish", "source": "t"},
        {"name": "Бьорнар", "style": "northern", "source": "t"},
        {"name": "Грохар", "style": "orcish", "source": "t"},
        {"name": "Харальд", "style": "northern", "source": "t"},
        {"name": "Могра", "style": "orcish", "source": "t"},
        {"name": "Сигурд", "style": "northern", "source": "t"},
        {"name": "Ургул", "style": "orcish", "source": "t"},
    ]
    splits = split(data)
    # (name, style) не должны пересекаться
    assert not (_keys(splits["train"]) & _keys(splits["val"]))
    assert not (_keys(splits["train"]) & _keys(splits["test"]))
    assert not (_keys(splits["val"]) & _keys(splits["test"]))
    # При этом «Эрик» может быть и в train, и в val — но в разных стилях.
    # Это допустимо, поэтому проверяем именно пары (name, style).


def test_real_leak_caught():
    """А вот если (name, style) одинаковый в train и val — это утечка."""
    # Если бы split положил один и тот же (name, style) в два сплита,
    # проверка должна была бы это заметить. Здесь мы эмулируем ситуацию
    # вручную — просто убеждаемся, что функция _keys работает.
    a = [{"name": "Эрик", "style": "northern", "source": "t"}]
    b = [{"name": "Эрик", "style": "northern", "source": "t"}]
    assert _keys(a) & _keys(b), "Одинаковый (name, style) должен пересекаться"
