"""Тесты для хранилища избранного."""
import pytest

from app import storage


@pytest.fixture
def tmp_root(tmp_path):
    """Изолированная папка для каждого теста."""
    return tmp_path


def test_load_empty(tmp_root):
    assert storage.load_favorites(root=tmp_root) == []


def test_add_and_load(tmp_root):
    storage.add_favorite("Аэлион", root=tmp_root)
    storage.add_favorite("Грохар", root=tmp_root)
    assert storage.load_favorites(root=tmp_root) == ["Аэлион", "Грохар"]


def test_add_duplicate_ignored(tmp_root):
    storage.add_favorite("Аэлион", root=tmp_root)
    storage.add_favorite("Аэлион", root=tmp_root)
    assert storage.load_favorites(root=tmp_root) == ["Аэлион"]


def test_add_empty_ignored(tmp_root):
    storage.add_favorite("", root=tmp_root)
    storage.add_favorite("   ", root=tmp_root)
    assert storage.load_favorites(root=tmp_root) == []


def test_remove_favorite(tmp_root):
    storage.add_favorite("Аэлион", root=tmp_root)
    storage.add_favorite("Грохар", root=tmp_root)
    storage.remove_favorite("Аэлион", root=tmp_root)
    assert storage.load_favorites(root=tmp_root) == ["Грохар"]


def test_remove_several(tmp_root):
    for n in ["А", "Б", "В"]:
        storage.add_favorite(n, root=tmp_root)
    storage.remove_favorites(["А", "В"], root=tmp_root)
    assert storage.load_favorites(root=tmp_root) == ["Б"]


def test_clear_favorites(tmp_root):
    storage.add_favorite("Аэлион", root=tmp_root)
    storage.clear_favorites(root=tmp_root)
    assert storage.load_favorites(root=tmp_root) == []


def test_export_txt(tmp_root):
    storage.add_favorite("Аэлион", root=tmp_root)
    storage.add_favorite("Грохар", root=tmp_root)
    out = tmp_root / "export.txt"
    storage.export_txt(out, root=tmp_root)
    content = out.read_text(encoding="utf-8")
    assert "Аэлион" in content
    assert "Грохар" in content
    assert content.count("\n") == 1  # 2 строки → 1 перенос


def test_export_csv(tmp_root):
    storage.add_favorite("Аэлион", root=tmp_root)
    out = tmp_root / "export.csv"
    storage.export_csv(out, root=tmp_root)
    content = out.read_text(encoding="utf-8")
    assert "name" in content
    assert "Аэлион" in content


def test_corrupted_json_returns_empty(tmp_root):
    path = tmp_root / "data" / "favorites.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a json {", encoding="utf-8")
    assert storage.load_favorites(root=tmp_root) == []
