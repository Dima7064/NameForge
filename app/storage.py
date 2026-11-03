
"""Хранение избранного (имена и города).

В source-режиме — в data/.
В .exe — в user_data/ рядом с NameForge.exe (сохраняется между запусками).
"""
import csv
import json
from pathlib import Path

from app.paths import get_user_data_dir


def _fav_path(root: Path | None = None) -> Path:
    if root is not None:
        return root / "data" / "favorites.json"
    return get_user_data_dir() / "favorites.json"


def _city_path(root: Path | None = None) -> Path:
    if root is not None:
        return root / "data" / "city_favorites.json"
    return get_user_data_dir() / "city_favorites.json"


# ============================================================
#                     Избранное: имена
# ============================================================

def load_favorites(root: Path | None = None) -> list[str]:
    path = _fav_path(root)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return list(json.load(f).get("favorites", []))
    except (json.JSONDecodeError, OSError):
        return []


def save_favorites(favorites: list[str], root: Path | None = None) -> None:
    path = _fav_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"favorites": favorites}, f, ensure_ascii=False, indent=2)


def add_favorite(name: str, root: Path | None = None) -> list[str]:
    if not name or not name.strip():
        return load_favorites(root)
    name = name.strip()
    favs = load_favorites(root)
    if name not in favs:
        favs.append(name)
        save_favorites(favs, root)
    return favs


def remove_favorite(name: str, root: Path | None = None) -> list[str]:
    favs = [f for f in load_favorites(root) if f != name.strip()]
    save_favorites(favs, root)
    return favs


def remove_favorites(names: list[str], root: Path | None = None) -> list[str]:
    names_set = {n.strip() for n in names}
    favs = [f for f in load_favorites(root) if f not in names_set]
    save_favorites(favs, root)
    return favs


def clear_favorites(root: Path | None = None) -> list[str]:
    save_favorites([], root)
    return []


def export_txt(path: Path, root: Path | None = None) -> Path:
    favs = load_favorites(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(favs), encoding="utf-8")
    return path


def export_csv(path: Path, root: Path | None = None) -> Path:
    favs = load_favorites(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for name in favs:
            writer.writerow([name])
    return path


# ============================================================
#                     Избранное: города
# ============================================================

def load_city_favorites(root: Path | None = None) -> list[str]:
    path = _city_path(root)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return list(json.load(f).get("favorites", []))
    except (json.JSONDecodeError, OSError):
        return []


def save_city_favorites(favorites: list[str], root: Path | None = None) -> None:
    path = _city_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"favorites": favorites}, f, ensure_ascii=False, indent=2)


def add_city_favorite(name: str, root: Path | None = None) -> list[str]:
    if not name or not name.strip():
        return load_city_favorites(root)
    name = name.strip()
    favs = load_city_favorites(root)
    if name not in favs:
        favs.append(name)
        save_city_favorites(favs, root)
    return favs


def remove_city_favorite(name: str, root: Path | None = None) -> list[str]:
    favs = [f for f in load_city_favorites(root) if f != name.strip()]
    save_city_favorites(favs, root)
    return favs


def clear_city_favorites(root: Path | None = None) -> list[str]:
    save_city_favorites([], root)
    return []


def export_city_txt(path: Path, root: Path | None = None) -> Path:
    favs = load_city_favorites(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(favs), encoding="utf-8")
    return path