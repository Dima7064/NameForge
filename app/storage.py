
"""Хранение избранного (имён и городов) и feedback."""
import csv
import json
import time
from pathlib import Path


def _root(root: Path | None) -> Path:
    if root is not None:
        return root
    return Path(__file__).resolve().parent.parent


# ============================================================
#                     Избранное: имена
# ============================================================

def _fav_path(root: Path | None = None) -> Path:
    return _root(root) / "data" / "favorites.json"


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
    path.write_text("\n".join(favs), encoding="utf-8")
    return path


def export_csv(path: Path, root: Path | None = None) -> Path:
    favs = load_favorites(root)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for name in favs:
            writer.writerow([name])
    return path


# ============================================================
#                     Избранное: города
# ============================================================

def _city_path(root: Path | None = None) -> Path:
    return _root(root) / "data" / "city_favorites.json"


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
    path.write_text("\n".join(favs), encoding="utf-8")
    return path


# ============================================================
#                          Feedback
# ============================================================

def _fb_path(root: Path | None = None) -> Path:
    return _root(root) / "data" / "feedback.json"


def load_feedback(root: Path | None = None) -> list[dict]:
    path = _fb_path(root)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return list(json.load(f).get("votes", []))
    except (json.JSONDecodeError, OSError):
        return []


def _save_feedback(votes: list[dict], root: Path | None = None) -> None:
    path = _fb_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"votes": votes}, f, ensure_ascii=False, indent=2)


def add_vote(name: str, style: str, vote: str,
             root: Path | None = None) -> dict[str, int]:
    if vote not in ("up", "down"):
        raise ValueError(f"vote должен быть 'up' или 'down', получено {vote!r}")
    votes = load_feedback(root)
    votes.append({
        "name": name, "style": style, "vote": vote,
        "ts": int(time.time()),
    })
    _save_feedback(votes, root)
    return vote_stats(name, root)


def vote_stats(name: str, root: Path | None = None) -> dict[str, int]:
    votes = load_feedback(root)
    up = sum(1 for v in votes if v["name"] == name and v["vote"] == "up")
    down = sum(1 for v in votes if v["name"] == name and v["vote"] == "down")
    return {"up": up, "down": down}


def clear_feedback(root: Path | None = None) -> None:
    _save_feedback([], root)