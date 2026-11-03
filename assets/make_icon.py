"""Генерирует assets/icon.ico — меч в золотом круге.

Запуск:
    python assets/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw


def make(path: Path) -> None:
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size / 256

    # Background circle
    cx, cy = size // 2, size // 2
    r = size // 2 - 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(31, 35, 43, 255))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(201, 162, 39, 255), width=int(6 * s))

    # Blade (silver, vertical)
    bl = int(cx - 12 * s)
    br = int(cx + 12 * s)
    bt = int(40 * s)
    bb = int(180 * s)
    draw.rectangle([bl, bt, br, bb], fill=(220, 220, 230, 255))

    # Blade tip (triangle)
    draw.polygon([(cx, int(22 * s)), (bl, bt), (br, bt)],
                 fill=(220, 220, 230, 255))

    # Fuller (dark line in middle of blade)
    draw.rectangle([cx - int(3 * s), bt + int(10 * s),
                    cx + int(3 * s), bb - int(10 * s)],
                   fill=(150, 150, 175, 255))

    # Crossguard (gold, horizontal)
    draw.rectangle([int(55 * s), int(182 * s), int(201 * s), int(202 * s)],
                   fill=(201, 162, 39, 255))

    # Grip (brown, vertical)
    draw.rectangle([cx - int(7 * s), int(202 * s),
                    cx + int(7 * s), int(234 * s)],
                   fill=(90, 60, 40, 255))

    # Pommel (gold, bottom circle)
    draw.ellipse([cx - int(13 * s), int(228 * s),
                  cx + int(13 * s), int(254 * s)],
                 fill=(201, 162, 39, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="ICO",
             sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
                    (128, 128), (256, 256)])
    print(f"✅ {path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    make(root / "icon.ico")
