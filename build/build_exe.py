
"""Сборка .exe через PyInstaller (onedir режим)."""
import sys
from pathlib import Path

import PyInstaller.__main__


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    for required in ("models/lstm.pt", "models/city_lstm.pt",
                     "models/classifier.pt", "models/scorers.pt",
                     "data/train.json"):
        if not (root / required).exists():
            print(f"❌ Отсутствует {required}. Запусти python -m ml.full_retrain.")
            sys.exit(1)

    icon_path = root / "assets" / "icon.ico"
    if not icon_path.exists():
        print("🖼  Генерирую иконку...")
        sys.path.insert(0, str(root / "assets"))
        from make_icon import make as make_icon
        make_icon(icon_path)

    sep = ";" if sys.platform == "win32" else ":"

    args = [
        str(root / "app" / "main.py"),
        "--name=NameForge",
        "--onedir",
        "--console",
        f"--icon={icon_path}",

        f"--add-data={root / 'models'}{sep}models",
        f"--add-data={root / 'data'}{sep}data",
        f"--add-data={root / 'assets'}{sep}assets",

        "--collect-all=gradio",
        "--collect-all=gradio_client",
        "--collect-all=groovy",
        "--collect-all=safehttpx",
        "--collect-all=huggingface_hub",
        "--collect-all=torch",
        "--collect-all=matplotlib",

        "--exclude-module=torchvision",
        "--exclude-module=torchaudio",
        "--exclude-module=tensorflow",
        "--exclude-module=IPython",
        "--exclude-module=jupyter",
        "--exclude-module=jupyterlab",
        "--exclude-module=notebook",
        "--exclude-module=tkinter",
        "--exclude-module=pytest",
        "--exclude-module=_pytest",
        "--exclude-module=pygments",

        f"--paths={root}",
        f"--distpath={root / 'dist'}",
        f"--workpath={root / 'build' / '_work'}",
        f"--specpath={root / 'build'}",
        "--noconfirm",
        "--clean",
    ]

    print("🔨 Запускаю PyInstaller...")
    print("   Это займёт 5-10 минут.\n")
    PyInstaller.__main__.run(args)
    print(f"\n✅ Готово: {root / 'dist' / 'NameForge' / 'NameForge.exe'}")


if __name__ == "__main__":
    main()