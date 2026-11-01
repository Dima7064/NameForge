
"""Точка входа NameForge (Gradio 6).

Запуск:
    python -m app.main
"""
import json
from pathlib import Path

from app.city_pipeline import CityPipeline
from app.pipeline import Pipeline
from app.ui import build_app


def pipeline_factory() -> Pipeline:
    return Pipeline(model_type="lstm")


def city_pipeline_factory() -> CityPipeline:
    return CityPipeline()


def _load_default_mode() -> str:
    cfg = Path(__file__).resolve().parent.parent / "config.json"
    if not cfg.exists():
        return "writer"
    try:
        with cfg.open("r", encoding="utf-8") as f:
            mode = json.load(f).get("mode", "writer")
        return mode if mode in ("writer", "defense") else "writer"
    except (json.JSONDecodeError, OSError):
        return "writer"


def main() -> None:
    root = Path(__file__).resolve().parent
    css_path = root.parent / "assets" / "theme.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    app = build_app(
        pipeline_factory,
        city_pipeline_factory=city_pipeline_factory,
        default_mode=_load_default_mode(),
    )
    app.launch(
        css=css,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()