
"""Точка входа Ономастикона. Работает и из исходников, и из .exe."""
import sys
import traceback

from app.city_pipeline import CityPipeline
from app.paths import get_install_dir
from app.pipeline import Pipeline
from app.ui import build_app


def pipeline_factory() -> Pipeline:
    return Pipeline(model_type="lstm")


def city_pipeline_factory() -> CityPipeline:
    return CityPipeline()


def main() -> None:
    try:
        css_path = get_install_dir() / "assets" / "theme.css"
        css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

        app = build_app(pipeline_factory,
                        city_pipeline_factory=city_pipeline_factory)
        app.launch(
            css=css,
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            show_error=True,
            quiet=False,
        )
    except Exception as e:
        print(f"\n❌ ОШИБКА ЗАПУСКА:\n{e}\n")
        traceback.print_exc()
        if getattr(sys, "frozen", False):
            input("Нажми Enter для закрытия...")
        sys.exit(1)


if __name__ == "__main__":
    main()