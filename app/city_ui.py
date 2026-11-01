
"""UI вкладки «Города»."""
import random
from pathlib import Path

import gradio as gr

from app import storage
from ml.lstm_gen import STYLES

MAX_SLOTS = 50

CITY_STYLE_LABELS = {
    "elven": "🧝 Эльфийские",
    "human": "🏰 Людские",
    "orcish": "🪓 Орочьи",
    "northern": "❄️ Северные",
    "southern": "☀️ Южные",
}


def build_city_tab(pipeline_factory, root: Path) -> None:
    pipeline_holder = {"p": None}

    def get_pipeline():
        if pipeline_holder["p"] is None:
            pipeline_holder["p"] = pipeline_factory()
        return pipeline_holder["p"]

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 🎛️ Параметры города")
            style_dd = gr.Dropdown(
                choices=[(CITY_STYLE_LABELS[s], s) for s in STYLES],
                value="northern",
                label="🎨 Стиль города",
            )
            syllables_slider = gr.Slider(
                1, 8, 2, step=1, label="🔤 Слоги",
                info="1–8",
            )
            count_slider = gr.Slider(
                1, MAX_SLOTS, 10, step=1, label="🔢 Количество",
            )
            with gr.Row():
                seed_tb = gr.Number(
                    label="🎲 Seed", value=None, precision=0, scale=3,
                )
                random_btn = gr.Button("🎲", scale=1, min_width=44)

            generate_btn = gr.Button(
                "✨ Сгенерировать города", variant="primary", size="lg",
            )
            status = gr.Markdown(
                value="_Готов к работе._",
                elem_classes=["status-bar"],
            )

        with gr.Column(scale=3, min_width=440):
            gr.Markdown("### 🏰 Города")
            result_slots = []
            for i in range(MAX_SLOTS):
                with gr.Row(visible=False) as row:
                    name_tb = gr.Textbox(
                        value="", show_label=False,
                        interactive=False, container=False,
                        scale=4, elem_classes=["card-name"],
                    )
                    star_btn = gr.Button(
                        "★", scale=0, min_width=48,
                        elem_classes=["star-btn"],
                    )
                result_slots.append({"row": row, "name": name_tb,
                                     "star": star_btn})

        with gr.Column(scale=1, min_width=260,
                       elem_classes=["favorites-panel"]):
            gr.Markdown("### ⭐ Избранные города")
            fav_display = gr.Textbox(
                value="", lines=10, show_label=False,
                interactive=False,
                placeholder="Пока пусто. Жми ★ на городе.",
            )
            clear_btn = gr.Button("🗑 Очистить всё", size="sm")
            export_txt_btn = gr.Button("📥 Экспорт в .txt", size="sm")
            download_file = gr.File(label="Скачать", visible=False)

    def refresh_favorites():
        favs = storage.load_city_favorites()
        if favs:
            return gr.update(value="\n".join(favs))
        return gr.update(value="")

    def on_random_click():
        return random.randint(0, 2**31 - 1)

    def on_generate(style, syllables, count, seed):
        try:
            pipeline = get_pipeline()
        except (FileNotFoundError, ValueError) as e:
            empty = []
            for _ in range(MAX_SLOTS):
                empty.extend([gr.update(visible=False), gr.update(value="")])
            empty.append(f"❌ {e}")
            yield tuple(empty)
            return

        try:
            results = pipeline.generate(
                style=style, syllables=int(syllables),
                count=int(count),
                seed=int(seed) if seed is not None else None,
            )
        except ValueError as e:
            empty = []
            for _ in range(MAX_SLOTS):
                empty.extend([gr.update(visible=False), gr.update(value="")])
            empty.append(f"❌ {e}")
            yield tuple(empty)
            return

        updates = []
        for i in range(MAX_SLOTS):
            if i < len(results):
                updates.extend([
                    gr.update(visible=True),
                    gr.update(value=results[i].name),
                ])
            else:
                updates.extend([
                    gr.update(visible=False),
                    gr.update(value=""),
                ])
        if results:
            updates.append(
                f"✅ Сгенерировано **{len(results)}** городов · "
                f"{CITY_STYLE_LABELS[style]} · слогов {syllables}"
            )
        else:
            updates.append(
                f"⚠️ Не удалось сгенерировать с {syllables} слогами."
            )
        yield tuple(updates)

    gen_outputs = []
    for slot in result_slots:
        gen_outputs.extend([slot["row"], slot["name"]])
    gen_outputs.append(status)

    generate_btn.click(
        fn=on_generate,
        inputs=[style_dd, syllables_slider, count_slider, seed_tb],
        outputs=gen_outputs,
    )

    random_btn.click(fn=on_random_click, inputs=None, outputs=seed_tb)

    def on_star_click(name):
        if not name:
            return gr.update(), gr.update(visible=False), "⚠️ Пустое имя"
        storage.add_city_favorite(name)
        return (
            refresh_favorites(),
            gr.update(visible=False),
            f"⭐ Город в избранное: **{name}**",
        )

    for slot in result_slots:
        slot["star"].click(
            fn=on_star_click,
            inputs=[slot["name"]],
            outputs=[fav_display, download_file, status],
        )

    def on_clear():
        storage.clear_city_favorites()
        return refresh_favorites(), gr.update(visible=False), "🗑 Очищено"

    def on_export_txt():
        favs = storage.load_city_favorites()
        if not favs:
            return gr.update(visible=False), "⚠️ Избранное пусто"
        out = root / "data" / "city_favorites.txt"
        storage.export_city_txt(out)
        return (
            gr.update(value=str(out), visible=True),
            f"📥 Экспортировано {len(favs)} городов в .txt",
        )

    clear_btn.click(
        fn=on_clear, inputs=None,
        outputs=[fav_display, download_file, status],
    )
    export_txt_btn.click(
        fn=on_export_txt, inputs=None,
        outputs=[download_file, status],
    )