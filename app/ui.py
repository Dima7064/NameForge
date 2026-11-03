
"""Ономастикон — UI writer-режима (только для пользователя).

Две вкладки:
  ⚔ Имена  — генерация имён (стиль, пол, слоги, количество, seed) + избранное
  🏰 Города — генерация городов (стиль, слоги, количество, seed) + избранное

Никаких метрик, объяснений, «режима для защиты» — только простой и удобный
интерфейс для писателя/геймдизайнера.
"""
import random
from pathlib import Path

import gradio as gr

from app import storage
from app.city_ui import build_city_tab
from app.paths import get_install_dir, get_user_data_dir
from ml.lstm_gen import STYLES

# ------------------------------------------------------------------
#  Название проекта. Меняй здесь — и оно поменяется везде (шапка, title).
#  Плейсхолдеры для подзаголовка — тоже здесь.
# ------------------------------------------------------------------
APP_TITLE = "Ономастикон"
APP_SUBTITLE = "ML-генератор фэнтези-имён и городов"
APP_ICON = "📖"

MAX_SLOTS = 50

STYLE_LABELS = {
    "elven": "🧝 Эльфийский",
    "human": "🏰 Люди",
    "orcish": "🪓 Орочий",
    "northern": "❄️ Северный",
    "southern": "☀️ Южный",
}


def _style_label(style: str) -> str:
    return STYLE_LABELS.get(style, style)


def build_app(pipeline_factory, city_pipeline_factory=None) -> gr.Blocks:
    install_dir = get_install_dir()
    user_dir = get_user_data_dir()

    pipeline_holder = {"pipeline": None}

    def get_pipeline():
        if pipeline_holder["pipeline"] is None:
            pipeline_holder["pipeline"] = pipeline_factory()
        return pipeline_holder["pipeline"]

    title_html = (
        f"<h1>{APP_ICON} {APP_TITLE}</h1>"
        f"<p style='color:#a89f8c;margin:0;font-size:0.95em;'>{APP_SUBTITLE}</p>"
    )

    with gr.Blocks(title=f"{APP_ICON} {APP_TITLE}") as app:
        # ---------- Шапка ----------
        with gr.Row(elem_classes=["app-title"]):
            with gr.Column(scale=1):
                gr.HTML(title_html)

        with gr.Tabs():
            # ============================================================
            #                   ВКЛАДКА «ИМЕНА»
            # ============================================================
            with gr.Tab("⚔ Имена", id="names_tab"):
                with gr.Row():
                    # --- ЛЕВАЯ: фильтры ---
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("### 🎛️ Параметры")
                        style_dd = gr.Dropdown(
                            choices=[(_style_label(s), s) for s in STYLES],
                            value="elven", label="🎨 Стиль",
                        )
                        gender_radio = gr.Radio(
                            choices=[("♂ Мужской", "male"),
                                     ("♀ Женский", "female"),
                                     ("🎲 Случайный", "random")],
                            value="male", label="⚧ Пол",
                        )
                        syllables_slider = gr.Slider(
                            1, 6, 2, step=1, label="🔤 Слоги",
                            info="1–6",
                        )
                        count_slider = gr.Slider(
                            1, MAX_SLOTS, 10, step=1, label="🔢 Количество",
                        )
                        with gr.Row():
                            seed_tb = gr.Number(
                                label="🎲 Seed", value=None,
                                precision=0, scale=3,
                            )
                            random_btn = gr.Button("🎲", scale=1, min_width=44)

                        generate_btn = gr.Button(
                            "✨ Сгенерировать", variant="primary", size="lg",
                        )
                        status = gr.Markdown(
                            value="_Готов к работе._",
                            elem_classes=["status-bar"],
                        )

                    # --- ЦЕНТР: карточки ---
                    with gr.Column(scale=3, min_width=440):
                        gr.Markdown("### 📜 Результаты")
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
                            result_slots.append({
                                "row": row, "name": name_tb, "star": star_btn,
                            })

                    # --- ПРАВАЯ: избранное ---
                    with gr.Column(scale=1, min_width=260,
                                   elem_classes=["favorites-panel"]):
                        gr.Markdown("### ⭐ Избранное")
                        fav_display = gr.Textbox(
                            value="", lines=10, show_label=False,
                            interactive=False,
                            placeholder="Пока пусто. Жми ★ на имени.",
                        )
                        fav_checkboxes = gr.CheckboxGroup(
                            choices=[], label="Выбрать для удаления",
                            visible=False,
                        )
                        with gr.Row():
                            remove_selected_btn = gr.Button(
                                "🗑 Удалить", visible=False, size="sm",
                            )
                            clear_all_btn = gr.Button(
                                "🗑 Очистить всё", size="sm",
                            )
                        with gr.Row():
                            export_txt_btn = gr.Button("📥 .txt", size="sm")
                            export_csv_btn = gr.Button("📥 .csv", size="sm")
                        download_file = gr.File(label="Скачать", visible=False)

            # ============================================================
            #                   ВКЛАДКА «ГОРОДА»
            # ============================================================
            with gr.Tab("🏰 Города", id="cities_tab"):
                if city_pipeline_factory is not None:
                    build_city_tab(city_pipeline_factory, user_dir)
                else:
                    gr.Markdown("⚠️ Генератор городов не подключён.")

        # ---------- Обработчики ----------

        def refresh_favorites_ui():
            favs = storage.load_favorites()
            if favs:
                return (gr.update(value="\n".join(favs)),
                        gr.update(choices=favs, value=[], visible=True),
                        gr.update(visible=True))
            return (gr.update(value=""),
                    gr.update(choices=[], value=[], visible=False),
                    gr.update(visible=False))

        def on_random_click():
            return random.randint(0, 2**31 - 1)

        def on_generate(style, gender, syllables, count, seed):
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
                    count=int(count), gender=gender,
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
                    updates.extend([gr.update(visible=True),
                                    gr.update(value=results[i].name)])
                else:
                    updates.extend([gr.update(visible=False),
                                    gr.update(value="")])

            if results:
                gender_icon = {"male": "♂", "female": "♀",
                               "random": "🎲"}.get(gender, "")
                updates.append(
                    f"✅ **{len(results)}** имён · "
                    f"{_style_label(style)} {gender_icon} · слогов {syllables}"
                )
            else:
                updates.append(f"⚠️ Не удалось с {syllables} слогами.")
            yield tuple(updates)

        gen_outputs = []
        for slot in result_slots:
            gen_outputs.extend([slot["row"], slot["name"]])
        gen_outputs.append(status)

        generate_btn.click(
            fn=on_generate,
            inputs=[style_dd, gender_radio, syllables_slider,
                    count_slider, seed_tb],
            outputs=gen_outputs,
        )
        random_btn.click(fn=on_random_click, inputs=None, outputs=seed_tb)

        def on_star_click(name):
            if not name:
                return (gr.update(), gr.update(), gr.update(),
                        gr.update(visible=False), "⚠️ Пустое имя")
            storage.add_favorite(name)
            text, cbs, btn = refresh_favorites_ui()
            return (text, cbs, btn, gr.update(visible=False),
                    f"⭐ В избранное: **{name}**")

        for slot in result_slots:
            slot["star"].click(
                fn=on_star_click, inputs=[slot["name"]],
                outputs=[fav_display, fav_checkboxes, remove_selected_btn,
                         download_file, status],
            )

        def on_remove_selected(selected):
            if not selected:
                return (gr.update(), gr.update(), gr.update(),
                        gr.update(visible=False), "⚠️ Нечего удалять")
            storage.remove_favorites(selected)
            text, cbs, btn = refresh_favorites_ui()
            return (text, cbs, btn, gr.update(visible=False),
                    f"🗑 Удалено: {len(selected)}")

        def on_clear_all():
            storage.clear_favorites()
            text, cbs, btn = refresh_favorites_ui()
            return (text, cbs, btn, gr.update(visible=False),
                    "🗑 Избранное очищено")

        def on_export_txt():
            favs = storage.load_favorites()
            if not favs:
                return gr.update(visible=False), "⚠️ Избранное пусто"
            out = user_dir / "export_favorites.txt"
            storage.export_txt(out)
            return (gr.update(value=str(out), visible=True),
                    f"📥 Экспортировано {len(favs)} в .txt")

        def on_export_csv():
            favs = storage.load_favorites()
            if not favs:
                return gr.update(visible=False), "⚠️ Избранное пусто"
            out = user_dir / "export_favorites.csv"
            storage.export_csv(out)
            return (gr.update(value=str(out), visible=True),
                    f"📥 Экспортировано {len(favs)} в .csv")

        remove_selected_btn.click(
            fn=on_remove_selected, inputs=[fav_checkboxes],
            outputs=[fav_display, fav_checkboxes, remove_selected_btn,
                     download_file, status],
        )
        clear_all_btn.click(
            fn=on_clear_all, inputs=None,
            outputs=[fav_display, fav_checkboxes, remove_selected_btn,
                     download_file, status],
        )
        export_txt_btn.click(fn=on_export_txt, inputs=None,
                             outputs=[download_file, status])
        export_csv_btn.click(fn=on_export_csv, inputs=None,
                             outputs=[download_file, status])

        app.load(fn=refresh_favorites_ui, inputs=None,
                 outputs=[fav_display, fav_checkboxes, remove_selected_btn])

    return app