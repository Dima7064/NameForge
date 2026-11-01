
"""Единый UI NameForge: writer + defense mode + researcher + города."""
import random
from pathlib import Path

import gradio as gr

from app import storage
from app.city_ui import build_city_tab
from app.researcher_ui import build_researcher_tab
from ml.lstm_gen import STYLES

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


def _fmt_meta(s) -> str:
    sc = s.scores
    return (
        f"🎨 Style **{sc['style']:.0%}** · "
        f"🔮 Novelty **{sc['novelty']:.0%}** · "
        f"📖 Plaus **{sc['plausibility']:.0%}** · "
        f"⭐ Total **{sc['total']:.2f}**  \n"
        f"📐 Слоги: {s.breakdown['syllables']} · "
        f"Паттерн: `{s.breakdown['pattern']}`"
    )


def _fmt_explanation(exp: dict) -> str:
    lines = [
        f"### 🔍 Разбор имени: **{exp['name']}**",
        "",
        f"**📐 Слоги:** `{exp['syllable_split']}` ({exp['syllable_count']})",
        f"**🔀 Паттерн CV:** `{exp['pattern']}`",
    ]
    if exp["diphthongs"]:
        lines.append(f"**🌊 Дифтонги:** {', '.join(exp['diphthongs'])}")
    lines.append(f"**🏁 Окончание:** `{exp['ending']}`")
    lines.append(f"**✅ Прошло фильтр:** {', '.join(exp['filters_passed'])}")

    sp = exp["style_probs"]
    top_style = max(sp, key=sp.get)
    lines.append("")
    lines.append("**📊 Вероятности по стилям:**")
    for style in STYLES:
        marker = " ←" if style == top_style else ""
        lines.append(f"- {_style_label(style)}: {sp[style]:.1%}{marker}")

    lines.append("")
    lines.append("**🎯 Скоры:**")
    sc = exp["scores"]
    lines.append(
        f"- Style: {sc['style']:.0%} · Novelty: {sc['novelty']:.0%} · "
        f"Plausibility: {sc['plausibility']:.0%} · "
        f"**Total: {sc['total']:.2f}**"
    )
    if exp["comments"]:
        lines.append("")
        lines.append("**💬 Комментарии:**")
        for c in exp["comments"]:
            lines.append(f"- {c}")
    return "\n".join(lines)


def build_app(pipeline_factory, city_pipeline_factory=None,
              default_mode: str = "writer") -> gr.Blocks:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    data_dir = root / "data"

    pipeline_holder = {"pipeline": None}
    state = {"count": 0}

    def get_pipeline():
        if pipeline_holder["pipeline"] is None:
            pipeline_holder["pipeline"] = pipeline_factory()
        return pipeline_holder["pipeline"]

    with gr.Blocks(title="⚔ Кузница Имён") as app:
        with gr.Row(elem_classes=["app-title"]):
            with gr.Column(scale=3):
                gr.HTML(
                    "<h1>⚔ Кузница Имён</h1>"
                    "<p style='color:#a89f8c;margin:0;font-size:0.95em;'>"
                    "✨ Генератор фэнтези-имён и городов</p>"
                )
            with gr.Column(scale=1, min_width=260):
                mode_switch = gr.Radio(
                    choices=[("✍️ Для себя", "writer"),
                             ("🔮 Для защиты", "defense")],
                    value=default_mode,
                    label="🎭 Режим",
                    elem_classes=["mode-switch"],
                )

        with gr.Tabs() as main_tabs:
            with gr.Tab("⚔ Генератор", id="gen_tab"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("### 🎛️ Фильтры")
                        style_dd = gr.Dropdown(
                            choices=[(_style_label(s), s) for s in STYLES],
                            value="elven", label="🎨 Стиль",
                        )
                        syllables_slider = gr.Slider(
                            1, 6, 2, step=1, label="🔤 Слоги",
                            info="1–6 (по данным датасета)",
                        )
                        count_slider = gr.Slider(
                            1, MAX_SLOTS, 10, step=1, label="🔢 Количество",
                        )
                        with gr.Row():
                            seed_tb = gr.Number(
                                label="🎲 Seed", value=None,
                                precision=0, scale=3,
                            )
                            random_btn = gr.Button(
                                "🎲", scale=1, min_width=44,
                            )

                        defense_left_col = gr.Column(visible=False)
                        with defense_left_col:
                            gr.Markdown("### ⚙️ Сэмплирование")
                            temperature_slider = gr.Slider(
                                0.1, 2.0, 0.8, step=0.05,
                                label="🌡️ Temperature",
                                info="<1 = уверенно, >1 = хаотично",
                            )
                            top_k_slider = gr.Slider(
                                0, 100, 40, step=1,
                                label="🎯 Top-k", info="0 = отключить",
                            )
                            top_p_slider = gr.Slider(
                                0.0, 1.0, 0.9, step=0.01,
                                label="📊 Top-p", info="1.0 = отключить",
                            )
                            with gr.Accordion("⚖️ Веса скоринга", open=False):
                                w_style_slider = gr.Slider(
                                    0.0, 1.0, 0.5, step=0.05,
                                    label="🎨 Style",
                                )
                                w_novelty_slider = gr.Slider(
                                    0.0, 1.0, 0.25, step=0.05,
                                    label="🔮 Novelty",
                                )
                                w_plaus_slider = gr.Slider(
                                    0.0, 1.0, 0.25, step=0.05,
                                    label="📖 Plausibility",
                                )
                                gr.Markdown(
                                    "_Веса нормализуются автоматически._"
                                )

                        generate_btn = gr.Button(
                            "✨ Сгенерировать", variant="primary", size="lg",
                        )
                        status = gr.Markdown(
                            value="_Готов к работе._",
                            elem_classes=["status-bar"],
                        )

                    with gr.Column(scale=3, min_width=440):
                        gr.Markdown("### 📜 Результаты")
                        result_slots = []
                        defense_slot_cols = []
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

                            with gr.Column(visible=False) as def_col:
                                meta_md = gr.Markdown(
                                    value="", elem_classes=["meta-block"],
                                )
                                with gr.Row():
                                    explain_btn = gr.Button(
                                        "📖 Объяснить", size="sm",
                                        scale=2,
                                        elem_classes=["explain-btn"],
                                    )
                                    up_btn = gr.Button(
                                        "👍", size="sm", scale=1,
                                        elem_classes=["feedback-btn"],
                                    )
                                    down_btn = gr.Button(
                                        "👎", size="sm", scale=1,
                                        elem_classes=["feedback-btn"],
                                    )

                            result_slots.append({
                                "row": row, "name": name_tb,
                                "star": star_btn, "def_col": def_col,
                                "meta": meta_md, "explain": explain_btn,
                                "up": up_btn, "down": down_btn,
                            })
                            defense_slot_cols.append(def_col)

                        explain_panel = gr.Markdown(
                            value="", elem_classes=["explain-panel"],
                        )

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
                        download_file = gr.File(
                            label="Скачать", visible=False,
                        )

            with gr.Tab("🏰 Города", id="city_tab"):
                if city_pipeline_factory is not None:
                    build_city_tab(city_pipeline_factory, root)
                else:
                    gr.Markdown(
                        "⚠️ Генератор городов не подключён."
                    )

            with gr.Tab("🔬 Исследователь", id="res_tab",
                        visible=False) as res_tab:
                build_researcher_tab(pipeline_factory, models_dir, data_dir)

        def refresh_favorites_ui():
            favs = storage.load_favorites()
            if favs:
                return (
                    gr.update(value="\n".join(favs)),
                    gr.update(choices=favs, value=[], visible=True),
                    gr.update(visible=True),
                )
            return (
                gr.update(value=""),
                gr.update(choices=[], value=[], visible=False),
                gr.update(visible=False),
            )

        def on_random_click():
            return random.randint(0, 2**31 - 1)

        def on_mode_change(mode):
            is_def = (mode == "defense")
            updates = [gr.update(visible=is_def)]
            for i in range(MAX_SLOTS):
                updates.append(gr.update(
                    visible=(is_def and i < state["count"])
                ))
            updates.append(gr.update() if is_def else gr.update(value=""))
            updates.append(gr.update(visible=is_def))
            if not is_def:
                updates.append(gr.update(selected="gen_tab"))
            else:
                updates.append(gr.update())
            return tuple(updates)

        mode_outputs = (
            [defense_left_col] + defense_slot_cols
            + [explain_panel, res_tab, main_tabs]
        )
        mode_switch.change(
            fn=on_mode_change, inputs=[mode_switch], outputs=mode_outputs,
        )

        def on_generate(mode, style, syllables, count, seed,
                        temperature, top_k, top_p,
                        w_style, w_novelty, w_plaus):
            try:
                pipeline = get_pipeline()
            except (FileNotFoundError, ValueError) as e:
                empty = []
                for _ in range(MAX_SLOTS):
                    empty.extend([
                        gr.update(visible=False), gr.update(value=""),
                        gr.update(value=""), gr.update(visible=False),
                    ])
                empty.extend([gr.update(value=""), f"❌ {e}"])
                yield tuple(empty)
                return

            is_def = (mode == "defense")
            wsum = max(1e-6, w_style + w_novelty + w_plaus)
            ws, wn, wp = w_style / wsum, w_novelty / wsum, w_plaus / wsum

            try:
                results = pipeline.generate(
                    style=style, syllables=int(syllables),
                    count=int(count),
                    seed=int(seed) if seed is not None else None,
                    temperature=float(temperature),
                    top_k=int(top_k), top_p=float(top_p),
                    w_style=ws, w_novelty=wn, w_plaus=wp,
                )
            except ValueError as e:
                empty = []
                for _ in range(MAX_SLOTS):
                    empty.extend([
                        gr.update(visible=False), gr.update(value=""),
                        gr.update(value=""), gr.update(visible=False),
                    ])
                empty.extend([gr.update(value=""), f"❌ {e}"])
                yield tuple(empty)
                return

            state["count"] = len(results)

            updates = []
            for i in range(MAX_SLOTS):
                if i < len(results):
                    updates.extend([
                        gr.update(visible=True),
                        gr.update(value=results[i].name),
                        gr.update(value=_fmt_meta(results[i])),
                        gr.update(visible=is_def),
                    ])
                else:
                    updates.extend([
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value=""),
                        gr.update(visible=False),
                    ])
            updates.append(gr.update(value=""))

            if results:
                mode_tag = "🔮 defense" if is_def else "✍️ writer"
                status_text = (
                    f"✅ **{len(results)}** имён · "
                    f"{_style_label(style)} · слогов {syllables} · {mode_tag}"
                )
            else:
                status_text = (
                    f"⚠️ Не удалось сгенерировать с {syllables} слогами."
                )
            updates.append(status_text)
            yield tuple(updates)

        gen_outputs = []
        for slot in result_slots:
            gen_outputs.extend([
                slot["row"], slot["name"], slot["meta"], slot["def_col"],
            ])
        gen_outputs.extend([explain_panel, status])

        generate_btn.click(
            fn=on_generate,
            inputs=[mode_switch, style_dd, syllables_slider, count_slider,
                    seed_tb, temperature_slider, top_k_slider, top_p_slider,
                    w_style_slider, w_novelty_slider, w_plaus_slider],
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

        def on_explain(name, style, mode, w_style, w_novelty, w_plaus):
            if not name:
                return gr.update(value="⚠️ Пустое имя")
            if mode != "defense":
                return gr.update(value="")
            try:
                pipeline = get_pipeline()
                wsum = max(1e-6, w_style + w_novelty + w_plaus)
                exp = pipeline.explain(
                    name, style,
                    w_style=w_style / wsum,
                    w_novelty=w_novelty / wsum,
                    w_plaus=w_plaus / wsum,
                )
                return gr.update(value=_fmt_explanation(exp))
            except Exception as e:
                return gr.update(value=f"❌ Ошибка: {e}")

        for slot in result_slots:
            slot["explain"].click(
                fn=on_explain,
                inputs=[slot["name"], style_dd, mode_switch,
                        w_style_slider, w_novelty_slider, w_plaus_slider],
                outputs=[explain_panel],
            )

        def make_vote_handler(vote: str):
            def handler(name, style):
                if not name:
                    return "⚠️ Пустое имя"
                stats = storage.add_vote(name, style, vote)
                emoji = "👍" if vote == "up" else "👎"
                return (f"{emoji} **{name}** — "
                        f"👍 {stats['up']} / 👎 {stats['down']}")
            return handler

        for slot in result_slots:
            slot["up"].click(
                fn=make_vote_handler("up"),
                inputs=[slot["name"], style_dd], outputs=[status],
            )
            slot["down"].click(
                fn=make_vote_handler("down"),
                inputs=[slot["name"], style_dd], outputs=[status],
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
            out = data_dir / "export_favorites.txt"
            storage.export_txt(out)
            return (gr.update(value=str(out), visible=True),
                    f"📥 Экспортировано {len(favs)} в .txt")

        def on_export_csv():
            favs = storage.load_favorites()
            if not favs:
                return gr.update(visible=False), "⚠️ Избранное пусто"
            out = data_dir / "export_favorites.csv"
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
        export_txt_btn.click(
            fn=on_export_txt, inputs=None,
            outputs=[download_file, status],
        )
        export_csv_btn.click(
            fn=on_export_csv, inputs=None,
            outputs=[download_file, status],
        )

        app.load(
            fn=refresh_favorites_ui, inputs=None,
            outputs=[fav_display, fav_checkboxes, remove_selected_btn],
        )

    return app