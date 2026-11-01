"""Визуализация посимвольных предсказаний LSTM (heatmap)."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from ml.lstm_gen import BOS, STYLE_TO_ID


def render_heatmap(model, itos: list[str], stoi: dict[str, int],
                   name: str, style: str) -> Image.Image:
    """Возвращает картинку: top-5 предсказаний на каждом шаге."""
    model.eval()
    seq = [BOS] + [stoi.get(c, 0) for c in name]
    x = torch.tensor([seq])
    s = torch.tensor([STYLE_TO_ID[style]])

    with torch.no_grad():
        logits, _ = model(x, s)
        probs = F.softmax(logits[0], dim=-1)  # (T, V)

    rows = []
    for i in range(len(seq) - 1):
        p = probs[i]
        top5_idx = torch.topk(p, 5).indices.tolist()
        top5 = [(itos[idx], p[idx].item()) for idx in top5_idx]
        prefix = name[:i] if i > 0 else "^"
        target = name[i] if i < len(name) else "$"
        rows.append({"prefix": prefix, "target": target, "top5": top5})

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5)),
                            facecolor="#151820")
    ax.set_facecolor("#0b0c10")

    for j in range(5):
        for i, row in enumerate(rows):
            if j < len(row["top5"]):
                c, p = row["top5"][j]
                intensity = min(1.0, p * 2.5)
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, color=(0.79, 0.63, 0.15, intensity)
                ))
                is_correct = (c == row["target"])
                text_color = "#ffffff" if intensity > 0.5 else "#e8e0d0"
                weight = "bold" if is_correct else "normal"
                ax.text(
                    j + 0.5, i + 0.5, f"{c}\n{p:.0%}",
                    ha="center", va="center", color=text_color,
                    fontsize=9, fontweight=weight,
                )

    ax.set_xlim(0, 5)
    ax.set_ylim(0, len(rows))
    ax.invert_yaxis()
    ax.set_xticks([x + 0.5 for x in range(5)])
    ax.set_xticklabels(["1-е", "2-е", "3-е", "4-е", "5-е"], color="#a89f8c")
    ax.set_yticks([y + 0.5 for y in range(len(rows))])
    ax.set_yticklabels(
        [f"{r['prefix']} → {r['target']}" for r in rows],
        color="#a89f8c", fontfamily="monospace",
    )
    ax.set_title(f"Предсказание: {name}", color="#c9a227", fontsize=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#3a4150")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#151820")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
