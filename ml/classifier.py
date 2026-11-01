"""Style classifier: char-CNN, 5 классов (elven / human / orcish / northern / southern).

Обучается на train, предсказывает P(style | name). Возвращает вероятности
для всех 5 стилей, из которых для конкретной задачи берётся P(target_style).

Архитектура маленькая (1 свёрточный слой): на 80 именах большая модель
мгновенно переобучается.
"""
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.lstm_gen import STYLES, STYLE_TO_ID


class StyleClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 32,
                 num_filters: int = 64, kernel_size: int = 3,
                 num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.dropout_p = dropout

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.conv = nn.Conv1d(emb_size, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) индексы. Возвращает logits (B, num_classes)."""
        emb = self.emb(x).transpose(1, 2)         # (B, emb, T)
        out = F.relu(self.conv(emb))              # (B, filters, T)
        mask = (x != 0).unsqueeze(1)              # (B, 1, T) — True для не-PAD
        # -1e9, а не -inf: иначе при полном PAD-ряду max() даёт -inf → NaN в loss.
        out = out.masked_fill(~mask, -1e9)
        out = out.max(dim=-1).values              # (B, filters)
        out = self.dropout(out)
        return self.fc(out)


# ---------- словарь ----------

def build_char_vocab(names: list[str]) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set("".join(names)))
    itos = ["<PAD>"] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi


def encode_for_classifier(name: str, stoi: dict[str, int]) -> list[int]:
    return [stoi.get(c, 0) for c in name]  # OOV → PAD, на практике не должно быть


# ---------- батчи ----------

def _make_batches(data: list[dict], batch_size: int, stoi: dict[str, int],
                  rng: random.Random):
    samples = [
        (encode_for_classifier(d["name"], stoi), STYLE_TO_ID[d["style"]])
        for d in data
    ]
    rng.shuffle(samples)
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        max_len = max(len(seq) for seq, _ in chunk)
        xs, ys = [], []
        for seq, style in chunk:
            padded = seq + [0] * (max_len - len(seq))
            xs.append(padded)
            ys.append(style)
        yield torch.tensor(xs), torch.tensor(ys)


# ---------- обучение ----------

def train_classifier(train_data: list[dict], val_data: list[dict],
                     vocab_data: list[dict] | None = None,
                     epochs: int = 100, lr: float = 1e-3, batch_size: int = 8,
                     patience: int = 20, seed: int = 42, verbose: bool = True):
    """Обучает classifier и возвращает (model, itos, stoi, history).

    history: dict с train_acc/val_acc по эпохам (для графиков на защите).
    """
    if vocab_data is None:
        vocab_data = train_data + val_data

    torch.manual_seed(seed)
    random.seed(seed)

    itos, stoi = build_char_vocab([d["name"] for d in vocab_data])
    model = StyleClassifier(vocab_size=len(itos))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    patience_counter = 0
    history = {"train_acc": [], "val_acc": []}

    def _accuracy(model, data):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in _make_batches(data, batch_size, stoi, random.Random(0)):
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / max(1, total)

    for epoch in range(1, epochs + 1):
        model.train()
        rng = random.Random(seed + epoch)
        for x, y in _make_batches(train_data, batch_size, stoi, rng):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        train_acc = _accuracy(model, train_data)
        val_acc = _accuracy(model, val_data)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"epoch {epoch:3d} | train acc {train_acc:.3f} | val acc {val_acc:.3f}")

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping на эпохе {epoch} (best val acc {best_val:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, itos, stoi, history


# ---------- предсказание ----------

@torch.no_grad()
def predict_proba(model: StyleClassifier, stoi: dict[str, int],
                  name: str) -> dict[str, float]:
    """Возвращает {style: probability} для всех 5 стилей. Сумма = 1."""
    model.eval()
    x = torch.tensor([encode_for_classifier(name, stoi)])
    probs = F.softmax(model(x), dim=-1).squeeze(0)
    return {STYLES[i]: probs[i].item() for i in range(len(STYLES))}


@torch.no_grad()
def predict_style(model: StyleClassifier, stoi: dict[str, int], name: str) -> str:
    """Возвращает стиль с максимальной вероятностью."""
    probs = predict_proba(model, stoi, name)
    return max(probs, key=probs.get)


# ---------- сохранение/загрузка ----------

def save_classifier(model: StyleClassifier, stoi: dict[str, int], path: Path) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "stoi": stoi,
        "config": {
            "vocab_size": model.vocab_size,
            "emb_size": model.emb_size,
            "num_filters": model.num_filters,
            "kernel_size": model.kernel_size,
            "num_classes": model.num_classes,
            "dropout": model.dropout_p,
        },
    }, path)


def load_classifier(path: Path) -> tuple[StyleClassifier, dict[str, int]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StyleClassifier(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stoi"]
