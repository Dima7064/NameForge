
"""Transformer-генератор фэнтези-имён с условием на стиль.

Посимоольная авторегрессивная модель: P(следующий символ | предыдущие + стиль).
Отличие от LSTM: вместо рекуррентности — causal self-attention.
На маленьких датасетах обычно проигрывает LSTM, но полезен для сравнения.
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2

STYLES = ["elven", "human", "orcish", "northern", "southern"]
STYLE_TO_ID = {s: i for i, s in enumerate(STYLES)}


# ---------- модель ----------

class PositionalEncoding(nn.Module):
    """Синусоидальный positional encoding (стандарт из «Attention is all you need»)."""

    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerNameGen(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.2, max_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.max_len = max_len

        self.char_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.style_emb = nn.Embedding(len(STYLES), d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        # enable_nested_tensor=False — отключаем предупреждение + nested tensor
        # не нужен для наших коротких последовательностей.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        """Causal mask (bool): True = «не обращать внимания».
        Позиция i не видит j > i (верхний треугольник = True)."""
        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, x: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        mask = self._causal_mask(T, x.device)
        # padding_mask: True = «не обращать внимания» (PAD-позиции)
        pad_mask = (x == PAD)
        emb = self.char_emb(x) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb) + self.style_emb(styles).unsqueeze(1)
        out = self.encoder(emb, mask=mask, src_key_padding_mask=pad_mask)
        return self.head(out)


# ---------- словарь ----------

def build_vocab(names: list[str]) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set("".join(names)))
    itos = ["<PAD>", "<BOS>", "<EOS>"] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi


def encode_name(name: str, stoi: dict[str, int]) -> list[int]:
    return [BOS] + [stoi[c] for c in name] + [EOS]


# ---------- батчи ----------

def make_batches(data: list[dict], batch_size: int, stoi: dict[str, int],
                 rng: random.Random):
    samples = [
        (encode_name(d["name"], stoi), STYLE_TO_ID[d["style"]]) for d in data
    ]
    rng.shuffle(samples)
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        max_len = max(len(seq) for seq, _ in chunk)
        xs, ys, styles = [], [], []
        for seq, style in chunk:
            padded = seq + [PAD] * (max_len - len(seq))
            xs.append(padded[:-1])
            ys.append(padded[1:])
            styles.append(style)
        yield torch.tensor(xs), torch.tensor(ys), torch.tensor(styles)


# ---------- обучение ----------

def train_transformer(train_data: list[dict], val_data: list[dict],
                      vocab_data: list[dict] | None = None,
                      d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                      dim_feedforward: int = 256, dropout: float = 0.2,
                      epochs: int = 100, lr: float = 1e-3, batch_size: int = 16,
                      patience: int = 15, seed: int = 42, verbose: bool = True):
    """Обучает Transformer и возвращает (model, itos, stoi, history)."""
    if vocab_data is None:
        vocab_data = train_data + val_data

    torch.manual_seed(seed)
    random.seed(seed)

    itos, stoi = build_vocab([d["name"] for d in vocab_data])
    model = TransformerNameGen(
        vocab_size=len(itos), d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    for epoch in range(1, epochs + 1):
        model.train()
        rng = random.Random(seed + epoch)
        train_losses = []
        for x, y, s in make_batches(train_data, batch_size, stoi, rng):
            optimizer.zero_grad()
            logits = model(x, s)
            loss = criterion(logits.reshape(-1, len(itos)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / max(1, len(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, s in make_batches(val_data, batch_size, stoi, random.Random(0)):
                logits = model(x, s)
                loss = criterion(logits.reshape(-1, len(itos)), y.reshape(-1))
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / max(1, len(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(float(torch.exp(torch.tensor(train_loss))))
        history["val_ppl"].append(float(torch.exp(torch.tensor(val_loss))))

        if verbose:
            print(f"epoch {epoch:3d} | "
                  f"train loss {train_loss:.4f} (ppl {history['train_ppl'][-1]:.2f}) | "
                  f"val loss {val_loss:.4f} (ppl {history['val_ppl'][-1]:.2f})")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping на эпохе {epoch} (best val loss {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, itos, stoi, history


# ---------- сэмплирование ----------

@torch.no_grad()
def sample(model: TransformerNameGen, itos: list[str], stoi: dict[str, int],
           style: str, temperature: float = 0.8, top_k: int = 40,
           top_p: float = 0.9, seed: int | None = None,
           max_len: int = 25) -> str:
    """Пошаговая генерация: на каждом шаге — forward по всей префиксной строке."""
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    seq = [BOS]
    style_id = STYLE_TO_ID[style]
    result = ""

    for _ in range(max_len):
        x = torch.tensor([seq])
        s = torch.tensor([style_id])
        logits = model(x, s)[:, -1, :] / max(1e-6, temperature)

        if top_k > 0 and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            threshold = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < threshold,
                                 torch.full_like(logits, float("-inf")),
                                 logits)

        probs = F.softmax(logits, dim=-1).squeeze(0)

        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum - sorted_probs >= top_p
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)

        # Служебные токены не должны попадать в результат:
        # PAD — паддинг, BOS — начало строки (не символ имени).
        probs[PAD] = 0.0
        probs[BOS] = 0.0
        if probs.sum() <= 0:
            break
        probs = probs / probs.sum()

        next_id = torch.multinomial(probs, 1, generator=generator).item()
        if next_id == EOS:
            break
        result += itos[next_id]
        seq.append(next_id)

    return result


# ---------- сохранение/загрузка ----------

def save_transformer(model: TransformerNameGen, itos: list[str],
                     stoi: dict[str, int], history: dict, path: Path) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "itos": itos,
        "stoi": stoi,
        "history": history,
        "config": {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "nhead": model.nhead,
            "num_layers": model.num_layers,
            "dim_feedforward": model.dim_feedforward,
            "dropout": model.dropout_p,
            "max_len": model.max_len,
        },
    }, path)


def load_transformer(path: Path) -> tuple[TransformerNameGen, list[str], dict[str, int]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TransformerNameGen(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["itos"], ckpt["stoi"]