"""LSTM-генератор фэнтези-имён с условием на стиль."""
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2

STYLES = ["elven", "human", "orcish", "northern", "southern"]
STYLE_TO_ID = {s: i for i, s in enumerate(STYLES)}


class LSTMNameGen(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128,
                 emb_size: int = 64, style_emb_size: int = 32,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.char_emb = nn.Embedding(vocab_size, emb_size, padding_idx=PAD)
        self.style_emb = nn.Embedding(len(STYLES), style_emb_size)
        self.lstm = nn.LSTM(
            input_size=emb_size + style_emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, styles, hidden=None):
        emb = self.char_emb(x)
        st = self.style_emb(styles).unsqueeze(1)
        st = st.expand(-1, x.size(1), -1)
        inp = torch.cat([emb, st], dim=-1)
        out, hidden = self.lstm(inp, hidden)
        logits = self.head(self.dropout(out))
        return logits, hidden


def build_vocab(names):
    chars = sorted(set("".join(names)))
    itos = ["<PAD>", "<BOS>", "<EOS>"] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi


def encode_name(name, stoi):
    return [BOS] + [stoi[c] for c in name] + [EOS]


def make_batches(data, batch_size, stoi, rng):
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


def train_lstm(train_data, val_data, vocab_data=None,
               epochs=150, lr=1e-3, batch_size=16,
               weight_decay=1e-5,
               patience=20, seed=42, verbose=True):
    if vocab_data is None:
        vocab_data = train_data + val_data

    torch.manual_seed(seed)
    random.seed(seed)

    itos, stoi = build_vocab([d["name"] for d in vocab_data])
    model = LSTMNameGen(vocab_size=len(itos))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
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
            logits, _ = model(x, s)
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
                logits, _ = model(x, s)
                loss = criterion(logits.reshape(-1, len(itos)), y.reshape(-1))
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / max(1, len(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(float(torch.exp(torch.tensor(train_loss))))
        history["val_ppl"].append(float(torch.exp(torch.tensor(val_loss))))

        if verbose and (epoch % 10 == 0 or epoch == 1):
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


@torch.no_grad()
def sample(model, itos, stoi, style, temperature=0.8, top_k=40,
           top_p=0.9, seed=None, max_len=40):
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    x = torch.tensor([[BOS]])
    s = torch.tensor([STYLE_TO_ID[style]])
    hidden = None
    result = ""

    for _ in range(max_len):
        logits, hidden = model(x, s, hidden)
        logits = logits[:, -1, :] / max(1e-6, temperature)

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

        probs[PAD] = 0.0
        probs[BOS] = 0.0
        if probs.sum() <= 0:
            break
        probs = probs / probs.sum()

        next_id = torch.multinomial(probs, 1, generator=generator).item()
        if next_id == EOS:
            break
        result += itos[next_id]
        x = torch.tensor([[next_id]])

    return result


def save_lstm(model, itos, stoi, history, path):
    torch.save({
        "model_state": model.state_dict(),
        "itos": itos,
        "stoi": stoi,
        "history": history,
        "config": {
            "vocab_size": len(itos),
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "emb_size": model.char_emb.embedding_dim,
            "style_emb_size": model.style_emb.embedding_dim,
            "dropout": model.dropout.p,
        },
    }, path)


def load_lstm(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LSTMNameGen(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["itos"], ckpt["stoi"]
