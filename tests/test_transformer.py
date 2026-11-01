"""Тесты для Transformer-генератора."""
import torch

from ml.transformer_gen import (
    BOS, EOS, TransformerNameGen, build_vocab, encode_name, sample, train_transformer,
)


# ---------- словарь ----------

def test_build_vocab_basic():
    itos, stoi = build_vocab(["Аб", "Ва"])
    assert itos[0] == "<PAD>"
    assert itos[1] == "<BOS>"
    assert itos[2] == "<EOS>"
    for ch in "АбВа":
        assert ch in stoi


def test_encode_name():
    _, stoi = build_vocab(["Аб"])
    encoded = encode_name("Аб", stoi)
    assert encoded[0] == BOS
    assert encoded[-1] == EOS


# ---------- forward ----------

def test_model_forward_shape():
    itos, _ = build_vocab(["Абв"])
    model = TransformerNameGen(
        vocab_size=len(itos), d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0.0,
    )
    x = torch.tensor([[BOS, 3, 4]])
    s = torch.tensor([0])
    logits = model(x, s)
    assert logits.shape == (1, 3, len(itos))


def test_model_forward_with_pad():
    """PAD в конце не должен приводить к NaN."""
    itos, _ = build_vocab(["Абв"])
    model = TransformerNameGen(
        vocab_size=len(itos), d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0.0,
    )
    x = torch.tensor([[BOS, 3, 4, 0, 0]])
    s = torch.tensor([0])
    logits = model(x, s)
    assert not torch.isnan(logits).any()


# ---------- обучение ----------

def _tiny_data():
    return [
        {"name": "Аэлион", "style": "elven"},
        {"name": "Сильван", "style": "elven"},
        {"name": "Роланд", "style": "human"},
        {"name": "Роберт", "style": "human"},
    ]


def test_train_short():
    data = _tiny_data()
    model, itos, stoi, history = train_transformer(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )
    assert len(history["train_loss"]) <= 3
    assert all(loss > 0 for loss in history["train_loss"])


def test_train_with_oov_in_val():
    train = [{"name": "Аб", "style": "human"}]
    val = [{"name": "За", "style": "human"}]
    model, itos, stoi, _ = train_transformer(
        train, val, vocab_data=train + val,
        epochs=2, lr=1e-2, patience=100, verbose=False,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )
    assert "З" in stoi


# ---------- сэмплирование ----------

def test_sample_returns_string():
    data = _tiny_data()
    model, itos, stoi, _ = train_transformer(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )
    name = sample(model, itos, stoi, style="elven", seed=42)
    assert isinstance(name, str)


def test_sample_deterministic():
    data = _tiny_data()
    model, itos, stoi, _ = train_transformer(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )
    n1 = sample(model, itos, stoi, style="elven", seed=42)
    n2 = sample(model, itos, stoi, style="elven", seed=42)
    assert n1 == n2


def test_sample_only_vocab_chars():
    data = _tiny_data()
    model, itos, stoi, _ = train_transformer(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )
    allowed = set("".join(d["name"] for d in data))
    for seed in range(10):
        name = sample(model, itos, stoi, style="elven", seed=seed)
        assert all(ch in allowed for ch in name), f"Символ вне словаря: {name!r}"
