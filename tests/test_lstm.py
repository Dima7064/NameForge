
"""Тесты для LSTM-генератора."""
import torch

from ml.lstm_gen import (
    BOS, EOS, LSTMNameGen, build_vocab, encode_name, sample, train_lstm,
)


# ---------- словарь ----------

def test_build_vocab_basic():
    itos, stoi = build_vocab(["Аб", "Ва"])
    assert itos[0] == "<PAD>"
    assert itos[1] == "<BOS>"
    assert itos[2] == "<EOS>"
    for ch in "АбВа":
        assert ch in stoi
    assert len(itos) == len(stoi)


def test_encode_name():
    _, stoi = build_vocab(["Аб"])
    encoded = encode_name("Аб", stoi)
    assert encoded[0] == BOS
    assert encoded[-1] == EOS
    assert len(encoded) == 4  # BOS + 2 буквы + EOS


# ---------- forward ----------

def test_model_forward_shape():
    itos, _ = build_vocab(["Абв"])
    model = LSTMNameGen(vocab_size=len(itos), hidden_size=16,
                        emb_size=8, style_emb_size=4, num_layers=1)
    x = torch.tensor([[BOS, 3, 4]])
    s = torch.tensor([0])
    logits, _ = model(x, s)
    assert logits.shape == (1, 3, len(itos))


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
    model, itos, stoi, history = train_lstm(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
    )
    assert len(history["train_loss"]) <= 3
    assert all(loss > 0 for loss in history["train_loss"])
    assert all(ppl > 1.0 for ppl in history["train_ppl"])


def test_train_with_oov_in_val():
    """Регрессионный тест: если в val есть символ, отсутствующий в train,
    обучение не падает (символ добавляется в vocab через vocab_data)."""
    train = [{"name": "Аб", "style": "human"}]
    val = [{"name": "За", "style": "human"}]  # 'З' нет в train
    model, itos, stoi, _ = train_lstm(
        train, val, vocab_data=train + val,
        epochs=2, lr=1e-2, patience=100, verbose=False,
    )
    assert "З" in stoi


# ---------- сэмплирование ----------

def test_sample_returns_string():
    data = _tiny_data()
    model, itos, stoi, _ = train_lstm(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
    )
    name = sample(model, itos, stoi, style="elven", seed=42)
    assert isinstance(name, str)


def test_sample_deterministic():
    data = _tiny_data()
    model, itos, stoi, _ = train_lstm(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
    )
    n1 = sample(model, itos, stoi, style="elven", seed=42)
    n2 = sample(model, itos, stoi, style="elven", seed=42)
    assert n1 == n2


def test_sample_only_vocab_chars():
    data = _tiny_data()
    model, itos, stoi, _ = train_lstm(
        data, data, epochs=3, lr=1e-2, patience=100, verbose=False,
    )
    allowed = set("".join(d["name"] for d in data))
    for seed in range(10):
        name = sample(model, itos, stoi, style="elven", seed=seed)
        assert all(ch in allowed for ch in name), f"Символ вне словаря: {name!r}"