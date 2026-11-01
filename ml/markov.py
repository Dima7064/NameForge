"""Марковская цепь фиксированного порядка (бейзлайн 1).

Модель посимвольная: P(следующий | последние k символов) = частота из train.
Если контекст длины k не встречался, откатываемся на k-1 (backoff).
"""
import random
from collections import Counter

START = "^"
END = "$"


class MarkovModel:
    def __init__(self, order: int = 2):
        if order < 1:
            raise ValueError("order должен быть >= 1")
        self.order = order
        # counts[ctx] = Counter(следующий символ)
        self.counts: dict[str, Counter] = {}

    def _fit_one(self, name: str) -> None:
        padded = START * (self.order - 1) + name + END
        for k in range(1, self.order):
            for i in range(len(padded) - k):
                ctx = padded[i:i + k]
                next_ch = padded[i + k]
                self.counts.setdefault(ctx, Counter())[next_ch] += 1

    def train(self, names: list[str]) -> None:
        self.counts = {}
        for name in names:
            self._fit_one(name)

    def _sample_next(self, rng: random.Random, ctx: str) -> str | None:
        """Возвращает следующий символ по backoff от полного контекста до 1."""
        for k in range(len(ctx), 0, -1):
            sub_ctx = ctx[-k:]
            if sub_ctx in self.counts:
                counter = self.counts[sub_ctx]
                chars = list(counter.keys())
                weights = list(counter.values())
                return rng.choices(chars, weights=weights)[0]
        return None

    def sample(self, rng: random.Random, max_len: int = 20) -> str:
        """Сгенерировать одно имя. Возвращает пустую строку, если не удалось."""
        result = ""
        ctx = START * (self.order - 1)
        for _ in range(max_len):
            next_ch = self._sample_next(rng, ctx)
            if next_ch is None or next_ch == END:
                break
            if next_ch == START:
                continue
            result += next_ch
            ctx = (ctx + next_ch)[-(self.order - 1):]
        return result
