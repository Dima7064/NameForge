"""Plausibility scorer: насколько имя правдоподобно как язык.

Использует char-bigram языковую модель с Laplace smoothing:
    P(имя) = Π P(c_i | c_{i-1}),  P(c | prev) = (count(prev,c) + α) / (count(prev) + α·V)

Калибровка: сырой log-prob нормируется z-score по train-именам и переводится
в [0,1] через сигмоиду. Так самые «естественные» train-имена получают ~0.5+,
а мусор — ближе к 0.
"""
import math


class BigramPlausibility:
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.counts: dict[str, int] = {}            # "ab" → сколько раз
        self.context_counts: dict[str, int] = {}    # "a" → сколько раз был левым символом
        self.vocab: set[str] = set()
        self.mean_lp: float = 0.0
        self.std_lp: float = 1.0

    def train(self, names: list[str]) -> None:
        self.vocab = set("".join(names))
        self.counts = {}
        self.context_counts = {}
        for name in names:
            padded = "^" + name + "$"
            for i in range(len(padded) - 1):
                pair = padded[i:i + 2]
                self.counts[pair] = self.counts.get(pair, 0) + 1
                self.context_counts[padded[i]] = self.context_counts.get(padded[i], 0) + 1

        # Калибровка по train
        lps = [self._log_prob(name) for name in names]
        if lps:
            self.mean_lp = sum(lps) / len(lps)
            var = sum((x - self.mean_lp) ** 2 for x in lps) / max(1, len(lps) - 1)
            self.std_lp = math.sqrt(var) if var > 0 else 1.0
        else:
            self.mean_lp = 0.0
            self.std_lp = 1.0

    def _log_prob(self, name: str) -> float:
        padded = "^" + name + "$"
        V = len(self.vocab) + 2  # +2 за ^ и $
        total = 0.0
        for i in range(len(padded) - 1):
            prev = padded[i]
            pair = padded[i:i + 2]
            c_pair = self.counts.get(pair, 0)
            c_prev = self.context_counts.get(prev, 0)
            p = (c_pair + self.smoothing) / (c_prev + self.smoothing * V)
            total += math.log(p)
        return total

    def score(self, name: str) -> float:
        """Возвращает [0,1]: больше — правдоподобнее."""
        if not name:
            return 0.0
        lp = self._log_prob(name)
        z = (lp - self.mean_lp) / self.std_lp if self.std_lp > 0 else 0.0
        # sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)
