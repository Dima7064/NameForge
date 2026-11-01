"""N-граммы с интерполированным Kneser-Ney (бейзлайн 2).

Формула (для контекста длины k):
    P(w | ctx) = max(c(ctx, w) − D, 0) / c(ctx) + λ(ctx) · P(w | ctx[1:])
    λ(ctx)     = D · N_unique(ctx) / c(ctx)
Для униграммы (пустой ctx) используется continuation counts:
    P(w) = |{v : c(v, w) > 0}| / Σ_w' |{v : c(v, w') > 0}|
"""
import random
from collections import Counter, defaultdict

START = "^"
END = "$"


class KneserNey:
    def __init__(self, order: int = 3, discount: float = 0.75):
        if order < 2:
            raise ValueError("order должен быть >= 2")
        self.order = order
        self.d = discount
        self.vocab: set[str] = set()
        # counts[ctx] = Counter(next_ch) — для ctx длины 1..order-1
        self.counts: dict[str, Counter] = {}
        self.totals: dict[str, int] = {}
        self.n_unique: dict[str, int] = {}
        # continuation counts для униграммы
        self.cont_counts: Counter = Counter()

    def train(self, names: list[str]) -> None:
        self.vocab = {END}
        self.counts = {}
        self.totals = {}
        self.n_unique = {}
        self.cont_counts = Counter()
        predecessors: dict[str, set[str]] = defaultdict(set)

        for name in names:
            padded = START * (self.order - 1) + name + END
            for ch in padded:
                self.vocab.add(ch)
            # n-граммы длины 1..order-1 → следующий символ
            for k in range(1, self.order):
                for i in range(len(padded) - k):
                    ctx = padded[i:i + k]
                    next_ch = padded[i + k]
                    if ctx not in self.counts:
                        self.counts[ctx] = Counter()
                        self.totals[ctx] = 0
                        self.n_unique[ctx] = 0
                    if next_ch not in self.counts[ctx]:
                        self.n_unique[ctx] += 1
                    self.counts[ctx][next_ch] += 1
                    self.totals[ctx] += 1
            # continuation counts: для каждого символа — уникальные предшественники
            for i in range(1, len(padded)):
                predecessors[padded[i]].add(padded[i - 1])
        for ch, preds in predecessors.items():
            self.cont_counts[ch] = len(preds)

    def prob(self, ctx: str, w: str) -> float:
        """P(w | ctx). ctx длины ≤ order-1; пустой ctx — униграмма."""
        if len(ctx) > self.order - 1:
            raise ValueError(f"ctx слишком длинный: {len(ctx)} > {self.order - 1}")
        if not ctx:
            total = sum(self.cont_counts.values())
            return self.cont_counts.get(w, 0) / total if total > 0 else 0.0
        if ctx not in self.counts:
            return self.prob(ctx[1:], w)
        c_ctx_w = self.counts[ctx].get(w, 0)
        c_ctx = self.totals[ctx]
        n_unique = self.n_unique[ctx]
        p_high = max(c_ctx_w - self.d, 0.0) / c_ctx
        lambda_ctx = self.d * n_unique / c_ctx
        p_low = self.prob(ctx[1:], w)
        return p_high + lambda_ctx * p_low

    def distribution(self, ctx: str) -> list[tuple[str, float]]:
        """Распределение по всему словарю. Гарантирует, что сумма = 1."""
        vocab = sorted(self.vocab)
        probs = [self.prob(ctx, w) for w in vocab]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return list(zip(vocab, probs))

    def sample(self, rng: random.Random, max_len: int = 20) -> str:
        result = ""
        for _ in range(max_len):
            ctx = (START * (self.order - 1) + result)[-(self.order - 1):]
            chars, weights = zip(*self.distribution(ctx))
            next_ch = rng.choices(chars, weights=weights)[0]
            if next_ch == END:
                break
            if next_ch == START:
                continue
            result += next_ch
        return result
