"""Novelty scorer: насколько имя ново относительно train-корпуса.

Метрика: 1 − max_jaccard(n-граммы имени, n-граммы train-имени).
    - Если имя точно совпадает с каким-то из train → novelty = 0.
    - Если пересечений n-грамм нет ни с одним train-именем → novelty = 1.
    - «Похоже на что-то из train, но не копия» → где-то между.

Берём n = 2 и 3 (биграммы и триграммы символов).
"""


class NoveltyScorer:
    def __init__(self, ngram_sizes: tuple[int, ...] = (2, 3)):
        self.ngram_sizes = ngram_sizes
        self.train_ngrams: list[set[str]] = []

    def train(self, names: list[str]) -> None:
        self.train_ngrams = [self._ngrams(n) for n in names]

    def _ngrams(self, name: str) -> set[str]:
        grams: set[str] = set()
        padded = "^" + name + "$"
        for n in self.ngram_sizes:
            if n > len(padded):
                continue
            for i in range(len(padded) - n + 1):
                grams.add(padded[i:i + n])
        return grams

    def score(self, name: str) -> float:
        """Возвращает [0,1]: 0 = точная копия из train, 1 = полностью новое."""
        grams = self._ngrams(name)
        if not grams or not self.train_ngrams:
            return 1.0
        max_sim = 0.0
        for tg in self.train_ngrams:
            inter = len(grams & tg)
            if inter == 0:
                continue
            union = len(grams | tg)
            if union > 0:
                sim = inter / union
                if sim > max_sim:
                    max_sim = sim
        return 1.0 - max_sim
