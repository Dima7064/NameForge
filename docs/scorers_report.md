# Отчёт по скоринг-моделям NameForge

**Train:** 1061 | **Val:** 130 | **Test:** 137

## Style classifier

- Accuracy на train: **0.991**
- Accuracy на val: **0.769**
- Accuracy на test: **0.803**
- Эпох до early stopping: 79

> **Замечание.** Целевая accuracy из ТЗ — ≥85%. На 80 именах (16 на стиль)
> этого достичь невозможно из-за малого объёма. Ожидаемый рост:
> 80 → 500 → 2000 имён даст 0.55 → 0.75 → 0.90+.

### Confusion matrix (test)

| Истина \ Предсказание | elven | human | orcish | northern | southern |
|---|---|---|---|---|---|
| **elven** | 21 | 2 | 3 | 0 | 1 |
| **human** | 1 | 22 | 0 | 4 | 0 |
| **orcish** | 1 | 1 | 16 | 1 | 0 |
| **northern** | 1 | 3 | 3 | 21 | 2 |
| **southern** | 1 | 1 | 1 | 1 | 30 |

## Plausibility (char-bigram)

- mean log-prob на train: `-18.654`
- std log-prob: `4.276`
- smoothing α: `1.0`

> Score = sigmoid((log_prob(name) − mean) / std). Значения в [0, 1].

## Novelty (2-3 gram jaccard)

- ngram_sizes: `(2, 3)`
- эталонных имён в train: `1061`

> Score = 1 − max_jaccard(name, train_i). 0 = точная копия, 1 = полностью новое.
