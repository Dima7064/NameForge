# Отчёт по скоринг-моделям NameForge

**Train:** 80 | **Val:** 10 | **Test:** 10

## Style classifier

- Accuracy на train: **0.887**
- Accuracy на val: **0.600**
- Accuracy на test: **0.800**
- Эпох до early stopping: 27

> **Замечание.** Целевая accuracy из ТЗ — ≥85%. На 80 именах (16 на стиль)
> этого достичь невозможно из-за малого объёма. Ожидаемый рост:
> 80 → 500 → 2000 имён даст 0.55 → 0.75 → 0.90+.

### Confusion matrix (test)

| Истина \ Предсказание | elven | human | orcish | northern | southern |
|---|---|---|---|---|---|
| **elven** | 2 | 0 | 0 | 0 | 0 |
| **human** | 0 | 2 | 0 | 0 | 0 |
| **orcish** | 0 | 0 | 1 | 0 | 1 |
| **northern** | 0 | 0 | 0 | 2 | 0 |
| **southern** | 0 | 1 | 0 | 0 | 1 |

## Plausibility (char-bigram)

- mean log-prob на train: `-20.612`
- std log-prob: `3.520`
- smoothing α: `1.0`

> Score = sigmoid((log_prob(name) − mean) / std). Значения в [0, 1].

## Novelty (2-3 gram jaccard)

- ngram_sizes: `(2, 3)`
- эталонных имён в train: `80`

> Score = 1 − max_jaccard(name, train_i). 0 = точная копия, 1 = полностью новое.
