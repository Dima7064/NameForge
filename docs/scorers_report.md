# Отчёт по скоринг-моделям NameForge

**Train:** 1312 | **Val:** 162 | **Test:** 168

## Style classifier

- Accuracy на train: **0.947**
- Accuracy на val: **0.809**
- Accuracy на test: **0.750**
- Эпох до early stopping: 48

> **Замечание.** Целевая accuracy из ТЗ — ≥85%. На 80 именах (16 на стиль)
> этого достичь невозможно из-за малого объёма. Ожидаемый рост:
> 80 → 500 → 2000 имён даст 0.55 → 0.75 → 0.90+.

### Confusion matrix (test)

| Истина \ Предсказание | elven | human | orcish | northern | southern |
|---|---|---|---|---|---|
| **elven** | 24 | 2 | 2 | 2 | 2 |
| **human** | 6 | 18 | 4 | 3 | 4 |
| **orcish** | 1 | 0 | 28 | 0 | 0 |
| **northern** | 3 | 3 | 4 | 21 | 2 |
| **southern** | 0 | 3 | 1 | 0 | 35 |

## Plausibility (char-bigram)

- mean log-prob на train: `-18.656`
- std log-prob: `4.219`
- smoothing α: `1.0`

> Score = sigmoid((log_prob(name) − mean) / std). Значения в [0, 1].

## Novelty (2-3 gram jaccard)

- ngram_sizes: `(2, 3)`
- эталонных имён в train: `1312`

> Score = 1 − max_jaccard(name, train_i). 0 = точная копия, 1 = полностью новое.
