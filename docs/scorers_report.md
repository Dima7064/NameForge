# Отчёт по скоринг-моделям NameForge

**Train:** 1962 | **Val:** 244 | **Test:** 250

## Style classifier

- Accuracy на train: **0.902**
- Accuracy на val: **0.779**
- Accuracy на test: **0.740**
- Эпох до early stopping: 46

> **Замечание.** Целевая accuracy из ТЗ — ≥85%. На 80 именах (16 на стиль)
> этого достичь невозможно из-за малого объёма. Ожидаемый рост:
> 80 → 500 → 2000 имён даст 0.55 → 0.75 → 0.90+.

### Confusion matrix (test)

| Истина \ Предсказание | elven | human | orcish | northern | southern |
|---|---|---|---|---|---|
| **elven** | 29 | 5 | 3 | 1 | 3 |
| **human** | 3 | 36 | 2 | 5 | 13 |
| **orcish** | 0 | 1 | 35 | 0 | 2 |
| **northern** | 1 | 12 | 1 | 34 | 1 |
| **southern** | 1 | 8 | 1 | 2 | 51 |

## Plausibility (char-bigram)

- mean log-prob на train: `-18.176`
- std log-prob: `4.317`
- smoothing α: `1.0`

> Score = sigmoid((log_prob(name) − mean) / std). Значения в [0, 1].

## Novelty (2-3 gram jaccard)

- ngram_sizes: `(2, 3)`
- эталонных имён в train: `1962`

> Score = 1 − max_jaccard(name, train_i). 0 = точная копия, 1 = полностью новое.
