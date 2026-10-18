# Relevance evaluation: Example AG

- Dataset: `evals/relevance/example-ag.jsonl` (40 items, 20 labelled relevant)
- Classifier: `openai/gpt-4o-mini`
- Gate: relevance >= 6

| Precision | Recall | F1 | Accuracy | Topic accuracy |
|---:|---:|---:|---:|---:|
| 1.00 | 0.95 | 0.97 | 0.97 | 1.00 |

## Threshold sweep

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0 | 0 | 20 | 1.00 | 1.00 | 1.00 |
| 2 | 20 | 0 | 0 | 20 | 1.00 | 1.00 | 1.00 |
| 3 | 20 | 0 | 0 | 20 | 1.00 | 1.00 | 1.00 |
| 4 | 20 | 0 | 0 | 20 | 1.00 | 1.00 | 1.00 |
| 5 | 20 | 0 | 0 | 20 | 1.00 | 1.00 | 1.00 |
| 6 **(configured)** | 19 | 0 | 1 | 20 | 1.00 | 0.95 | 0.97 |
| 7 | 16 | 0 | 4 | 20 | 1.00 | 0.80 | 0.89 |
| 8 | 9 | 0 | 11 | 20 | 1.00 | 0.45 | 0.62 |
| 9 | 1 | 0 | 19 | 20 | 1.00 | 0.05 | 0.10 |
| 10 | 0 | 0 | 20 | 20 | 0.00 | 0.00 | 0.00 |

## Misclassified at threshold 6 (1)

| Item | Expected | Score | Topic | Title | Reason |
|---|---|---:|---|---|---|
| r18 | relevant | 5 | rates_currency | ECB cuts deposit rate by a quarter point | The ECB's rate cut could affect financing costs and export margins for Example AG. |
