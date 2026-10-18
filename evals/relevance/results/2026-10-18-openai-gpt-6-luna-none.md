# Relevance evaluation: Example AG

- Dataset: `evals/relevance/example-ag.jsonl` (40 items, 20 labelled relevant)
- Classifier: `openai/gpt-6-luna`
- Settings: temperature provider default, reasoning effort none
- Gate: relevance >= 6
- Runs: 5, at most 8 requests in parallel

| Run | Precision | Recall | F1 | Accuracy | Topic accuracy | LLM calls (failed) | Cost (USD) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 40 (0) | 0.0065 | 7.5 |
| 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 40 (0) | 0.0065 | 7.7 |
| 3 | 1.00 | 0.95 | 0.97 | 0.97 | 1.00 | 40 (0) | 0.0065 | 7.7 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 40 (0) | 0.0065 | 7.4 |
| 5 | 1.00 | 0.95 | 0.97 | 0.97 | 1.00 | 40 (0) | 0.0065 | 7.4 |
| **Mean** | 1.00 | 0.98 | 0.99 | 0.99 | 1.00 | 40.0 (0.0) | 0.0065 | 7.5 |

Per 1,000 headlines (mean over runs): $0.16, 1,370,000 input and 52,470 output tokens. Costs apply the configured list prices to every token (no cached-input discount).

## Threshold sweep (all 5 runs pooled)

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 2 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 3 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 4 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 5 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 6 **(configured)** | 98 | 0 | 2 | 100 | 1.00 | 0.98 | 0.99 |
| 7 | 67 | 0 | 33 | 100 | 1.00 | 0.67 | 0.80 |
| 8 | 50 | 0 | 50 | 100 | 1.00 | 0.50 | 0.67 |
| 9 | 15 | 0 | 85 | 100 | 1.00 | 0.15 | 0.26 |
| 10 | 6 | 0 | 94 | 100 | 1.00 | 0.06 | 0.11 |

## Misclassified at threshold 6 (1 of 40 items)

| Item | Expected | Wrong in | Scores | Topic | Title | Reason |
|---|---|---:|---|---|---|---|
| r18 | relevant | 2/5 | 6, 6, 5, 6, 5 | rates_currency | ECB cuts deposit rate by a quarter point | The ECB rate cut is a relevant euro-area financing signal, though no direct effect on Example AG is specified. |
