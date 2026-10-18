# Relevance evaluation: Example AG

- Dataset: `evals/relevance/example-ag.jsonl` (40 items, 20 labelled relevant)
- Classifier: `openai/gpt-4o-mini`
- Settings: temperature 0, reasoning effort provider default
- Gate: relevance >= 6
- Runs: 5, at most 8 requests in parallel

| Run | Precision | Recall | F1 | Accuracy | Topic accuracy | LLM calls (failed) | Cost (USD) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.95 | 0.97 | 0.97 | 1.00 | 40 (0) | 0.0093 | 4.5 |
| 2 | 1.00 | 0.95 | 0.97 | 0.97 | 1.00 | 40 (0) | 0.0093 | 4.4 |
| 3 | 1.00 | 0.90 | 0.95 | 0.95 | 1.00 | 40 (0) | 0.0093 | 4.8 |
| 4 | 1.00 | 0.90 | 0.95 | 0.95 | 1.00 | 40 (0) | 0.0093 | 4.3 |
| 5 | 1.00 | 0.95 | 0.97 | 0.97 | 1.00 | 40 (0) | 0.0093 | 4.5 |
| **Mean** | 1.00 | 0.93 | 0.96 | 0.97 | 1.00 | 40.0 (0.0) | 0.0093 | 4.5 |

Per 1,000 headlines (mean over runs): $0.23, 1,371,000 input and 42,925 output tokens. Costs apply the configured list prices to every token (no cached-input discount).

## Threshold sweep (all 5 runs pooled)

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98 | 0 | 2 | 100 | 1.00 | 0.98 | 0.99 |
| 2 | 98 | 0 | 2 | 100 | 1.00 | 0.98 | 0.99 |
| 3 | 98 | 0 | 2 | 100 | 1.00 | 0.98 | 0.99 |
| 4 | 98 | 0 | 2 | 100 | 1.00 | 0.98 | 0.99 |
| 5 | 98 | 0 | 2 | 100 | 1.00 | 0.98 | 0.99 |
| 6 **(configured)** | 93 | 0 | 7 | 100 | 1.00 | 0.93 | 0.96 |
| 7 | 80 | 0 | 20 | 100 | 1.00 | 0.80 | 0.89 |
| 8 | 50 | 0 | 50 | 100 | 1.00 | 0.50 | 0.67 |
| 9 | 5 | 0 | 95 | 100 | 1.00 | 0.05 | 0.10 |
| 10 | 0 | 0 | 100 | 100 | 0.00 | 0.00 | 0.00 |

## Misclassified at threshold 6 (2 of 40 items)

| Item | Expected | Wrong in | Scores | Topic | Title | Reason |
|---|---|---:|---|---|---|---|
| r17 | relevant | 2/5 | 6, 6, 2, 2, 6 | none | Investor buys Austrian maker of implant components | The article discusses an acquisition in the precision machining sector but lacks direct relevance to Example AG's supply chain or customers. |
| r18 | relevant | 5/5 | 5, 5, 5, 5, 5 | rates_currency | ECB cuts deposit rate by a quarter point | The ECB's rate cut could affect financing costs and export margins for Example AG. |
