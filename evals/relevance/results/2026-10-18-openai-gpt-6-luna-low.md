# Relevance evaluation: Example AG

- Dataset: `evals/relevance/example-ag.jsonl` (40 items, 20 labelled relevant)
- Classifier: `openai/gpt-6-luna`
- Settings: temperature provider default, reasoning effort low
- Gate: relevance >= 6
- Runs: 5, at most 8 requests in parallel

| Run | Precision | Recall | F1 | Accuracy | Topic accuracy | LLM calls (failed) | Cost (USD) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.90 | 0.95 | 0.95 | 0.94 | 40 (0) | 0.0069 | 9.0 |
| 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 40 (0) | 0.0070 | 9.1 |
| 3 | 1.00 | 0.95 | 0.97 | 0.97 | 0.95 | 40 (0) | 0.0069 | 8.9 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 40 (0) | 0.0069 | 10.7 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 40 (0) | 0.0070 | 8.7 |
| **Mean** | 1.00 | 0.97 | 0.98 | 0.98 | 0.97 | 40.0 (0.0) | 0.0069 | 9.3 |

Per 1,000 headlines (mean over runs): $0.17, 1,370,000 input and 71,110 output tokens. Costs apply the configured list prices to every token (no cached-input discount).

## Threshold sweep (all 5 runs pooled)

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 2 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 3 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 4 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 5 | 100 | 0 | 0 | 100 | 1.00 | 1.00 | 1.00 |
| 6 **(configured)** | 97 | 0 | 3 | 100 | 1.00 | 0.97 | 0.98 |
| 7 | 71 | 0 | 29 | 100 | 1.00 | 0.71 | 0.83 |
| 8 | 44 | 0 | 56 | 100 | 1.00 | 0.44 | 0.61 |
| 9 | 17 | 0 | 83 | 100 | 1.00 | 0.17 | 0.29 |
| 10 | 6 | 0 | 94 | 100 | 1.00 | 0.06 | 0.11 |

## Misclassified at threshold 6 (2 of 40 items)

| Item | Expected | Wrong in | Scores | Topic | Title | Reason |
|---|---|---:|---|---|---|---|
| r11 | relevant | 2/5 | 5, 6, 5, 6, 6 | trade_regulation | Washington tightens export controls on precision machine tools | New export licensing requirements for precision machine tools are an indirect trade-policy signal for industrial manufacturing. |
| r18 | relevant | 1/5 | 5, 6, 6, 6, 6 | rates_currency | ECB cuts deposit rate by a quarter point | The ECB rate cut is a relevant macro signal that may affect financing costs and the euro, but no direct company impact is stated. |
