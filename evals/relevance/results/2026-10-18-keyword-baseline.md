# Relevance evaluation: Example AG

- Dataset: `evals/relevance/example-ag.jsonl` (40 items, 20 labelled relevant)
- Classifier: `fake/keyword-heuristic`
- Gate: relevance >= 6
- Runs: 1, at most 8 requests in parallel

| Run | Precision | Recall | F1 | Accuracy | Topic accuracy | LLM calls (failed) | Cost (USD) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.88 | 0.70 | 0.78 | 0.80 | 0.93 | 40 (0) | 0.0000 | 0.1 |

Per 1,000 headlines (mean over runs): $0.00, 0 input and 0 output tokens. Costs apply the configured list prices to every token (no cached-input discount).

## Threshold sweep

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 11 | 2 | 9 | 0.62 | 0.90 | 0.73 |
| 2 | 18 | 11 | 2 | 9 | 0.62 | 0.90 | 0.73 |
| 3 | 18 | 11 | 2 | 9 | 0.62 | 0.90 | 0.73 |
| 4 | 18 | 11 | 2 | 9 | 0.62 | 0.90 | 0.73 |
| 5 | 18 | 11 | 2 | 9 | 0.62 | 0.90 | 0.73 |
| 6 **(configured)** | 14 | 2 | 6 | 18 | 0.88 | 0.70 | 0.78 |
| 7 | 10 | 0 | 10 | 20 | 1.00 | 0.50 | 0.67 |
| 8 | 10 | 0 | 10 | 20 | 1.00 | 0.50 | 0.67 |
| 9 | 3 | 0 | 17 | 20 | 1.00 | 0.15 | 0.26 |
| 10 | 1 | 0 | 19 | 20 | 1.00 | 0.05 | 0.10 |

## Misclassified at threshold 6 (8 of 40 items)

| Item | Expected | Wrong in | Scores | Topic | Title | Reason |
|---|---|---:|---|---|---|---|
| r06 | relevant | 1/1 | 1 | none | Fire halts anodising line used by dozens of metalworking firms | no configured keyword or watchlist entity found |
| r08 | relevant | 1/1 | 5 | input_costs | European power prices spike as cold snap meets low wind output | keyword hits: 1 |
| r10 | relevant | 1/1 | 5 | input_costs | EU opens anti-dumping probe into imported aluminium extrusions | keyword hits: 1 |
| r11 | relevant | 1/1 | 5 | trade_regulation | Washington tightens export controls on precision machine tools | keyword hits: 1 |
| r12 | relevant | 1/1 | 1 | none | Certification backlog: device makers wait up to 18 months for approval | no configured keyword or watchlist entity found |
| r17 | relevant | 1/1 | 5 | competitors | Investor buys Austrian maker of implant components | keyword hits: 1 |
| n01 | irrelevant | 1/1 | 6 | supplier_customer_health | Football club enters administration as bankruptcy looms | keyword hits: 2 |
| n03 | irrelevant | 1/1 | 6 | trade_regulation | New mobile tariffs: operator cuts tariff prices for students | keyword hits: 2 |
