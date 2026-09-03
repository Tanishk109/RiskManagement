# Final Held-Out Evaluation

Generated from the final chronological 15% of the local labeled IEEE-CIS training data.
The saved CatBoost candidate and validation-selected thresholds were used without retraining.

## Model-derived results

- Test transactions: 88,581
- Fraud transactions: 3,083
- Precision at block threshold: 0.382498
- Recall at block threshold: 0.351606
- F1: 0.366402
- Average precision / PR-AUC: 0.382920
- ROC-AUC: 0.851372
- TP / FP / TN / FN: 1084 / 1750 / 83748 / 1999
- APPROVE / REVIEW / BLOCK: 84693 / 1054 / 2834

## Frozen validation operating point

- Scenario: Scenario B — Moderate merchant
- Review threshold: 0.1750
- Block threshold: 0.2500
- Selection reason: lowest estimated cost feasible under a 2% review-capacity limit.

## Estimated business cost

All monetary values below use illustrative merchant assumptions, not industry facts:

- Fraud loss: 346942.20 INR
- False-positive cost: 81533.13 INR
- Manual review cost: 26350.00 INR
- Total estimated cost: 454825.32 INR
