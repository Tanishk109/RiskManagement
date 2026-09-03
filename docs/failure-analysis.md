# Failure Analysis

The validation-only Logistic Regression baseline has now been analyzed at the fixed descriptive threshold 0.50. This is not held-out performance and does not support an operational rule or threshold.

- False positives: 83; false negatives: 2,825; true positives: 217; true negatives: 85,456.
- False-negative `TransactionAmt`: total 481,947.732, median 77.000, P90 445.000, P95 653.590, maximum 5,191.000.
- The ten highest-value false negatives range from 1,795.800 to 5,191.000; all are `ProductCD=W` and lack an identity-table match.
- Supported zero-recall slices are `ProductCD=W`, `ProductCD=S`, `card4=discover`, and the fixed `TransactionAmt >= 500` bucket.
- Identity-unavailable fraud recall is 0.003281 across 1,524 fraud rows. Conversely, 82 of 83 false positives have identity available, reinforcing the risk that identity features partly encode the enrichment process.

These findings describe the selected unweighted model specifically at threshold 0.50. Its threshold-independent AP is 0.231434, but its recall at 0.50 is only 0.071335; threshold behavior must not be confused with ranking quality. Full counts, category comparisons, examples, and support filters are in `artifacts/reports/baseline_error_analysis.md`. No rules were created.

## CatBoost validation candidate

The selected identity-free CatBoost candidate improves every frozen Logistic Regression failure slice at threshold 0.50, but does not eliminate the weaknesses:

- `ProductCD=W` recall rises from 0 to 0.047811 over 1,485 fraud rows.
- Amount `>=500` recall rises from 0 to 0.258929 over 224 fraud rows.
- `card4=discover` recall rises from 0 to 0.196721 over 122 fraud rows.
- `ProductCD=S` recall rises from 0 to 0.168224 over 107 fraud rows.
- Identity-unavailable recall rises from 0.003281 to 0.056430 over 1,524 fraud rows.
- Missing-`DeviceType` recall rises from 0.004507 to 0.059240 over 1,553 fraud rows.

CatBoost still misses 2,304 validation fraud rows totaling 378,333.733 in `TransactionAmt`; median is 85.796, P90 400.000, P95 554.000, and maximum 3,822.950. Relative to Logistic Regression, this is 521 fewer false negatives and 103,613.999 less false-negative amount at the same descriptive threshold. The remaining `ProductCD=W` and identity-unavailable/missing-device recall is especially weak and must not be concealed by aggregate AP gains. No rules or threshold decisions were derived from these errors.

## Provisional three-way validation point

Under **ILLUSTRATIVE MERCHANT ASSUMPTIONS** for Scenario B, the provisional validation operating point uses review threshold 0.175 and block threshold 0.250. This is not held-out performance or a final threshold. It routes 148 fraud rows to review and 1,070 to block, while 1,824 fraud rows totaling 289,883.247 in `TransactionAmt` remain approved.

The earlier failure slices remain material even after adding `REVIEW`: `ProductCD=W` has 1,237/72/176 approved/reviewed/blocked fraud rows; amount `>=500` has 119/11/94; `card4=discover` has 59/11/52; and `ProductCD=S` has 58/11/38. The highest-value approved fraud remains TransactionID 3,455,844 at amount 3,822.95 and score 0.008379. These are validation residuals for later controlled rule-candidate analysis; no rules were implemented here.

## Final temporal test failure pattern

The frozen rule-free policy was evaluated once on the later 88,581-row held-out partition. At the 0.250 block threshold it produced 1,084 true positives, 1,750 false positives, and 1,999 false negatives, for precision 0.382498 and recall 0.351606. The review band contained 1,054 additional rows.

The main temporal failure is precision stability rather than block recall: validation block recall was 0.351742, while held-out recall is 0.351606, but precision fell from 0.616005 to 0.382498 as block volume rose from 1.960917% to 3.199332%. Estimated false-positive cost increased from INR 33,596.47 on validation to INR 81,533.13 on the equally sized held-out partition under the same illustrative assumptions. Expected fraud-amount capture also fell from 41.162812% to 36.206639%.

This is final test evidence of temporal generalization weakness. It is documented rather than repaired: no post-test threshold, feature, model, or rule tuning is permitted against this held-out result.
