# Logistic Baseline Validation Error Analysis

All results use the selected Logistic Regression validation baseline and the fixed 0.50 classification threshold. No rule or operational threshold was derived from these errors.

## Outcome counts and amounts

| Outcome | Rows | Amount total | Amount median | P90 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| False Negatives | 2,825 | 481,947.732 | 77.000 | 445.000 | 653.590 | 5,191.000 |
| True Positives | 217 | 14,959.856 | 83.859 | 102.714 | 119.994 | 300.000 |
| False Positives | 83 | 3,669.486 | 31.660 | 88.162 | 157.093 | 176.448 |
| True Negatives | 85,456 | 11,446,916.620 | 67.950 | 261.950 | 424.950 | 5,279.950 |

## False negatives versus True positives

### ProductCD

- False negatives: W=52.57%, C=33.49%, R=5.42%, H=4.74%, S=3.79%
- True positives: C=94.01%, R=3.23%, H=2.76%

### card4

- False negatives: visa=65.66%, mastercard=28.99%, discover=4.32%, american express=1.03%
- True positives: visa=59.45%, mastercard=40.55%

### card6

- False negatives: debit=54.80%, credit=45.20%
- True positives: credit=64.98%, debit=35.02%

### identity_available

- False negatives: False=53.77%, True=46.23%
- True positives: True=97.70%, False=2.30%

### DeviceType

- False negatives: <MISSING>=54.73%, desktop=25.84%, mobile=19.43%
- True positives: desktop=54.84%, mobile=41.94%, <MISSING>=3.23%


## False positives versus True negatives

### ProductCD

- False positives: C=98.80%, R=1.20%
- True negatives: W=82.86%, C=9.24%, R=3.75%, H=2.59%, S=1.57%

### card4

- False positives: visa=54.22%, mastercard=45.78%
- True negatives: visa=65.20%, mastercard=32.97%, discover=0.98%, american express=0.84%, <MISSING>=0.00%

### card6

- False positives: credit=61.45%, debit=38.55%
- True negatives: debit=79.67%, credit=20.33%, <MISSING>=0.00%, debit or credit=0.00%

### identity_available

- False positives: True=98.80%, False=1.20%
- True negatives: False=83.55%, True=16.45%

### DeviceType

- False positives: mobile=53.01%, desktop=45.78%, <MISSING>=1.20%
- True negatives: <MISSING>=84.02%, desktop=9.19%, mobile=6.79%

## High-value false negatives

These are actual validation errors retained for inspection only; no rules were created.

| TransactionID | Amount | Probability | ProductCD | card4 | card6 | Identity | DeviceType |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 3409708 | 5,191.000 | 0.420636 | W | visa | debit | False | None |
| 3455844 | 3,822.950 | 0.240944 | W | visa | credit | False | None |
| 3436956 | 3,260.050 | 0.077770 | W | mastercard | debit | False | None |
| 3421871 | 3,190.000 | 0.184220 | W | visa | debit | False | None |
| 3418094 | 2,963.950 | 0.169609 | W | visa | credit | False | None |
| 3418107 | 2,963.950 | 0.177516 | W | visa | credit | False | None |
| 3442600 | 2,681.000 | 0.145367 | W | visa | debit | False | None |
| 3455964 | 2,268.390 | 0.124291 | W | visa | credit | False | None |
| 3488473 | 2,161.000 | 0.062708 | W | visa | debit | False | None |
| 3476245 | 1,795.800 | 0.023479 | W | visa | debit | False | None |

## Weakest meaningful recall slices

Only slices with at least 50 actual fraud rows are included.
Amount buckets are Fixed, left-closed ranges: 0–25, 25–50, 50–100, 100–250, 250–500, 500+.

| Slice | Category | Fraud support | TP | FN | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| `ProductCD` | W | 1,485 | 0 | 1,485 | 0.000000 |
| `TransactionAmt_fixed_bucket` | 500+ | 224 | 0 | 224 | 0.000000 |
| `card4` | discover | 122 | 0 | 122 | 0.000000 |
| `ProductCD` | S | 107 | 0 | 107 | 0.000000 |
| `identity_available` | False | 1,524 | 5 | 1,519 | 0.003281 |
| `DeviceType` | <MISSING> | 1,553 | 7 | 1,546 | 0.004507 |
| `TransactionAmt_fixed_bucket` | 250–500 | 263 | 2 | 261 | 0.007605 |
| `TransactionAmt_fixed_bucket` | 100–250 | 757 | 29 | 728 | 0.038309 |
| `ProductCD` | H | 140 | 6 | 134 | 0.042857 |
| `ProductCD` | R | 160 | 7 | 153 | 0.043750 |
| `card6` | debit | 1,624 | 76 | 1,548 | 0.046798 |
| `card4` | visa | 1,984 | 129 | 1,855 | 0.065020 |
| `TransactionAmt_fixed_bucket` | 25–50 | 644 | 53 | 591 | 0.082298 |
| `TransactionAmt_fixed_bucket` | 0–25 | 360 | 30 | 330 | 0.083333 |
| `card4` | mastercard | 907 | 88 | 819 | 0.097023 |
