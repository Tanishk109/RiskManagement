# Local IEEE-CIS data

MerchantShield uses the labeled `train_transaction.csv` and `train_identity.csv` files from the IEEE-CIS Fraud Detection Kaggle competition. Raw and derived data are protected local inputs and are ignored by git.

## Kaggle CLI

1. Accept the competition rules in your own Kaggle account.
2. Configure the Kaggle CLI outside this repository.
3. Download and extract locally:

```bash
mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

Do not add `kaggle.json` to the project.

## Manual download

Download the competition archive from Kaggle after accepting its terms, extract it, and place only these required labeled files at:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Do not use Kaggle's original `test_transaction.csv` or `test_identity.csv` for the local held-out evaluation: they have no fraud labels. MerchantShield creates its held-out test from the later chronological portion of the labeled training files.

## Validate

```bash
make data-check
```

The command calculates row counts, fraud count/rate, `TransactionDT` boundaries, identity coverage, and join validity from the actual local files. It prints `STATUS: VALID` only when required columns and uniqueness checks pass.

## Redistribution

Before committing, publishing, or deploying any extracted IEEE-CIS row, review the current Kaggle competition/data rules. The default project keeps raw data, processed rows, prediction exports, and seeded evaluation cases local only.
