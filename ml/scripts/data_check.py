from __future__ import annotations

from common import RAW_DATA
from merchantshield_ml.data import load_ieee_cis


def main() -> None:
    try:
        _, report = load_ieee_cis(
            RAW_DATA / "train_transaction.csv",
            RAW_DATA / "train_identity.csv",
            feature_names=[],
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"STATUS: INVALID\n{exc}")
        raise SystemExit(2) from exc
    print(f"Transaction rows:        {report.transaction_rows:,}")
    print(f"Identity rows:           {report.identity_rows:,}")
    print(f"Merged rows:             {report.merged_rows:,}")
    print(f"Fraud rows:              {report.fraud_rows:,}")
    print(f"Fraud percentage:        {report.fraud_percentage:.6f}%")
    print(f"TransactionDT min:       {report.transaction_dt_min:.0f}")
    print(f"TransactionDT max:       {report.transaction_dt_max:.0f}")
    print(f"Identity coverage:       {report.identity_coverage:.6f}%")
    print(f"STATUS: {report.status}")


if __name__ == "__main__":
    main()
