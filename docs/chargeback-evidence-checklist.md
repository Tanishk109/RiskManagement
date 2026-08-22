# Chargeback evidence checklist

MerchantShield uses this explicit internal checklist to organize evidence supplied by a merchant. It is a preparation aid, **not** a representation of payment-network rules, legal advice, or a prediction that a dispute will be won.

| Merchant-selected dispute category | Expected evidence categories |
| --- | --- |
| Item not received | Invoice, tracking, proof of delivery |
| Duplicate | Invoice, customer communication |
| Refund not received | Refund evidence, customer communication |
| Cancelled recurring | Customer communication, merchant policy |
| Not as described | Invoice, customer communication, merchant policy |
| Other | Invoice |

Completeness is calculated as `present expected categories / expected categories`. A higher value means only that more checklist categories are represented. It is never shown as success probability, network acceptance, or realized loss prevention.

Evidence files are limited to PDF, PNG, and JPEG, at 10 MB per file. PostgreSQL stores case, draft, and file metadata; uploaded bytes are handled by a replaceable file-storage adapter and are not stored in the database. The local adapter writes under `data/uploads/chargebacks/`, which is ignored by Git.

Generated drafts use only merchant-entered case fields and evidence filenames/categories. File contents are not interpreted. Missing checklist items are stated explicitly. A human must edit or approve a draft before export, and MerchantShield has no automatic bank or network submission endpoint.
