# Merchant rules

Rules are deliberately empty before real validation error analysis. MerchantShield does not ship plausible-sounding textbook fraud rules without evidence.

Each future rule must use only information available before the transaction is scored and document:

- the observed validation failure;
- rows affected;
- fraud caught;
- legitimate rows affected;
- cost impact under the named merchant assumptions;
- why `REVIEW` or `BLOCK` is justified.

Allowed actions are `NONE`, `REVIEW`, and `BLOCK`. Initial rules should normally escalate uncertain cases to `REVIEW`. Labels, future chargebacks, and future activity are forbidden as rule inputs.
