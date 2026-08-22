# Abuse-Ring Sentinel

The Abuse-Ring Sentinel is a defensive NetworkX linkage analysis over the chronological IEEE-CIS validation partition. It identifies suspicious linked components for review; it does not confirm fraud rings or infer real-world identities.

## Data and attributes

The service reads transaction amount plus the six documented dataset fields below and joins them one-to-one to the frozen CatBoost validation probability and model version:

| Source field | Product wording |
| --- | --- |
| `card4` | card4 shared network attribute |
| `card6` | card6 shared card-type attribute |
| `P_emaildomain` | purchaser email-domain attribute |
| `R_emaildomain` | recipient email-domain attribute |
| `DeviceType` | device-type attribute |
| `DeviceInfo` | dataset-provided device-info attribute |

These attributes do not establish a card number, account, person, or common owner. MerchantShield retains the source field name and does not invent a meaning for a masked identifier.

## Graph construction

The graph is bipartite: one node class represents transactions and the other represents eligible shared attribute values. An edge means only `SHARES ATTRIBUTE`. By default, a value must appear on at least two validation transactions and no more than 50. Values above that maximum are suppressed because common email domains, device classes, and card attributes would otherwise form weak, non-specific giant components.

Connected components must contain at least two transactions. A component enters the suspicious list only when it meets the visible minimum high-risk transaction count and high-risk share based on the existing provisional review threshold. Components are ranked by high-risk transaction amount, high-risk count, and connectivity. None of these rules is a fraud label or identity inference.

## Real validation observation

With the default configuration, the real validation graph considered 88,581 rows and produced 4,337 linked transaction nodes, 519 eligible shared-attribute nodes, and 4,615 edges. It suppressed 63 over-common attribute values. Of 465 connected components with at least two transactions, 83 met the configurable suspicious-component rule; the API returns the top 25. These are validation observations, not detector-performance metrics.

## Evaluation and safety boundary

Evaluation status is **Not evaluated yet**. The graph service never reads `isFraud` or `actual_label`, rejects held-out artifact paths before file access, uses NetworkX only, and has no Neo4j dependency. The transaction search accepts a validation `TransactionID` and displays only its eligible shared-attribute neighborhood. A displayed link or component can be coincidental and must not be described as confirmed coordinated abuse.
