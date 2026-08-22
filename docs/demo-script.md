# Demo Script

## Before presenting

Confirm that the UI says either “Held-out results loaded” with a real final artifact or “Not evaluated yet.” Never present fixture test behavior as fraud performance.

## Evidence-backed path

1. Open Overview and state that the displayed numbers come from transactions later in time than training data.
2. Open one programmatically selected correct high-confidence fraud case and explain the score using source feature names.
3. Open a genuine false positive or false negative and say clearly: “The model gets this case wrong.”
4. If a validation-derived rule genuinely escalates a model miss, show the model, rule, and ground truth sequence. Otherwise do not imply that example exists.
5. Open Cost Lab, move the review and block thresholds, and explain the resulting recall, false-positive, review-volume, and merchant-cost changes.
6. Finish: “The goal isn't maximum fraud recall. The goal is minimum merchant loss under an acceptable customer-friction level.”

## Current checkout

The real dataset has passed local validation and EDA and has chronological train/validation/test partitions, but no model or held-out performance evaluation exists. Demonstrate the honest empty-state behavior, threshold invariant, API validation, and persistence tests; do not quote precision, recall, cost, savings, or latency.
