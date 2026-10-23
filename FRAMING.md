# Framing: Rawaj Pay mule detection

## Decision

Rawaj Pay needs to identify potentially suspicious transactions that may be linked to mule activity. The model should help the fraud team decide which transactions to investigate first. Since the team has only five analysts, the system must prioritize the most suspicious cases instead of overwhelming them with alerts. False positives waste analysts' time, while false negatives allow suspicious transactions to go undetected.

## Metric

Precision and recall are important because the system must balance the number of suspicious transactions detected against the number of legitimate transactions incorrectly flagged. Precision@5 is particularly relevant because the team can prioritize five cases for review. Accuracy alone is not sufficient because it can hide poor performance on a minority class. The decision threshold should reflect the trade-off between false positives and false negatives, and should be evaluated using the available labels and business costs.

## Baseline

The baseline is to evaluate the existing risk scores and determine a suitable threshold for flagging potentially suspicious transactions. The threshold should balance precision and recall while considering the cost of false positives and false negatives. Since Rawaj Pay has only five analysts, the system should prioritize the most suspicious cases without overwhelming the team. Any new approach should be compared against this baseline using the same labelled data and evaluation metrics. Before evaluating performance, the transaction data must be audited for missing values, duplicates, invalid timestamps, and potential target leakage.

## Classical ML or LLM

Classical machine learning is the more suitable starting point because Rawaj Pay's transaction data is structured and the goal is to assign risk scores to transactions. A classification model can learn patterns from labelled examples and be evaluated using precision, recall, and Precision@5. An LLM is not necessary for the core detection task because it does not primarily involve understanding or generating natural language. The final choice should be based on measured performance, data quality, interpretability, and the cost of incorrect decisions.