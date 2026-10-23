# AI usage

## Tools I used, and for which functions

I used ChatGPT to understand the task requirements, help implement and review `audit()` in `rawaj/audit.py` and the metric functions in `rawaj/metrics.py`, and discuss edge cases and expected behavior.

## One place where the AI was wrong, incomplete, or not what the spec said

One suggested approach for detecting constant columns did not fully account for the specification's treatment of missing values. I noticed the issue by comparing the suggestion with the task requirements, which treat different missing-value representations as one missing state. I reviewed the implementation to make sure the missing-value mask was used consistently.

## The functions I wrote without AI, and why

I implemented the functions in `rawaj/metrics.py` myself, except for `precision_at_k()`, `best_threshold()`, and `threshold_for_precision()`, where I used AI assistance. I wanted to understand the core evaluation metrics and implement their logic myself. For `rawaj/audit.py`, I used AI assistance during implementation, then reviewed the code against the specification and ran the public tests. I am responsible for understanding and checking the final implementation.