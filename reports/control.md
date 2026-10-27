# ModelPulse audit report

Generated 2026-10-27 19:57 UTC by ModelPulse v1.0.0.

## 1. Verdict

**Neither drift nor decline.** No material drift and no material performance change were found. This does not prove the model is reliable - only that these checks did not find a problem in this batch.

- Drift status: **Stable** (0 of 17 features drifted; 0 more were statistically significant but below the effect-size threshold).
- Decision Tree: ROC-AUC 0.920 -> 0.920, F1 0.664 -> 0.664; material decline: **no**.
- Naive Bayes: ROC-AUC 0.874 -> 0.874, F1 0.609 -> 0.609; material decline: **no**.

## 2. Experiment settings

| Setting | Value |
|---|---|
| Scenario | Unchanged control |
| Scenario description | The held-out test set with no modification. Any drift flagged here is a false alarm, so this scenario checks the detector itself. |
| Severity | 0.00 (0 = none, 1 = most severe) |
| Features modified | none |
| Random seed | 2026 |
| Reference batch | training set, 8,631 sessions |
| Current batch | modified held-out test set, 3,699 sessions |
| Dataset SHA-256 | `b3055ee355f59134...` |
| Decision thresholds | Decision Tree 0.41, Naive Bayes 0.94 (chosen on validation data) |
| Decision Tree hyper-parameters | {'max_depth': 6, 'min_samples_leaf': 100} |

Pre-registered alerting rules:

- Feature drift: Benjamini-Hochberg adjusted p < 0.05 **and** KS D >= 0.1 (numerical) or total variation distance >= 0.1 (categorical).
- Batch status "Drift detected" when >= 20% of features drift; "Warning" when at least one does.
- Material performance decline: ROC-AUC drop >= 0.03 or F1 drop >= 0.05, with the paired bootstrap 95% CI (300 replicates) excluding zero.

Transformation log:

- Scenario: Unchanged control; severity 0.00; seed 2026; rows 3699.
- No transformation applied.

## 3. Data quality

| Check | Result |
|---|---|
| Rows in current batch | 3,699 |
| Missing cells | 0 (0 rows affected) |
| Values violating domain rules | 0 |
| Values outside training range | 5 |
| Category levels unseen in training | 1 |

| Schema check | Result | Detail |
|---|---|---|
| All 17 input features present | pass | ok |
| No unexpected columns | pass | ok |
| Numerical features are numeric | pass | ok |
| Labels available | pass | performance can be evaluated |

## 4. Detected changes

| Feature | Test | Effect | BH-adjusted p | Drifted | Change |
|---|---|---|---|---|---|
| TrafficType | Chi-square (Monte-Carlo) | TVD = 0.023 | 0.862 | no | '1' 20.2% -> 19.1% |
| ProductRelated | Kolmogorov-Smirnov | KS D = 0.020 | 0.862 | no | mean 32 -> 31; median 18 -> 18 |
| OperatingSystems | Chi-square (Monte-Carlo) | TVD = 0.019 | 0.862 | no | '3' 21.1% -> 19.9% |
| Region | Chi-square | TVD = 0.018 | 0.862 | no | '9' 3.9% -> 4.6% |
| BounceRates | Kolmogorov-Smirnov | KS D = 0.018 | 0.862 | no | mean 0.0221 -> 0.0224; median 0.0033 -> 0.00267 |
| ProductRelated_Duration | Kolmogorov-Smirnov | KS D = 0.018 | 0.862 | no | mean 1.2e+03 -> 1.19e+03; median 602 -> 591 |
| VisitorType | Chi-square | TVD = 0.017 | 0.449 | no | 'Returning_Visitor' 86.1% -> 84.4% |
| Month | Chi-square | TVD = 0.016 | 0.862 | no | 'Oct' 4.2% -> 5.1% |
| Administrative_Duration | Kolmogorov-Smirnov | KS D = 0.016 | 0.862 | no | mean 79.3 -> 84.4; median 8 -> 6 |
| Administrative | Kolmogorov-Smirnov | KS D = 0.014 | 0.862 | no | mean 2.32 -> 2.3; median 1 -> 1 |
| Browser | Chi-square (Monte-Carlo) | TVD = 0.014 | 0.862 | no | '4' 5.7% -> 6.7% |
| ExitRates | Kolmogorov-Smirnov | KS D = 0.014 | 0.862 | no | mean 0.0429 -> 0.0435; median 0.025 -> 0.0259 |
| Informational_Duration | Kolmogorov-Smirnov | KS D = 0.014 | 0.862 | no | mean 35.4 -> 32.2; median 0 -> 0 |
| Informational | Kolmogorov-Smirnov | KS D = 0.010 | 0.998 | no | mean 0.51 -> 0.49; median 0 -> 0 |
| SpecialDay | Kolmogorov-Smirnov | KS D = 0.008 | 0.998 | no | mean 0.0622 -> 0.0596; median 0 -> 0 |
| PageValues | Kolmogorov-Smirnov | KS D = 0.008 | 0.998 | no | mean 5.82 -> 6.06; median 0 -> 0 |
| Weekend | Chi-square | TVD = 0.008 | 0.862 | no | 'False' 77.0% -> 76.2% |

## 5. Model performance on identical sessions

### Decision Tree

| Metric | Baseline | Current | Change | 95% paired CI of change |
|---|---|---|---|---|
| Accuracy | 0.896 | 0.896 | +0.000 | [0.000, 0.000] |
| Precision | 0.665 | 0.665 | +0.000 | [0.000, 0.000] |
| Recall | 0.663 | 0.663 | +0.000 | [0.000, 0.000] |
| F1-score | 0.664 | 0.664 | +0.000 | [0.000, 0.000] |
| ROC-AUC | 0.920 | 0.920 | +0.000 | [0.000, 0.000] |
| Brier score (lower is better) | 0.072 | 0.072 | +0.000 | [0.000, 0.000] |

McNemar test on paired correctness: 0 sessions correct only before the shift, 0 only after (p = 1.000).

### Naive Bayes

| Metric | Baseline | Current | Change | 95% paired CI of change |
|---|---|---|---|---|
| Accuracy | 0.849 | 0.849 | +0.000 | [0.000, 0.000] |
| Precision | 0.508 | 0.508 | +0.000 | [0.000, 0.000] |
| Recall | 0.760 | 0.760 | +0.000 | [0.000, 0.000] |
| F1-score | 0.609 | 0.609 | +0.000 | [0.000, 0.000] |
| ROC-AUC | 0.874 | 0.874 | +0.000 | [0.000, 0.000] |
| Brier score (lower is better) | 0.217 | 0.217 | +0.000 | [0.000, 0.000] |

McNemar test on paired correctness: 0 sessions correct only before the shift, 0 only after (p = 1.000).

## 6. Warnings

- None.

## 7. Recommended investigations

1. 5 values lie outside the training range, so the model is extrapolating for those sessions.

## 8. Limitations

- Data drift does not automatically mean the model is failing.
- No detected drift does not prove that the model remains reliable.
- A decline in metrics on a simulated batch is evidence about that experiment, not proof of production performance.
- Synthetic experiments are not equivalent to monitoring a real deployed service.
- Statistical significance is not the same as business significance.
- PCA plots are exploratory visualisations, not a definitive reliability test.

---

Data: Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, UCI Machine Learning Repository, CC BY 4.0).
