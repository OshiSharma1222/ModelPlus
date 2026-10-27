# ModelPulse audit report

Generated 2026-10-27 20:40 UTC by ModelPulse v1.0.0.

## 1. Verdict

**Data-quality issue with performance decline.** 50.0% of sessions have missing values and 0 values break domain rules. The observed values may look like the reference, but the model is scoring imputed inputs. The performance drop is most plausibly caused by the broken inputs rather than by a real change in shopper behaviour.

- Drift status: **Stable** (0 of 17 features drifted; 0 more were statistically significant but below the effect-size threshold).
- Decision Tree: ROC-AUC 0.920 → 0.807, F1 0.664 → 0.511; material decline: **yes**.
- Naive Bayes: ROC-AUC 0.874 → 0.816, F1 0.609 → 0.504; material decline: **yes**.

## 2. Experiment settings

| Setting | Value |
|---|---|
| Scenario | Missing-value injection |
| Scenario description | Blanks out up to 50% of values (severity 1) of the selected features at random (MCAR). The model sees training-median / mode imputations instead. Simulates a broken upstream pipeline. |
| Severity | 0.60 (0 = none, 1 = most severe) |
| Features modified | PageValues, ExitRates |
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

- Scenario: Missing-value injection; severity 0.60; seed 2026; rows 3699.
- PageValues: 1110 values (30%) set to missing completely at random.
- ExitRates: 1110 values (30%) set to missing completely at random.

## 3. Data quality

| Check | Result |
|---|---|
| Rows in current batch | 3,699 |
| Missing cells | 2,220 (1,851 rows affected) |
| Values violating domain rules | 0 |
| Values outside training range | 4 |
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
| TrafficType | Chi-square (Monte-Carlo) | TVD = 0.023 | 0.827 | no | '1' 20.2% → 19.1% |
| ExitRates | Kolmogorov-Smirnov | KS D = 0.020 | 0.827 | no | mean 0.0429 → 0.0435; median 0.025 → 0.0264 |
| ProductRelated | Kolmogorov-Smirnov | KS D = 0.020 | 0.827 | no | mean 32 → 31; median 18 → 18 |
| OperatingSystems | Chi-square (Monte-Carlo) | TVD = 0.019 | 0.827 | no | '3' 21.1% → 19.9% |
| Region | Chi-square | TVD = 0.018 | 0.827 | no | '9' 3.9% → 4.6% |
| BounceRates | Kolmogorov-Smirnov | KS D = 0.018 | 0.827 | no | mean 0.0221 → 0.0224; median 0.0033 → 0.00267 |
| ProductRelated_Duration | Kolmogorov-Smirnov | KS D = 0.018 | 0.827 | no | mean 1.2e+03 → 1.19e+03; median 602 → 591 |
| VisitorType | Chi-square | TVD = 0.017 | 0.449 | no | 'Returning_Visitor' 86.1% → 84.4% |
| Month | Chi-square | TVD = 0.016 | 0.862 | no | 'Oct' 4.2% → 5.1% |
| Administrative_Duration | Kolmogorov-Smirnov | KS D = 0.016 | 0.843 | no | mean 79.3 → 84.4; median 8 → 6 |
| Administrative | Kolmogorov-Smirnov | KS D = 0.014 | 0.862 | no | mean 2.32 → 2.3; median 1 → 1 |
| Browser | Chi-square (Monte-Carlo) | TVD = 0.014 | 0.827 | no | '4' 5.7% → 6.7% |
| Informational_Duration | Kolmogorov-Smirnov | KS D = 0.014 | 0.862 | no | mean 35.4 → 32.2; median 0 → 0 |
| Informational | Kolmogorov-Smirnov | KS D = 0.010 | 1.000 | no | mean 0.51 → 0.49; median 0 → 0 |
| SpecialDay | Kolmogorov-Smirnov | KS D = 0.008 | 1.000 | no | mean 0.0622 → 0.0596; median 0 → 0 |
| Weekend | Chi-square | TVD = 0.008 | 0.827 | no | 'False' 77.0% → 76.2% |
| PageValues | Kolmogorov-Smirnov | KS D = 0.007 | 1.000 | no | mean 5.82 → 5.99; median 0 → 0 |

## 5. Model performance on identical sessions

### Decision Tree

| Metric | Baseline | Current | Change | 95% paired CI of change |
|---|---|---|---|---|
| Accuracy | 0.896 | 0.870 | -0.026 | [-0.033, -0.019] |
| Precision | 0.665 | 0.612 | -0.053 | [-0.073, -0.030] |
| Recall | 0.663 | 0.439 | -0.224 | [-0.258, -0.189] |
| F1-score | 0.664 | 0.511 | -0.153 | [-0.184, -0.125] |
| ROC-AUC | 0.920 | 0.807 | -0.113 | [-0.138, -0.092] |
| Brier score (lower = better) | 0.072 | 0.098 | +0.026 | [0.021, 0.032] |

McNemar test on paired correctness: 128 sessions correct only before the shift, 32 only after (p = < 0.001).

### Naive Bayes

| Metric | Baseline | Current | Change | 95% paired CI of change |
|---|---|---|---|---|
| Accuracy | 0.849 | 0.829 | -0.020 | [-0.027, -0.013] |
| Precision | 0.508 | 0.457 | -0.051 | [-0.069, -0.034] |
| Recall | 0.760 | 0.563 | -0.198 | [-0.231, -0.163] |
| F1-score | 0.609 | 0.504 | -0.105 | [-0.127, -0.079] |
| ROC-AUC | 0.874 | 0.816 | -0.058 | [-0.070, -0.046] |
| Brier score (lower = better) | 0.217 | 0.225 | +0.008 | [0.004, 0.011] |

McNemar test on paired correctness: 118 sessions correct only before the shift, 43 only after (p = < 0.001).

## 6. Warnings

- 2,220 missing values were imputed.
- Decision Tree shows a material performance decline.
- Naive Bayes shows a material performance decline.

## 7. Recommended investigations

1. 2,220 values were missing and were imputed with training medians/modes. Trace the upstream pipeline for the affected columns before trusting predictions for those sessions.
2. 4 values lie outside the training range, so the model is extrapolating for those sessions.
3. Retrain or recalibrate only after confirming the shift persists on newly labelled data; a single simulated batch is not enough evidence.
4. Decision Tree Brier score worsened by 0.026: predicted probabilities are less trustworthy - review calibration.

## 8. Limitations

- Data drift does not automatically mean the model is failing.
- No detected drift does not prove that the model remains reliable.
- A decline in metrics on a simulated batch is evidence about that experiment, not proof of production performance.
- Synthetic experiments are not equivalent to monitoring a real deployed service.
- Statistical significance is not the same as business significance.
- PCA plots are exploratory visualisations, not a definitive reliability test.

---

Data: Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, UCI Machine Learning Repository, CC BY 4.0).
