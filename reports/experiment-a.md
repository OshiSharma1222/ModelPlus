# ModelPulse audit report

Generated 2026-10-27 20:40 UTC by ModelPulse v1.0.0.

## 1. Verdict

**Drift without measurable decline.** The inputs changed, but on these sessions the model's quality did not drop beyond the pre-registered limits. Treat the drift as a warning to monitor, not as evidence of failure.

- Drift status: **Warning** (2 of 17 features drifted; 1 more were statistically significant but below the effect-size threshold).
- Decision Tree: ROC-AUC 0.920 → 0.923, F1 0.664 → 0.662; material decline: **no**.
- Naive Bayes: ROC-AUC 0.874 → 0.872, F1 0.609 → 0.595; material decline: **no**.

## 2. Experiment settings

| Setting | Value |
|---|---|
| Scenario | Numerical shift |
| Scenario description | Multiplies the selected numerical features by up to 3x (severity 1). Counts are rounded and bounded rates are clipped to their valid range. Simulates longer browsing sessions or a change in how time is logged. |
| Severity | 0.80 (0 = none, 1 = most severe) |
| Features modified | Administrative_Duration, Informational_Duration, ProductRelated_Duration |
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

- Scenario: Numerical shift; severity 0.80; seed 2026; rows 3699.
- Administrative_Duration: multiplied by 2.60.
- Informational_Duration: multiplied by 2.60.
- ProductRelated_Duration: multiplied by 2.60.

## 3. Data quality

| Check | Result |
|---|---|
| Rows in current batch | 3,699 |
| Missing cells | 0 (0 rows affected) |
| Values violating domain rules | 0 |
| Values outside training range | 47 |
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
| ProductRelated_Duration | Kolmogorov-Smirnov | KS D = 0.261 | < 0.001 | **yes** | mean 1.2e+03 → 3.09e+03; median 602 → 1.54e+03 |
| Administrative_Duration | Kolmogorov-Smirnov | KS D = 0.173 | < 0.001 | **yes** | mean 79.3 → 219; median 8 → 15.6 |
| Informational_Duration | Kolmogorov-Smirnov | KS D = 0.045 | < 0.001 | no | mean 35.4 → 83.8; median 0 → 0 |
| TrafficType | Chi-square (Monte-Carlo) | TVD = 0.023 | 0.752 | no | '1' 20.2% → 19.1% |
| ProductRelated | Kolmogorov-Smirnov | KS D = 0.020 | 0.637 | no | mean 32 → 31; median 18 → 18 |
| OperatingSystems | Chi-square (Monte-Carlo) | TVD = 0.019 | 0.472 | no | '3' 21.1% → 19.9% |
| Region | Chi-square | TVD = 0.018 | 0.752 | no | '9' 3.9% → 4.6% |
| BounceRates | Kolmogorov-Smirnov | KS D = 0.018 | 0.684 | no | mean 0.0221 → 0.0224; median 0.0033 → 0.00267 |
| VisitorType | Chi-square | TVD = 0.017 | 0.112 | no | 'Returning_Visitor' 86.1% → 84.4% |
| Month | Chi-square | TVD = 0.016 | 0.858 | no | 'Oct' 4.2% → 5.1% |
| Administrative | Kolmogorov-Smirnov | KS D = 0.014 | 0.858 | no | mean 2.32 → 2.3; median 1 → 1 |
| Browser | Chi-square (Monte-Carlo) | TVD = 0.014 | 0.637 | no | '4' 5.7% → 6.7% |
| ExitRates | Kolmogorov-Smirnov | KS D = 0.014 | 0.858 | no | mean 0.0429 → 0.0435; median 0.025 → 0.0259 |
| Informational | Kolmogorov-Smirnov | KS D = 0.010 | 0.998 | no | mean 0.51 → 0.49; median 0 → 0 |
| SpecialDay | Kolmogorov-Smirnov | KS D = 0.008 | 0.998 | no | mean 0.0622 → 0.0596; median 0 → 0 |
| PageValues | Kolmogorov-Smirnov | KS D = 0.008 | 0.998 | no | mean 5.82 → 6.06; median 0 → 0 |
| Weekend | Chi-square | TVD = 0.008 | 0.684 | no | 'False' 77.0% → 76.2% |

## 5. Model performance on identical sessions

### Decision Tree

| Metric | Baseline | Current | Change | 95% paired CI of change |
|---|---|---|---|---|
| Accuracy | 0.896 | 0.897 | +0.001 | [-0.001, 0.003] |
| Precision | 0.665 | 0.672 | +0.007 | [0.001, 0.015] |
| Recall | 0.663 | 0.652 | -0.010 | [-0.018, -0.003] |
| F1-score | 0.664 | 0.662 | -0.002 | [-0.008, 0.005] |
| ROC-AUC | 0.920 | 0.923 | +0.003 | [-0.000, 0.006] |
| Brier score (lower = better) | 0.072 | 0.072 | -0.001 | [-0.001, 0.000] |

McNemar test on paired correctness: 6 sessions correct only before the shift, 9 only after (p = 0.607).

### Naive Bayes

| Metric | Baseline | Current | Change | 95% paired CI of change |
|---|---|---|---|---|
| Accuracy | 0.849 | 0.837 | -0.012 | [-0.016, -0.008] |
| Precision | 0.508 | 0.483 | -0.025 | [-0.034, -0.015] |
| Recall | 0.760 | 0.774 | +0.014 | [0.002, 0.027] |
| F1-score | 0.609 | 0.595 | -0.014 | [-0.023, -0.005] |
| ROC-AUC | 0.874 | 0.872 | -0.002 | [-0.003, -0.001] |
| Brier score (lower = better) | 0.217 | 0.238 | +0.021 | [0.019, 0.022] |

McNemar test on paired correctness: 58 sessions correct only before the shift, 13 only after (p = < 0.001).

## 6. Warnings

- Drift warning: ProductRelated_Duration, Administrative_Duration.

## 7. Recommended investigations

1. Prioritise Administrative_Duration: these drifted features are among the Decision Tree's five most important inputs.
2. 47 values lie outside the training range, so the model is extrapolating for those sessions.
3. Naive Bayes Brier score worsened by 0.021: predicted probabilities are less trustworthy - review calibration.

## 8. Limitations

- Data drift does not automatically mean the model is failing.
- No detected drift does not prove that the model remains reliable.
- A decline in metrics on a simulated batch is evidence about that experiment, not proof of production performance.
- Synthetic experiments are not equivalent to monitoring a real deployed service.
- Statistical significance is not the same as business significance.
- PCA plots are exploratory visualisations, not a definitive reliability test.

---

Data: Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, UCI Machine Learning Repository, CC BY 4.0).
