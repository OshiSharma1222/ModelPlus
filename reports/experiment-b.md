# ModelPulse audit report

Generated 2026-10-27 19:57 UTC by ModelPulse v1.0.0.

## 1. Verdict

**Performance decline without detected drift.** Feature distributions look unchanged, yet the model performs worse. This pattern points to a change in the relationship between features and outcome (concept drift), which marginal drift tests cannot detect.

- Drift status: **Stable** (0 of 17 features drifted; 0 more were statistically significant but below the effect-size threshold).
- Decision Tree: ROC-AUC 0.920 -> 0.669, F1 0.664 -> 0.276; material decline: **yes**.
- Naive Bayes: ROC-AUC 0.874 -> 0.740, F1 0.609 -> 0.369; material decline: **yes**.

## 2. Experiment settings

| Setting | Value |
|---|---|
| Scenario | Feature-outcome relationship shift |
| Scenario description | Shuffles the selected features among a random fraction of sessions (severity). Each feature's marginal distribution is unchanged, but its link to the purchase outcome is broken. Simulates concept drift that marginal drift tests cannot see. |
| Severity | 0.80 (0 = none, 1 = most severe) |
| Features modified | PageValues |
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

- Scenario: Feature-outcome relationship shift; severity 0.80; seed 2026; rows 3699.
- PageValues: values shuffled jointly among 2959 sessions (80%); marginal distributions preserved exactly.

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
| Accuracy | 0.896 | 0.775 | -0.121 | [-0.136, -0.109] |
| Precision | 0.665 | 0.275 | -0.390 | [-0.430, -0.347] |
| Recall | 0.663 | 0.278 | -0.385 | [-0.429, -0.338] |
| F1-score | 0.664 | 0.276 | -0.387 | [-0.425, -0.349] |
| ROC-AUC | 0.920 | 0.669 | -0.252 | [-0.275, -0.227] |
| Brier score (lower is better) | 0.072 | 0.156 | +0.084 | [0.076, 0.094] |

McNemar test on paired correctness: 598 sessions correct only before the shift, 149 only after (p = < 0.001).

### Naive Bayes

| Metric | Baseline | Current | Change | 95% paired CI of change |
|---|---|---|---|---|
| Accuracy | 0.849 | 0.774 | -0.075 | [-0.087, -0.063] |
| Precision | 0.508 | 0.325 | -0.183 | [-0.212, -0.156] |
| Recall | 0.760 | 0.427 | -0.334 | [-0.377, -0.288] |
| F1-score | 0.609 | 0.369 | -0.240 | [-0.271, -0.209] |
| ROC-AUC | 0.874 | 0.740 | -0.134 | [-0.149, -0.117] |
| Brier score (lower is better) | 0.217 | 0.272 | +0.055 | [0.047, 0.063] |

McNemar test on paired correctness: 420 sessions correct only before the shift, 143 only after (p = < 0.001).

## 6. Warnings

- Decision Tree shows a material performance decline.
- Naive Bayes shows a material performance decline.

## 7. Recommended investigations

1. 5 values lie outside the training range, so the model is extrapolating for those sessions.
2. Performance fell without marginal drift. Investigate whether the feature-outcome relationship has changed (e.g. segment-level error analysis) and collect fresh labels before retraining.
3. Retrain or recalibrate only after confirming the shift persists on newly labelled data; a single simulated batch is not enough evidence.
4. Decision Tree Brier score worsened by 0.084: predicted probabilities are less trustworthy - review calibration.
5. Naive Bayes Brier score worsened by 0.055: predicted probabilities are less trustworthy - review calibration.

## 8. Limitations

- Data drift does not automatically mean the model is failing.
- No detected drift does not prove that the model remains reliable.
- A decline in metrics on a simulated batch is evidence about that experiment, not proof of production performance.
- Synthetic experiments are not equivalent to monitoring a real deployed service.
- Statistical significance is not the same as business significance.
- PCA plots are exploratory visualisations, not a definitive reliability test.

---

Data: Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, UCI Machine Learning Repository, CC BY 4.0).
