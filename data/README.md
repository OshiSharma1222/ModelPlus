# Data

ModelPulse uses the **Online Shoppers Purchasing Intention Dataset**.

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset #468 |
| URL | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Authors | C. Okan Sakar, Yomi Kastro |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Rows | 12,330 sessions |
| Features | 17 (10 numerical, 7 categorical) |
| Target | `Revenue` (purchase made: 15.5% positive) |

The raw CSV is **not** committed to this repository. Download it with:

```bash
python scripts/download_data.py
```

This places `online_shoppers_intention.csv` in this folder. The app and the
experiment script also download it automatically the first time they run.

## How the data is used

1. The source data is split **once** into a stratified 70% training set and a
   30% held-out test set (seed `2026`).
2. 20% of the training set is held back as a validation set, used only to
   choose each model's decision threshold.
3. The training set is the **reference** distribution for drift detection.
4. Every simulated "current batch" is a transformed copy of the untouched
   test set. Labels are never modified, so baseline and shifted performance
   are always measured on the same sessions.

Citation:

> Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of
> online shoppers' purchasing intention using multilayer perceptron and LSTM
> recurrent neural networks. *Neural Computing & Applications* 31, 6893–6908
> (2019).
