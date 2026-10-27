"""Baseline (Decision Tree) and comparison (Naive Bayes) classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import config
from .evaluation import best_f1_threshold


def one_hot(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals. Columns are stable because the
    preprocessor fixes the category set of every categorical feature."""
    return pd.get_dummies(X[config.FEATURES], columns=config.CATEGORICAL_FEATURES, dtype=float)


class MixedNaiveBayes:
    """Naive Bayes with Gaussian likelihoods for numerical features and
    Laplace-smoothed categorical likelihoods for categorical features."""

    def __init__(self, alpha: float = 1.0, var_smoothing: float = 1e-6):
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MixedNaiveBayes":
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        Xn = X[config.NUMERIC_FEATURES].to_numpy(dtype=float)
        eps = self.var_smoothing * Xn.var(axis=0).max()
        self.means_ = np.vstack([Xn[y == c].mean(axis=0) for c in self.classes_])
        self.vars_ = np.vstack([Xn[y == c].var(axis=0) + eps for c in self.classes_])
        self.cat_log_probs_ = {}
        for col in config.CATEGORICAL_FEATURES:
            cats = list(X[col].cat.categories)
            table = np.zeros((2, len(cats)))
            for ci, c in enumerate(self.classes_):
                counts = X.loc[y == c, col].value_counts().reindex(cats, fill_value=0)
                table[ci] = (counts.to_numpy() + self.alpha) / (counts.sum() + self.alpha * len(cats))
            self.cat_log_probs_[col] = (cats, np.log(table))
        return self

    def joint_log_likelihood(self, X: pd.DataFrame) -> np.ndarray:
        Xn = X[config.NUMERIC_FEATURES].to_numpy(dtype=float)
        jll = np.tile(self.log_prior_, (len(X), 1))
        for ci in range(2):
            jll[:, ci] += -0.5 * np.sum(
                np.log(2 * np.pi * self.vars_[ci]) + (Xn - self.means_[ci]) ** 2 / self.vars_[ci],
                axis=1,
            )
        for col, (cats, log_table) in self.cat_log_probs_.items():
            codes = pd.Categorical(X[col], categories=cats).codes
            jll += log_table[:, codes].T
        return jll

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))


@dataclass
class FittedModel:
    key: str
    label: str
    estimator: object
    threshold: float
    columns: list = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of a purchase for preprocessed inputs ``X``."""
        if self.key == "tree":
            enc = one_hot(X).reindex(columns=self.columns, fill_value=0.0)
            return self.estimator.predict_proba(enc)[:, 1]
        return self.estimator.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        """Importance per original feature (tree only; one-hot columns summed)."""
        if self.key != "tree":
            return pd.Series(dtype=float)
        imp = pd.Series(self.estimator.feature_importances_, index=self.columns)
        # One-hot columns are named "<feature>_<level>"; map them back.
        owners = [
            col if col in config.NUMERIC_FEATURES
            else next(f for f in config.CATEGORICAL_FEATURES if col.startswith(f + "_"))
            for col in self.columns
        ]
        return imp.groupby(owners).sum().sort_values(ascending=False)


def train_decision_tree(X: pd.DataFrame, y: np.ndarray, seed: int = config.SEED):
    enc = one_hot(X)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=seed, class_weight=None),
        param_grid={"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [10, 25, 50, 100]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=1,
    )
    search.fit(enc, y)
    return search.best_estimator_, list(enc.columns), search.best_params_


def train_models(
    fit_X: pd.DataFrame,
    fit_y: np.ndarray,
    val_X: pd.DataFrame,
    val_y: np.ndarray,
    seed: int = config.SEED,
) -> tuple[dict[str, FittedModel], dict]:
    """Fit both classifiers and pick each decision threshold on validation data."""
    tree, columns, best_params = train_decision_tree(fit_X, fit_y, seed)
    tree_model = FittedModel("tree", config.MODEL_LABELS["tree"], tree, 0.5, columns)
    tree_model.threshold = best_f1_threshold(val_y, tree_model.predict_proba(val_X))

    nb = MixedNaiveBayes().fit(fit_X, fit_y)
    nb_model = FittedModel("nb", config.MODEL_LABELS["nb"], nb, 0.5)
    nb_model.threshold = best_f1_threshold(val_y, nb_model.predict_proba(val_X))

    info = {"tree_params": best_params}
    return {"tree": tree_model, "nb": nb_model}, info
