"""
TASK 6 - Capacity, overfitting and underfitting.                            (4 marks)

Run:  python task06_capacity_overfitting/task06_capacity.py

Sections
  A. Capacity sweeps on one fixed data scale (5,000 images):
        - softmax regression with L2 regularisation (C = 1e-4 ... 1e4)
        - k-NN with k = 1 ... 1001
        - decision tree with max_depth = 1 ... 32 (plus fully grown)
     -> train error, validation error, generalisation gap, regime labels.
  B. Same sweeps on raw-4096 vs PCA-100 features (how compression changes capacity).
  C. Why picking the hyper-parameter on the TEST set biases the reported error:
     10 repeated splits, best-on-test vs best-on-validation vs k-fold CV.

Outputs: results/task06_results.json, results/figures/task06_*.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

SCALE = 5000
C_GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]   # capacity up <-
K_GRID = [1, 3, 5, 11, 31, 101, 301, 1001]
D_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, None]                  # None = fully grown


def regime(train_err, val_err):
    """Label the regime from the two error levels and their gap."""
    gap = val_err - train_err
    if train_err > 0.55 and val_err > 0.55:
        return "underfitting (high bias): hypothesis space too small / too constrained"
    if gap < 0.03:
        return "well-fit: small gap, both errors near their floor"
    if gap < 0.08:
        return "mild overfitting: gap opening but validation still improving"
    return "overfitting (high variance): training error pushed far below validation"


def sweep_logreg(Xtr, ytr, Xva, yva):
    rows = []
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=400, solver="lbfgs", random_state=0)
        m.fit(Xtr, ytr)
        te = 1 - accuracy_score(ytr, m.predict(Xtr))
        ve = 1 - accuracy_score(yva, m.predict(Xva))
        rows.append({"param": C, "train_err": float(te), "val_err": float(ve),
                     "gap": float(ve - te), "regime": regime(te, ve)})
        print(f"    C={C:<9} train_err {te:.4f}  val_err {ve:.4f}  gap {ve - te:+.4f}  "
              f"| {rows[-1]['regime'].split(':')[0]}")
    return rows


def sweep_knn(Xtr, ytr, Xva, yva):
    # one neighbour search serves every k
    nb = KNeighborsClassifier(n_neighbors=max(K_GRID), n_jobs=-1)
    nb.fit(Xtr, ytr)
    _, idx = nb.kneighbors(Xva, n_neighbors=max(K_GRID))
    ytr_ = ytr
    rows = []
    for k in K_GRID:
        votes = np.apply_along_axis(lambda c: np.bincount(c, minlength=len(np.unique(ytr_)))
                                    .argmax(), 1, ytr_[idx[:, :k]])
        n_probe = min(600, len(Xtr))
        te_k = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(Xtr, ytr)
        te = 1 - accuracy_score(ytr[:n_probe], te_k.predict(Xtr[:n_probe]))
        ve = 1 - accuracy_score(yva, votes)
        rows.append({"param": k, "train_err": float(te), "val_err": float(ve),
                     "gap": float(ve - te), "regime": regime(te, ve)})
        print(f"    k={k:<6} train_err {te:.4f}  val_err {ve:.4f}  gap {ve - te:+.4f}  "
              f"| {rows[-1]['regime'].split(':')[0]}")
    return rows


def sweep_tree(Xtr, ytr, Xva, yva):
    rows = []
    for d in D_GRID:
        m = DecisionTreeClassifier(max_depth=d, random_state=0)
        m.fit(Xtr, ytr)
        te = 1 - accuracy_score(ytr, m.predict(Xtr))
        ve = 1 - accuracy_score(yva, m.predict(Xva))
        rows.append({"param": d if d is not None else "full", "train_err": float(te),
                     "val_err": float(ve), "gap": float(ve - te),
                     "regime": regime(te, ve), "leaves": int(m.get_n_leaves())})
        print(f"    depth={str(d):<5} train_err {te:.4f}  val_err {ve:.4f}  gap {ve - te:+.4f}  "
              f"| {rows[-1]['regime'].split(':')[0]}")
    return rows


def section_AB(X64, y):
    print("\n=== A/B. Capacity sweeps (raw-4096 vs PCA-100) ==========================")
    idx = cm.stratified_subset(y, SCALE)
    tr, va, te = cm.stratified_split(y[idx])
    X = cm.flatten01(np.asarray(X64[idx]))
    yy = y[idx]
    pca = PCA(n_components=100, svd_solver="randomized", random_state=0).fit(X[tr])
    feats = {"raw-4096": (X, None), "PCA-100": (pca.transform(X), None)}

    out = {"scale": SCALE, "splits": {"train": len(tr), "val": len(va), "test": len(te)},
           "features": {}}
    for fname, (A, _) in feats.items():
        print(f"\n  --- features: {fname} ---")
        out["features"][fname] = {
            "softmax_regression_vs_C": sweep_logreg(A[tr], yy[tr], A[va], yy[va]),
            "knn_vs_k": sweep_knn(A[tr], yy[tr], A[va], yy[va]),
            "tree_vs_depth": sweep_tree(A[tr], yy[tr], A[va], yy[va]),
        }

    # ---- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for r, fname in enumerate(feats):
        d = out["features"][fname]
        specs = [("softmax_regression_vs_C", C_GRID, "log", "C (regularisation strength)"),
                 ("knn_vs_k", K_GRID, "log", "k (neighbours)"),
                 ("tree_vs_depth", None, "linear", "tree depth")]
        for c, (key, grid, xs, xlabel) in enumerate(specs):
            rows = d[key]
            x = np.arange(len(rows))
            tr_e = [r_["train_err"] for r_ in rows]
            va_e = [r_["val_err"] for r_ in rows]
            ax[r, c].plot(x, tr_e, "-o", ms=4, label="training error")
            ax[r, c].plot(x, va_e, "-s", ms=4, label="validation error")
            ax[r, c].fill_between(x, tr_e, va_e, alpha=0.18)
            ax[r, c].set_xticks(x)
            ax[r, c].set_xticklabels([str(r_["param"]) for r_ in rows], rotation=45, fontsize=7)
            ax[r, c].set_xlabel(xlabel)
            ax[r, c].set_ylabel("0/1 error")
            ax[r, c].set_title(f"{fname}: {key.replace('_', ' ')}", fontsize=9)
            ax[r, c].legend(fontsize=7)
    fig.suptitle("Task 6: capacity sweeps - shaded band = generalisation gap")
    cm.save_fig(fig, "task06_capacity")
    return out


def section_C(X64, y, n_seeds=10, scale_n=2000):
    """Test-set hyper-parameter selection bias."""
    print("\n=== C. Selecting the hyper-parameter on the test set ====================")
    best_test, best_val, cv_est, fixed_c = [], [], [], []
    for seed in range(n_seeds):
        idx = cm.stratified_subset(y, scale_n, seed=seed)
        tr, va, te = cm.stratified_split(y[idx], seed=seed)
        X = cm.flatten01(np.asarray(X64[idx]))
        yy = y[idx]
        acc_test, acc_val = [], []
        for C in C_GRID:
            m = LogisticRegression(C=C, max_iter=300, solver="lbfgs", random_state=0)
            m.fit(X[tr], yy[tr])
            acc_test.append(accuracy_score(yy[te], m.predict(X[te])))
            acc_val.append(accuracy_score(yy[va], m.predict(X[va])))
        best_test.append(max(acc_test))                     # dishonest: picked on the test set
        i_va = int(np.argmax(acc_val))
        best_val.append(acc_test[i_va])                     # honest: picked on validation
        # k-fold cross-validation estimate of the chosen C (on train+val only)
        pool = np.concatenate([tr, va])
        kcv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        clf = LogisticRegression(C=C_GRID[i_va], max_iter=300, solver="lbfgs", random_state=0)
        cv_est.append(float(cross_val_score(clf, X[pool], yy[pool], cv=kcv, scoring="accuracy").mean()))
        fixed_c.append(acc_test[C_GRID.index(1.0)])         # fixed default, no selection
    out = {
        "mean best-on-TEST  (biased)": round(float(np.mean(best_test)), 4),
        "std  best-on-TEST": round(float(np.std(best_test)), 4),
        "mean best-on-VALIDATION (honest)": round(float(np.mean(best_val)), 4),
        "std  best-on-VALIDATION": round(float(np.std(best_val)), 4),
        "mean 5-fold CV estimate (train+val)": round(float(np.mean(cv_est)), 4),
        "std  5-fold CV": round(float(np.std(cv_est)), 4),
        "mean fixed C=1 (no selection)": round(float(np.mean(fixed_c)), 4),
        "selection bias (best-on-test - best-on-val)": round(float(np.mean(best_test)
                                                                    - np.mean(best_val)), 4),
        "runs": n_seeds, "scale": scale_n,
    }
    for k, v in out.items():
        print(f"  {k:<45}: {v}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4))
    bars = ["best chosen\non test\n(biased)", "best chosen\non validation",
            "5-fold CV\nestimate", "fixed C=1\n(no selection)"]
    vals = [out["mean best-on-TEST  (biased)"], out["mean best-on-VALIDATION (honest)"],
            out["mean 5-fold CV estimate (train+val)"], out["mean fixed C=1 (no selection)"]]
    errs = [0.0, out["std  best-on-VALIDATION"], out["std  5-fold CV"], 0.0]
    ax.bar(bars, vals, yerr=errs, capsize=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(min(vals) - 0.03, max(vals) + 0.02)
    ax.set_ylabel("reported test accuracy")
    ax.set_title("Task 6: hyper-parameter selection on the test set inflates the result")
    cm.save_fig(fig, "task06_selection_bias")
    return out


def main():
    print("#" * 78)
    print("# TASK 6 - capacity / overfitting / underfitting")
    print("#" * 78)
    X64, y, classes = cm.load_tomato()
    res = {}
    res["A_B_capacity"] = section_AB(X64, y)
    res["C_selection_bias"] = section_C(X64, y)
    cm.save_json("task06_results", res)
    print("\nTASK 6 complete -> results/task06_results.json")


if __name__ == "__main__":
    main()
