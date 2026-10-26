"""
TASK 7 - Decompose error into bias and variance; MLE vs MAP.               (4 marks)

Run:  python task07_bias_variance/task07_bias_variance.py

Sections
  A. Bootstrap experiment: 20 replicas of the training set (sample with
     replacement) for a LOW-capacity model (strongly regularised linear
     classifier) and a HIGH-capacity model (fully grown tree), evaluated on one
     fixed test set -> bias^2, variance and the residual (noise) term, for both
     one-hot squared error and 0/1 error.
  B. Why the measured error does not equal bias^2 + variance + noise exactly:
     the missing terms are quantified.
  C. MLE vs MAP for a class proportion under a Beta(a,b) prior: how the estimate
     differs, and how the prior's influence decays with sample size (real counts
     drawn from the data set).

Outputs: results/task07_results.json, results/figures/task07_*.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

REPLICAS = 20          # >= 10 required by the brief
SCALE = 3000
SEED = 7


def make_models(C_low, high):
    low = LogisticRegression(C=C_low, max_iter=400, solver="lbfgs", random_state=0)
    hi = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=0) \
        if high == "tree" else None
    return {"low-capacity (C=1e-4 linear)": low, "high-capacity (fully grown tree)": hi}


def bootstrap(Xpool, ypool, Xte, yte, C, kind, n_rep=REPLICAS):
    rng = np.random.default_rng(SEED)
    n = len(Xpool)
    preds = np.empty((n_rep, len(yte)), dtype=np.int64)
    for r in range(n_rep):
        b = rng.integers(0, n, size=n)                     # bootstrap sample
        if kind == "low":
            m = LogisticRegression(C=C, max_iter=400, solver="lbfgs", random_state=0)
        else:
            m = DecisionTreeClassifier(random_state=0)
        m.fit(Xpool[b], ypool[b])
        preds[r] = m.predict(Xte)
    return preds


def decompose(preds, yte, n_classes):
    """bias^2 / variance / residual for one-hot predictions and for 0/1 error."""
    onehot = np.zeros((len(preds), len(yte), n_classes))
    onehot[np.arange(len(preds))[:, None], np.arange(len(yte))[None, :], preds] = 1.0
    pbar = onehot.mean(0)                                  # mean prediction distribution
    y_true = cm.onehot(yte, n_classes)

    bias2_soft = float(np.mean(np.sum((pbar - y_true) ** 2, axis=1))) / n_classes
    var_soft = float(np.mean(np.sum((onehot - pbar[None, :, :]) ** 2, axis=2).mean(0))) / n_classes

    mode = pbar.argmax(1)
    err_per_rep = float(np.mean(preds != yte[None, :]))     # measured 0/1 error
    bias2_hard = float(np.mean(mode != yte))                # modal predictor is wrong
    var_hard = float(np.mean(preds != mode[None, :]))       # replicas disagree with mode
    residual = err_per_rep - bias2_hard - var_hard          # what bias+variance missed

    return {"measured_error_01": err_per_rep,
            "bias2_hard": bias2_hard, "variance_hard": var_hard,
            "residual_noise_term": float(residual),
            "bias2_onehot": bias2_soft, "variance_onehot": var_soft,
            "sum_bias2_var_onehot": float(bias2_soft + var_soft),
            "mean_onehot_MSE": float(np.mean((onehot - y_true[None, :, :]) ** 2)),
            "replica_agreement": float(np.mean(preds == mode[None, :]))}


def section_A(X64, y, classes):
    print("\n=== A. Bootstrap bias-variance experiment ({} replicas) ================".format(REPLICAS))
    idx = cm.stratified_subset(y, SCALE)
    tr, va, te = cm.stratified_split(y[idx], seed=SEED)
    X = cm.flatten01(np.asarray(X64[idx]))
    yy = y[idx]
    pca = PCA(n_components=100, svd_solver="randomized", random_state=0).fit(X[tr])
    Xp = pca.transform(X)
    pool = np.concatenate([tr, va])                        # train on everything but the test set
    C = len(classes)

    out = {"replicas": REPLICAS, "scale": SCALE, "models": {}}
    all_preds = {}
    for kind, cname, tag in (("low", 1e-4, "low-capacity (L2-regularised linear, C=1e-4)"),
                             ("high", None, "high-capacity (fully grown decision tree)")):
        preds = bootstrap(Xp[pool], yy[pool], Xp[te], yy[te], C=cname, kind=kind)
        all_preds[tag] = preds
        d = decompose(preds, yy[te], C)
        out["models"][tag] = d
        print(f"  {tag}")
        print(f"      measured 0/1 error (mean over replicas) {d['measured_error_01']:.4f}")
        print(f"      bias^2 (modal predictor wrong)          {d['bias2_hard']:.4f}")
        print(f"      variance (replica disagreement)         {d['variance_hard']:.4f}")
        print(f"      residual = err - bias^2 - variance      {d['residual_noise_term']:+.4f}")
        print(f"      one-hot MSE = bias^2 + variance exactly: {d['mean_onehot_MSE']:.4f} "
              f"= {d['bias2_onehot']:.4f} + {d['variance_onehot']:.4f}")
        print(f"      replica agreement with the modal class  {d['replica_agreement']:.4f}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = list(out["models"])
    comp = ["bias2_hard", "variance_hard", "residual_noise_term"]
    labels = ["bias^2", "variance", "residual (noise + interaction)"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bottoms = np.zeros(len(tags))
    for c, lab in zip(comp, labels):
        v = np.array([out["models"][t][c] for t in tags])
        ax[0].bar([t.split(" (")[0] for t in tags], v, bottom=bottoms, label=lab)
        bottoms += v
    for i, t in enumerate(tags):
        ax[0].text(i, bottoms[i] + 0.01, f"measured\n{out['models'][t]['measured_error_01']:.3f}",
                   ha="center", fontsize=8)
    ax[0].set_ylabel("0/1 error contribution")
    ax[0].set_title("Error decomposition (test error of each replica)")
    ax[0].legend(fontsize=7)

    xs = np.arange(REPLICAS)
    for t in tags:
        per_rep = np.mean(all_preds[t] != yy[te][None, :], axis=1)
        ax[1].plot(xs, per_rep, "o-", ms=3, label=t.split(" (")[0])
    ax[1].set_xlabel("bootstrap replica"); ax[1].set_ylabel("0/1 error of that replica")
    ax[1].set_title("Spread of the replicas = variance")
    ax[1].legend(fontsize=7)
    cm.save_fig(fig, "task07_bias_variance")
    return out


def section_B(out):
    print("\n=== B. Does measured error = bias^2 + variance + noise? ==================")
    exp = {}
    for tag, d in out["models"].items():
        lhs = d["measured_error_01"]
        rhs = d["bias2_hard"] + d["variance_hard"] + d["residual_noise_term"]
        # a *theoretical* additive estimate that ignores the interaction term
        rhs_no_noise = d["bias2_hard"] + d["variance_hard"]
        exp[tag] = {"measured": round(lhs, 4),
                    "bias2+variance": round(rhs_no_noise, 4),
                    "gap": round(lhs - rhs_no_noise, 4),
                    "bias2+variance+residual": round(rhs, 4)}
        print(f"  {tag}")
        print(f"      measured error            {lhs:.4f}")
        print(f"      bias^2 + variance         {rhs_no_noise:.4f}   (gap {lhs - rhs_no_noise:+.4f})")
        print(f"      + residual term           {rhs:.4f}   (identity closes by construction)")
    print("  sources of the gap:")
    print("   - noise: test labels themselves carry irreducible uncertainty (annotation")
    print("     error, disease symptoms that are genuinely ambiguous); PlantVillage labels")
    print("     are largely clean so this term is small here but never exactly zero")
    print("   - interaction: 0/1 loss is not additive - a replica that is wrong where the")
    print("     modal predictor is also wrong does not add a second error")
    print("   - finite test set: every term is an estimate with ~1/sqrt(n_test) noise")
    return exp


def section_C(y, classes):
    print("\n=== C. MLE vs MAP for a class proportion under a Beta prior =============")
    counts = cm.class_counts(y, len(classes))
    p_true = counts / counts.sum()
    # use the minority and majority tomato classes as concrete cases
    minority, majority = int(np.argmin(counts)), int(np.argmax(counts))
    prior = (2.0, 2.0)                                    # Beta(2,2): gentle shrink to 0.5
    a, b = prior
    print(f"  prior Beta(a={a}, b={b})   mean a/(a+b) = {a/(a+b):.3f}, "
          f"variance = {a*b/((a+b)**2*(a+b+1)):.4f}")
    rows = {}
    for tag, c in (("minority", minority), ("majority", majority)):
        p = p_true[c]
        print(f"\n  class '{classes[c]}'  true proportion p = {p:.4f}  ({counts[c]} images)")
        rows[classes[c]] = {"true_p": float(p), "samples": {}}
        for n in (10, 50, 100, 500, 1000, 5000):
            rng = np.random.default_rng(0)
            k = rng.binomial(n, p)                        # how many images of this class
            mle = k / n
            map_ = (k + a - 1) / (n + a + b - 2)          # Beta posterior mode
            mean = (k + a) / (n + a + b)                  # posterior mean
            rows[classes[c]]["samples"][str(n)] = {
                "k": int(k), "MLE": float(mle), "MAP": float(map_), "post_mean": float(mean),
                "|MLE-p|": float(abs(mle - p)), "|MAP-p|": float(abs(map_ - p))}
            print(f"    n={n:<5} k={k:<5}  MLE {mle:.4f}  MAP {map_:.4f}  "
                  f"post.mean {mean:.4f}   |MLE-p| {abs(mle-p):.4f}  |MAP-p| {abs(map_-p):.4f}")
    # average |error| over many repetitions: where does the prior help?
    print("\n  RMSE of MLE vs MAP over 200 repeated samples (minority class):")
    p = p_true[minority]
    rmse = {}
    for n in (10, 50, 100, 500, 1000, 5000):
        rng = np.random.default_rng(1)
        ks = rng.binomial(n, p, size=200)
        mles = ks / n
        maps = (ks + a - 1) / (n + a + b - 2)
        rmse[n] = {"MLE": float(np.sqrt(np.mean((mles - p) ** 2))),
                   "MAP": float(np.sqrt(np.mean((maps - p) ** 2))),
                   "prior_shift": float(abs((a - 1) / (a + b - 2) - p))}
        print(f"    n={n:<5} RMSE(MLE) {rmse[n]['MLE']:.4f}   RMSE(MAP) {rmse[n]['MAP']:.4f}   "
              f"{'MAP better' if rmse[n]['MAP'] < rmse[n]['MLE'] else 'MLE better'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ns = list(rmse)
    ax[0].plot(ns, [rmse[n]["MLE"] for n in ns], "-o", label="RMSE(MLE)")
    ax[0].plot(ns, [rmse[n]["MAP"] for n in ns], "-s", label="RMSE(MAP)  Beta(2,2)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("sample size n")
    ax[0].set_ylabel("RMSE of the proportion estimate")
    ax[0].set_title("Prior helps for small n, vanishes for large n")
    ax[0].legend()
    ns2 = [10, 50, 100, 500, 1000, 5000]
    mles = [rows[classes[minority]]["samples"][str(n)]["MLE"] for n in ns2]
    maps = [rows[classes[minority]]["samples"][str(n)]["MAP"] for n in ns2]
    ax[1].plot(ns2, mles, "-o", label="MLE = k/n")
    ax[1].plot(ns2, maps, "-s", label="MAP = (k+a-1)/(n+a+b-2)")
    ax[1].axhline(p, ls="--", c="k", label=f"true p = {p:.3f}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("sample size n")
    ax[1].set_ylabel("estimated class proportion")
    ax[1].set_title("One real draw per sample size (minority class)")
    ax[1].legend(fontsize=8)
    cm.save_fig(fig, "task07_mle_map")
    return {"prior": {"a": a, "b": b}, "classes": rows, "rmse": {str(k): v for k, v in rmse.items()},
            "formulae": {"MLE": "p_hat = k/n",
                         "MAP": "p_map = (k + a - 1)/(n + a + b - 2)  [Beta-Binomial posterior mode]",
                         "posterior_mean": "(k + a)/(n + a + b)"}}


def main():
    print("#" * 78)
    print("# TASK 7 - bias-variance decomposition and MLE vs MAP")
    print("#" * 78)
    X64, y, classes = cm.load_tomato()
    res = {"A_bootstrap": section_A(X64, y, classes)}
    res["B_gap_analysis"] = section_B(res["A_bootstrap"])
    res["C_mle_vs_map"] = section_C(y, classes)
    cm.save_json("task07_results", res)
    print("\nTASK 7 complete -> results/task07_results.json")


if __name__ == "__main__":
    main()
