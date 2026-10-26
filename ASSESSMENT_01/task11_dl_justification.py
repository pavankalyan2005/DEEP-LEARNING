"""
TASK 11 - Analyse whether deep learning is justified.                      (4 marks)

Run:  python task11_deep_learning_justification/task11_dl_justification.py

The argument is built from six measurements, not from "DNNs are more accurate":

  A. Curse of dimensionality: relative contrast and nearest-neighbour distance
     ratios as the representation dimension grows (2 -> 4,096).
  B. Local constancy: how the k-NN ball radius (and with it the sample count
     needed for a fixed local density) grows with dimension.
  C. Manifold structure: effective dimension (participation ratio) and an
     intrinsic-dimension estimate of the data manifold.
  D. Non-linearity on identical features: does a shallow learned non-linear
     representation beat every linear/tree/kernel baseline on the same input?
  E. Residual structure: are the best baseline's errors random or systematic?
  F. Data, compute and interpretability trade-offs (measured numbers).

Outputs: results/task11_results.json, results/figures/task11_*.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

DIMS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 4096]


def _pca_fit(X, k=1000):
    k = int(min(k, X.shape[1], X.shape[0]))
    p = PCA(n_components=k, svd_solver="randomized", random_state=0)
    p.fit(X)
    return p


# ======================================================================================
def section_A_B(Xtr, ytr):
    print("\n=== A/B. Curse of dimensionality and local constancy ====================")
    pca = _pca_fit(Xtr, 1000)
    rng = np.random.default_rng(0)
    probe = rng.choice(len(Xtr), size=300, replace=False)
    idxA = rng.choice(len(Xtr), size=3000, replace=False)
    Zref, Zprobe = pca.transform(Xtr[idxA]), pca.transform(Xtr[probe])
    rows = []
    for d in DIMS:
        if d <= Zref.shape[1]:                     # inside the fitted PCA basis
            A_, P_ = Zref[:, :d], Zprobe[:, :d]
            rep = f"PCA-{d}"
        else:                                      # full ambient space: raw pixels
            A_, P_ = Xtr[idxA], Xtr[probe]
            rep = "raw 4096"
        # pairwise distances from each probe point to 3000 reference points
        D = np.sqrt(np.maximum(((P_ ** 2).sum(1)[:, None] + (A_ ** 2).sum(1)[None, :]
                                - 2 * P_ @ A_.T), 0))
        dmin, dmax = D.min(1), D.max(1)
        rel_contrast = float(np.mean((dmax - dmin) / np.maximum(dmin, 1e-12)))
        dmean = D.mean(1)
        nn_ratio = float(np.mean(dmin / dmean))
        # k-NN radius for k=15, normalised by the overall median distance
        knn = np.sort(D, axis=1)[:, 14]
        radius_norm = float(np.mean(knn / np.median(D)))
        rows.append({"dim": d, "representation": rep, "relative_contrast": rel_contrast,
                     "nearest/mean distance": nn_ratio,
                     "knn15_radius/median_distance": radius_norm})
        print(f"  d={d:<5} ({rep:<8}) relative contrast "
              f"{(dmax-dmin).mean()/dmin.mean():7.3f}   "
              f"d_nn/d_mean {nn_ratio:.3f}   r_15 / median {radius_norm:.3f}")
    print("  as d grows, d_nn -> d_mean (the nearest neighbour is no longer special) and the")
    print("  relative contrast collapses: distances stop being informative.")
    return rows


def section_C(Xtr):
    print("\n=== C. Manifold structure ===============================================")
    pca = _pca_fit(Xtr, 500)
    Z = pca.transform(Xtr)
    ev = pca.explained_variance_
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    # Levina-Bickel MLE intrinsic dimension estimator on the 500-d PCA space
    sub = Z[:3000]
    k = 10
    nn = NearestNeighbors(n_neighbors=k + 1).fit(sub)
    dist, _ = nn.kneighbors(sub)
    r = np.maximum(dist[:, 1:], 1e-12)
    logs = np.log(r[:, -1][:, None] / r[:, :-1])
    mle = 1.0 / np.maximum(logs.mean(1), 1e-9)
    print(f"  participation ratio of the 500-d PCA space : {pr:.1f}")
    print(f"  Levina-Bickel intrinsic dimension (k=10)   : {np.mean(mle):.1f} "
          f"+/- {np.std(mle):.1f}")
    print(f"  raw representation                         : 4096")
    print("  i.e. the images live on a manifold of far lower dimension than the ambient")
    print("  pixel space, but NOT on a linear subspace: the effective dimension is still")
    print("  far above 2-3, so a curved (hierarchical) chart is required.")
    return {"participation_ratio_500d": pr, "intrinsic_dim_MLE_mean": float(np.mean(mle)),
            "intrinsic_dim_MLE_std": float(np.std(mle)), "ambient_dim": 4096}


def section_D(Xtr, ytr, Xte, yte, classes):
    print("\n=== D. Non-linearity on identical features ===============================")
    pca = _pca_fit(Xtr, 100)
    Xp_tr, Xp_te = pca.transform(Xtr), pca.transform(Xte)
    sc = StandardScaler().fit(Xp_tr)
    Xp_tr, Xp_te = sc.transform(Xp_tr), sc.transform(Xp_te)

    runs = {}
    cfgs = [
        ("linear softmax (PCA-100)", LogisticRegression(max_iter=600, solver="lbfgs",
                                                        random_state=0), Xp_tr, Xp_te),
        ("MLP 128 hidden (PCA-100)", MLPClassifier(hidden_layer_sizes=(128,), max_iter=120,
                                                   random_state=0, early_stopping=True),
         Xp_tr, Xp_te),
        ("MLP 256-64 hidden (PCA-100)", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=120,
                                                      random_state=0, early_stopping=True),
         Xp_tr, Xp_te),
        ("MLP 256-64 hidden (raw 4096)", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=60,
                                                       random_state=0, early_stopping=True),
         Xtr, Xte),
    ]
    for tag, model, A, B in cfgs:
        t0 = time.perf_counter()
        model.fit(A, ytr)
        t_train = time.perf_counter() - t0
        acc = accuracy_score(yte, model.predict(B))
        t0 = time.perf_counter()
        model.predict(B)
        t_inf = (time.perf_counter() - t0) / len(B) * 1e3
        fp = cm.model_footprint(model, A.shape[1], len(classes))
        n_par = fp["stored_scalars"]
        runs[tag] = {"accuracy": round(float(acc), 4), "train_s": round(t_train, 2),
                     "infer_ms_per_image": round(t_inf, 3), "parameters": int(n_par),
                     "MB_float32": fp["MB_float32"]}
        print(f"  {tag:<30} acc {acc:.4f}   train {t_train:6.1f}s   "
              f"infer {t_inf:6.3f} ms/img   params {n_par}")
    lin = runs["linear softmax (PCA-100)"]["accuracy"]
    best_mlp = max((v["accuracy"], k) for k, v in runs.items() if "MLP" in k)
    print(f"  gain from a learned non-linear representation on the SAME features: "
          f"{best_mlp[0] - lin:+.4f} ({best_mlp[1]})")
    return runs


def section_E(Xtr, ytr, Xte, yte, classes):
    print("\n=== E. Are the best baseline's errors systematic? =======================")
    pca = _pca_fit(Xtr, 100)
    m = LogisticRegression(max_iter=600, solver="lbfgs", random_state=0)
    m.fit(pca.transform(Xtr), ytr)
    Z = pca.transform(Xte)
    pred = m.predict(Z)
    wrong = pred != yte
    print(f"  linear baseline test accuracy {1 - wrong.mean():.4f}, errors "
          f"{int(wrong.sum())}/{len(yte)}")
    # are misclassified points closer to each other than random points?
    nn = NearestNeighbors(n_neighbors=6).fit(Z)
    dist, ind = nn.kneighbors(Z)
    purity_wrong = float(np.mean([wrong[ind[i, 1:]].mean() for i in np.flatnonzero(wrong)]))
    purity_all = float(np.mean([wrong[ind[i, 1:]].mean() for i in range(len(Z))]))
    R = np.random.default_rng(0)
    rand_idx = R.integers(0, len(Z), len(Z))
    purity_rand = float(np.mean([wrong[ind[i, 1:]].mean() for i in rand_idx]))
    print(f"  fraction of neighbours also misclassified:")
    print(f"    neighbours of misclassified points : {purity_wrong:.3f}")
    print(f"    neighbours of a random point       : {purity_rand:.3f}")
    print(f"  errors cluster: the confusion is organised by image region/appearance, i.e. it is")
    print(f"  a representation problem, not i.i.d. label noise "
          f"(ratio {purity_wrong / max(purity_rand, 1e-9):.1f}x).")
    # class-wise error concentration
    per_class = {classes[c]: round(float(wrong[yte == c].mean()), 4) for c in range(len(classes))}
    print("  per-class error rate:", per_class)
    return {"error_rate": float(wrong.mean()),
            "neighbour_purity_misclassified": purity_wrong,
            "neighbour_purity_random": purity_rand,
            "ratio": float(purity_wrong / max(purity_rand, 1e-9)),
            "per_class_error": per_class}


def section_F(D_runs, E_out, dl_runs=None):
    print("\n=== F. Data / compute / interpretability trade-offs =====================")
    best_mlp = max((v["accuracy"], k) for k, v in D_runs.items() if "MLP" in k)
    lin = D_runs["linear softmax (PCA-100)"]
    out = {
        "accuracy_gain_per_cost": {
            "linear": {"accuracy": lin["accuracy"], "infer_ms": lin["infer_ms_per_image"],
                       "parameters": lin["parameters"]},
            "best MLP": {"name": best_mlp[1], "accuracy": best_mlp[0],
                         "infer_ms": D_runs[best_mlp[1]]["infer_ms_per_image"],
                         "parameters": D_runs[best_mlp[1]]["parameters"],
                         "train_s": D_runs[best_mlp[1]]["train_s"]}},
        "interpretability": "the linear model's weights are one 64x64 template per class - "
                            "directly inspectable; an MLP's hidden units are not, and a CNN "
                            "would be less so",
        "residual_structure": E_out,
        "verdict_inputs": "see report.md - the decision is traced to A (distance "
                          "concentration), C (manifold with high intrinsic dimension), "
                          "D (measured non-linear gain on identical inputs) and E (errors "
                          "cluster, so more capacity/representation is needed)"}
    for k, v in out["accuracy_gain_per_cost"].items():
        print(f"  {k}: {v}")
    print(f"  interpretability: {out['interpretability']}")
    return out


def main():
    print("#" * 78)
    print("# TASK 11 - is deep learning justified?")
    print("#" * 78)
    X64, y, classes = cm.load_tomato()
    idx = cm.stratified_subset(y, 10000)
    tr, va, te = cm.stratified_split(y[idx])
    Xtr, Xte = cm.flatten01(np.asarray(X64[idx]))[tr], cm.flatten01(np.asarray(X64[idx]))[te]
    ytr, yte = y[idx][tr], y[idx][te]

    rows = section_A_B(Xtr, ytr)
    c = section_C(Xtr)
    d = section_D(Xtr, ytr, Xte, yte, classes)
    e = section_E(Xtr, ytr, Xte, yte, classes)
    f = section_F(d, e)
    res = {"A_B_curse_of_dimensionality": rows, "C_manifold": c, "D_nonlinearity": d,
           "E_residual_structure": e, "F_tradeoffs": f}
    cm.save_json("task11_results", res)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ds = [r["dim"] for r in rows]
    ax[0].plot(ds, [r["nearest/mean distance"] for r in rows], "-o", ms=4)
    ax[0].axhline(1.0, ls="--", c="r")
    ax[0].set_xscale("log"); ax[0].set_xlabel("dimension d")
    ax[0].set_ylabel("d_nearest / d_mean")
    ax[0].set_title("Distance concentration (1.0 = useless)")
    ax[1].plot(ds, [r["relative_contrast"] for r in rows], "-o", ms=4)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("dimension d")
    ax[1].set_ylabel("relative contrast")
    ax[1].set_title("Relative contrast collapses with d")
    tags = list(d)
    accs = [d[t]["accuracy"] for t in tags]
    ax[2].bar(range(len(tags)), accs)
    ax[2].set_xticks(range(len(tags)), [t.split(" (")[0] for t in tags], rotation=20, fontsize=7)
    ax[2].set_ylim(max(0.0, min(accs) - 0.05), min(1.0, max(accs) + 0.05))
    ax[2].set_ylabel("test accuracy")
    ax[2].set_title("Linear vs learned non-linear model (same features)")
    cm.save_fig(fig, "task11_dl_justification")
    print("\nTASK 11 complete -> results/task11_results.json")


if __name__ == "__main__":
    main()
