"""
TASK 1 - Analyse the problem structure and data representation.            (4 marks)

Run:  python task01_problem_representation/task01.py

Sections
  A. Decompose AgriVision into learning-task components (K, H, L, A) and count
     the scalars involved at every scale.
  B. Show the organisation of data as scalars / vectors / matrices / tensors
     (one image, mini-batch of 32, labels, weights) and the memory cost of each.
  C. Compute L1, L2 and L-infinity norms of the SAME feature vectors, show how they
     differ on real images, and match each norm to an analysis purpose.
  D. Measure the effect of every preprocessing step:
       D1 resizing (32 / 64 / 128 px),  D2 normalisation ([0,1] vs [0,255]),
       D3 flattening (translation sensitivity), D4 data split (stratified vs naive),
       D5 grayscale (what colour information is discarded).

Outputs: results/task01_results.json, results/figures/task01_*.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

OUT = cm.save_json


# ======================================================================================
# A. learning-task decomposition
# ======================================================================================
def section_A(X, y, classes) -> dict:
    print("\n=== A. Learning-task components =========================================")
    N, D = X.shape
    C = len(classes)
    counts = cm.class_counts(y, C)
    props = counts / counts.sum()

    parts = {
        "Learning task": "multi-class single-label image classification (discriminative)",
        "Input dimension K": D,                       # 4096 raw pixel features
        "Input per sample": f"matrix 64x64 = {D} scalar intensities",
        "Output space": f"{C} mutually exclusive crop-disease labels",
        "Hypothesis space": "softmax decision functions h(x) = argmax Wx+b",
        "Loss": "0/1 loss (evaluation) ~ cross-entropy (optimisation)",
        "Performance measure": "accuracy + macro-F1 (class imbalance aware)",
        "Data size N": int(N),
        "Class proportions": {classes[i]: round(float(props[i]), 4) for i in range(C)},
        "Imbalance ratio max/min": round(float(counts.max() / max(counts[counts > 0].min(), 1)), 2),
        "Bayes-ish floor (majority) ": round(float(props.max()), 4),
        "Random-guess accuracy (uniform)": round(1.0 / C, 4),
        "Linear parameters (C x (K+1))": int(C * (D + 1)),
    }
    for k, v in parts.items():
        print(f"  {k:<32}: {v}")
    return parts


# ======================================================================================
# B. representation / shapes
# ======================================================================================
def section_B(X, y, classes) -> dict:
    print("\n=== B. How the data is organised (shapes + cost) =========================")
    C = len(classes)
    one = X[0:1]                       # (1,4096)
    batch = X[:32]                     # (32,4096)
    batch_t = X[:32].reshape(32, 64, 64)
    yb = y[:32]
    yb_oh = cm.onehot(yb, C)
    W = np.zeros((C, X.shape[1] + 1))  # weights + bias

    shapes = {
        "one pixel": "scalar (uint8 0..255 -> float32 0..1)",
        "one image (flattened)": list(one.shape),
        "one image (spatial matrix)": [64, 64],
        "mini-batch of 32": list(batch.shape),
        "mini-batch spatial tensor": list(batch_t.shape),
        "labels of 32": list(yb.shape) + ["int64 class index"],
        "labels one-hot": list(yb_oh.shape),
        "weight matrix + bias": list(W.shape),
        "whole training matrix": [int(X.shape[0]), int(X.shape[1])],
        "an RGB version would be": [int(X.shape[0]), 64, 64, 3],
    }
    mem = {
        "one image uint8 (KB)": round(one.astype(np.uint8).nbytes / 1e3, 3),
        "one image float32 (KB)": round(one.astype(np.float32).nbytes / 1e3, 3),
        "batch-32 float32 (KB)": round(batch.astype(np.float32).nbytes / 1e3, 2),
        "whole set uint8 (MB)": round(len(X) * 4096 / 1e6, 1),
        "whole set float32 (MB)": round(X.astype(np.float32).nbytes / 1e6, 1),
        "softmax weights float32 (KB)": round(W.astype(np.float32).nbytes / 1e3, 2),
    }
    for k, v in shapes.items():
        print(f"  {k:<32}: {v}")
    for k, v in mem.items():
        print(f"  {k:<32}: {v}")
    return {"shapes": shapes, "memory": mem,
            "why": {
                "matrix per image": "keeps 2-D structure only if left un-flattened; "
                                    "flattening (n,4096) lets linear algebra operate on pixels "
                                    "as features but destroys neighbourhood relations",
                "batch tensor (32,64,64)": "mini-batch is the unit of SGD; 32 is small enough "
                                           "for memory and large enough for a stable gradient",
                "labels as integers": "sparse, loss needs one-hot/probability form (32,C)",
                "weights C x (K+1)": "one decision hyperplane per class -> 40,970 numbers, "
                                     "the object that must fit inside the phone budget"}}


# ======================================================================================
# C. norms of the same feature vector
# ======================================================================================
def section_C(X, y, classes) -> dict:
    print("\n=== C. L1 / L2 / L-inf norms of the same vectors =========================")
    C = len(classes)
    Xd = X.reshape(len(X), -1).astype(np.float64)   # X is already in [0,1] (flatten01)
    n1 = np.abs(Xd).sum(1)                      # because pixels >= 0, L1 = sum
    n2 = np.linalg.norm(Xd, ord=2, axis=1)
    ni = np.abs(Xd).max(1)

    # worked example on one real image: print its brightest pixel and totals
    i = int(np.argmax(n2))
    vec = Xd[i]
    j = int(np.argmax(vec))
    print(f"  example image #{i} ({classes[y[i]]}):")
    print(f"    L1  = sum |x_j|        = {n1[i]:.4f}   (total intensity / leaf coverage)")
    print(f"    L2  = sqrt(sum x_j^2)  = {n2[i]:.4f}   (signal energy)")
    print(f"    Linf= max |x_j|        = {ni[i]:.4f}   at pixel {j} "
          f"(row {j // 64}, col {j % 64}) - brightest pixel")
    print(f"    ordering check  Linf <= L2 <= L1 : "
          f"{ni[i]:.4f} <= {n2[i]:.4f} <= {n1[i]:.4f} -> "
          f"{bool(ni[i] <= n2[i] + 1e-12 <= n1[i] + 1e-12)}")

    # why the three norms disagree: two images with near-identical L2 but different L1/Linf
    order = np.argsort(n2)
    a, b = int(order[len(order) // 3]), int(order[len(order) // 2])
    diff = {"img_a": {"class": classes[y[a]], "L1": round(n1[a], 3), "L2": round(n2[a], 3),
                      "Linf": round(ni[a], 3)},
            "img_b": {"class": classes[y[b]], "L1": round(n1[b], 3), "L2": round(n2[b], 3),
                      "Linf": round(ni[b], 3)},
            "note": "similar L2 (energy) but different L1 (coverage) and Linf (hot pixel)"}

    # per-class statistics
    stats = {}
    print("  per-class mean +/- std:")
    for c in range(C):
        m = y == c
        stats[classes[c]] = {
            "L1": [round(float(n1[m].mean()), 3), round(float(n1[m].std()), 3)],
            "L2": [round(float(n2[m].mean()), 3), round(float(n2[m].std()), 3)],
            "Linf": [round(float(ni[m].mean()), 3), round(float(ni[m].std()), 3)],
        }
        print(f"    {classes[c][:30]:<31} L1 {n1[m].mean():.3f}+/-{n1[m].std():.3f}  "
              f"L2 {n2[m].mean():.3f}+/-{n2[m].std():.3f}  "
              f"Linf {ni[m].mean():.3f}+/-{ni[m].std():.3f}")

    # discrimination power of each norm (one-way ANOVA-style: between/within variance ratio)
    def sep(v):
        gm = v.mean()
        between = sum((v[y == c].mean() - gm) ** 2 * (y == c).sum() for c in range(C)) / len(v)
        within = sum(((v[y == c] - v[y == c].mean()) ** 2).sum() for c in range(C)) / len(v)
        return float(between / (within + 1e-12))
    sep_ratios = {"L1": sep(n1), "L2": sep(n2), "Linf": sep(ni)}
    print("  between/within class variance ratio (higher = norm separates classes better):")
    for k, v in sep_ratios.items():
        print(f"    {k:<5}: {v:.5f}")

    purposes = {
        "L1": "sparse/robust views: total leaf coverage, background suppression, LASSO-style "
              "sparsity, robustness to a few bright artefacts (it dilutes them over 4096 terms)",
        "L2": "Euclidean geometry: nearest-neighbour search, PCA least squares, weight decay - "
              "the norm whose unit ball matches the gradient steps we take",
        "Linf": "worst-case / adversarial analysis: a single saturated pixel (specular "
                "highlight, sensor defect) dominates it - useful for sensor-noise bounds",
    }
    print("  purposes:")
    for k, v in purposes.items():
        print(f"    {k:<5}: {v}")

    # figure: norm distributions per class
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    short = cm.short_names(classes)
    for ai, (name, v) in enumerate([("L1", n1), ("L2", n2), ("Linf", ni)]):
        data = [v[y == c] for c in range(C)]
        ax[ai].boxplot(data, tick_labels=[short[c] for c in range(C)], showfliers=False)
        ax[ai].set_title(f"{name} norm per class")
        ax[ai].tick_params(axis="x", rotation=90, labelsize=7)
    cm.save_fig(fig, "task01_norms")

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    sc = ax.scatter(n1, ni, c=y, s=3, cmap="tab10", alpha=0.5)
    ax.set_xlabel("L1 (total intensity)")
    ax.set_ylabel("Linf (brightest pixel)")
    ax.set_title("Same image, two norms: L1 tracks coverage, Linf tracks hot pixels")
    fig.colorbar(sc, ax=ax, ticks=range(C), label="class")
    cm.save_fig(fig, "task01_norms_scatter")

    return {"example": diff, "per_class": stats, "class_separation_ratio": sep_ratios,
            "purposes": purposes,
            "inequality": "||x||inf <= ||x||2 <= ||x||1 holds for every image"}


# ======================================================================================
# D. preprocessing effects
# ======================================================================================
def _fast_clf(Xtr, ytr, scale=None, max_iter=1000):
    m = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=0)
    m.fit(Xtr, ytr)
    return m


def section_D_resolution(scale_n=1500) -> dict:
    """D1: what does resizing to 64x64 cost?  (same first `scale_n` tomato images at each size)"""
    print("\n=== D1. Effect of resizing / resolution ==================================")
    res = {}
    for size in (32, 64, 128):
        Xs, ys, _ = cm.stream_class_images(cm.TOMATO_PREFIX, size=size, mode="L", n=scale_n)
        tr, _, te = cm.stratified_split(ys)
        Xf = cm.flatten01(Xs)
        m = _fast_clf(Xf[tr], ys[tr])
        acc = accuracy_score(ys[te], m.predict(Xf[te]))
        res[f"{size}x{size} ({size*size} feat)"] = {
            "test_acc": round(float(acc), 4), "n_iter": int(m.n_iter_[0]),
            "converged": bool(m.n_iter_[0] < 1000)}
        print(f"  {size}x{size: <3} = {size*size: <5} features -> test acc {acc:.4f}  "
              f"(n_iter={int(m.n_iter_[0])})")
    return res


def section_D_scaling(classes, scale_n=1000) -> dict:
    """D2: pixel scaling changes the conditioning of X and hence optimisation."""
    print("\n=== D2. Effect of normalisation ([0,1] vs [0,255]) =======================")
    X64, y, _ = cm.load_tomato()
    idx = cm.stratified_subset(y, scale_n)
    tr, _, te = cm.stratified_split(y[idx])
    X01 = cm.flatten01(np.asarray(X64[idx]))
    Xraw = X01 * 255.0

    out = {}
    for name, Xf in (("[0,1]", X01), ("[0,255]", Xraw)):
        m = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=0)
        m.fit(Xf[tr], y[idx][tr])
        # gradient norm of the cross-entropy at the returned solution
        C = len(classes)
        P = cm.softmax_stable(Xf[te] @ m.coef_.T + m.intercept_.T)
        Y = cm.onehot(y[idx][te], C)
        g = (P - Y).T @ Xf[te] / len(te)
        acc = accuracy_score(y[idx][te], m.predict(Xf[te]))
        out[name] = {"test_acc": round(float(acc), 4), "n_iter": int(m.n_iter_[0]),
                     "grad_norm_at_solution": float(np.linalg.norm(g)),
                     "col_norm_ratio": float(np.linalg.norm(Xf, axis=0).max() /
                                             max(np.linalg.norm(Xf, axis=0).min(), 1e-9))}
        print(f"  pixels {name:<8} n_iter={int(m.n_iter_[0]):>3}  test acc {acc:.4f}  "
              f"||grad||={np.linalg.norm(g):.4f}  max/min column norm="
              f"{out[name]['col_norm_ratio']:.1f}")
    return out


def section_D_flattening(scale_n=1000) -> dict:
    """D3: flattening a 2-D image into a vector makes Euclidean distance translation-fragile."""
    print("\n=== D3. Effect of flattening (translation destroys the metric) ==========")
    X64, y, classes = cm.load_tomato()
    idx = cm.stratified_subset(y, scale_n)
    X = cm.flatten01(np.asarray(X64[idx]))

    shift = np.roll(np.asarray(X64[idx]), shift=3, axis=2).reshape(len(idx), -1) / 255.0
    d_shift = np.linalg.norm(X - shift, axis=1)
    d_same, d_diff = [], []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(idx), size=300, replace=False):
        same = np.flatnonzero(y[idx] == y[idx][i])
        same = same[same != i]
        diff = np.flatnonzero(y[idx] != y[idx][i])
        d_same.append(np.linalg.norm(X[i] - X[rng.choice(same)]))
        d_diff.append(np.linalg.norm(X[i] - X[rng.choice(diff)]))
    d_same, d_diff = np.array(d_same), np.array(d_diff)
    ratio = float(d_shift.mean() / d_same.mean())
    if ratio > 1.0:
        consequence = ("a 3-pixel shift is FARTHER than a different leaf of the same class "
                       f"({ratio:.2f}x): flattened-pixel Euclidean distance is not translation "
                       "invariant")
    else:
        consequence = (f"a 3-pixel shift alone costs {ratio*100:.0f}% of the average within-class "
                       "distance, even though the leaf is identical: flattened-pixel Euclidean "
                       "distance conflates translation with identity (it is not translation "
                       "invariant)")
    out = {
        "mean L2 to same image shifted 3 px": round(float(d_shift.mean()), 3),
        "mean L2 to random same-class image": round(float(d_same.mean()), 3),
        "mean L2 to random different-class image": round(float(d_diff.mean()), 3),
        "ratio shift/same-class": round(ratio, 2),
        "consequence": consequence,
    }
    for k, v in out.items():
        print(f"  {k:<38}: {v}")
    return out


def section_D_split(scale_n=1000, n_seeds=10) -> dict:
    """D4: stratified vs naive random split."""
    print("\n=== D4. Effect of the data split =========================================")
    X64, y, classes = cm.load_tomato()
    idx = cm.stratified_subset(y, scale_n)
    X = cm.flatten01(np.asarray(X64[idx]))
    yy = y[idx]
    C = len(classes)
    strat_acc, naive_acc = [], []
    strat_dev, naive_dev = [], []
    target = cm.class_counts(yy, C) / len(yy)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(yy))
        cut = int(0.7 * len(yy))
        # naive: random 70/30 with no stratification
        tr_n, te_n = perm[:cut], perm[cut:]
        # stratified helper
        tr_s, _, te_s = cm.stratified_split(yy, seed=seed)
        for accs, tr, te, devs in ((strat_acc, tr_s, te_s, strat_dev),
                                   (naive_acc, tr_n, te_n, naive_dev)):
            m = LogisticRegression(max_iter=200, solver="lbfgs", random_state=0)
            m.fit(X[tr], yy[tr])
            accs.append(accuracy_score(yy[te], m.predict(X[te])))
            p = cm.class_counts(yy[te], C) / len(te)
            devs.append(float(np.abs(p - target).sum() / 2))       # total variation distance
    out = {
        "stratified: mean acc": round(float(np.mean(strat_acc)), 4),
        "stratified: std acc": round(float(np.std(strat_acc)), 4),
        "naive: mean acc": round(float(np.mean(naive_acc)), 4),
        "naive: std acc": round(float(np.std(naive_acc)), 4),
        "stratified: mean class-proportion shift (TV distance)": round(float(np.mean(strat_dev)), 4),
        "naive: mean class-proportion shift (TV distance)": round(float(np.mean(naive_dev)), 4),
        "stratified accs": [round(float(a), 4) for a in strat_acc],
        "naive accs": [round(float(a), 4) for a in naive_acc],
    }
    for k, v in out.items():
        print(f"  {k:<55}: {v}")
    return out


def section_D_gray(n=1000) -> dict:
    """D5: how much information does dropping colour remove?"""
    print("\n=== D5. Effect of the grayscale conversion =================================")
    Xg, yg, _ = cm.stream_class_images(cm.TOMATO_PREFIX, size=64, mode="L", n=n)
    Xc, _, _ = cm.stream_class_images(cm.TOMATO_PREFIX, size=64, mode="RGB", n=n)
    tr, _, te = cm.stratified_split(yg)

    # how much of the pixel variance lives in colour rather than luminance?
    S = Xc.astype(np.float32)
    R, G, B = S[..., 0], S[..., 1], S[..., 2]
    Y = (R + G + B) / 3.0
    chroma1, chroma2 = R - G, G - B
    tot = Y.var() + chroma1.var() + chroma2.var()
    chroma_frac = float((chroma1.var() + chroma2.var()) / tot)

    Xg_f = cm.flatten01(Xg)
    acc_gray = accuracy_score(yg[te], _fast_clf(Xg_f[tr], yg[tr]).predict(Xg_f[te]))
    Xrgb = S.reshape(len(S), -1) / 255.0
    acc_rgb = accuracy_score(yg[te], _fast_clf(Xrgb[tr], yg[tr]).predict(Xrgb[te]))
    out = {
        "chroma share of pixel variance": round(chroma_frac, 4),
        "gray-64 test acc (scale 1000)": round(float(acc_gray), 4),
        "RGB-64 test acc (scale 1000)": round(float(acc_rgb), 4),
        "features gray": 4096, "features RGB": 12288,
        "note": "RGB buys accuracy but triples the input dimension and the deployment cost",
    }
    for k, v in out.items():
        print(f"  {k:<38}: {v}")
    return out


# ======================================================================================
def main():
    print("#" * 78)
    print("# TASK 1 - problem structure and data representation")
    print("#" * 78)
    X, y, classes = cm.load_tomato()          # 10-class tomato subset, 64x64 grayscale
    X = cm.flatten01(np.asarray(X))           # (n,4096) raw feature matrix - the representation
                                              # the model actually sees (Task 1, section D3)
    results = {
        "dataset": {"name": "PlantVillage tomato subset", "images": int(len(y)),
                    "classes": classes, "input": "64x64 grayscale -> 4096 features"},
    }
    results["A_decomposition"] = section_A(X, y, classes)
    results["B_representation"] = section_B(X, y, classes)
    results["C_norms"] = section_C(X, y, classes)
    results["D1_resolution"] = section_D_resolution()
    results["D2_normalisation"] = section_D_scaling(classes)
    results["D3_flattening"] = section_D_flattening()
    results["D4_split"] = section_D_split()
    results["D5_grayscale"] = section_D_gray()
    OUT("task01_results", results)
    print("\nTASK 1 complete -> results/task01_results.json")


if __name__ == "__main__":
    main()
