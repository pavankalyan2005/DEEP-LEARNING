"""
TASK 5 - Different baseline learning strategies.                           (4 marks)

Run:  python task05_baselines/task05_baselines.py

Four structurally different learning strategies are compared, each on RAW
4,096-dimensional pixels and on PCA-compressed (100 components) features,
at the four data scales 1,000 / 5,000 / 10,000 / full:

  1. softmax regression   - linear parametric, global hyperplanes, C(d+1) parameters
  2. k-NN (k = 15)        - non-parametric, instance memory, local constant model
  3. random forest        - tree/axis-aligned partition model, piecewise-constant
  4. Nystroem RBF + linear- explicit kernel method (approximate RKHS)

Reported per run: accuracy, macro-F1, training time, per-image inference time,
stored scalars and float32 memory footprint.

Outputs: results/task05_results.json, results/figures/task05_*.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

# Nystroem/LinearSVC gives calibrated-ish scores through the decision function;
# we derive probabilities with a softmax so Task 3-style analysis stays possible.
MODELS = ["softmax regression", "k-NN (k=15)", "random forest", "RBF kernel (Nystroem)"]


def build(name: str):
    if name == "softmax regression":
        return make_pipeline(StandardScaler(with_std=False),
                             LogisticRegression(max_iter=800, solver="lbfgs", random_state=0))
    if name == "k-NN (k=15)":
        return KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    if name == "random forest":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=0,
                                      n_jobs=-1)
    if name == "RBF kernel (Nystroem)":
        return make_pipeline(StandardScaler(with_std=False),
                             Nystroem(n_components=300, random_state=0),
                             LinearSVC(C=1.0, max_iter=3000, dual=True, random_state=0))
    raise KeyError(name)


def infer_ms(model, X, n_probe=200, single=False) -> float:
    """Mean inference time in ms per image (batched, or single-image calls)."""
    X = X[:n_probe]
    if single:                                   # per-call latency as a phone would see it
        t0 = time.perf_counter()
        for i in range(len(X)):
            model.predict(X[i:i + 1])
        return (time.perf_counter() - t0) / len(X) * 1e3
    model.predict(X)                              # warm-up
    t0 = time.perf_counter()
    model.predict(X)
    return (time.perf_counter() - t0) / len(X) * 1e3


def run_one(model_name, feat_name, Xtr, ytr, Xte, yte, names):
    m = build(model_name)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    t_train = time.perf_counter() - t0
    yhat = m.predict(Xte)
    acc = accuracy_score(yte, yhat)
    f1 = f1_score(yte, yhat, average="macro", zero_division=0)
    rep = cm.classification_report(yte, yhat, names)     # per-class P/R/F1 + confusion matrix
    t_batch = infer_ms(m, Xte, n_probe=min(500, len(Xte)))
    t_single = infer_ms(m, Xte, n_probe=min(200, len(Xte)), single=True)
    foot = cm.model_footprint(m, Xtr.shape[1], len(names))
    # kernel / forest / linear: count what is actually stored in float32
    row = {"features": feat_name, "model": model_name,
           "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
           "train_s": round(t_train, 2),
           "infer_batch_ms": round(t_batch, 3), "infer_single_ms": round(t_single, 3),
           "stored_scalars": foot["stored_scalars"], "model_MB_f32": foot["MB_float32"],
           "per_class": rep["per_class"], "confusion": rep["confusion"]}
    print(f"    {feat_name:<12} {model_name:<24} acc {acc:.4f}  F1 {f1:.4f}  "
          f"train {t_train:6.2f}s  infer {t_batch:6.3f} ms/img  "
          f"{foot['MB_float32']:7.3f} MB")
    return row, m


def main():
    print("#" * 78)
    print("# TASK 5 - three (four) distinct learning strategies, raw vs PCA features")
    print("#" * 78)
    X64, y, classes = cm.load_tomato()
    names = classes
    out = {"classes": names, "runs": []}

    for label, n in cm.make_scales(len(y)):
        idx = cm.stratified_subset(y, n)
        tr, va, te = cm.stratified_split(y[idx])
        Xall = cm.flatten01(np.asarray(X64[idx]))
        yy = y[idx]
        print(f"\n=== scale {label}: n={len(idx)} (train {len(tr)}, test {len(te)}) ===")
        pca = PCA(n_components=100, svd_solver="randomized", random_state=0).fit(Xall[tr])
        feats = {"raw-4096": (Xall, Xall[te]),
                 "PCA-100": (pca.transform(Xall), pca.transform(Xall[te]))}
        for feat_name, (A, Ate) in feats.items():
            for mname in MODELS:
                try:
                    row, _ = run_one(mname, feat_name, A[tr], yy[tr], Ate, yy[te], names)
                    row["scale"] = label
                    row["n_images"] = int(len(idx))
                    out["runs"].append(row)
                except Exception as exc:                      # noqa: BLE001
                    print(f"    {feat_name} {mname} FAILED: {type(exc).__name__}: {exc}")
                    out["runs"].append({"scale": label, "features": feat_name,
                                        "model": mname, "error": f"{type(exc).__name__}: {exc}"})
        cm.save_json("task05_results", out)        # checkpoint after every scale

    # ---- figures ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in out["runs"] if "accuracy" in r]
    scales = [s for s, _ in cm.make_scales(len(y))]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for mname in MODELS:
        for feat, style in (("raw-4096", "-o"), ("PCA-100", "--s")):
            ys = [next((r["accuracy"] for r in runs
                        if r["scale"] == s and r["model"] == mname and r["features"] == feat), None)
                  for s in scales]
            ax[0].plot(scales, ys, style, label=f"{mname} / {feat}", ms=5, lw=1.2)
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("data scale"); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("Accuracy vs data scale"); ax[0].legend(fontsize=6.5)
    ax[0].tick_params(axis="x", rotation=20)

    for mname in MODELS:
        ys = [next((r["train_s"] for r in runs
                    if r["scale"] == s and r["model"] == mname and r["features"] == "PCA-100"), None)
              for s in scales]
        ax[1].plot(scales, ys, "-o", label=mname, ms=5)
    ax[1].set_yscale("log"); ax[1].set_xlabel("data scale"); ax[1].set_ylabel("seconds")
    ax[1].set_title("Training cost (PCA-100 features)"); ax[1].legend(fontsize=7)
    ax[1].tick_params(axis="x", rotation=20)

    for mname in MODELS:
        ys = [next((r["infer_single_ms"] for r in runs
                    if r["scale"] == s and r["model"] == mname and r["features"] == "PCA-100"), None)
              for s in scales]
        ax[2].plot(scales, ys, "-o", label=mname, ms=5)
    ax[2].axhline(50, ls="--", c="r", label="50 ms budget (Day 4)")
    ax[2].set_xlabel("data scale"); ax[2].set_ylabel("ms per image")
    ax[2].set_title("Single-image inference (PCA-100)"); ax[2].legend(fontsize=7)
    ax[2].tick_params(axis="x", rotation=20)
    cm.save_fig(fig, "task05_baselines")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"{r['model'].split(' (')[0]}\n{r['features']}" for r in runs
              if r["scale"] == scales[-1]]
    vals = [r["model_MB_f32"] for r in runs if r["scale"] == scales[-1]]
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)), labels, fontsize=7)
    ax.set_ylabel("stored model size (MB, float32)")
    ax.set_title(f"Deployment footprint at full scale (n={scales[-1]})")
    cm.save_fig(fig, "task05_footprint")

    cm.save_json("task05_results", out)
    print("\nTASK 5 complete -> results/task05_results.json")


if __name__ == "__main__":
    main()
