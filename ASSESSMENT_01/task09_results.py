"""
TASK 9 - Organise and examine experimental performance.                    (3 marks)

Run:  python task09_results_organisation/task09_results.py

Reads the experiment log written by Task 5 (all models x all data scales x raw
and PCA features) and turns it into organised evidence:

  A. Accuracy / macro-F1 tables for every model at 1,000 / 5,000 / 10,000 / full.
  B. Cost tables: training time, batched and single-image inference, stored
     parameter count and float32 memory footprint.
  C. Per-class precision / recall / F1 and the confusion matrix of the best
     configuration, with the most confusable class pairs ranked.
  D. Why those pairs are confused - distance between the class-mean images
     (image evidence) for every class pair.
  E. Effect of dataset size and of compression, in numbers.
  Markdown tables are written to results/task09_tables.md.

Outputs: results/task09_results.json, results/task09_tables.md,
         results/figures/task09_*.png
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

MODELS = ["softmax regression", "k-NN (k=15)", "random forest", "RBF kernel (Nystroem)"]


def load_task05() -> dict:
    res = cm.load_json("task05_results")
    if res is None or not res.get("runs"):
        print("[note] results/task05_results.json not found - running Task 5 first ...")
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "task05_baselines", "task05_baselines.py")], check=True)
        res = cm.load_json("task05_results")
    return res


def get(runs, scale, model, feat, key):
    for r in runs:
        if r.get("scale") == scale and r.get("model") == model and r.get("features") == feat:
            return r.get(key)
    return None


# ======================================================================================
def tables(runs, scales, names) -> str:
    md = ["# Task 9 - organised experimental results", ""]
    md += ["## A. Accuracy and macro-F1 by data scale", "",
           "| data scale | model | features | accuracy | macro-F1 |",
           "|---|---|---|---|---|"]
    for s in scales:
        for m in MODELS:
            for f in ("raw-4096", "PCA-100"):
                a, g = get(runs, s, m, f, "accuracy"), get(runs, s, m, f, "macro_f1")
                md.append(f"| {s} | {m} | {f} | {a} | {g} |")
    md += ["", "## B. Cost at each scale (PCA-100 features)", "",
           "| data scale | model | train s | batch ms/img | single ms/img | stored scalars | MB (f32) |",
           "|---|---|---|---|---|---|---|"]
    for s in scales:
        for m in MODELS:
            r = [get(runs, s, m, "PCA-100", k) for k in
                 ("train_s", "infer_batch_ms", "infer_single_ms",
                  "stored_scalars", "model_MB_f32")]
            md.append(f"| {s} | {m} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    md += ["", "## B2. Cost on raw 4,096 features at full scale", "",
           "| model | train s | batch ms/img | single ms/img | MB (f32) |",
           "|---|---|---|---|---|"]
    for m in MODELS:
        r = [get(runs, scales[-1], m, "raw-4096", k) for k in
             ("train_s", "infer_batch_ms", "infer_single_ms", "model_MB_f32")]
        md.append(f"| {m} | {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    md += ["", "## E. Effect of compression (raw - PCA-100) on accuracy", "",
           "| data scale | " + " | ".join(MODELS) + " |",
           "|---|" + "---|" * len(MODELS)]
    for s in scales:
        cells = []
        for m in MODELS:
            a, p = get(runs, s, m, "raw-4096", "accuracy"), get(runs, s, m, "PCA-100", "accuracy")
            cells.append(f"{(p - a):+.4f}" if (a is not None and p is not None) else "-")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(md) + "\n"


def confusion_analysis(runs, scales, names):
    print("\n=== C. Confusion analysis ===============================================")
    best = None
    for r in runs:
        if r.get("scale") == scales[-1] and r.get("features") == "raw-4096" and "confusion" in r:
            if best is None or r["accuracy"] > best["accuracy"]:
                best = r
    C = np.array(best["confusion"], dtype=float)
    off = C - np.diag(np.diag(C))
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                pairs.append((int(off[i, j]), names[i], names[j]))
    pairs.sort(reverse=True)
    print(f"  best model at full scale: {best['model']} (acc {best['accuracy']})")
    print("  most confusable pairs (row predicted as column):")
    for cnt, a, b in pairs[:8]:
        print(f"    {a:<26} -> {b:<26} : {int(cnt)} images")

    # per-class metrics of the best model
    pc = best["per_class"]
    print("  per-class metrics (best model, full scale):")
    for n in names:
        d = pc[n]
        print(f"    {n:<30} P {d['precision']:.3f}  R {d['recall']:.3f}  "
              f"F1 {d['f1']:.3f}  n={d['support']}")
    return {"best": best, "top_confusions": [{"from": a, "to": b, "count": c}
                                             for c, a, b in pairs[:10]]}


def visual_distance(X64, y, names):
    """D. image-level evidence: how similar are the class-mean images?"""
    print("\n=== D. Distance between class-mean images ===============================")
    X = cm.flatten01(np.asarray(X64))
    means = np.stack([X[y == c].mean(0) for c in range(len(names))])
    D = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            D[i, j] = np.linalg.norm(means[i] - means[j])
    np.fill_diagonal(D, np.inf)
    pairs = sorted(((D[i, j], names[i], names[j]) for i in range(len(names))
                    for j in range(i + 1, len(names))), key=lambda t: t[0])
    print("  most similar classes by mean-image distance:")
    for d, a, b in pairs[:6]:
        print(f"    {a:<26} ~ {b:<26}  L2(mean images) = {d:.3f}")
    print("  most different:")
    for d, a, b in pairs[-3:]:
        print(f"    {a:<26} ! {b:<26}  L2(mean images) = {d:.3f}")
    return D, pairs


def figures(runs, scales, names, best, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = cm.short_names(names)

    # accuracy vs scale, one line per model, raw vs PCA
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for m in MODELS:
        for f, st in (("raw-4096", "-o"), ("PCA-100", "--s")):
            ys = [get(runs, s, m, f, "accuracy") for s in scales]
            ax[0].plot(scales, ys, st, ms=5, lw=1.2, label=f"{m}/{f}")
    ax[0].set_ylim(0, 1.02); ax[0].set_xlabel("data scale")
    ax[0].set_ylabel("test accuracy"); ax[0].set_title("Accuracy vs data set size")
    ax[0].legend(fontsize=6); ax[0].tick_params(axis="x", rotation=20)

    # accuracy vs single-image inference time (full scale)
    for i, f in (("raw-4096", "o"), ("PCA-100", "s")):
        xs = [get(runs, scales[-1], m, f, "infer_single_ms") for m in MODELS]
        ys = [get(runs, scales[-1], m, f, "accuracy") for m in MODELS]
        ax[1].scatter(xs, ys, s=70, marker=f, label=f)
        for m, x, y in zip(MODELS, xs, ys):
            if x is not None:
                ax[1].annotate(m.split(" (")[0], (x, y), fontsize=7,
                               xytext=(4, 4), textcoords="offset points")
    ax[1].axvline(50, ls="--", c="r", label="50 ms budget (Day 4)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("single-image inference (ms, log)")
    ax[1].set_ylabel("test accuracy (full scale)")
    ax[1].set_title("Accuracy vs latency"); ax[1].legend(fontsize=7)
    cm.save_fig(fig, "task09_accuracy_cost")

    # confusion heatmap of the best model
    C = np.array(best["confusion"])
    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(Cn, cmap="Blues")
    ax.set_xticks(range(len(names)), [n[:14] for n in short], rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), [n[:14] for n in short], fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if C[i, j] > 0:
                ax.text(j, i, int(C[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if Cn[i, j] > 0.55 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalised) - {best['model']}, "
                 f"{best['features']}, full scale")
    fig.colorbar(im, ax=ax, fraction=0.046)
    cm.save_fig(fig, "task09_confusion")

    # class-mean similarity matrix
    fig, ax = plt.subplots(figsize=(7, 5.8))
    Dm = D.copy(); np.fill_diagonal(Dm, 0)
    im = ax.imshow(Dm, cmap="viridis")
    ax.set_xticks(range(len(names)), [n[:14] for n in short], rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), [n[:14] for n in short], fontsize=7)
    ax.set_title("L2 distance between class-mean images (lighter = more similar)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    cm.save_fig(fig, "task09_class_similarity")


def main():
    print("#" * 78)
    print("# TASK 9 - organised experimental performance")
    print("#" * 78)
    data = load_task05()
    runs = [r for r in data["runs"] if "accuracy" in r]
    scales = sorted({r["scale"] for r in runs},
                    key=lambda s: {"full": 3, "1000": 0, "5000": 1, "10000": 2}.get(s, 9))
    X64, y, names = cm.load_tomato()

    md = tables(runs, scales, names)
    conf = confusion_analysis(runs, scales, names)
    D, pairs = visual_distance(X64, y, names)
    figures(runs, scales, names, conf["best"], D)

    # markdown: append per-class + confusion narrative
    best = conf["best"]
    md += "\n## C. Per-class precision / recall / F1 (best model, full scale)\n\n"
    md += "| class | precision | recall | F1 | support |\n|---|---|---|---|---|\n"
    for n in names:
        d = best["per_class"][n]
        md += (f"| {n} | {d['precision']:.3f} | {d['recall']:.3f} | "
               f"{d['f1']:.3f} | {d['support']} |\n")
    md += "\n### Most confusable pairs (count of images)\n\n| from | to | count |\n|---|---|---|\n"
    for p in conf["top_confusions"]:
        md += f"| {p['from']} | {p['to']} | {p['count']} |\n"
    md += "\n### Closest class-mean images (why those pairs are confused)\n\n"
    md += "| class A | class B | L2 distance of mean images |\n|---|---|---|\n"
    for d, a, b in pairs[:8]:
        md += f"| {a} | {b} | {d:.3f} |\n"
    with open(os.path.join(cm.RESULTS, "task09_tables.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n[save] results/task09_tables.md")

    # accuracy retained vs data size and compression, summarised numerically
    growth = {}
    for m in MODELS:
        first = get(runs, scales[0], m, "raw-4096", "accuracy")
        last = get(runs, scales[-1], m, "raw-4096", "accuracy")
        growth[m] = round(float(last - first), 4) if first and last else None
    print("  accuracy gained from 1,000 -> full images (raw features):")
    for m, g in growth.items():
        print(f"    {m:<26} {g:+.4f}")

    cm.save_json("task09_results", {
        "scales": scales, "growth_1000_to_full": growth,
        "best_model_full_scale": {k: v for k, v in best.items() if k != "per_class"},
        "top_confusions": conf["top_confusions"],
        "class_mean_distances": {f"{names[i]} ~ {names[j]}": float(D[i, j])
                                 for _, i_, j_ in pairs[:8]
                                 for i, j in [(names.index(i_), names.index(j_))]},
        "n_runs": len(runs)})
    print("\nTASK 9 complete -> results/task09_results.json + task09_tables.md")


if __name__ == "__main__":
    main()
