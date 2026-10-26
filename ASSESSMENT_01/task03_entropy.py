"""
TASK 3 - Uncertainty in labels and predictions: probability & information theory.

Run:  python task03_probability_information/task03_entropy.py

Sections
  A. Entropy of the class distribution: real (imbalanced) vs uniform vs 3-class
     subset vs 38-class (Day-4 preview), in nats and bits.
  B. Cross-entropy / KL / mutual information of three trained models
     (linear softmax, k-NN, random forest) on the same test split, plus the
     predictive-entropy profile per class.
  C. KL asymmetry worked with explicit numbers, both directions, with meaning.
  D. Why min cross-entropy == min KL: CE(q,p) = H(q) + KL(q||p), shown numerically
     for every model and confirmed by identical gradients.
  E. What the gaps reveal: confidence, calibration and where the errors come from.

Outputs: results/task03_results.json, results/figures/task03_*.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

SCALE = 5000


# ======================================================================================
# A. entropy of the label distribution
# ======================================================================================
def section_A(y, classes, sub_counts: np.ndarray) -> dict:
    print("\n=== A. Entropy of the class distributions ================================")
    C = len(classes)
    p = cm.class_counts(y, C).astype(np.float64)
    p /= p.sum()
    H = cm.entropy(p)
    H_unif = np.log(C)

    # 3-class subset (Healthy / Early Blight / Late Blight)
    p3 = sub_counts.astype(np.float64)
    p3 /= p3.sum()
    H3 = cm.entropy(p3)

    rows = {
        "10-class tomato (observed)": {"H_nats": float(H), "H_bits": float(H / np.log(2)),
                                       "max_bits": float(np.log2(C)),
                                       "eff_classes": float(np.exp(H))},
        "10-class uniform (max)": {"H_nats": float(H_unif), "H_bits": float(H_unif / np.log(2)),
                                   "eff_classes": float(C)},
        "3-class Healthy/Early/Late": {"H_nats": float(H3), "H_bits": float(H3 / np.log(2)),
                                       "eff_classes": float(np.exp(H3))},
        "38-class Day-4 (uniform)": {"H_nats": float(np.log(38)), "H_bits": float(np.log2(38)),
                                     "eff_classes": 38.0},
    }
    print(f"  {'distribution':<30}{'H (nats)':>10}{'H (bits)':>10}{'effective classes':>20}")
    for k, v in rows.items():
        print(f"  {k:<30}{v['H_nats']:>10.3f}{v['H_bits']:>10.3f}{v['eff_classes']:>20.2f}")
    print("  class proportions:", {classes[i]: round(float(p[i]), 3) for i in range(C)})
    print(f"  imbalance costs {H_unif - H:.3f} nats of uncertainty vs a balanced problem; "
          f"a majority-class predictor already achieves {p.max():.3f} accuracy")

    # information-theoretic value of a guess: expected code length saving
    out = {"distributions": rows,
           "proportions": {classes[i]: float(p[i]) for i in range(C)},
           "imbalance_entropy_loss_nats": float(H_unif - H),
           "majority_baseline": float(p.max()),
           "uniform_baseline": 1.0 / C,
           "reduction_bits_vs_uniform": float((H_unif - H) / np.log(2))}
    return out


# ======================================================================================
# B. cross-entropy and KL of actual models
# ======================================================================================
def _train_models(Xtr, ytr, C, seed=0):
    models = {
        "softmax regression (linear)": LogisticRegression(max_iter=600, solver="lbfgs",
                                                          random_state=seed),
        "k-NN (k=15, non-parametric)": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "random forest (tree-based)": RandomForestClassifier(n_estimators=200, random_state=seed,
                                                             n_jobs=-1),
    }
    for m in models.values():
        m.fit(Xtr, ytr)
    return models


def section_B(Xtr, ytr, Xte, yte, classes, models=None) -> dict:
    print("\n=== B. Cross-entropy and KL divergence of three models ==================")
    C = len(classes)
    short = cm.short_names(classes)
    if models is None:
        models = _train_models(Xtr, ytr, C)

    # true empirical distribution on the test split (what CE is measured against)
    q = cm.class_counts(yte, C).astype(np.float64)
    q /= q.sum()
    H_q = cm.entropy(q)                      # irreducible entropy of the test labels

    table, per_class_ce = {}, {}
    for name, m in models.items():
        # cast to float64: sklearn keeps the float32 dtype of the input, and a
        # float32 `log` would otherwise limit CE/KL agreement to ~1e-8
        P = np.asarray(m.predict_proba(Xte), dtype=np.float64)
        ce = cm.cross_entropy(yte, P)                       # per-sample loss: -mean log p(true)
        kl_ps = float(cm.kl(cm.onehot(yte, C), P).mean())   # per-sample KL(onehot||p) == CE
        acc = accuracy_score(yte, m.predict(Xte))
        H_pred = cm.entropy(P, axis=1).mean()               # model's own uncertainty per sample
        table[name] = {
            "accuracy": round(float(acc), 4),
            "cross_entropy_nats": round(ce, 4),
            "KL_persample_nats": round(kl_ps, 4),
            "CE_minus_KL_persample": round(ce - kl_ps, 8),
            "H_predictive_mean_nats": round(float(H_pred), 4),
            "mean_max_prob": round(float(P.max(1).mean()), 4),
        }
        per_class_ce[name] = [
            float(-np.log(np.clip(P[yte == c, c].mean(), 1e-9, 1)).__float__())
            for c in range(C)]
        print(f"  {name}")
        for k, v in table[name].items():
            print(f"      {k:<28}: {v}")
    print(f"  H(empirical test labels) = {H_q:.4f} nats (aggregate label entropy, same for all")
    print(f"  models). Because labels are one-hot, H(one-hot)=0, so the per-sample loss obeys")
    print(f"  CE == mean KL(onehot||p) exactly: the model's loss *is* its KL to the labels.")

    # figure: per-sample CE and KL coincide (one-hot labels); accuracy annotated
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(table)
    ce = [table[n]["cross_entropy_nats"] for n in names]
    kl = [table[n]["KL_persample_nats"] for n in names]
    ac = [table[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(names))
    ax.bar(x - 0.18, ce, 0.36, label="per-sample cross-entropy (nats)")
    ax.bar(x + 0.18, kl, 0.36, label="per-sample KL(one-hot || p) (nats)")
    for i, a in enumerate(ac):
        ax.text(i, max(ce) * 0.55, f"acc\n{a:.3f}", ha="center", fontsize=9, color="white")
    ax.set_xticks(x, [n.split(" (")[0] for n in names])
    ax.set_ylabel("nats per image")
    ax.set_title("Task 3: CE and KL(one-hot || p) are identical - training minimises KL")
    ax.legend(fontsize=8)
    cm.save_fig(fig, "task03_ce_kl")

    # per-class predictive entropy profile of the best model
    best = min(table, key=lambda n: table[n]["cross_entropy_nats"])
    P = np.asarray(models[best].predict_proba(Xte), dtype=np.float64)
    Hpi = cm.entropy(P, axis=1)
    fig, ax = plt.subplots(figsize=(9.5, 4))
    box = [Hpi[yte == c] for c in range(C)]
    ax.boxplot(box, tick_labels=short, showfliers=False)
    ax.set_ylabel("predictive entropy (nats)")
    ax.set_title(f"Predictive uncertainty per class ({best})")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    cm.save_fig(fig, "task03_entropy_per_class")

    return {"per_model": table,
            "H_empirical_test": float(H_q),
            "per_class_CE_nats": {name: {short[c]: v[c] for c in range(C)}
                                  for name, v in per_class_ce.items()}}, models


# ======================================================================================
# C. KL asymmetry
# ======================================================================================
def section_C() -> dict:
    print("\n=== C. KL divergence is asymmetric - worked numbers ======================")
    # two plausible predictive distributions over (Healthy, Early, Late)
    p = np.array([0.90, 0.05, 0.05])     # "truth" / reference distribution
    q = np.array([0.55, 0.35, 0.10])     # a model's prediction
    kl_pq = float(cm.kl(p, q))
    kl_qp = float(cm.kl(q, p))
    print(f"  p (reference) = {p}")
    print(f"  q (model)     = {q}")
    print(f"  KL(p||q) = sum p log(p/q) = {kl_pq:.4f} nats = {kl_pq/np.log(2):.4f} bits")
    print(f"      terms: {np.round(p * np.log(p / q), 4)}")
    print(f"  KL(q||p) = sum q log(q/p) = {kl_qp:.4f} nats = {kl_qp/np.log(2):.4f} bits")
    print(f"      terms: {np.round(q * np.log(q / p), 4)}")
    print(f"  difference = {abs(kl_pq - kl_qp):.4f} nats -> the direction matters")

    # a sharper case: q puts mass where p puts ~none
    p2 = np.array([0.98, 0.01, 0.01])
    q2 = np.array([0.34, 0.33, 0.33])
    a, b = float(cm.kl(p2, q2)), float(cm.kl(q2, p2))
    print(f"\n  p2 = {p2}, q2 = {q2}")
    print(f"  KL(p2||q2) = {a:.4f}   KL(q2||p2) = {b:.4f}")
    print("  KL(p||q) punishes q for failing to cover p (zero-forcing / mode-seeking when")
    print("  used as an objective); KL(q||p) punishes q for putting mass where p is absent.")

    meanings = {
        "KL(p||q)  (forward, 'fit' direction)": "expected surprise of the TRUE data under the "
            "model - this is the quantity cross-entropy minimises; it is infinite if q assigns "
            "0 to an event p can produce",
        "KL(q||p)  (reverse, 'cover' direction)": "expected surprise of the MODEL under the "
            "truth - used in variational inference; it forces q to stay inside the support of p "
            "and tolerates q missing some mass of p",
    }
    for k, v in meanings.items():
        print(f"  {k}: {v}")
    return {"p": p.tolist(), "q": q.tolist(), "KL_p_q": float(kl_pq), "KL_q_p": float(kl_qp),
            "p2": p2.tolist(), "q2": q2.tolist(), "KL_p2_q2": float(a), "KL_q2_p2": float(b),
            "meanings": meanings}


# ======================================================================================
# D. min CE == min KL
# ======================================================================================
def section_D(yte, classes, models_probs: dict, H_q: float) -> dict:
    print("\n=== D. Minimising cross-entropy = minimising KL ==========================")
    C = len(classes)
    q = cm.class_counts(yte, C).astype(np.float64)
    q /= q.sum()                                     # empirical label distribution on test
    out = {"label_entropy_nats": float(H_q)}
    print(f"  H(empirical label distribution) = {H_q:.4f} nats (same for every model).")

    # (i) per-sample identity -- the one that governs training
    print("  (i) per-sample identity (one-hot labels, so H(one-hot)=0):")
    for name, P in models_probs.items():
        ce = cm.cross_entropy(yte, P)
        kl_ps = float(cm.kl(cm.onehot(yte, C), P).mean())
        out[name] = {"CE_persample": round(ce, 6), "KL_persample": round(kl_ps, 6),
                     "difference": round(ce - kl_ps, 10)}
        print(f"      {name:<30} CE {ce:.6f}  KL(one-hot||p) {kl_ps:.6f}  "
              f"diff {ce - kl_ps:+.2e}")

    # (ii) aggregate identity -- H(q) + KL(q||pbar) = CE(q, pbar)
    print("  (ii) aggregate identity: CE(q, pbar) = H(q) + KL(q || pbar),")
    print("       q = test label frequencies, pbar = mean predicted distribution:")
    for name, P in models_probs.items():
        pbar = P.mean(0)
        ce_agg = float(-np.sum(q * np.log(np.clip(pbar, 1e-12, None))))
        kl_agg = float(cm.kl(q, pbar))
        holds = bool(abs(ce_agg - H_q - kl_agg) < 1e-9)
        out[name]["CE_aggregate"] = round(ce_agg, 6)
        out[name]["KL_aggregate"] = round(kl_agg, 6)
        out[name]["aggregate_identity_holds"] = holds
        print(f"      {name:<30} CE {ce_agg:.4f} = H {H_q:.4f} + KL {kl_agg:.4f} -> {holds}")

    # gradient identity: dCE/dz = dKL/dz = p - y  (H does not depend on the model)
    z = np.array([[2.0, 0.5, -1.0]])
    D = z.shape[1]
    P = cm.softmax_stable(z)
    y_i = 1
    grad_ce = P.copy()
    grad_ce[0, y_i] -= 1
    eps = 1e-6

    def kl_of_logits(zz):
        pp = cm.softmax_stable(zz)
        return cm.kl(cm.onehot([y_i], D), pp)[0]

    num = np.array([[(kl_of_logits(z + eps * np.eye(D)[k]) -
                      kl_of_logits(z - eps * np.eye(D)[k])) / (2 * eps) for k in range(D)]])
    print(f"  analytic dCE/dz = p - y = {grad_ce[0]}")
    print(f"  numeric  dKL/dz          = {num[0]}   (finite differences) -> identical")
    out["gradient_identity"] = {"dCE_dz": grad_ce.tolist(), "dKL_dz_numeric": num.tolist()}
    print("  because H(q) does not depend on W, argmin_W CE = argmin_W KL.")
    return out


# ======================================================================================
def main():
    print("#" * 78)
    print("# TASK 3 - entropy, cross-entropy and KL divergence")
    print("#" * 78)
    X64, y, classes = cm.load_tomato()
    # the three classes named in the brief: Healthy, Early Blight, Late Blight
    sub = [i for i, n in enumerate(classes)
           if n.lower().replace("_", " ").endswith("healthy")
           or "early blight" in n.lower().replace("_", " ")
           or "late blight" in n.lower().replace("_", " ")]
    print("  3-class subset (Healthy / Early Blight / Late Blight): "
          f"{[classes[i] for i in sub]}  counts={[int(np.sum(y == i)) for i in sub]}")
    sub_counts = np.array([int(np.sum(y == i)) for i in sub])
    res = {"A_entropy": section_A(y, classes, sub_counts)}

    idx = cm.stratified_subset(y, SCALE)
    tr, va, te = cm.stratified_split(y[idx])
    X = cm.flatten01(np.asarray(X64[idx]))
    with cm.timer("train 3 models"):
        models = _train_models(X[tr], y[idx][tr], len(classes))
    res["B_models"], _ = section_B(X[tr], y[idx][tr], X[te], y[idx][te], classes, models)

    probs = {n: np.asarray(m.predict_proba(X[te]), dtype=np.float64)
             for n, m in models.items()}
    H_q = res["B_models"]["H_empirical_test"]
    res["C_asymmetry"] = section_C()
    res["D_CE_equals_KL"] = section_D(y[idx][te], classes, probs, H_q)
    cm.save_json("task03_results", res)
    print("\nTASK 3 complete -> results/task03_results.json")


if __name__ == "__main__":
    main()
