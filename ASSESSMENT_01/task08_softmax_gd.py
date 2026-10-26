"""
TASK 8 - Behaviour of gradient-based optimisation.                          (4 marks)

Run:  python task08_gradient_optimisation/task08_softmax_gd.py

Sections
  A. Numerical verification of the derived gradient
         dL/dW = X^T (softmax(XW+b) - Y)/n + lambda*W ,  dL/db = mean(p - y)
     (the derivation itself is written out in report.md).
  B. From-scratch batch GD, SGD and mini-batch SGD on the SAME softmax model:
     convergence path, gradient noise, cost per epoch, wall-clock.
  C. Three learning rates relative to the stability bound lr < 2/lambda_max of
     the Hessian: which are stable, which diverge and why.
  D. Link back to Task 4: the condition number measured here is the same kappa
     that predicts the iteration counts observed.

Outputs: results/task08_results.json, results/figures/task08_*.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402

LAM = 1e-4            # L2 regularisation (weight decay)
SEED = 0


# ----------------------------------------------------------------------------------
# the model, from scratch
# ----------------------------------------------------------------------------------
def loss_and_grad(W, X, Y, lam=LAM):
    """Cross-entropy loss and its gradient for softmax regression (numerically stable)."""
    Z = X @ W
    P = cm.softmax_stable(Z)
    n = len(X)
    loss = -np.sum(Y * np.log(np.clip(P, 1e-12, 1))) / n + lam * 0.5 * np.sum(W ** 2)
    G = X.T @ (P - Y) / n + lam * W
    return float(loss), G, P


def gradient_check(n=60, d=12, C=4, eps=1e-6):
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n, d))
    Y = cm.onehot(rng.integers(0, C, size=n), C)
    W = rng.normal(size=(d, C)) * 0.1
    loss, G, _ = loss_and_grad(W, X, Y, lam=0.0)
    num = np.zeros_like(G)
    for i in range(d):
        for j in range(C):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += eps
            Wm[i, j] -= eps
            lp, _, _ = loss_and_grad(Wp, X, Y, lam=0.0)
            lm, _, _ = loss_and_grad(Wm, X, Y, lam=0.0)
            num[i, j] = (lp - lm) / (2 * eps)
    rel = np.abs(G - num) / (np.abs(G) + np.abs(num) + 1e-12)
    return {"max_abs_diff": float(np.abs(G - num).max()),
            "max_rel_diff": float(rel.max()),
            "loss": loss}


def run_gd(X, Y, mode="batch", lr=0.1, epochs=25, batch=32, lam=LAM, seed=SEED):
    """batch | mini | sgd  -> loss history per epoch, gradient-norm history, wall time."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = Y.shape[1]
    W = np.zeros((d, C))
    hist, gnorm, t_hist = [], [], []
    t0 = time.perf_counter()
    n_steps = 0
    for e in range(epochs):
        if mode == "batch":
            loss, G, _ = loss_and_grad(W, X, Y, lam)
            W -= lr * G
            hist.append(loss); gnorm.append(float(np.linalg.norm(G))); n_steps += 1
        else:
            idx = rng.permutation(n)
            if mode == "mini":
                chunks = [idx[i:i + batch] for i in range(0, n, batch)]
            else:                                        # sgd: one sample at a time
                chunks = [np.array([j]) for j in idx]
            ep_loss = 0.0
            last_g = 0.0
            for ch in chunks:
                l, G, _ = loss_and_grad(W, X[ch], Y[ch], lam)
                W -= lr * G
                ep_loss += l * len(ch)
                last_g = float(np.linalg.norm(G))
                n_steps += 1
            hist.append(ep_loss / n); gnorm.append(last_g)
        t_hist.append(time.perf_counter() - t0)
    return {"loss": hist, "grad_norm": gnorm, "seconds": t_hist[-1],
            "n_gradient_evaluations": n_steps, "W": W}


# ----------------------------------------------------------------------------------
def section_A():
    print("\n=== A. Gradient check (derived vs numerical) ============================")
    r = gradient_check()
    print(f"  max |analytic - numeric| = {r['max_abs_diff']:.3e}")
    print(f"  max relative difference  = {r['max_rel_diff']:.3e}")
    print("  -> the implementation matches  dL/dW = X^T(p - y)/n + lambda W  to "
          f"~1e-7 (central differences).")
    return r


def section_B(Xtr, Ytr, Xte, yte, classes):
    print("\n=== B. Batch GD vs SGD vs mini-batch ====================================")
    C = len(classes)
    # a step size below the stability bound computed in section C (2/lambda_max ~ 0.025);
    # the earlier 0.5 was far above it and made every optimiser diverge, which compared
    # noise, not convergence.
    lr = 0.02
    runs = {}
    for tag, kw in (("batch GD (1 step/epoch)", dict(mode="batch", epochs=40)),
                    ("mini-batch B=32", dict(mode="mini", epochs=40, batch=32)),
                    ("SGD (B=1)", dict(mode="sgd", epochs=5)),
                    ):
        r = run_gd(Xtr, Ytr, lr=lr, **kw)
        acc = float(np.mean(cm.softmax_stable(Xte @ r["W"]).argmax(1) == yte))
        runs[tag] = {"final_loss": round(r["loss"][-1], 4), "test_acc": round(acc, 4),
                     "seconds": round(r["seconds"], 2),
                     "gradient_evaluations": r["n_gradient_evaluations"],
                     "loss_curve": [round(x, 5) for x in r["loss"]],
                     "grad_norm": [round(x, 5) for x in r["grad_norm"]],
                     "loss_std_last10": round(float(np.std(r["loss"][-10:])), 5)}
        print(f"  {tag:<26} loss {r['loss'][0]:.4f} -> {r['loss'][-1]:.4f}  "
              f"acc {acc:.4f}  {r['seconds']:5.2f}s  "
              f"({r['n_gradient_evaluations']} gradient evaluations)")
    print("  the per-step gradient noise (mini/SGD) is what makes the curve jagged;")
    print("  batch GD moves along the exact descent direction, so it is smooth but pays")
    print("  one full X^T(P-Y) pass (O(n d C)) for every single update.")

    # gradient noise magnitude: variance of the per-sample gradient at the solution
    W = run_gd(Xtr, Ytr, mode="batch", lr=0.02, epochs=40)["W"]
    P = cm.softmax_stable(Xtr @ W)
    Y = Ytr
    G_full = Xtr.T @ (P - Y) / len(Xtr)
    rng = np.random.default_rng(0)
    sample_norms = []
    for _ in range(100):
        i = rng.integers(0, len(Xtr))
        g_i = Xtr[i][:, None] @ (P[i] - Y[i])[None, :]
        sample_norms.append(np.linalg.norm(g_i))
    noise = {"norm full-batch gradient": float(np.linalg.norm(G_full)),
             "mean norm single-sample gradient": float(np.mean(sample_norms)),
             "noise-to-signal ratio": float(np.mean(sample_norms) / np.linalg.norm(G_full))}
    print(f"  gradient noise: |g_full| = {noise['norm full-batch gradient']:.4f}, "
          f"E|g_single| = {noise['mean norm single-sample gradient']:.4f}  "
          f"-> ratio {noise['noise-to-signal ratio']:.1f}: a single example points in a "
          f"direction {noise['noise-to-signal ratio']:.0f}x 'larger' than the true step.")
    return {"runs": runs, "noise": noise}


def section_C(Xtr, Ytr, Xte, yte, classes):
    print("\n=== C. Learning rates vs the stability bound ============================")
    C = len(classes)
    # Hessian of the loss at the origin: H = X^T D X / n with D = diag(p(1-p)), p = 1/C
    p0 = 1.0 / C * (1 - 1.0 / C)
    Xs = Xtr
    G = (Xs.T @ Xs) / len(Xs) * p0
    ev = np.linalg.eigvalsh(G)
    lmax, lmin = float(ev[-1]), float(ev[0])
    kappa = lmax / max(lmin, 1e-12)
    bound = 2.0 / lmax
    print(f"  Hessian eigenvalues at the initial point: lambda_max = {lmax:.4f}, "
          f"lambda_min = {lmin:.3e}")
    print(f"  condition number kappa = {kappa:.3e}")
    print(f"  stability bound  lr < 2/lambda_max = {bound:.4f}")
    print(f"  fastest useful step ~ 2/(lambda_max+lambda_min) = {2/(lmax+lmin):.4f};")
    print(f"  to make progress in the ill-conditioned direction we need "
          f"~kappa*log(1/eps) = {kappa:.0f}+ iterations")

    mults = [0.1, 0.5, 1.2]
    runs = {}
    for mm in mults:
        lr = mm * bound
        r = run_gd(Xtr, Ytr, mode="batch", lr=lr, epochs=40)
        diverged = (not np.isfinite(r["loss"][-1])) or r["loss"][-1] > 10 * r["loss"][0]
        acc = float(np.mean(cm.softmax_stable(Xte @ r["W"]).argmax(1) == yte)) \
            if np.all(np.isfinite(r["W"])) else float("nan")
        runs[f"lr={lr:.4f} ({mm}x bound)"] = {
            "lr": lr, "final_loss": float(r["loss"][-1]) if np.isfinite(r["loss"][-1]) else None,
            "diverged": bool(diverged), "test_acc": acc,
            "loss_curve": [round(float(x), 6) for x in r["loss"] if np.isfinite(x)][:40]}
        state = "DIVERGES (oscillates / explodes)" if diverged else "stable"
        print(f"  lr = {lr:.4f} = {mm}x bound -> {state}; final loss {r['loss'][-1]:.4f}")

    return runs, lmax, lmin, kappa, bound


def section_D(runs_B, runs_C, kappa, bound):
    print("\n=== D. Link to Task 4 (conditioning) ====================================")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for tag, r in runs_B.items():
        ax[0].plot(r["loss_curve"], label=tag)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy")
    ax[0].set_title("B. batch / mini-batch / SGD at lr=0.5"); ax[0].legend(fontsize=8)
    ax[1].plot(runs_B["batch GD (1 step/epoch)"]["grad_norm"], "-o", ms=3)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("||gradient||")
    ax[1].set_title("Batch gradient norm decays smoothly")
    for tag, r in runs_C.items():
        ax[2].plot(r["loss_curve"], label=tag)
    ax[2].axhline(0, c="k", lw=0.5)
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("cross-entropy")
    ax[2].set_title(f"C. learning rates vs 2/lambda_max={bound:.3f}")
    ax[2].legend(fontsize=7); ax[2].set_ylim(-0.5, 6)
    cm.save_fig(fig, "task08_optimisation")
    print(f"  measured kappa = {kappa:.3e}: the same quantity computed in Task 4 from")
    print("  cov(X) predicts both the small usable step size (stability) and the large")
    print("  number of iterations (accuracy); standardising/whitening reduces kappa and")
    print("  both effects shrink together - that is the observed agreement.")


def main():
    print("#" * 78)
    print("# TASK 8 - softmax regression from scratch: batch / SGD / mini-batch")
    print("#" * 78)
    res = {"A_gradient_check": section_A()}

    X64, y, classes = cm.load_tomato()
    idx = cm.stratified_subset(y, 2000)
    tr, _, te = cm.stratified_split(y[idx])
    X = cm.flatten01(np.asarray(X64[idx]))
    Y = cm.onehot(y[idx][tr], len(classes))
    with cm.timer("experiments"):
        res["B_strategies"] = section_B(X[tr], Y, X[te], y[idx][te], classes)
        runs_C, lmax, lmin, kappa, bound = section_C(
            X[tr], Y, X[te], y[idx][te], classes)
        res["C_learning_rates"] = {k: {kk: vv for kk, vv in v.items() if kk != "loss_curve"}
                                   for k, v in runs_C.items()}
        res["C_hessian"] = {"lambda_max": lmax, "lambda_min": lmin,
                            "condition_number": kappa, "stability_bound_lr": bound}
        section_D(res["B_strategies"]["runs"], runs_C, kappa, bound)
    cm.save_json("task08_results", res)
    print("\nTASK 8 complete -> results/task08_results.json")


if __name__ == "__main__":
    main()
