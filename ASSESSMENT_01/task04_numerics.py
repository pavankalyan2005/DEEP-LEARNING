"""
TASK 4 - Diagnose numerical computation issues.                            (3 marks)

Run:  python task04_numerics/task04_numerics.py

Sections
  A. Reproduce the failure: naive softmax / naive log-softmax on z = [1000, 999, 995]
     and on a sweep of logit magnitudes; trace how nan/inf enters the gradient.
  B. The fix: softmax(z) = softmax(z - max z), log-sum-exp = m + log sum exp(z-m).
     Show the result is mathematically unchanged and matches exact arithmetic.
  C. Condition number of the feature covariance matrix for raw / standardised /
     PCA-whitened features, and how it controls the stable learning rate and the
     convergence rate of gradient descent (with a measured demonstration).

Outputs: results/task04_results.json, results/figures/task04_*.png
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402


# ======================================================================================
# A. the failure
# ======================================================================================
def section_A() -> dict:
    print("\n=== A. Reproduced failure: naive softmax =================================")
    z = np.array([1000.0, 999.0, 995.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        naive = cm.softmax_naive(z)
        warn_msg = [str(x.message) for x in w]
    print(f"  z                       = {z}")
    print(f"  np.exp(z)               = {np.exp(z)}            <- overflow at exp(709.78)")
    print(f"  naive softmax(z)        = {naive}  <- inf/inf = nan, probabilities are destroyed")
    print(f"  warnings raised         = {warn_msg}")

    # underflow case: everything is too small
    z_neg = np.array([-1000.0, -1005.0, -1100.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        naive_neg = cm.softmax_naive(z_neg)
        warn_neg = [str(x.message) for x in w]
    print(f"\n  z                       = {z_neg}")
    print(f"  np.exp(z)               = {np.exp(z_neg)}")
    print(f"  naive softmax(z)        = {naive_neg}  <- 0/0 = nan (underflow)")
    print(f"  warnings raised         = {warn_neg}")

    # where does it start to break? sweep the largest logit
    maxima = np.array([0, 100, 300, 500, 700, 709, 709.78, 710, 800, 1000, 10000])
    sweep = []
    for m in maxima:
        zz = np.array([m, m - 1, m - 5])
        with np.errstate(over="ignore", invalid="ignore"):
            p = cm.softmax_naive(zz)
        ok = np.isfinite(p).all()
        sweep.append({"max_z": float(m), "exp_overflow": bool(m > 709.782),
                      "naive_softmax_ok": bool(ok), "result": p.tolist()})
        print(f"    max_z={m:>9}  exp overflows={bool(m > 709.782)!s:<5} "
              f"naive softmax finite={ok}")

    # the failure propagates into the gradient of cross-entropy
    Y = cm.onehot([0, 1, 2], 3)
    with np.errstate(invalid="ignore"):
        P = cm.softmax_naive(np.array([[1000.0, 999.0, 995.0]])) * np.ones((3, 1))
    grad = P - Y
    print(f"\n  gradient of the CE loss dL/dz = softmax(z) - y = {np.nan_to_num(grad[0])}"
          f"  -> contains {np.sum(~np.isfinite(grad))} non-finite entries")
    print("  one nan in the gradient is written into the weights, and every later")
    print("  prediction is nan as well: the failure is permanent once it occurs.")

    # naive log-softmax underflow (used when computing the loss itself)
    z2 = np.array([-800.0, -900.0, -1000.0])
    naive_log = -np.log(np.exp(z2).sum()) + z2        # naive log p
    with np.errstate(divide="ignore"):
        stable_log = z2 - cm.logsumexp(z2)
    print(f"\n  naive log p on z={z2}: {naive_log}   (exp underflows -> log 0 = -inf)")
    print(f"  stable  log p          : {stable_log}")
    return {"z_overflow": z.tolist(), "naive_softmax": np.asarray(naive).tolist(),
            "z_underflow": z_neg.tolist(), "naive_softmax_underflow": np.asarray(naive_neg).tolist(),
            "sweep": sweep,
            "failure_trace": "exp(z) overflows for z > 709.78 (float64) -> inf; inf/inf = nan "
                             "in softmax; nan propagates into dL/dz = p - y and then into W",
            "naive_log_softmax": np.asarray(naive_log).tolist(),
            "stable_log_softmax": np.asarray(stable_log).tolist()}


# ======================================================================================
# B. the correction
# ======================================================================================
def section_B() -> dict:
    print("\n=== B. Why subtracting max(z) works =====================================")
    z = np.array([1000.0, 999.0, 995.0])
    m = z.max()
    zs = z - m                                   # <= 0 for every component
    p_stable = cm.softmax_stable(z)
    p_shifted = np.exp(zs) / np.exp(zs).sum()
    print(f"  z - max(z) = {zs}   (all <= 0  ->  exp() <= 1  ->  cannot overflow)")
    print(f"  stable softmax      = {p_stable}")
    print(f"  shifted-by-hand     = {p_shifted}")
    print(f"  max |difference|    = {np.abs(p_stable - p_shifted).max():.3e}")

    # mathematical invariance: softmax(z) = exp(z_i)/sum exp(z_j)
    #                        = exp(z_i - c)/sum exp(z_j - c)   for any constant c
    err_invar = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        zz = rng.normal(0, 500, size=5)
        a = cm.softmax_stable(zz)
        b = cm.softmax_stable(zz - 123.456)
        err_invar.append(np.abs(a - b).max())
    print(f"  invariance softmax(z) == softmax(z - c) verified on 200 random vectors, "
          f"max deviation {max(err_invar):.3e}")

    # exact value with high-precision arithmetic (decimal, no float64 overflow)
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    ez = [Decimal(float(v)).exp() for v in z]
    tot = sum(ez)
    exact = [float(e / tot) for e in ez]
    print(f"  exact (50-digit decimal)  = {exact}")
    print(f"  stable float64            = {p_stable.tolist()}")
    print(f"  max abs error vs exact    = {max(abs(a - b) for a, b in zip(exact, p_stable)):.3e}")

    # log-sum-exp trick for the loss
    lz = np.array([1000.0, 999.0, 995.0])
    naive_lse = np.log(np.exp(lz).sum())
    stable_lse = cm.logsumexp(lz)
    print(f"\n  log-sum-exp of {lz}")
    print(f"    naive  log(sum(exp(z))) = {naive_lse}   (inf: exp overflowed first)")
    print(f"    stable m + log sum exp(z-m) = {stable_lse}  (finite)")
    # and for very negative values
    lz2 = np.array([-800.0, -900.0, -1000.0])
    with np.errstate(under="ignore"):
        naive2 = np.log(np.exp(lz2).sum())
    stable2 = cm.logsumexp(lz2)
    print(f"    on {lz2}: naive = {naive2}, stable = {stable2}")
    print("    m + log sum exp(z-m) keeps every exponent <= 0, so exp can only underflow")
    print("    on terms that are already negligible; the largest term is always exp(0)=1,")
    print("    so the sum is >= 1 and the log is finite and accurate.")
    return {"stable_softmax": p_stable.tolist(), "exact": exact,
            "invariance_max_deviation": float(max(err_invar)),
            "logsumexp_naive": float(naive_lse), "logsumexp_stable": float(stable_lse),
            "logsumexp_negative_naive": float(naive2), "logsumexp_negative_stable": float(stable2),
            "derivation": "exp(z_i - m)/sum_j exp(z_j - m) = exp(z_i)/exp(m)*sum-normaliser = "
                          "exp(z_i)/sum_j exp(z_j)  (the factor exp(-m) cancels numerator and "
                          "denominator), so probabilities are identical while all exponents "
                          "are <= 0."}


# ======================================================================================
# C. conditioning
# ======================================================================================
def _gd_curve(X, y, C, lr, iters=60):
    """Plain batch gradient descent on softmax regression - returns the loss curve."""
    n, d = X.shape
    W = np.zeros((d, C))
    Y = cm.onehot(y, C)
    hist = []
    for _ in range(iters):
        P = cm.softmax_stable(X @ W)
        loss = -np.mean(np.log(np.clip(P[np.arange(n), y], 1e-12, 1)))
        g = X.T @ (P - Y) / n
        W -= lr * g
        hist.append(float(loss))
    return hist


def _gd_train(X, y, C, lr, iters=60):
    """Batch GD on softmax regression - returns the fitted weight matrix."""
    n, d = X.shape
    W = np.zeros((d, C))
    Y = cm.onehot(y, C)
    for _ in range(iters):
        P = cm.softmax_stable(X @ W)
        W -= lr * (X.T @ (P - Y) / n)
    return W


def section_C(Xtr, ytr, Xte, yte, classes) -> dict:
    print("\n=== C. Condition number of the feature covariance ========================")
    C = len(classes)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xs = (Xtr - mu) / sd                      # standardised
    Zs = (Xte - mu) / sd
    # PCA whitening: eigenbasis with unit variance
    cov = np.cov(Xtr, rowvar=False)
    evals, evecs = np.linalg.eigh(np.clip(cov, 0, None))
    evals = np.clip(evals[::-1], 1e-12, None)
    evecs = evecs[:, ::-1]
    keep = 200
    Xw = ((Xtr - Xtr.mean(0)) @ evecs[:, :keep]) / np.sqrt(evals[:keep])
    Zw = ((Xte - Xtr.mean(0)) @ evecs[:, :keep]) / np.sqrt(evals[:keep])

    k_raw = np.linalg.cond(np.cov(Xtr, rowvar=False))
    k_std = np.linalg.cond(np.cov(Xs, rowvar=False))
    k_wht = np.linalg.cond(np.cov(Xw, rowvar=False))
    print(f"  cond(covariance) raw 4096-d     : {k_raw:.3e}")
    print(f"  cond(covariance) standardised   : {k_std:.3e}")
    print(f"  cond(covariance) PCA-whitened   : {k_wht:.3e}   (-> ~1 by construction)")
    print(f"  sqrt(kappa) = ratio of extreme learning rates: "
          f"raw {np.sqrt(k_raw):.1f}x, standardised {np.sqrt(k_std):.1f}x")

    # measured: same batch GD, same number of iterations, three feature scalings
    lrs = [0.1, 1.0, 10.0]
    curves = {}
    for tag, A, B in (("raw [0,1]", Xtr, Xte), ("standardised", Xs, Zs),
                      ("PCA-whitened (k=200)", Xw, Zw)):
        curves[tag] = {}
        for lr in lrs:
            h = _gd_curve(A, ytr, C, lr=lr, iters=60)
            curves[tag][lr] = h
            if not np.isfinite(h[-1]) or h[-1] > 2.0 * h[0]:
                status = "diverged (loss more than doubled)"
            elif h[-1] > h[0]:
                status = "oscillating (final above start)"
            elif abs(h[0] - h[-1]) < 1e-3:
                status = "flat"
            else:
                status = "converging"
            print(f"    {tag:<22} lr={lr:<5} loss {h[0]:.4f} -> {h[-1]:.4f}  [{status}]")
    print(f"  stable step size must satisfy  lr < 2/lambda_max(Hessian)  with")
    print(f"  lambda_max ~ ||x||^2 p(1-p)/n ; slowest mode needs lr ~ 1/lambda_min, so the")
    print(f"  iteration count scales with cond = lambda_max/lambda_min ~ {k_raw:.1e} on raw "
          f"pixels.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for i, (tag, d) in enumerate(curves.items()):
        for lr, h in d.items():
            ax[i].plot(h, label=f"lr={lr}")
        ax[i].set_title(f"{tag}\ncond={k_raw if i==0 else (k_std if i==1 else k_wht):.1e}",
                        fontsize=9)
        ax[i].set_xlabel("GD iteration"); ax[i].legend(fontsize=8)
        ax[i].set_yscale("log"); ax[i].set_ylim(0.1, 30)
    ax[0].set_ylabel("cross-entropy loss")
    fig.suptitle("Task 4: conditioning sets the usable learning rate and the convergence speed")
    cm.save_fig(fig, "task04_conditioning")

    # test accuracy of the three representations after a fixed GD budget (lr tuned per case)
    accs = {}
    for tag, A, B in (("raw [0,1]", Xtr, Xte), ("standardised", Xs, Zs),
                      ("PCA-whitened (k=200)", Xw, Zw)):
        best = 0.0
        for lr in (0.01, 0.1, 1.0):
            W = _gd_train(A, ytr, C, lr=lr, iters=60)
            acc = float(np.mean(cm.softmax_stable(B @ W).argmax(1) == yte))
            best = max(best, acc)
        accs[tag] = round(best, 4)
        print(f"  {tag:<22} best test accuracy after 60 GD steps: {best:.4f}")
    return {"condition_number": {"raw": float(k_raw), "standardised": float(k_std),
                                 "whitened": float(k_wht)},
            "gd_curves": {t: {str(lr): [float(x) for x in h] for lr, h in d.items()}
                          for t, d in curves.items()},
            "stable_lr_bound": "lr < 2/lambda_max, iterations ~ O(kappa log(1/eps))",
            "gd_test_accuracy_after_60_steps": accs}


# ======================================================================================
def main():
    print("#" * 78)
    print("# TASK 4 - numerical failures, corrections and conditioning")
    print("#" * 78)
    res = {"A_failure": section_A(), "B_correction": section_B()}

    X64, y, classes = cm.load_tomato()
    idx = cm.stratified_subset(y, 1000)            # small scale: conditioning demo only
    tr, _, te = cm.stratified_split(y[idx])
    X = cm.flatten01(np.asarray(X64[idx]))
    res["C_conditioning"] = section_C(X[tr], y[idx][tr], X[te], y[idx][te], classes)
    cm.save_json("task04_results", res)
    print("\nTASK 4 complete -> results/task04_results.json")


if __name__ == "__main__":
    main()
