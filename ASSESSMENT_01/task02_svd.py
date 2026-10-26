"""
TASK 2 - Examine the structure of the feature space with linear algebra.    (5 marks)

Run:  python task02_feature_space_svd/task02_svd.py

Sections
  A. Hand-worked eigen-decomposition of a REAL 2x2 covariance matrix (closed form
     vs numpy, verified by reconstruction A = Q L Q^T) plus a 3x3 SVD check.
  B. Full spectrum of the 4096 x 4096 pixel covariance matrix: eigenvalue decay,
     redundancy between neighbouring pixels, effective dimensionality.
  C. Relation between eigenvalues, singular values and explained variance
     (lambda_i = sigma_i^2 / (n-1)), verified numerically on the data matrix.
  D. Reconstruction error vs number of components k -> the compression trade-off,
     and measured classification accuracy vs k.
  E. What the leading principal components look like when reshaped to 64x64.
  F. Cost of the decomposition as the data set grows (measured wall-clock).

Outputs: results/task02_results.json, results/figures/task02_*.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as cm  # noqa: E402


# ======================================================================================
# A. worked examples
# ======================================================================================
def section_A(X64, y) -> dict:
    print("\n=== A. Hand-worked decomposition ========================================")
    # take two real pixel locations and build the 2x2 covariance of the centred data
    X = cm.flatten01(np.asarray(X64)).astype(np.float64)
    i, j = 64 * 32 + 32, 64 * 48 + 48        # centre of the leaf, and a darker corner
    Z = X[:, [i, j]]
    Z = Z - Z.mean(0)
    n = len(Z)
    a = float(Z[:, 0] @ Z[:, 0] / (n - 1))   # var(x_i)
    b = float(Z[:, 0] @ Z[:, 1] / (n - 1))   # cov(x_i, x_j)
    d = float(Z[:, 1] @ Z[:, 1] / (n - 1))   # var(x_j)
    C = np.array([[a, b], [b, d]])

    tr, det = a + d, a * d - b * b
    disc = np.sqrt((a - d) ** 2 + 4 * b ** 2)
    lam1 = (tr + disc) / 2
    lam2 = (tr - disc) / 2
    # eigenvector for lam1 (closed form)
    v1 = np.array([b, lam1 - a]) if abs(b) > 1e-15 else np.array([1.0, 0.0])
    v1 /= np.linalg.norm(v1)
    v2 = np.array([-v1[1], v1[0]])
    Q = np.column_stack([v1, v2])
    L = np.diag([lam1, lam2])
    recon = Q @ L @ Q.T
    err = np.abs(recon - C).max()

    print(f"  pixels: #{i} (centre) and #{j} (edge)")
    print(f"  C = [[{a:.6f}, {b:.6f}],\n      [{b:.6f}, {d:.6f}]]")
    print(f"  closed form:  lambda = ((a+d) +/- sqrt((a-d)^2+4b^2))/2")
    print(f"               sqrt term = {disc:.6f}")
    print(f"  hand-computed eigenvalues : lambda1={lam1:.6f}, lambda2={lam2:.6f}")
    print(f"  numpy     eigenvalues     : {np.linalg.eigvalsh(C)[::-1]}")
    print(f"  trace check  lam1+lam2 = {lam1 + lam2:.6f}  vs  a+d = {tr:.6f}")
    print(f"  det  check   lam1*lam2 = {lam1 * lam2:.6f}  vs  ad-b^2 = {det:.6f}")
    print(f"  reconstruction max |Q L Q^T - C| = {err:.2e}")

    # 3x3 SVD worked on a tiny block (4 images x 3 pixels) so the factors are visible
    A = np.asarray(X[:6, [i, j, 64 * 16 + 16]], dtype=np.float64)
    A = A - A.mean(0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print("\n  3x3 sub-problem: A = U diag(s) V^T  on 6 images x 3 pixels")
    print(f"    singular values s = {np.round(s, 6)}")
    print(f"    s^2/(n-1)         = {np.round(s ** 2 / (len(A) - 1), 6)}  (= eigenvalues of "
          f"the 3x3 covariance)")
    print(f"    eigvalsh(cov)     = {np.round(np.linalg.eigvalsh(A.T @ A / (len(A) - 1))[::-1], 6)}")
    print(f"    reconstruction max error = {np.abs(U[:, :3] @ np.diag(s) @ Vt - A).max():.2e}")

    return {"covariance_2x2": {"a_var": a, "b_cov": b, "d_var": d,
                               "hand_eigenvalues": [lam1, lam2],
                               "numpy_eigenvalues": np.linalg.eigvalsh(C).tolist()[::-1],
                               "reconstruction_error": float(err),
                               "eigenvector_1": v1.tolist()},
            "svd_3x3": {"singular_values": s.tolist(),
                        "sigma2_over_n_minus_1": (s ** 2 / (len(A) - 1)).tolist(),
                        "reconstruction_error":
                            float(np.abs(U[:, :3] @ np.diag(s) @ Vt - A).max())}}


# ======================================================================================
# B/C/D/E/F - full spectrum
# ======================================================================================
def section_spectrum(Xtr: np.ndarray, Xte: np.ndarray, ytr, yte, n_total: int) -> dict:
    print("\n=== B/C. Spectrum of the 4096-d pixel representation =====================")
    mu = Xtr.mean(0)
    Z = Xtr - mu
    cov = (Z.T @ Z) / (len(Z) - 1)               # 4096 x 4096

    t0 = time.perf_counter()
    evals, evecs = np.linalg.eigh(cov)           # ascending
    t_eig = time.perf_counter() - t0
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0, None)

    var = evals / evals.sum()
    cum = np.cumsum(var)
    pr = float(evals.sum() ** 2 / (evals ** 2).sum())     # participation ratio
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    k999 = int(np.searchsorted(cum, 0.999) + 1)

    # redundancy: correlation of neighbouring vs random pixel pairs
    rng = np.random.default_rng(0)
    samp = rng.choice(len(Z), size=min(2000, len(Z)), replace=False)
    Zs = Z[samp]
    sd = Zs.std(0) + 1e-12
    def corr(p, q):
        return float(np.mean((Zs[:, p] - Zs[:, p].mean()) * (Zs[:, q] - Zs[:, q].mean())
                             / (sd[p] * sd[q])))
    nb, rnd = [], []
    for _ in range(400):
        p = int(rng.integers(0, 4096))
        q = p + (1 if p % 64 < 63 else -1) + (64 if p < 4032 else -64)   # right/down neighbour
        nb.append(corr(p, q))
        q2 = int(rng.integers(0, 4096))
        if q2 != p:
            rnd.append(corr(p, q2))
    print(f"  eigenvalue decay: top1 {var[0]:.4f}, top10 {cum[9]:.4f}, "
          f"k95={k95}, k99={k99}, k99.9={k999} (of 4096)")
    print(f"  effective dimension (participation ratio) = {pr:.1f}")
    print(f"  mean |corr| neighbouring pixels {np.mean(np.abs(nb)):.3f}  vs  "
          f"random pairs {np.mean(np.abs(rnd)):.3f}  -> redundancy")

    # C: verify sigma^2/(n-1) = lambda on a randomised SVD of the centred matrix
    from sklearn.utils.extmath import randomized_svd
    U, sv, _ = randomized_svd(Z, n_components=200, random_state=0, n_iter=5)
    lam_svd = sv ** 2 / (len(Z) - 1)
    rel = np.abs(lam_svd - evals[:200]) / (evals[:200] + 1e-15)
    print(f"  sigma_i^2/(n-1) vs lambda_i : max rel. difference over first 200 = "
          f"{rel.max():.2e}")

    # figure: scree + cumulative
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(np.arange(1, 401), var[:400], ".-", ms=3)
    ax[0].axhline(1.0 / 4096, ls="--", c="grey", label="uniform (1/4096)")
    ax[0].set_xlabel("component i"); ax[0].set_ylabel("explained variance ratio (log)")
    ax[0].set_title("Scree plot: pixel redundancy"); ax[0].legend()
    ax[1].plot(cum[:1000], lw=1.5)
    for k, c in ((k95, "r"), (k99, "g")):
        ax[1].axvline(k, ls="--", c=c, label=f"k={k} -> {int(cum[k-1]*100)}%")
        ax[1].axhline(cum[k - 1], ls=":", c=c)
    ax[1].set_xlabel("number of components k"); ax[1].set_ylabel("cumulative explained variance")
    ax[1].set_title("Cumulative information vs k"); ax[1].legend()
    cm.save_fig(fig, "task02_spectrum")

    # D: reconstruction error vs k
    # sum of discarded eigenvalues = per-image MSE (summed over the 4096 pixels);
    # divide by 4096 to report the per-pixel MSE whose RMSE lives on the 0-1 scale.
    print("\n=== D. Reconstruction error vs k ========================================")
    ks = [1, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 4096]
    rec = {k: float(evals[k:].sum() / 4096.0) for k in ks if k <= 4096}   # per-pixel MSE
    base = float(evals.sum())
    print("   k   | retained var | reconstruction MSE (per pixel) | RMSE on 0-1 scale")
    for k in ks:
        if k > 4096:
            continue
        mse = rec[k]
        print(f"  {k:>4} |    {cum[k-1]:6.4f}    |          {mse:.6f}          | {np.sqrt(mse):.4f}")

    # measured accuracy vs k (compression actually used by a model)
    print("   accuracy of a linear classifier as a function of k:")
    acc = {}
    for k in (10, 25, 50, 100, 200, 400, 800):
        Ztr = (Xtr - mu) @ evecs[:, :k]
        Zte = (Xte - mu) @ evecs[:, :k]
        m = LogisticRegression(max_iter=400, solver="lbfgs", random_state=0)
        m.fit(Ztr, ytr)
        acc[k] = round(float(accuracy_score(yte, m.predict(Zte))), 4)
        print(f"    k={k:<5} var={cum[k-1]*100:6.2f}%  test acc {acc[k]:.4f}")

    # E: leading PCs as images
    fig, ax = plt.subplots(2, 6, figsize=(13, 4.6))
    for t in range(12):
        v = evecs[:, t].reshape(64, 64)
        v = (v - v.min()) / (v.max() - v.min() + 1e-12)
        ax[t // 6, t % 6].imshow(v, cmap="viridis")
        ax[t // 6, t % 6].set_title(f"PC{t+1}  λ={var[t]*100:.1f}%", fontsize=8)
        ax[t // 6, t % 6].axis("off")
    fig.suptitle("Leading principal components (mean image removed)")
    cm.save_fig(fig, "task02_components")

    return {"eigenvalue_decay": {"var_explained_top1": float(var[0]),
                                 "top10": float(cum[9]), "k_for_95pct": k95,
                                 "k_for_99pct": k99, "k_for_99_9pct": k999,
                                 "participation_ratio": pr},
            "redundancy": {"mean_abs_corr_neighbours": float(np.mean(np.abs(nb))),
                           "mean_abs_corr_random_pairs": float(np.mean(np.abs(rnd)))},
            "svd_vs_eigen_max_rel_diff_first200": float(rel.max()),
            "reconstruction_mse_per_pixel": {str(k): rec[k] for k in ks if k <= 4096},
            "cumulative_variance_at_k": {str(k): float(cum[k - 1]) for k in ks if k <= 4096},
            "accuracy_vs_k": {str(k): v for k, v in acc.items()},
            "eigh_seconds": t_eig}


def section_cost(scales) -> dict:
    print("\n=== F. Cost of the decomposition vs data size ============================")
    rows = []
    X64, y, _ = cm.load_tomato()
    for name, n in scales:
        idx = cm.stratified_subset(y, n)
        tr, _, _ = cm.stratified_split(y[idx])
        X = cm.flatten01(np.asarray(X64[idx]))[tr]
        mu = X.mean(0)
        Z = X - mu
        t0 = time.perf_counter()
        cov = (Z.T @ Z) / (len(Z) - 1)
        t_cov = time.perf_counter() - t0
        t0 = time.perf_counter()
        np.linalg.eigh(cov)
        t_eig = time.perf_counter() - t0
        t0 = time.perf_counter()
        from sklearn.decomposition import PCA
        PCA(n_components=100, svd_solver="randomized", random_state=0).fit(X)
        t_pca = time.perf_counter() - t0
        row = {"n_train": int(len(X)), "covariance_s": round(t_cov, 2),
               "eigh_4096_s": round(t_eig, 2), "randomized_pca100_s": round(t_pca, 2)}
        rows.append(row)
        print(f"  n={len(X):>6}  cov(O(n d^2)) {t_cov:6.2f}s   eigh(O(d^3)) {t_eig:6.2f}s   "
              f"randomized-PCA-100 {t_pca:6.2f}s")
    theory = {"covariance": "O(n d^2) = n*4096^2 flops -> grows linearly in n",
              "eigendecomposition": "O(d^3) = 4096^3 flops -> constant in n, ~6.9e10 flops",
              "randomized_svd": "O(n d k) for k components -> the practical choice",
              "memory": "covariance is d^2 = 16.8M doubles = 134 MB; data matrix is n*d*4"}
    for k, v in theory.items():
        print(f"  {k:<22}: {v}")
    return {"measurements": rows, "theory": theory}


# ======================================================================================
def main():
    print("#" * 78)
    print("# TASK 2 - feature-space structure via eigendecomposition / SVD")
    print("#" * 78)
    X64, y, classes = cm.load_tomato()
    res = {"A_worked_example": section_A(X64, y)}

    idx = cm.stratified_subset(y, 10000)            # scale 10,000 for the main analysis
    tr, va, te = cm.stratified_split(y[idx])
    X = cm.flatten01(np.asarray(X64[idx]))
    with cm.timer("PCA machinery"):
        spec = section_spectrum(X[tr], X[te], y[idx][tr], y[idx][te], len(idx))
    res["B_F_spectrum"] = spec
    res["F_cost"] = section_cost([(s, n) for s, n in cm.make_scales(len(y))])
    cm.save_json("task02_results", res)
    print("\nTASK 2 complete -> results/task02_results.json")


if __name__ == "__main__":
    main()
