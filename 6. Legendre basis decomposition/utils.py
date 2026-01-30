import pandas as pd
import numpy as np
from numpy.polynomial.legendre import legvander

def subspace_distance(U, V, metric="projection"):
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    sin2 = 1.0 - s**2

    if metric == "chordal":
        return float(np.sqrt(np.sum(sin2)))
    elif metric == "projection":
        return float(np.sqrt(2.0 * np.sum(sin2)))
    else:
        raise ValueError("metric must be 'projection' or 'chordal'")

def orthonormal_legendre_basis(x_norm, K):
    Phi = legvander(np.asarray(x_norm, float), K - 1)  # (p, K)
    Q, _ = np.linalg.qr(Phi)                           # (p, K)
    return Q

def rolling_pca_vs_legendre_distance(
    Ylog: pd.DataFrame,
    x_norm,
    T=720,
    step=24,
    Ks=(1, 2, 3, 4, 5),
    metric="projection",
    time_index="end",   # "end" matches rolling-std indexing; or "start"
):
    times = pd.to_datetime(Ylog.index, utc=True)
    Xall = Ylog.to_numpy(dtype=float)
    n, p = Xall.shape
    Ks = list(Ks)

    Q_leg = {K: orthonormal_legendre_basis(x_norm, K) for K in Ks}

    t_list = []
    dist = {K: [] for K in Ks}

    for s in range(0, n - T + 1, step):
        X = Xall[s:s+T, :]
        X = X - X.mean(axis=0, keepdims=True)   # window-centering (no look-ahead)

        # PCA via SVD; tick-space PCs are rows of Vt
        _, _, Vt = np.linalg.svd(X, full_matrices=False)

        # choose time stamp for this window
        if time_index == "end":
            t_list.append(times[s + T - 1])
        elif time_index == "start":
            t_list.append(times[s])
        else:
            raise ValueError("time_index must be 'start' or 'end'")

        # compute distance for each K
        for K in Ks:
            U_pca = Vt[:K, :].T      # (p, K), orthonormal columns
            dK = subspace_distance(U_pca, Q_leg[K], metric=metric)
            dist[K].append(dK)

    out = pd.DataFrame({K: dist[K] for K in Ks}, index=pd.DatetimeIndex(t_list, name="window_time"))
    return out
def legendre_scores(Ylog, x_norm, deg=5, center_time=True, ridge=1e-10):
    mL, B_leg, Phi_leg, Yhat_leg, R_leg = orthopoly_decompose(
        Ylog, x_norm, deg=deg, kind="legendre", center_time=center_time, ridge=ridge
    )
    return dict(
        legendre=dict(m=mL, scores=B_leg, Phi=Phi_leg, Yhat=Yhat_leg, resid=R_leg),
    )

def orthopoly_decompose(
    Ylog,                 # array-like (n_t, n_ticks) OR pd.DataFrame with tick columns
    x_norm,               # array-like (n_ticks,), must be in [-1, 1]
    deg=5,                # highest degree; number of factors = deg+1
    kind="legendre",      # "legendre" or "chebyshev"
    center_time=True,     # subtract m(x) across time (like PCA centering)
    ridge=1e-10,          # small ridge for numerical stability
    index=None, 
    alpha = 0.5,          # optional time index if Ylog is ndarray
):
    if isinstance(Ylog, pd.DataFrame):
        Y = Ylog.to_numpy(dtype=float)
        if index is None:
            index = Ylog.index
    else:
        Y = np.asarray(Ylog, dtype=float)
        if index is None:
            index = pd.RangeIndex(Y.shape[0])

    x = np.asarray(x_norm, dtype=float)
    if np.any(x < -1 - 1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError("x_norm must lie in [-1,1].")

    n_t, n_ticks = Y.shape
    if x.shape[0] != n_ticks:
        raise ValueError("x_norm length must match n_ticks (number of columns in Ylog).")

    # --- mean curve and centering (matches the paper's 'centered data' assumption)
    if center_time:
        m = Y.mean(axis=0)
        X = Y - m
    else:
        m = np.zeros(n_ticks)
        X = Y

    # --- build Vandermonde / design matrix Phi(x)
    if kind.lower().startswith("leg"):
        Phi = legvander(x, deg)    # (n_ticks, deg+1)
        basis_name = "Legendre"
    else:
        raise ValueError("kind must be 'legendre'.")

    # --- solve for coefficients B in least squares sense for all t:
    # X ≈ B Phi^T  ->  B = X Phi (Phi^T Phi)^{-1}
    G = Phi.T @ Phi
    G = G + ridge * np.eye(G.shape[0])
    Ginv = np.linalg.inv(G)

    B = (X @ Phi) @ Ginv               # (n_t, deg+1)
    Yhat = (B @ Phi.T) + m             # (n_t, n_ticks)
    R = Y - Yhat

    # package coefficients as DataFrame for convenient "score through time"
    cols = [f"{basis_name}_deg{j}" for j in range(deg + 1)]
    B_df = pd.DataFrame(B, index=index, columns=cols)

    return m, B_df, Phi, Yhat, R
if __name__ == '__main__':
    pass