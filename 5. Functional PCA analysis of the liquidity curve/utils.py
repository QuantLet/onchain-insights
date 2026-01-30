import pandas as pd
import matplotlib.dates as mdates
import numpy as np

def plot_liquidity_surface(df, TT, TT_end, centered=False, ax=None):
    d = df[df.hour > TT]
    d = d[d.hour <= TT_end]
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, unit = 's')
    if centered:
        WINDOW = 50
        d["tick_norm"] = (d["tickLower"].astype(float) - 0.0) / WINDOW
        d["tick_norm"] = d["tick_norm"].clip(-1, 1)
    else:
        WINDOW = 50
        d["tick_norm"] = (d["tickLower"].astype(float) - d['poolTick'].astype(float)) / WINDOW
        d["tick_norm"] = d["tick_norm"].clip(-1, 1)

    # Log liquidity (use log1p to handle zeros)
    d["logL"] = np.log1p(d["active_liquidity_L"].astype(float))

    # Time x tick grid
    surf = (
        d.pivot_table(index="timestamp", columns="tick_norm", values="logL", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )

    t = mdates.date2num(surf.index.to_pydatetime())
    x = surf.columns.to_numpy(dtype=float)
    T, X = np.meshgrid(t, x, indexing="ij")
    Z = surf.to_numpy(dtype=float)

    # fig = plt.figure(figsize=(12, 7))
    # ax = fig.add_subplot(111, projection="3d")

    s = ax.plot_surface(T, X, Z, cmap="coolwarm", linewidth=0, antialiased=True)
    if centered:
        centertxt = "peg tick"
    else:
        centertxt = "current price tick"

    ax.set_ylabel(f"Normalised tick (±50 from {centertxt})")
    ax.set_xlabel("Time")
    ax.set_zlabel("log(liquidity)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=mdates.UTC))
    # fig.colorbar(s, ax=ax, shrink=0.6, pad=0.1, label="log(L)")
    ax.view_init(azim = 295)
    return d

def pca_factor_decomposition(Y, K=5, center=True):
    Y = np.asarray(Y, dtype=float)
    n_t, n_ticks = Y.shape
    if K > min(n_t, n_ticks):
        raise ValueError("K must be <= min(n_t, n_ticks).")

    
    if center:
        m = Y.mean(axis=0)
        X = Y - m
    else:
        m = np.zeros(n_ticks)
        X = Y

    
    U_time, S, Vt = np.linalg.svd(X, full_matrices=False)

    
    U = Vt[:K, :]                    # (K, n_ticks), rows are u_k(x)

   
    B = X @ U.T                      # (n_t, K) since U has orthonormal rows
   
    X_hat = B @ U                    # (n_t, n_ticks)
    Y_hat = X_hat + m
    R = Y - Y_hat

    eigvals_all = (S**2) / (n_t - 1)
    eigvals = eigvals_all[:K]
    cpve = eigvals.sum() / eigvals_all.sum()

    return m, U, B, Y_hat, R, eigvals, cpve

def rolling_pca_eigs_and_cpve(Ylog: pd.DataFrame, T=400, step=1, Ks=(3,4,5,6,7)):
    times = Ylog.index
    Xall = Ylog.to_numpy(dtype=float)
    n, p = Xall.shape

    starts, ends, eigs_list, cpve_dict, vecs_list = [], [], [], {K: [] for K in Ks}, [] 

    for s in range(0, n - T + 1, step):
        X = Xall[s:s+T, :]                      # (T, p)
        X = X - X.mean(axis=0, keepdims=True)   # center columns (ticks)

        # SVD: eigenvalues of covariance = S^2/(T-1), already sorted descending
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        eigs = (S**2) / (T - 1)
        vecs = Vt[:p, :]                    # (n_vec, p)  PCs in tick space

        starts.append(times[s])
        ends.append(times[s + T - 1])
        eigs_list.append(eigs)
        vecs_list.append(vecs)

        c = np.cumsum(eigs) / np.sum(eigs)
        for K in Ks:
            cpve_dict[K].append(c[min(K-1, len(c)-1)])

    starts = pd.to_datetime(starts, utc=True)
    ends   = pd.to_datetime(ends, utc=True)
    eigs_arr = np.vstack(eigs_list)  # (n_windows, p)
    vecs_arr = np.stack(vecs_list, axis=0)  # (n_windows, n_vec, p)
    cpve = {K: np.array(v) for K, v in cpve_dict.items()}
    return starts, ends, eigs_arr, vecs_arr, cpve


if __name__ == '__main__':
    pass