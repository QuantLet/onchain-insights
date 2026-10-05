# ============================================================
# Section 5.2.10:
# Interpretability of ANGEL probabilistic model
# ============================================================
#
# ANGEL diagnostic semantics:
#
#   selection_weights:
#       Main interpretable variable-selection diagnostic.
#       Shape usually (B,V), sometimes (B,H,V).
#       Sparsemax/entmax gives exact zeros for excluded variables.
#
#   expected_open:
#       Compatibility alias for selection_weights.
#
#   effective_selection_used:
#       Compatibility alias for selection_weights.
#
#   hard_gates:
#       Compatibility stub. In ANGEL this is all ones and should NOT be
#       interpreted as hard L0 selection.
#
#   l0_penalty:
#       Actually entropy penalty, not hard-concrete L0 penalty.
#
#   selector_temperature:
#       Learned sparsemax/entmax temperature.
#
#   cross_attn_maps / cross_attn_mean_layers / selected_cross_attn:
#       Optional cross-attention diagnostics.
#
# ============================================================

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict

from benchmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    compute_abs_threshold_event_prob,
    _resolve_horizons,
    model_paths,
)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, *args, **kwargs):
        return x if x is not None else range(0)


# ============================================================
# 1. Feature-name and paper-specific group helpers
# ============================================================

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, list):
        vals = []
        for z in x:
            if z is not None:
                vals.append(np.asarray(z, dtype=np.float64))
        if len(vals) == 0:
            return None
        try:
            return np.stack(vals, axis=0)
        except Exception:
            return vals
    return np.asarray(x, dtype=np.float64)


def extract_selection_weights_for_audit(A):
    diag = A.get("diagnostics", {})
    for key in [
        "selection_weights",
        "effective_selection_used",
        "expected_open",
        "effective_selection_norm",
        "expected_effective_selection",
    ]:
        if key in diag:
            arr = _to_numpy(diag[key])
            if arr is not None and not isinstance(arr, list):
                return key, arr
    raise KeyError(f"No selection diagnostic found. Available keys: {list(diag.keys())}")


def extract_cross_attention_for_audit(A):
    diag = A.get("diagnostics", {})
    for key in [
        "cross_attn_maps",
        "cross_attn_mean_layers",
        "selected_cross_attn",
        "selected_cross_attn_norm",
    ]:
        if key in diag:
            arr = _to_numpy(diag[key])
            if arr is not None:
                return key, arr
    return None, None


def normalize_selection_to_BV(sel, A, horizons_ahead=24):
    """
    Normalize selection diagnostic to (B,V).
    """
    sel = np.asarray(sel, dtype=np.float64)
    B, H = np.asarray(A["true"]).shape
    h_idx, _ = _resolve_horizons(A, horizons_ahead)

    if sel.ndim == 2:
        # (B,V)
        return sel

    if sel.ndim == 3:
        # Could be (B,H,V) or (B,T,V)
        if sel.shape[0] == B and sel.shape[1] == H:
            return np.nanmean(sel[:, h_idx, :], axis=1)
        return np.nanmean(sel, axis=1)

    if sel.ndim == 4:
        # Could be (B,H,T,V)
        if sel.shape[0] == B and sel.shape[1] == H:
            return np.nanmean(sel[:, h_idx, :, :], axis=(1, 2))
        return np.nanmean(sel, axis=(1, 2))

    raise ValueError(f"Unsupported selection shape: {sel.shape}")


def normalize_attention_to_BHV(attn, A):
    """
    Normalize attention to (B,H,V), preserving horizon dimension.

    Supports common shapes:
      - (B,H,V)
      - (B,heads,H,V)
      - (layers,B,H,V)
      - (layers,B,heads,H,V)
      - (B,layers,heads,H,V)

    Returns:
      attn_BHV : np.ndarray or None
    """
    if attn is None:
        return None

    if isinstance(attn, list):
        vals = []
        for x in attn:
            if x is not None:
                vals.append(np.asarray(x, dtype=np.float64))
        if len(vals) == 0:
            return None
        attn = np.stack(vals, axis=0)

    X = np.asarray(attn, dtype=np.float64)
    B, H = np.asarray(A["true"]).shape

    # (B,H,V)
    if X.ndim == 3:
        if X.shape[0] == B and X.shape[1] == H:
            return X

        # (layers,B,V) -> no horizon; repeat impossible
        if X.shape[1] == B:
            return None

    # (B,heads,H,V) or (B,H,heads,V)
    if X.ndim == 4:
        if X.shape[0] == B:
            if X.shape[2] == H:
                # (B,heads,H,V)
                return np.nanmean(X, axis=1)
            if X.shape[1] == H:
                # (B,H,heads,V)
                return np.nanmean(X, axis=2)

        if X.shape[1] == B:
            # (layers,B,H,V)
            if X.shape[2] == H:
                return np.nanmean(X, axis=0)

    # (layers,B,heads,H,V) or (B,layers,heads,H,V)
    if X.ndim == 5:
        if X.shape[1] == B and X.shape[3] == H:
            # (layers,B,heads,H,V)
            return np.nanmean(X, axis=(0, 2))

        if X.shape[0] == B and X.shape[3] == H:
            # (B,layers,heads,H,V)
            return np.nanmean(X, axis=(1, 2))

    print(f"Could not normalize attention shape {X.shape} to (B,H,V).")
    return None


def effective_number_from_weights(W, eps=1e-12):
    """
    Effective number of active features:
      exp(entropy(normalized weights)).
    """
    W = np.asarray(W, dtype=np.float64)
    W = np.clip(W, 0.0, None)
    W = W / np.maximum(W.sum(axis=-1, keepdims=True), eps)
    ent = -np.sum(W * np.log(W + eps), axis=-1)
    return np.exp(ent)


def audit_angel_cross_attention(
    A,
    feature_names=None,
    horizons_ahead=24,
    active_threshold=1e-8,
    top_k=15,
):
    """
    Audit whether cross-attention is genuinely using one feature, or whether
    the apparent one-feature pattern comes from sparse selection / saved
    selected-cross-attention diagnostics.
    """
    print("\n====================================================")
    print("ANGEL cross-attention diagnostic audit")
    print("====================================================")

    sel_key, sel_raw = extract_selection_weights_for_audit(A)
    attn_key, attn_raw = extract_cross_attention_for_audit(A)

    print(f"Selection key used: {sel_key}, raw shape={getattr(sel_raw, 'shape', None)}")
    print(f"Attention key used:  {attn_key}, raw shape={getattr(attn_raw, 'shape', None)}")

    sel_BV = normalize_selection_to_BV(sel_raw, A, horizons_ahead=horizons_ahead)

    B, V = sel_BV.shape
    names = get_feature_names_from_A(A, n_features=V, feature_names=feature_names)

    # Selection sparsity
    sel_row_sum = np.nansum(sel_BV, axis=1)
    sel_nnz = np.sum(sel_BV > active_threshold, axis=1)
    sel_eff_n = effective_number_from_weights(sel_BV)

    print("\nSelection-weight diagnostics")
    print("----------------------------")
    print(f"Shape after normalization: {sel_BV.shape}")
    print(f"Mean row sum: {np.nanmean(sel_row_sum):.6f}")
    print(f"Median row sum: {np.nanmedian(sel_row_sum):.6f}")
    print(f"Mean nonzero selected vars: {np.nanmean(sel_nnz):.3f}")
    print(f"Median nonzero selected vars: {np.nanmedian(sel_nnz):.3f}")
    print(f"Mean effective number selected: {np.nanmean(sel_eff_n):.3f}")
    print(f"Median effective number selected: {np.nanmedian(sel_eff_n):.3f}")
    print(f"Share exactly/near one-hot: {np.mean(sel_nnz <= 1):.3%}")

    sel_mean = np.nanmean(sel_BV, axis=0)
    sel_freq = np.nanmean(sel_BV > active_threshold, axis=0)

    top_sel = np.argsort(-sel_mean)[:top_k]

    print("\nTop variables by average selection weight")
    print("-----------------------------------------")
    for rank, j in enumerate(top_sel, 1):
        print(
            f"{rank:2d}. {names[j]:40s} "
            f"mean_w={sel_mean[j]:.6f} "
            f"freq={sel_freq[j]:.3f}"
        )

    # Attention diagnostics
    attn_BHV = normalize_attention_to_BHV(attn_raw, A)

    if attn_BHV is None:
        print("\nNo usable raw cross-attention array could be normalized to (B,H,V).")
        print("If you only saved selected_cross_attn_norm, the diagnostic may already be multiplied by selection weights.")
        return {
            "selection_BV": sel_BV,
            "attention_BHV": None,
        }

    B2, H, V2 = attn_BHV.shape
    B_common = min(B, B2)

    sel_BV = sel_BV[:B_common]
    attn_BHV = attn_BHV[:B_common]

    if V2 != V:
        print(f"\nWarning: selection V={V}, attention V={V2}. Feature alignment may be wrong.")
        V_common = min(V, V2)
        sel_BV = sel_BV[:, :V_common]
        attn_BHV = attn_BHV[:, :, :V_common]
        names = names[:V_common]
        V = V_common

    attn_row_sum = np.nansum(attn_BHV, axis=-1)
    attn_nnz = np.sum(attn_BHV > active_threshold, axis=-1)
    attn_eff_n = effective_number_from_weights(attn_BHV)

    print("\nRaw cross-attention diagnostics")
    print("-------------------------------")
    print(f"Shape after normalization: {attn_BHV.shape}")
    print(f"Mean attention row sum over V: {np.nanmean(attn_row_sum):.6f}")
    print(f"Median attention row sum over V: {np.nanmedian(attn_row_sum):.6f}")
    print(f"Mean nonzero attended vars per horizon: {np.nanmean(attn_nnz):.3f}")
    print(f"Median nonzero attended vars per horizon: {np.nanmedian(attn_nnz):.3f}")
    print(f"Mean effective attended vars: {np.nanmean(attn_eff_n):.3f}")
    print(f"Median effective attended vars: {np.nanmedian(attn_eff_n):.3f}")
    print(f"Share one-hot attention rows: {np.mean(attn_nnz <= 1):.3%}")

    # Effective attention = attention * selection
    eff_BHV = attn_BHV * sel_BV[:, None, :]
    eff_sum = np.nansum(eff_BHV, axis=-1, keepdims=True)
    eff_norm_BHV = eff_BHV / np.maximum(eff_sum, 1e-12)

    eff_nnz = np.sum(eff_BHV > active_threshold, axis=-1)
    eff_eff_n = effective_number_from_weights(eff_BHV)

    print("\nEffective selected cross-attention diagnostics")
    print("----------------------------------------------")
    print("Defined as raw_cross_attention * selection_weights.")
    print(f"Mean nonzero effective vars per horizon: {np.nanmean(eff_nnz):.3f}")
    print(f"Median nonzero effective vars per horizon: {np.nanmedian(eff_nnz):.3f}")
    print(f"Mean effective number after selection × attention: {np.nanmean(eff_eff_n):.3f}")
    print(f"Median effective number after selection × attention: {np.nanmedian(eff_eff_n):.3f}")
    print(f"Share one-feature effective rows: {np.mean(eff_nnz <= 1):.3%}")

    attn_mean = np.nanmean(attn_BHV, axis=(0, 1))
    eff_mean = np.nanmean(eff_norm_BHV, axis=(0, 1))

    top_attn = np.argsort(-attn_mean)[:top_k]
    top_eff = np.argsort(-eff_mean)[:top_k]

    print("\nTop variables by raw cross-attention")
    print("------------------------------------")
    for rank, j in enumerate(top_attn, 1):
        print(
            f"{rank:2d}. {names[j]:40s} "
            f"mean_attn={attn_mean[j]:.6f} "
            f"mean_sel={sel_mean[j]:.6f}"
        )

    print("\nTop variables by effective selection × attention")
    print("------------------------------------------------")
    for rank, j in enumerate(top_eff, 1):
        print(
            f"{rank:2d}. {names[j]:40s} "
            f"mean_eff_attn={eff_mean[j]:.6f} "
            f"mean_sel={sel_mean[j]:.6f} "
            f"mean_raw_attn={attn_mean[j]:.6f}"
        )

    return {
        "selection_key": sel_key,
        "attention_key": attn_key,
        "selection_BV": sel_BV,
        "attention_BHV": attn_BHV,
        "effective_attention_BHV": eff_BHV,
        "effective_attention_norm_BHV": eff_norm_BHV,
        "feature_names": names,
        "selection_nnz": sel_nnz,
        "attention_nnz": attn_nnz,
        "effective_attention_nnz": eff_nnz,
    }

def get_feature_names_from_A(A, n_features=None, feature_names=None):
    """
    Extract covariate names from saved prediction dictionary.

    ANGEL's covariate branch uses enc_in - 1 variables, because the last
    input channel is the target. Therefore names here should correspond only
    to covariates, not the target.
    """
    if feature_names is not None:
        names = list(feature_names)
    elif "covariate_names" in A:
        names = list(A["covariate_names"])
    elif "feature_names" in A:
        names = list(A["feature_names"])
    elif "var_names" in A:
        names = list(A["var_names"])
    else:
        if n_features is None:
            raise ValueError("n_features is required when feature names are unavailable.")
        names = [f"feature_{j:03d}" for j in range(n_features)]

    names = [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in names
    ]

    if n_features is not None and len(names) != n_features:
        print(
            f"Warning: number of feature names ({len(names)}) does not match "
            f"diagnostic feature dimension ({n_features}). Using generic names."
        )
        names = [f"feature_{j:03d}" for j in range(n_features)]

    return names


def build_paper_feature_groups(features):
    """
    Build paper-specific feature groups for Section 5.2.10.
    """
    features = [str(f) for f in features]
    feature_set = set(features)

    feature_groups = OrderedDict()

    feature_groups["Uniswap V3 liquidity-curve shape"] = [
        f for f in features
        if (
            f.startswith("Gegenbauer_")
            or f.startswith("E_")
            or ("ratio" in f)
            or f in [
                "tangent_up",
                "tangent_down",
                "swap_size_imbalance",
                "tvlUSD_100",
                "tvlUSD_500",
            ]
        )
    ]

    feature_groups["Curve 3pool liquidity conditions"] = [
        f for f in [
            "w_USDC",
            "w_USDT",
            "curve_entropy",
            "gauge_share_3crv",
            "totalValueLockedUSD",
        ]
        if f in feature_set
    ]

    feature_groups["Broader market conditions"] = [
        f for f in features
        if (
            f.startswith("eth_")
            or f.startswith("btc_")
            or f in [
                "usd_index",
                "fx_volatility",
                "fear_greed_index",
            ]
        )
    ]

    feature_groups["Historical peg deviation"] = [
        f for f in features
        if f == "depeg_bps" or f.startswith("depeg_bps_lag")
    ]

    feature_groups["Liquidity ownership and position structure"] = [
        f for f in [
            "hhi_24h_rolling_mean",
            "tick_width_24h_rolling_median",
            "n_in_range_log_return",
            "weighted_mean_age_hours",
        ]
        if f in feature_set
    ]

    feature_groups["Trading velocity and flows"] = [
        f for f in [
            "swap_count_100",
            "swap_count_500",
            "net_amountUSD_100",
            "net_amountUSD_500",
            "net_amount0",
            "hourlyVolumeUSD",
        ]
        if f in feature_set
    ]

    feature_groups["AAVE lending market conditions"] = [
        f for f in [
            "supplied_USD_usdt",
            "utilisation_rate_usdt",
            "supplied_USD_usdc",
            "utilisation_rate_usdc",
            "liquidation_USD",
        ]
        if f in feature_set
    ]

    return feature_groups


def build_feature_group_mapping(
    feature_names,
    custom_group_map=None,
    assign_unmatched_to="Other",
    verbose=True,
):
    feature_names = [str(f) for f in feature_names]
    custom_group_map = custom_group_map or {}

    paper_groups = build_paper_feature_groups(feature_names)

    feature_to_group = {
        f: assign_unmatched_to
        for f in feature_names
    }

    duplicate_assignments = []

    for group_name, group_features in paper_groups.items():
        for f in group_features:
            if f not in feature_to_group:
                continue

            if feature_to_group[f] != assign_unmatched_to:
                duplicate_assignments.append(
                    {
                        "Feature": f,
                        "Existing group": feature_to_group[f],
                        "New group ignored": group_name,
                    }
                )
                continue

            feature_to_group[f] = group_name

    for f, g in custom_group_map.items():
        f = str(f)
        if f in feature_to_group:
            feature_to_group[f] = str(g)

    group_map_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature group": [feature_to_group[f] for f in feature_names],
    })

    if verbose:
        print("\nSection 5.2.10 feature-group coverage:")
        print("======================================")
        coverage = (
            group_map_df
            .groupby("Feature group", as_index=False)
            .agg(N_features=("Feature", "count"))
            .sort_values("N_features", ascending=False)
        )
        print(coverage.to_string(index=False))

        n_other = int(np.sum(group_map_df["Feature group"] == assign_unmatched_to))
        if n_other > 0:
            print(f"\nWarning: {n_other} features assigned to '{assign_unmatched_to}'.")
            print(
                group_map_df.loc[
                    group_map_df["Feature group"] == assign_unmatched_to,
                    "Feature",
                ]
                .head(80)
                .to_string(index=False)
            )

        if len(duplicate_assignments) > 0:
            print("\nWarning: duplicate feature-group assignments detected.")
            print("The first group in OrderedDict order was kept.")
            print(pd.DataFrame(duplicate_assignments).to_string(index=False))

    return group_map_df


# ============================================================
# 2. Diagnostics extraction
# ============================================================

def print_diagnostic_keys(A):
    if "diagnostics" not in A:
        print("No A['diagnostics'] found.")
        return

    print("\nAvailable A['diagnostics'] keys:")
    print("================================")
    for k, v in A["diagnostics"].items():
        shape = getattr(v, "shape", None)
        if isinstance(v, list):
            shape = f"list[{len(v)}]"
            if len(v) > 0:
                shape += f", first shape={getattr(v[0], 'shape', None)}"
        print(f"{k:40s} shape={shape}")


def get_diag_array(A, key, required=False):
    diag = A.get("diagnostics", None)

    if diag is None:
        if required:
            raise KeyError(
                "A['diagnostics'] not found. "
                "Rerun testing with --save_test_diagnostics 1."
            )
        return None

    if key not in diag:
        if required:
            raise KeyError(
                f"A['diagnostics']['{key}'] not found. "
                f"Available keys: {list(diag.keys())}"
            )
        return None

    val = diag[key]

    # Lists are handled separately for attention maps.
    if isinstance(val, list):
        return val

    try:
        return np.asarray(val, dtype=np.float64)
    except Exception:
        return val


def choose_selection_weight_array(A):
    """
    ANGEL-specific selection diagnostic.

    Priority:
      1. selection_weights
      2. effective_selection_used
      3. expected_open
      4. effective_selection_norm
      5. expected_effective_selection

    For ANGEL:
      - selection_weights is the real interpretable object.
      - expected_open is only a compatibility alias.
      - hard_gates should be ignored.
    """
    priority = [
        "selection_weights",
        "effective_selection_used",
        "expected_open",
        "effective_selection_norm",
        "expected_effective_selection",
    ]

    for key in priority:
        arr = get_diag_array(A, key, required=False)
        if arr is not None and not isinstance(arr, list):
            return key, np.asarray(arr, dtype=np.float64)

    raise KeyError(
        "No suitable ANGEL selection-weight diagnostic found. Expected one of: "
        f"{priority}"
    )


def choose_effective_selection_array(A):
    """
    Optional effective/applied selection diagnostic.

    In ANGEL this is usually identical to selection_weights.
    """
    priority = [
        "effective_selection_used",
        "effective_selection_norm",
        "expected_effective_selection",
        "selection_weights",
    ]

    for key in priority:
        arr = get_diag_array(A, key, required=False)
        if arr is not None and not isinstance(arr, list):
            return key, np.asarray(arr, dtype=np.float64)

    return None, None


def choose_attention_array(A):
    """
    Try to find a cross-attention diagnostic.

    Possible keys depend on Baseclass_forecast saving code.
    """
    priority = [
        "selected_cross_attn_norm",
        "selected_cross_attn",
        "cross_attn_mean_layers",
        "cross_attn_maps",
    ]

    for key in priority:
        arr = get_diag_array(A, key, required=False)
        if arr is not None:
            return key, arr

    return None, None


def choose_hard_gate_array_if_informative(A, active_threshold=1e-6):
    """
    In ANGEL hard_gates is a compatibility stub of all ones.
    This function returns None if hard_gates carries no information.
    """
    hard = get_diag_array(A, "hard_gates", required=False)

    if hard is None or isinstance(hard, list):
        return None

    hard = np.asarray(hard, dtype=np.float64)

    finite = hard[np.isfinite(hard)]
    if finite.size == 0:
        return None

    # If all hard gates are essentially one, ignore them.
    if np.nanmin(finite) > 1.0 - 1e-8 and np.nanmax(finite) <= 1.0 + 1e-8:
        print(
            "Info: hard_gates appears to be all ones. "
            "Ignoring hard_gates because ANGEL does not use L0 hard gates."
        )
        return None

    # If no variation, also ignore.
    if np.nanstd(finite) < 1e-12:
        print("Info: hard_gates has no variation. Ignoring.")
        return None

    return hard


# ============================================================
# 3. Shape normalization helpers
# ============================================================

def normalize_feature_diag_to_BF(arr, A=None, horizons_ahead=24):
    """
    Normalize a variable-level diagnostic to shape (B,F).

    Supported:
      - (B,F)
      - (B,H,F)
      - (B,T,F)
      - (B,H,T,F)

    If a dimension matches forecast horizon H, select / average the requested
    horizons. Otherwise average over middle dimensions.
    """
    if arr is None:
        return None

    X = np.asarray(arr, dtype=np.float64)

    if X.ndim == 2:
        return X

    if A is not None:
        H = np.asarray(A["true"]).shape[1]
        h_idx, _ = _resolve_horizons(A, horizons_ahead)
    else:
        H = None
        h_idx = None

    if X.ndim == 3:
        B, D, F = X.shape

        if H is not None and D == H:
            return np.nanmean(X[:, h_idx, :], axis=1)

        return np.nanmean(X, axis=1)

    if X.ndim == 4:
        B, D1, D2, F = X.shape

        if H is not None and D1 == H:
            return np.nanmean(X[:, h_idx, :, :], axis=(1, 2))

        return np.nanmean(X, axis=(1, 2))

    raise ValueError(f"Unsupported feature diagnostic shape {X.shape}")


def normalize_attention_diag_to_BF(attn, A=None, horizons_ahead=24):
    """
    Normalize cross-attention diagnostic to shape (B,V).

    Supports common shapes:
      - list of layer tensors
      - (B,H,V)
      - (B,heads,H,V)
      - (layers,B,H,V)
      - (layers,B,heads,H,V)
      - (B,layers,heads,H,V)

    The last dimension is assumed to be variables V.
    """
    if attn is None:
        return None

    # Convert list of layer tensors into stacked array.
    if isinstance(attn, list):
        vals = []
        for x in attn:
            if x is None:
                continue
            vals.append(np.asarray(x, dtype=np.float64))
        if len(vals) == 0:
            return None
        try:
            X = np.stack(vals, axis=0)  # likely (L,B,...,V)
        except Exception:
            print("Warning: could not stack attention list. Skipping attention diagnostic.")
            return None
    else:
        X = np.asarray(attn, dtype=np.float64)

    if A is not None:
        B_true, H_true = np.asarray(A["true"]).shape
        h_idx, _ = _resolve_horizons(A, horizons_ahead)
    else:
        B_true, H_true = None, None
        h_idx = None

    # Already (B,V)
    if X.ndim == 2:
        return X

    # (B,H,V)
    if X.ndim == 3:
        if B_true is not None and X.shape[0] == B_true and X.shape[1] == H_true:
            return np.nanmean(X[:, h_idx, :], axis=1)

        # Maybe (layers,B,V)
        if B_true is not None and X.shape[1] == B_true:
            return np.nanmean(X, axis=0)

        # Otherwise average middle dimension.
        return np.nanmean(X, axis=1)

    # (B,heads,H,V) or (layers,B,H,V)
    if X.ndim == 4:
        if B_true is not None and X.shape[0] == B_true:
            # (B,heads,H,V)
            if X.shape[2] == H_true:
                return np.nanmean(X[:, :, h_idx, :], axis=(1, 2))
            # (B,H,heads,V)
            if X.shape[1] == H_true:
                return np.nanmean(X[:, h_idx, :, :], axis=(1, 2))
            return np.nanmean(X, axis=(1, 2))

        if B_true is not None and X.shape[1] == B_true:
            # (layers,B,H,V)
            if X.shape[2] == H_true:
                return np.nanmean(X[:, :, h_idx, :], axis=(0, 2))
            # (layers,B,heads,V)
            return np.nanmean(X, axis=(0, 2))

        # Fallback: assume last dim V, find B axis.
        return np.nanmean(X, axis=tuple(range(X.ndim - 1))[1:])

    # (layers,B,heads,H,V) or (B,layers,heads,H,V)
    if X.ndim == 5:
        if B_true is not None and X.shape[1] == B_true:
            # (layers,B,heads,H,V)
            if X.shape[3] == H_true:
                return np.nanmean(X[:, :, :, h_idx, :], axis=(0, 2, 3))
            return np.nanmean(X, axis=(0, 2, 3))

        if B_true is not None and X.shape[0] == B_true:
            # (B,layers,heads,H,V)
            if X.shape[3] == H_true:
                return np.nanmean(X[:, :, :, h_idx, :], axis=(1, 2, 3))
            return np.nanmean(X, axis=(1, 2, 3))

    print(f"Warning: unsupported attention diagnostic shape {X.shape}. Skipping.")
    return None


def normalize_feature_diag_to_BHF(arr, A=None):
    """
    Normalize a variable-level diagnostic to shape (B, H, F), keeping the
    horizon dimension intact.  Returns None when no horizon dimension can be
    identified (e.g. the raw array is already (B, F)).
    """
    if arr is None:
        return None

    X = np.asarray(arr, dtype=np.float64)

    if X.ndim == 2:
        return None  # (B, F) — no horizon

    if A is not None:
        B_true, H_true = np.asarray(A["true"]).shape
    else:
        B_true, H_true = None, None

    if X.ndim == 3:
        if H_true is not None and X.shape[0] == B_true and X.shape[1] == H_true:
            return X  # already (B, H, F)
        return None

    if X.ndim == 4:
        if H_true is not None and X.shape[0] == B_true and X.shape[1] == H_true:
            # (B, H, T, F) — average over T
            return np.nanmean(X, axis=2)
        return None

    return None


def align_B_length(X, A):
    if X is None:
        return None

    B_pred = np.asarray(A["true"]).shape[0]
    B_x = X.shape[0]
    B = min(B_pred, B_x)

    if B_pred != B_x:
        print(
            f"Warning: diagnostic length {B_x} differs from prediction length {B_pred}. "
            f"Using first {B} rows."
        )

    return X[:B]


def row_entropy_from_weights(W, eps=1e-12):
    """
    Entropy and effective number of selected variables from row-wise weights.
    Assumes W rows sum approximately to one.
    """
    W = np.asarray(W, dtype=np.float64)
    W_pos = np.clip(W, 0.0, None)

    row_sum = np.sum(W_pos, axis=1, keepdims=True)
    W_norm = W_pos / np.maximum(row_sum, eps)

    entropy = -np.sum(W_norm * np.log(W_norm + eps), axis=1)
    eff_n = np.exp(entropy)

    V = W.shape[1]
    normalized_entropy = entropy / np.log(max(V, 2))

    return entropy, normalized_entropy, eff_n


# ============================================================
# 4. Feature and group summary tables
# ============================================================

def build_feature_selection_table_angel(
    A,
    horizons_ahead=24,
    active_threshold=1e-6,
    feature_names=None,
    custom_group_map=None,
):
    """
    Build ANGEL feature-level interpretability table.

    Primary importance measure:
      selection_weights

    Selection frequency:
      fraction of test windows where selection_weight > active_threshold.

    Because sparsemax/entmax weights sum to one, group-level sums are usually
    more meaningful than group-level averages.
    """
    sel_key, sel_raw = choose_selection_weight_array(A)
    eff_key, eff_raw = choose_effective_selection_array(A)
    hard_raw = choose_hard_gate_array_if_informative(A, active_threshold=active_threshold)

    attn_key, attn_raw = choose_attention_array(A)

    sel_BF = normalize_feature_diag_to_BF(
        sel_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )
    sel_BF = align_B_length(sel_BF, A)

    eff_BF = normalize_feature_diag_to_BF(
        eff_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )
    eff_BF = align_B_length(eff_BF, A)

    hard_BF = normalize_feature_diag_to_BF(
        hard_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )
    hard_BF = align_B_length(hard_BF, A)

    attn_BF = normalize_attention_diag_to_BF(
        attn_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )
    attn_BF = align_B_length(attn_BF, A)

    n_features = sel_BF.shape[1]

    names = get_feature_names_from_A(
        A,
        n_features=n_features,
        feature_names=feature_names,
    )

    group_map = build_feature_group_mapping(
        names,
        custom_group_map=custom_group_map,
    )
    groups = group_map["Feature group"].tolist()

    entropy, normalized_entropy, eff_n = row_entropy_from_weights(sel_BF)

    print("\nANGEL selector sparsity summary:")
    print("================================")
    print(f"Selection diagnostic used: {sel_key}")
    print(f"Mean row sum: {np.nanmean(np.nansum(sel_BF, axis=1)):.6f}")
    print(f"Mean entropy: {np.nanmean(entropy):.4f}")
    print(f"Mean normalized entropy: {np.nanmean(normalized_entropy):.4f}")
    print(f"Mean effective number of selected variables: {np.nanmean(eff_n):.3f}")
    print(f"Median effective number of selected variables: {np.nanmedian(eff_n):.3f}")

    rows = []

    for j in range(n_features):
        wj = sel_BF[:, j]

        if eff_BF is not None:
            ej = eff_BF[:, j]
        else:
            ej = None

        if hard_BF is not None:
            hj = hard_BF[:, j]
        else:
            hj = None

        if attn_BF is not None and attn_BF.shape[1] == n_features:
            aj = attn_BF[:, j]
        else:
            aj = None

        row = {
            "Feature": names[j],
            "Feature group": groups[j],
            "Selection diagnostic used": sel_key,
            "Effective diagnostic used": eff_key,
            "Attention diagnostic used": attn_key,
            "Selection frequency": float(np.nanmean(wj > active_threshold)),
            "Average selection weight": float(np.nanmean(wj)),
            "Median selection weight": float(np.nanmedian(wj)),
            "P90 selection weight": float(np.nanquantile(wj, 0.90)),
            "P99 selection weight": float(np.nanquantile(wj, 0.99)),
            "Average effective selection": float(np.nanmean(ej)) if ej is not None else np.nan,
            "Mean hard gate": float(np.nanmean(hj)) if hj is not None else np.nan,
            "Mean cross-attention": float(np.nanmean(aj)) if aj is not None else np.nan,
            "Selection weight std": float(np.nanstd(wj)),
        }

        rows.append(row)

    feature_table = pd.DataFrame(rows)

    feature_table = feature_table.sort_values(
        [
            "Average selection weight",
            "Selection frequency",
            "P90 selection weight",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selector_summary = pd.DataFrame({
        "Metric": [
            "Mean row sum",
            "Mean entropy",
            "Mean normalized entropy",
            "Mean effective number selected",
            "Median effective number selected",
            "P90 effective number selected",
            "P99 effective number selected",
        ],
        "Value": [
            float(np.nanmean(np.nansum(sel_BF, axis=1))),
            float(np.nanmean(entropy)),
            float(np.nanmean(normalized_entropy)),
            float(np.nanmean(eff_n)),
            float(np.nanmedian(eff_n)),
            float(np.nanquantile(eff_n, 0.90)),
            float(np.nanquantile(eff_n, 0.99)),
        ],
    })

    return (
        feature_table,
        group_map,
        sel_BF,
        hard_BF,
        eff_BF,
        attn_BF,
        selector_summary,
    )


def build_group_selection_table_angel(feature_table):
    """
    Aggregate feature-level ANGEL diagnostics by group.

    Important:
      Since ANGEL selection weights sum to one across variables, group-level
      total selection mass is more meaningful than average per-feature weight.
    """
    g = (
        feature_table
        .groupby("Feature group", as_index=False)
        .agg(
            **{
                "Group average selection weight": ("Average selection weight", "mean"),
                "Group total selection mass": ("Average selection weight", "sum"),
                "Group selection frequency": ("Selection frequency", "mean"),
                "Group average effective selection": ("Average effective selection", "mean"),
                "Group average cross-attention": ("Mean cross-attention", "mean"),
                "N features": ("Feature", "count"),
            }
        )
    )

    total_mass = g["Group total selection mass"].sum()
    if np.isfinite(total_mass) and total_mass > 0:
        g["Group selection mass share"] = g["Group total selection mass"] / total_mass
    else:
        g["Group selection mass share"] = np.nan

    g = g.sort_values(
        "Group total selection mass",
        ascending=False,
    ).reset_index(drop=True)

    return g


# ============================================================
# 5. High-risk episode selection
# ============================================================

def get_time_index_for_horizon(A, horizon_idx=0):
    y = np.asarray(A["true"])
    B = y.shape[0]

    possible_keys = [
        "timestamp",
        "timestamps",
        "time",
        "times",
        "test_times",
        "ds",
        "index",
        "datetime",
        "datetimes",
    ]

    for key in possible_keys:
        if key not in A:
            continue

        t = np.asarray(A[key])

        if t.ndim == 1 and len(t) == B:
            try:
                return pd.to_datetime(t)
            except Exception:
                return t

        if t.ndim == 2 and t.shape[0] == B:
            h = min(horizon_idx, t.shape[1] - 1)
            try:
                return pd.to_datetime(t[:, h])
            except Exception:
                return t[:, h]

    if "meta" in A and isinstance(A["meta"], pd.DataFrame):
        if "cutoff" in A["meta"].columns:
            try:
                return pd.to_datetime(A["meta"]["cutoff"].values[:B])
            except Exception:
                return A["meta"]["cutoff"].values[:B]

    return np.arange(B)


def select_high_risk_episode_center(
    A,
    horizons_ahead=24,
    threshold=15.0,
    manual_center=None,
    prefer_realized_event=True,
):
    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    if len(h_idx) != 1:
        raise ValueError("Use a single horizon for case-study plots, e.g. horizons_ahead=24.")

    h = int(h_idx[0])

    if manual_center is not None:
        return int(manual_center), horizon_label

    y = np.asarray(A["true"], dtype=np.float64)
    p_depeg = compute_abs_threshold_event_prob(A, abs_threshold=threshold)

    y_h = y[:, h]
    p_h = p_depeg[:, h]

    valid = np.isfinite(y_h) & np.isfinite(p_h)

    if prefer_realized_event:
        event = valid & (np.abs(y_h) >= threshold)

        if np.any(event):
            idx = np.where(event)[0]
            center = idx[np.nanargmax(p_h[idx])]
            return int(center), horizon_label

    idx = np.where(valid)[0]

    if len(idx) == 0:
        raise ValueError("No valid observations for high-risk episode selection.")

    center = idx[np.nanargmax(p_h[idx])]
    return int(center), horizon_label


def build_high_risk_episode_frames_angel(
    A,
    feature_table,
    selection_BF,
    effective_BF=None,
    attention_BF=None,
    horizons_ahead=24,
    threshold=15.0,
    manual_center=None,
    window_pre=72,
    window_post=72,
    top_n_features=25,
):
    """
    Build data frames for high-risk ANGEL interpretability case study.
    """
    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    if len(h_idx) != 1:
        raise ValueError("Use a single horizon for case-study plots, e.g. horizons_ahead=24.")

    h = int(h_idx[0])

    center_idx, _ = select_high_risk_episode_center(
        A,
        horizons_ahead=horizons_ahead,
        threshold=threshold,
        manual_center=manual_center,
        prefer_realized_event=True,
    )

    y = np.asarray(A["true"], dtype=np.float64)
    p_depeg = compute_abs_threshold_event_prob(A, abs_threshold=threshold)

    B = min(y.shape[0], selection_BF.shape[0], p_depeg.shape[0])

    y = y[:B]
    p_depeg = p_depeg[:B]
    selection_BF = selection_BF[:B]

    if effective_BF is not None:
        effective_BF = effective_BF[:B]

    if attention_BF is not None and attention_BF.shape[0] >= B:
        attention_BF = attention_BF[:B]
    else:
        attention_BF = None

    center_idx = int(np.clip(center_idx, 0, B - 1))

    lo = max(0, center_idx - window_pre)
    hi = min(B, center_idx + window_post + 1)

    time_index = get_time_index_for_horizon(A, horizon_idx=h)
    time_index = time_index[:B]

    base = pd.DataFrame({
        "idx": np.arange(lo, hi),
        "t": time_index[lo:hi],
        "realized": y[lo:hi, h],
        "p_depeg": p_depeg[lo:hi, h],
    })

    ft = feature_table.copy()
    ft = ft.sort_values(
        [
            "Average selection weight",
            "Selection frequency",
            "P90 selection weight",
        ],
        ascending=False,
    )
    top_features = ft.head(top_n_features)["Feature"].tolist()

    all_features = feature_table["Feature"].tolist()
    feature_to_col = {f: j for j, f in enumerate(all_features)}
    cols = [feature_to_col[f] for f in top_features]

    selection_mat = selection_BF[lo:hi, :][:, cols].T

    if effective_BF is not None:
        effective_mat = effective_BF[lo:hi, :][:, cols].T
    else:
        effective_mat = None

    if attention_BF is not None and attention_BF.shape[1] == selection_BF.shape[1]:
        attention_mat = attention_BF[lo:hi, :][:, cols].T
    else:
        attention_mat = None

    records = []

    feature_group_lookup = (
        feature_table[["Feature", "Feature group"]]
        .drop_duplicates()
        .set_index("Feature")["Feature group"]
        .to_dict()
    )

    for local_t, global_idx in enumerate(range(lo, hi)):
        for row_j, f in enumerate(top_features):
            rec = {
                "idx": global_idx,
                "t": time_index[global_idx],
                "Feature": f,
                "Feature group": feature_group_lookup.get(f, "Unknown"),
                "Selection weight": selection_mat[row_j, local_t],
            }

            if effective_mat is not None:
                rec["Effective selection"] = effective_mat[row_j, local_t]

            if attention_mat is not None:
                rec["Cross-attention"] = attention_mat[row_j, local_t]

            records.append(rec)

    selection_long = pd.DataFrame(records)

    return {
        "base": base,
        "selection_long": selection_long,
        "top_features": top_features,
        "selection_matrix": selection_mat,
        "effective_selection_matrix": effective_mat,
        "attention_matrix": attention_mat,
        "center_idx": center_idx,
        "window_lo": lo,
        "window_hi": hi,
        "horizon_label": horizon_label,
    }


# ============================================================
# 6. Plotting
# ============================================================

def plot_variable_selection_weights_angel(
    feature_table,
    out_path=None,
    top_n=30,
    title="Variable selection weights",
):
    df = feature_table.head(top_n).copy()
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(df))))

    ax.barh(
        df["Feature"],
        df["Average selection weight"],
        color="tab:blue",
        alpha=0.85,
    )

    ax.set_xlabel("Average sparse selection weight")
    ax.set_title(title)
    ax.set_xlim(0, max(0.01, float(np.nanmax(df["Average selection weight"])) * 1.10))
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=240, bbox_inches="tight")

    return fig, ax


def plot_variable_selection_frequency_angel(
    feature_table,
    out_path=None,
    top_n=30,
    title="Variable selection frequency",
):
    df = feature_table.head(top_n).copy()
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(df))))

    ax.barh(
        df["Feature"],
        df["Selection frequency"],
        color="tab:purple",
        alpha=0.85,
    )

    ax.set_xlabel("Fraction of test windows with weight > threshold")
    ax.set_title(title)
    ax.set_xlim(0, max(1.0, float(np.nanmax(df["Selection frequency"])) * 1.05))
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=240, bbox_inches="tight")

    return fig, ax


def plot_group_selection_mass_angel(
    group_table,
    out_path=None,
    title="Selection mass by feature group",
):
    df = group_table.copy()
    df = df.sort_values("Group total selection mass", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.48 * len(df))))

    ax.barh(
        df["Feature group"],
        df["Group total selection mass"],
        color="royalblue",
        alpha=0.85,
    )

    ax.set_xlabel("Total average sparse selection mass")
    ax.set_title(title)
    ax.set_xlim(0, max(0.01, float(np.nanmax(df["Group total selection mass"])) * 1.10))
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=240, bbox_inches="tight", transparent = True)

    return fig, ax


def plot_selected_covariates_high_risk_episode_angel(
    episode,
    feature_table,
    group_table,
    out_path=None,
    threshold=15.0,
    title="Selected covariates during high-risk forecasts",
    heatmap_kind="selection",
):
    """
    Three-panel ANGEL interpretability plot.

    Panel A:
        group selection mass

    Panel B:
        sparse selection weights / effective selection / cross-attention

    Panel C:
        predicted depeg probability and realized deviation
    """
    base = episode["base"]
    top_features = episode["top_features"]
    center_idx = episode["center_idx"]

    if heatmap_kind == "effective" and episode["effective_selection_matrix"] is not None:
        M = episode["effective_selection_matrix"]
        cbar_label = "Effective selection"
        panel_b_title = "Panel B: effective variable selection during episode"
    elif heatmap_kind == "attention" and episode["attention_matrix"] is not None:
        M = episode["attention_matrix"]
        cbar_label = "Cross-attention"
        panel_b_title = "Panel B: cross-attention to selected variables during episode"
    else:
        M = episode["selection_matrix"]
        cbar_label = "Sparse selection weight"
        panel_b_title = "Panel B: sparse variable-selection weights during episode"

    fig = plt.figure(figsize=(13, 10))

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.1, 2.1, 1.2],
        hspace=0.38,
    )

    # -----------------------------
    # Panel A
    # -----------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    g = group_table.copy()
    g = g.sort_values("Group total selection mass", ascending=True)

    ax1.barh(
        g["Feature group"],
        g["Group total selection mass"],
        color="tab:green",
        alpha=0.85,
    )

    ax1.set_xlabel("Total average sparse selection mass")
    ax1.set_title("Panel A: variable-selection mass by feature group")
    ax1.grid(axis="x", alpha=0.25)

    # -----------------------------
    # Panel B
    # -----------------------------
    ax2 = fig.add_subplot(gs[1, 0])

    vmax = np.nanmax(M)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = ax2.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )

    ax2.set_yticks(np.arange(len(top_features)))
    ax2.set_yticklabels(top_features, fontsize=8)

    n_time = len(base)
    xticks = np.linspace(0, n_time - 1, min(8, n_time), dtype=int)

    xlabels = []
    for i in xticks:
        val = base.iloc[i]["t"]
        if isinstance(val, pd.Timestamp):
            xlabels.append(val.strftime("%Y-%m-%d\n%H:%M"))
        else:
            xlabels.append(str(val))

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, fontsize=8)

    local_center = int(center_idx - int(base["idx"].iloc[0]))

    if 0 <= local_center < n_time:
        ax2.axvline(
            local_center,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="Episode center",
        )

    ax2.set_title(panel_b_title)

    cbar = fig.colorbar(
        im,
        ax=ax2,
        orientation="vertical",
        fraction=0.018,
        pad=0.01,
    )
    cbar.set_label(cbar_label)

    # -----------------------------
    # Panel C
    # -----------------------------
    ax3 = fig.add_subplot(gs[2, 0])

    x = np.arange(n_time)

    ax3.plot(
        x,
        base["p_depeg"].astype(float).values,
        color="tab:red",
        linewidth=1.9,
        label=rf"$P(|\Delta p|>{threshold:g}\mathrm{{bps}})$",
    )

    ax3.set_ylabel("Predicted tail probability", color="tab:red")
    ax3.tick_params(axis="y", labelcolor="tab:red")
    ax3.set_ylim(bottom=0.0)
    ax3.grid(alpha=0.25)

    ax3b = ax3.twinx()

    ax3b.plot(
        x,
        base["realized"].astype(float).values,
        color="black",
        linewidth=1.5,
        label="Realized deviation",
    )

    ax3b.axhline(threshold, color="gray", linestyle=":", linewidth=1.0)
    ax3b.axhline(-threshold, color="gray", linestyle=":", linewidth=1.0)
    ax3b.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    ax3b.set_ylabel("Realized deviation, bps", color="black")
    ax3b.tick_params(axis="y", labelcolor="black")

    if 0 <= local_center < n_time:
        ax3.axvline(
            local_center,
            color="red",
            linestyle="--",
            linewidth=1.2,
        )

    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xlabels, fontsize=8)
    ax3.set_title("Panel C: predicted tail probability and realized deviation")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path is not None:
        fig.savefig(out_path, dpi=260, bbox_inches="tight")

    return fig


def plot_selection_heatmap_testset(
    weight_BF,
    feature_names,
    A=None,
    out_path=None,
    top_n=15,
    title="Selection weights over test set",
    cmap="coolwarm",
    cbar_label="Weight",
    max_display_samples=2000,
):
    """
    Heatmap of selection weights / effective selection over the entire test set.
    Rows = top_n features (sorted by mean weight), columns = test samples.
    """
    B, V = weight_BF.shape
    mean_weights = np.nanmean(weight_BF, axis=0)
    top_idx = np.argsort(-mean_weights)[:top_n]
    top_names = [feature_names[i] for i in top_idx]

    if B > max_display_samples:
        step = B // max_display_samples
        sample_idx = np.arange(0, B, step)
    else:
        sample_idx = np.arange(B)

    mat = weight_BF[sample_idx][:, top_idx].T  # (top_n, n_display)

    if A is not None:
        time_index = get_time_index_for_horizon(A, horizon_idx=0)
        time_index = time_index[:B][sample_idx]
    else:
        time_index = sample_idx

    n_display = mat.shape[1]
    fig_width = max(12, min(n_display / 25, 40))
    fig_height = max(4, top_n * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    pos_vals = mat[mat > 0]
    vmax = float(np.nanpercentile(pos_vals, 99)) if pos_vals.size > 0 else 1.0

    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )

    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(top_names, fontsize=8)

    xtick_count = min(10, n_display)
    xticks = np.linspace(0, n_display - 1, xtick_count, dtype=int)
    xlabels = []
    for i in xticks:
        val = time_index[i]
        if isinstance(val, pd.Timestamp):
            xlabels.append(val.strftime("%Y-%m-%d"))
        else:
            xlabels.append(str(val))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Test sample")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.018, pad=0.01)
    cbar.set_label(cbar_label)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=240, bbox_inches="tight")

    return fig, ax


def plot_effective_selection_depeg_horizon(
    A,
    eff_raw,
    feature_names,
    out_path=None,
    top_n=15,
    n_samples=5,
    threshold=15.0,
    title="Effective selection over forecast horizon — depeg samples",
    cmap="coolwarm",
):
    """
    For a handful of realized depeg samples, plot effective_selection_used
    as a heatmap over the forecast horizon.

    Rows = top_n features, columns = horizon steps.
    One sub-panel per selected depeg sample.
    Falls back to single-column display when no horizon dimension is found.
    """
    eff_BHF = normalize_feature_diag_to_BHF(eff_raw, A)
    has_horizon = eff_BHF is not None

    if not has_horizon:
        eff_BF = normalize_feature_diag_to_BF(eff_raw, A)
        if eff_BF is None:
            print("Warning: could not extract effective selection for depeg horizon plot.")
            return None
        eff_BHF = eff_BF[:, np.newaxis, :]  # treat (B, F) as (B, 1, F)

    B, H, V = eff_BHF.shape

    y = np.asarray(A["true"], dtype=np.float64)[:B]
    depeg_mask = np.any(np.abs(y) >= threshold, axis=1)
    depeg_idx = np.where(depeg_mask)[0]

    if len(depeg_idx) == 0:
        print(
            f"Warning: no depeg samples found with |deviation| >= {threshold} bps. "
            "Skipping depeg horizon plot."
        )
        return None

    if len(depeg_idx) <= n_samples:
        selected = depeg_idx
    else:
        step = max(1, len(depeg_idx) // n_samples)
        selected = depeg_idx[::step][:n_samples]

    n_sel = len(selected)

    mean_w = np.nanmean(eff_BHF.reshape(-1, V), axis=0)
    top_idx = np.argsort(-mean_w)[:top_n]
    top_names = [feature_names[i] for i in top_idx]

    sample_mats = [eff_BHF[idx][:, top_idx].T for idx in selected]  # each (top_n, H)

    all_vals = np.concatenate([m.ravel() for m in sample_mats])
    pos_vals = all_vals[all_vals > 0]
    vmax = float(np.nanpercentile(pos_vals, 99)) if pos_vals.size > 0 else 1.0

    fig, axes = plt.subplots(
        1,
        n_sel,
        figsize=(max(5 * n_sel, 12), max(5, top_n * 0.45 + 2.0)),
        sharey=True,
    )
    if n_sel == 1:
        axes = [axes]

    time_index = get_time_index_for_horizon(A, horizon_idx=0)[:B]

    h_tick_count = min(H, 12)
    h_ticks = np.linspace(0, H - 1, h_tick_count, dtype=int)
    if has_horizon:
        h_labels = [f"h+{h + 1}" for h in h_ticks]
    else:
        h_labels = ["—"]

    last_im = None
    for ax, idx, mat in zip(axes, selected, sample_mats):
        last_im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )

        ax.set_xticks(h_ticks)
        ax.set_xticklabels(h_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Forecast horizon")

        t_val = time_index[idx]
        t_str = (
            t_val.strftime("%Y-%m-%d %H:%M")
            if isinstance(t_val, pd.Timestamp)
            else str(t_val)
        )
        max_dev = float(np.nanmax(np.abs(y[idx])))
        ax.set_title(f"Sample {idx}\n{t_str}\nmax|dev|={max_dev:.1f} bps", fontsize=8)

    axes[0].set_yticks(np.arange(top_n))
    axes[0].set_yticklabels(top_names, fontsize=8)

    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes,
            orientation="vertical",
            fraction=0.018,
            pad=0.02,
            label="Effective selection",
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    if out_path is not None:
        fig.savefig(out_path, dpi=240, bbox_inches="tight")

    return fig


# ============================================================
# 7. Main wrapper
# ============================================================

def make_section_5210_interpretability_outputs_angel(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima/section_5210_interpretability_angel",
    model="ANGEL",
    horizons_ahead=24,
    threshold=15.0,
    active_threshold=1e-6,
    feature_names=None,
    custom_group_map=None,
    manual_center=None,
    window_pre=72,
    window_post=72,
    top_n_features=25,
    heatmap_kind="selection",
    print_keys=True,
):
    """
    Create Section 5.2.10 interpretability tables and plots for ANGEL.

    heatmap_kind:
      "selection" -> Panel B uses selection_weights
      "effective" -> Panel B uses effective_selection_used if available
      "attention" -> Panel B uses cross-attention diagnostic if available
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if runs is None:
        if model_paths is None:
            raise ValueError("Pass either runs or model_paths.")
        runs = load_forecast_pickles(model_paths)
        check_common_grid_and_shape(runs)

    if model not in runs:
        raise ValueError(f"model={model} not found in runs. Available: {list(runs.keys())}")

    A = runs[model]

    if "diagnostics" not in A:
        raise KeyError(
            f"{model} preds_test_set.pkl does not contain A['diagnostics']. "
            "Rerun test with --save_test_diagnostics 1."
        )

    if print_keys:
        print_diagnostic_keys(A)

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    (
        feature_table,
        group_map,
        selection_BF,
        hard_BF,
        effective_BF,
        attention_BF,
        selector_summary,
    ) = build_feature_selection_table_angel(
        A,
        horizons_ahead=horizons_ahead,
        active_threshold=active_threshold,
        feature_names=feature_names,
        custom_group_map=custom_group_map,
    )

    group_table = build_group_selection_table_angel(feature_table)

    episode = build_high_risk_episode_frames_angel(
        A,
        feature_table=feature_table,
        selection_BF=selection_BF,
        effective_BF=effective_BF,
        attention_BF=attention_BF,
        horizons_ahead=horizons_ahead,
        threshold=threshold,
        manual_center=manual_center,
        window_pre=window_pre,
        window_post=window_post,
        top_n_features=top_n_features,
    )

    model_tag = model.replace(" ", "_").replace("/", "_")
    horizon_tag = horizon_label.replace(",", "_").replace("-", "_")

    # -----------------------------
    # Save tables
    # -----------------------------
    feature_table.to_csv(
        out_dir / f"section_5210_feature_selection_table_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    group_table.to_csv(
        out_dir / f"section_5210_group_selection_table_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    group_map.to_csv(
        out_dir / f"section_5210_feature_group_mapping_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    selector_summary.to_csv(
        out_dir / f"section_5210_selector_sparsity_summary_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    episode["base"].to_csv(
        out_dir / f"section_5210_high_risk_episode_base_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    episode["selection_long"].to_csv(
        out_dir / f"section_5210_high_risk_episode_selection_long_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    # -----------------------------
    # Save plots
    # -----------------------------
    plot_variable_selection_weights_angel(
        feature_table,
        out_path=out_dir / f"section_5210_variable_selection_weights_{model_tag}_{horizon_tag}.png",
        top_n=top_n_features,
        title=f"Average sparse variable-selection weights: {model}, horizons {horizon_label}",
    )

    plot_variable_selection_frequency_angel(
        feature_table,
        out_path=out_dir / f"section_5210_variable_selection_frequency_{model_tag}_{horizon_tag}.png",
        top_n=top_n_features,
        title=f"Sparse variable-selection frequency: {model}, horizons {horizon_label}",
    )

    plot_group_selection_mass_angel(
        group_table,
        out_path=out_dir / f"section_5210_group_selection_mass_{model_tag}_{horizon_tag}.png",
        title=f"Selection mass by feature group: {model}",
    )

    plot_selected_covariates_high_risk_episode_angel(
        episode,
        feature_table=feature_table,
        group_table=group_table,
        out_path=out_dir / f"section_5210_selected_covariates_high_risk_episode_{model_tag}_{horizon_tag}.png",
        threshold=threshold,
        title=f"Selected covariates during high-risk forecasts: {model}, horizons {horizon_label}",
        heatmap_kind=heatmap_kind,
    )

    # --------------------------------------------------
    # Test-set heatmaps: selection_weights and
    # effective_selection_used (top-15 vars, coolwarm)
    # --------------------------------------------------
    feat_names = feature_table["Feature"].tolist()

    plot_selection_heatmap_testset(
        selection_BF,
        feature_names=feat_names,
        A=A,
        out_path=out_dir / f"section_5210_testset_heatmap_selection_weights_{model_tag}_{horizon_tag}.png",
        top_n=15,
        title=f"selection_weights over test set — top 15 vars: {model}",
        cmap="Reds",
        cbar_label="Sparse selection weight",
    )

    if effective_BF is not None:
        plot_selection_heatmap_testset(
            effective_BF,
            feature_names=feat_names,
            A=A,
            out_path=out_dir / f"section_5210_testset_heatmap_effective_selection_{model_tag}_{horizon_tag}.png",
            top_n=15,
            title=f"effective_selection_used over test set — top 15 vars: {model}",
            cmap="Reds",
            cbar_label="Effective selection",
        )

    # --------------------------------------------------
    # Depeg samples: effective_selection over horizon
    # (top-15 vars, coolwarm, one panel per depeg sample)
    # --------------------------------------------------
    _, eff_raw_for_horizon = choose_effective_selection_array(A)
    if eff_raw_for_horizon is not None:
        plot_effective_selection_depeg_horizon(
            A,
            eff_raw=eff_raw_for_horizon,
            feature_names=feat_names,
            out_path=out_dir / f"section_5210_depeg_horizon_effective_selection_{model_tag}_{horizon_tag}.png",
            top_n=15,
            n_samples=5,
            threshold=threshold,
            title=f"effective_selection_used over forecast horizon — depeg samples: {model}",
            cmap="Reds",
        )

    print("\n==============================================")
    print(f"Section 5.2.10 ANGEL interpretability outputs for {model}")
    print(f"Horizons: {horizon_label}")
    print("==============================================")

    print("\nSelector sparsity summary:")
    print(selector_summary.to_string(index=False))

    print("\nTop selected features:")
    display_cols = [
        "Feature",
        "Feature group",
        "Selection frequency",
        "Average selection weight",
        "P90 selection weight",
        "Average effective selection",
        "Mean cross-attention",
    ]
    display_cols = [c for c in display_cols if c in feature_table.columns]

    print(
        feature_table[display_cols]
        .head(25)
        .to_string(index=False)
    )

    print("\nSelection mass by feature group:")
    print(group_table.to_string(index=False))

    print(f"\nSelected high-risk episode center index: {episode['center_idx']}")

    return {
        "runs": runs,
        "feature_table": feature_table,
        "group_table": group_table,
        "feature_group_mapping": group_map,
        "selector_summary": selector_summary,
        "episode": episode,
        "selection_BF": selection_BF,
        "hard_BF": hard_BF,
        "effective_selection_BF": effective_BF,
        "attention_BF": attention_BF,
        "out_dir": out_dir,
    }


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    
    section_5210_result = make_section_5210_interpretability_outputs_angel(
        model_paths=model_paths,
        out_dir="./comparison_nn_vs_arima/section_5210_interpretability_angel",

        # Change this to the key used in your model_paths dict.
        # For example, if model_paths has "SAINT": ..., use model="SAINT".
        # If you renamed the new model "ANGEL", use model="ANGEL".
        model="SAINT",

        horizons_ahead=24,
        threshold=15.0,

        # sparsemax gives exact zeros, so 1e-8 or 1e-6 are both reasonable.
        active_threshold=1e-6,

        feature_names=None,
        custom_group_map=None,
        manual_center=None,
        window_pre=72,
        window_post=72,
        top_n_features=25,

        # Options:
        #   "selection" -> Panel B uses selection_weights
        #   "effective" -> Panel B uses effective_selection_used
        #   "attention" -> Panel B uses cross-attention if available
        heatmap_kind="selection",

        print_keys=True,
    )
    A = section_5210_result["runs"]["SAINT"]  # or "ANGEL", depending on your key

    audit = audit_angel_cross_attention(
        A,
        feature_names=None,
        horizons_ahead=24,
        active_threshold=1e-8,
        top_k=20,
    )

    feature_selection_table = section_5210_result["feature_table"]
    group_selection_table = section_5210_result["group_table"]
    high_risk_episode = section_5210_result["episode"]