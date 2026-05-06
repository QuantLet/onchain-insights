import pandas as pd
import matplotlib.pyplot as plt
import os 
from datetime import timedelta
import numpy as np

def build_hourly_position_snapshots(df, hourly_blocks, position_cols=None):
    """
    Convert mint/burn event dataframe into hourly snapshots of active positions.

    Returns a long dataframe with one row per (hour, active position), including
    the position's liquidity at that hour.
    """
    if position_cols is None:
        position_cols = ["owner", "tickLower", "tickUpper"]

    # ----------------------------
    # 1) Keep only what we need
    # ----------------------------
    e = df[position_cols + ["blockNumber", "liquidityAmount", "liquiditySign"]].copy()

    # Use Python ints to avoid int64 overflow issues on large liquidity values
    e["liquidityAmount"] = e["liquidityAmount"].map(int)
    e["liquiditySign"] = e["liquiditySign"].map(int)

    # Signed liquidity delta: mint = +amount, burn = -amount
    e["delta_liquidity"] = e["liquidityAmount"] * e["liquiditySign"]

    # --------------------------------------------------------
    # 2) Collapse to one delta per position per block
    #    (hourly snapshots are block-end state, so intra-block
    #     event ordering does not matter here)
    # --------------------------------------------------------
    block_deltas = (
        e.groupby(position_cols + ["blockNumber"], as_index=False)["delta_liquidity"]
         .sum()
         .sort_values(position_cols + ["blockNumber"])
         .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 3) Compute liquidity after each block for each position
    # --------------------------------------------------------
    block_deltas["liquidity"] = (
        block_deltas.groupby(position_cols)["delta_liquidity"].cumsum()
    )

    # Next block where this position changes again
    block_deltas["next_block"] = (
        block_deltas.groupby(position_cols)["blockNumber"].shift(-1)
    )

    # Only keep states where position is active
    states = block_deltas[block_deltas["liquidity"] > 0].copy()

    # --------------------------------------------------------
    # 4) Map block intervals to hour intervals using hourly_blocks
    # --------------------------------------------------------
    hb = (
        hourly_blocks[["hour_utc", "block_number"]]
        .sort_values("block_number")
        .reset_index(drop=True)
        .copy()
    )
    hb["hour_idx"] = np.arange(len(hb))

    hour_blocks = hb["block_number"].to_numpy()
    n_hours = len(hb)

    # For each active state:
    #   active for all hours where
    #   hour_block >= blockNumber and hour_block < next_block
    states["start_idx"] = np.searchsorted(
        hour_blocks,
        states["blockNumber"].to_numpy(),
        side="left"
    )

    next_block_filled = states["next_block"].fillna(np.iinfo(np.int64).max).to_numpy()
    states["end_idx"] = np.searchsorted(
        hour_blocks,
        next_block_filled,
        side="left"
    )

    # Open-ended states remain active through the last available hour
    states["end_idx"] = states["end_idx"].clip(upper=n_hours)

    # Remove zero-length hour ranges
    states = states[states["start_idx"] < states["end_idx"]].copy()

    if states.empty:
        return pd.DataFrame(
            columns=["hour_idx", "hour_utc", "block_number"] + position_cols + ["liquidity"]
        )

    # --------------------------------------------------------
    # 5) Explode each active state into one row per active hour
    # --------------------------------------------------------
    lengths = (states["end_idx"] - states["start_idx"]).astype(int).to_numpy()

    repeated_states = states.loc[
        states.index.repeat(lengths),
        position_cols + ["liquidity"]
    ].copy()

    repeated_states["hour_idx"] = np.concatenate([
        np.arange(s, e) for s, e in zip(states["start_idx"], states["end_idx"])
    ])

    # Join hour metadata
    snapshots = repeated_states.merge(
        hb[["hour_idx", "hour_utc", "block_number"]],
        on="hour_idx",
        how="left"
    )

    snapshots = snapshots[
        ["hour_idx", "hour_utc", "block_number"] + position_cols + ["liquidity"]
    ].sort_values(["hour_idx"] + position_cols).reset_index(drop=True)

    return snapshots

def liquidity_curve_on_ticks(
    df: pd.DataFrame,
    peg_tick: int,
    half_width: int = 50,
    tick_spacing: int = 1,
    tick_lower_col: str = "tickLower",
    tick_upper_col: str = "tickUpper",
    liquidity_col: str = "liquidity",
) -> pd.DataFrame:
    """
    Build the *active liquidity* curve L(t) over ticks in [peg_tick-half_width, peg_tick+half_width],
    where a position contributes liquidity on [tickLower, tickUpper) (inclusive of lower, exclusive of upper).

    Returns a DataFrame with columns: tick, active_liquidity

    Notes:
      - If you want an exact curve, sample at tick_spacing=1 (or ensure your sampling includes all boundary ticks).
      - liquidity is assumed additive (same units as stored in the subgraph).
    """
    if df.empty:
        return pd.DataFrame({"tick": [], "active_liquidity": []})

    start = int(peg_tick) - int(half_width)
    end   = int(peg_tick) + int(half_width)

    x = df[[tick_lower_col, tick_upper_col, liquidity_col]].copy()
    x[tick_lower_col] = pd.to_numeric(x[tick_lower_col], errors="coerce").astype("Int64")
    x[tick_upper_col] = pd.to_numeric(x[tick_upper_col], errors="coerce").astype("Int64")
    # liquidity is often a big integer encoded as string
    x[liquidity_col] = pd.to_numeric(x[liquidity_col], errors="coerce").fillna(0).astype("int64")

    # keep only positions that overlap the range at all
    x = x[(x[tick_upper_col] > start) & (x[tick_lower_col] <= end)]
    if x.empty:
        ticks = list(range(start, end + 1, tick_spacing))
        return pd.DataFrame({"tick": ticks, "active_liquidity": [0] * len(ticks)})

    # initial liquidity active at 'start'
    initial = x[(x[tick_lower_col] <= start) & (x[tick_upper_col] > start)][liquidity_col].sum()

    # deltas inside (start, end]
    add = (
        x[(x[tick_lower_col] > start) & (x[tick_lower_col] <= end)]
        .groupby(tick_lower_col)[liquidity_col]
        .sum()
    )
    sub = (
        x[(x[tick_upper_col] > start) & (x[tick_upper_col] <= end)]
        .groupby(tick_upper_col)[liquidity_col]
        .sum()
        .mul(-1)
    )
    deltas = add.add(sub, fill_value=0).to_dict()  # tick -> net change at that tick

    ticks = list(range(start, end + 1, tick_spacing))

    active = []
    L = int(initial)
    for i, t in enumerate(ticks):
        if i > 0:
            L += int(deltas.get(t, 0))
        active.append(L)

    return pd.DataFrame({"tick": ticks, "active_liquidity": active})

def plot_missing_liquidity(positions, pos_mev, hourly_curve, block_nbr, non_nfpm = False):
    pos = positions.query("block_number == @block_nbr")
    curve = hourly_curve.query(f'block == {block_nbr}').copy()
    curve_df = liquidity_curve_on_ticks(pos, peg_tick=0, half_width=50, tick_spacing=1)
    if non_nfpm:
        pos_mev_temp = pos_mev.query("block_number == @block_nbr")
        non_nfpm_curve = liquidity_curve_on_ticks(pos_mev_temp, peg_tick=0, half_width=50, tick_spacing=1)
    curve_df['tickLower'] = curve_df['tick'] 
    curve_df['tickUpper'] = curve_df['tick'] + 1
    plt.figure(figsize=(12, 6))
    plt.bar((curve["tickLower"] + 3*curve["tickUpper"]) / 4, curve["active_liquidity_L"], width = 0.5, color = 'cornflowerblue', alpha =1., label = 'Liquidity Curve')
    if non_nfpm:
        plt.bar((3*curve_df["tickLower"] + curve_df["tickUpper"]) / 4, curve_df["active_liquidity"] + non_nfpm_curve["active_liquidity"].fillna(0), width = 0.5, alpha =.8, color = 'skyblue', zorder = -1, label = 'Direct SC Mints')
    plt.bar((3*curve_df["tickLower"] + curve_df["tickUpper"]) / 4, curve_df["active_liquidity"], width = 0.5, alpha =1., color = 'forestgreen', label = 'Non Fungible Positions')
    plt.bar((curve["tickLower"] + curve["tickUpper"]) / 2, curve["active_liquidity_L"], width = 1, facecolor = 'None', lw = 2, edgecolor = 'black', alpha =1)
    plt.vlines(curve['poolTick'] -1/2, ymin=0, ymax=2*curve['active_liquidity_L'].max(), colors='gray', linestyles='dashed', label='current pool tick')
    plt.xlabel('Tick')
    plt.ylabel('Active Liquidity')
    plt.xlim(-20, 20)
    plt.ylim(0, 1.1*curve['active_liquidity_L'].max())
    # plt.title('Liquidity Curve Comparison')
    plt.legend(bbox_to_anchor=(0.04, -.1), ncols = 4, frameon=False, loc='upper left')
    plt.savefig(f'./liquidity_curve_{block_nbr}.png', dpi=300, bbox_inches='tight', transparent = True)

def _first_present(cols, candidates, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in columns: {list(cols)}")
    return None


def _prepare_hourly_blocks(hourly_blocks):
    hb = hourly_blocks.copy()
    if "hour_utc" not in hb.columns and "hour" in hb.columns:
        hb = hb.rename(columns={"hour": "hour_utc"})

    block_col = _first_present(hb.columns, ["block_number", "block", "blockNumber"])
    hb["hour_utc"] = pd.to_datetime(hb["hour_utc"], utc=True)
    hb["block_number"] = pd.to_numeric(hb[block_col], errors="coerce")
    hb = hb.dropna(subset=["hour_utc", "block_number"]).copy()
    hb["block_number"] = hb["block_number"].astype("int64")
    hb = hb.sort_values(["hour_utc", "block_number"]).drop_duplicates("hour_utc", keep="last")
    return hb.reset_index(drop=True)


def _prepare_pool(pool):
    p = pool.copy()
    if "hour_utc" not in p.columns and "hour" in p.columns:
        p = p.rename(columns={"hour": "hour_utc"})
    p["hour_utc"] = pd.to_datetime(p["hour_utc"], utc=True)

    tick_col = _first_present(p.columns, ["tick", "currentTick", "tick_current", "poolTick"])
    liq_col = _first_present(p.columns, ["poolLiquidity", "pooLiquidity", "pool_liquidity"], required=False)

    out_cols = ["hour_utc", tick_col]
    rename_map = {tick_col: "current_tick"}
    if liq_col is not None:
        out_cols.append(liq_col)
        rename_map[liq_col] = "pool_liquidity"

    p = p[out_cols].copy().rename(columns=rename_map)
    p["current_tick"] = pd.to_numeric(p["current_tick"], errors="coerce")
    if "pool_liquidity" in p.columns:
        p["pool_liquidity"] = pd.to_numeric(p["pool_liquidity"], errors="coerce")
    return p.sort_values("hour_utc").drop_duplicates("hour_utc", keep="last").reset_index(drop=True)


def _prepare_nfpm_positions(pos):
    nfpm = pos.copy()
    if "hour_utc" not in nfpm.columns and "hour" in nfpm.columns:
        nfpm = nfpm.rename(columns={"hour": "hour_utc"})

    nfpm["hour_utc"] = pd.to_datetime(nfpm["hour_utc"], utc=True)
    nfpm["liquidity"] = pd.to_numeric(nfpm["liquidity"], errors="coerce")
    nfpm["tickLower"] = pd.to_numeric(nfpm["tickLower"], errors="coerce")
    nfpm["tickUpper"] = pd.to_numeric(nfpm["tickUpper"], errors="coerce")
    nfpm = nfpm.dropna(subset=["hour_utc", "liquidity", "tickLower", "tickUpper"]).copy()

    owner_col = _first_present(nfpm.columns, ["owner", "ownerAddress", "wallet"], required=False)
    if owner_col is None:
        nfpm["owner"] = "unknown"
    else:
        nfpm["owner"] = nfpm[owner_col].astype(str).str.lower()

    id_col = _first_present(nfpm.columns, ["id", "tokenId", "position_id"], required=False)
    if id_col is None:
        nfpm["id"] = np.arange(len(nfpm)).astype(str)
    else:
        nfpm["id"] = nfpm[id_col].astype(str)

    nfpm["tickLower"] = nfpm["tickLower"].astype("int64")
    nfpm["tickUpper"] = nfpm["tickUpper"].astype("int64")
    nfpm = nfpm[nfpm["liquidity"] > 0].copy()
    return nfpm[["hour_utc", "owner", "id", "tickLower", "tickUpper", "liquidity"]]


def _prepare_non_nfpm_events(non_nfpm_events):
    ev = non_nfpm_events.copy()

    owner_col = _first_present(ev.columns, ["owner", "ownerAddress", "wallet"])
    ev["owner"] = ev[owner_col].astype(str).str.lower()

    block_col = _first_present(ev.columns, ["blockNumber", "block_number", "block"])
    ev["blockNumber"] = pd.to_numeric(ev[block_col], errors="coerce")

    tick_l_col = _first_present(ev.columns, ["tickLower", "tick_lower"])
    tick_u_col = _first_present(ev.columns, ["tickUpper", "tick_upper"])
    ev["tickLower"] = pd.to_numeric(ev[tick_l_col], errors="coerce")
    ev["tickUpper"] = pd.to_numeric(ev[tick_u_col], errors="coerce")

    liq_amt_col = _first_present(ev.columns, ["liquidityAmount", "liquidity", "liquidity_amount"])
    liq_sign_col = _first_present(ev.columns, ["liquiditySign", "sign", "direction"])
    ev["delta_liquidity"] = pd.to_numeric(ev[liq_amt_col], errors="coerce").fillna(0.0) * pd.to_numeric(ev[liq_sign_col], errors="coerce").fillna(0.0)

    time_col = _first_present(ev.columns, ["timestamp", "event_time", "time"], required=False)
    if time_col is not None:
        if np.issubdtype(ev[time_col].dtype, np.number):
            ev["event_time"] = pd.to_datetime(ev[time_col], unit="s", utc=True, errors="coerce")
        else:
            ev["event_time"] = pd.to_datetime(ev[time_col], utc=True, errors="coerce")
    else:
        ev["event_time"] = pd.NaT

    log_col = _first_present(ev.columns, ["logIndex", "log_index"], required=False)
    if log_col is None:
        ev["logIndex"] = 0
    else:
        ev["logIndex"] = pd.to_numeric(ev[log_col], errors="coerce").fillna(0)

    tx_col = _first_present(ev.columns, ["transactionHash", "tx_hash", "event_key"], required=False)
    if tx_col is None:
        ev["event_key"] = np.arange(len(ev)).astype(str)
    else:
        ev["event_key"] = ev[tx_col].astype(str)

    ev = ev.dropna(subset=["blockNumber", "tickLower", "tickUpper", "owner"]).copy()
    ev["blockNumber"] = ev["blockNumber"].astype("int64")
    ev["logIndex"] = ev["logIndex"].astype("int64")
    ev["tickLower"] = ev["tickLower"].astype("int64")
    ev["tickUpper"] = ev["tickUpper"].astype("int64")

    ev["position_key"] = ev["owner"] + "|" + ev["tickLower"].astype(str) + "|" + ev["tickUpper"].astype(str)
    ev = ev.sort_values(["blockNumber", "logIndex", "event_key"]).reset_index(drop=True)
    return ev[["blockNumber", "logIndex", "event_key", "event_time", "owner", "tickLower", "tickUpper", "position_key", "delta_liquidity"]]


def _build_non_nfpm_open_snapshots(hb, ev):
    state = {}
    rows = []
    j = 0
    recs = ev.to_dict("records")

    for _, snap in hb.iterrows():
        snap_hour = snap["hour_utc"]
        snap_block = int(snap["block_number"])

        while j < len(recs) and int(recs[j]["blockNumber"]) <= snap_block:
            e = recs[j]
            key = e["position_key"]
            prev = state.get(key)
            prev_liq = 0.0 if prev is None else float(prev["liquidity"])
            new_liq = prev_liq + float(e["delta_liquidity"])

            if new_liq <= 0:
                if key in state:
                    del state[key]
            else:
                state[key] = {
                    "owner": e["owner"],
                    "tickLower": int(e["tickLower"]),
                    "tickUpper": int(e["tickUpper"]),
                    "liquidity": new_liq,
                }
            j += 1

        for key, st in state.items():
            rows.append(
                {
                    "hour_utc": snap_hour,
                    "owner": st["owner"],
                    "id": key,
                    "tickLower": st["tickLower"],
                    "tickUpper": st["tickUpper"],
                    "liquidity": st["liquidity"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["hour_utc", "owner", "id", "tickLower", "tickUpper", "liquidity"])
    return pd.DataFrame(rows)


def _assign_events_to_snapshot_hour(ev, hb):
    right = hb[["hour_utc", "block_number"]].sort_values("block_number").copy()
    left = ev.sort_values("blockNumber").copy()

    assigned = pd.merge_asof(
        left,
        right,
        left_on="blockNumber",
        right_on="block_number",
        direction="forward",
        allow_exact_matches=True,
    )
    return assigned.dropna(subset=["hour_utc"]).copy()


def _compute_rolling_whale_flags(owner_flows, all_hours, whale_top_pct=0.05, rolling_window_hours=24):
    if owner_flows.empty:
        return pd.DataFrame(columns=["hour_utc", "owner", "is_whale", "rolling_1d_volume", "owner_rank", "n_owners", "top_n"])

    flow = owner_flows.copy()
    flow["gross_flow"] = flow["mint_flow"] + flow["burn_flow"]

    owners = flow[["owner"]].drop_duplicates().copy()
    owners["_k"] = 1
    hours = all_hours[["hour_utc"]].drop_duplicates().copy()
    hours["_k"] = 1

    panel = (
        hours.merge(owners, on="_k", how="inner")
        .drop(columns=["_k"])
        .merge(flow[["hour_utc", "owner", "gross_flow"]], on=["hour_utc", "owner"], how="left")
        .fillna({"gross_flow": 0.0})
        .sort_values(["owner", "hour_utc"])
        .reset_index(drop=True)
    )

    panel["rolling_1d_volume"] = (
        panel.groupby("owner")["gross_flow"]
        .rolling(window=int(rolling_window_hours), min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    panel = panel.sort_values(["hour_utc", "rolling_1d_volume"], ascending=[True, False]).copy()
    panel["owner_rank"] = panel.groupby("hour_utc").cumcount() + 1
    panel["n_owners"] = panel.groupby("hour_utc")["owner"].transform("size")
    panel["top_n"] = np.ceil(panel["n_owners"] * float(whale_top_pct)).astype(int).clip(lower=1)

    panel["is_whale"] = (panel["owner_rank"] <= panel["top_n"]) & (panel["rolling_1d_volume"] > 0)

    return panel[["hour_utc", "owner", "is_whale", "rolling_1d_volume", "owner_rank", "n_owners", "top_n"]]


def compute_jit_liquidity(ev, hb):
    """Public wrapper for JIT detection."""
    return _compute_jit_liquidity(ev, hb)


def _compute_jit_liquidity(ev, hb):
    """
    Detect JIT (Just-In-Time) liquidity: positions where a mint AND burn
    occur in the **same block**.

    Returns:
        jit_hourly: DataFrame[hour_utc, jit_liquidity, total_non_nfpm_minted]
        jit_details: per-position JIT detail DataFrame for drill-down
    """
    # Group events by block and position key
    block_pos = (
        ev.groupby(["blockNumber", "position_key", "owner", "tickLower", "tickUpper"])
        .agg(
            mint_liq=("delta_liquidity", lambda s: s[s > 0].sum()),
            burn_liq=("delta_liquidity", lambda s: -s[s < 0].sum()),
            n_events=("delta_liquidity", "size"),
            min_logIndex=("logIndex", "min"),
            max_logIndex=("logIndex", "max"),
        )
        .reset_index()
    )

    # Total minted liquidity per block (all non-NFPM positions)
    block_total_minted = (
        block_pos[block_pos["mint_liq"] > 0]
        .groupby("blockNumber", as_index=False)["mint_liq"]
        .sum()
        .rename(columns={"mint_liq": "block_total_minted"})
    )

    # JIT = both mint and burn in the same block for the same position
    jit = block_pos[(block_pos["mint_liq"] > 0) & (block_pos["burn_liq"] > 0)].copy()
    # JIT volume = min(minted, burned) in that block
    jit["jit_liquidity"] = jit[["mint_liq", "burn_liq"]].min(axis=1)
    # Sandwiched events: log indices between the first mint and last burn
    jit["n_sandwiched_events"] = (jit["max_logIndex"] - jit["min_logIndex"] - 1).clip(lower=0)

    # Map blocks to snapshot hours
    right = hb[["hour_utc", "block_number"]].sort_values("block_number").copy()

    # Map total minted to hours
    total_minted_h = pd.merge_asof(
        block_total_minted.sort_values("blockNumber"),
        right, left_on="blockNumber", right_on="block_number",
        direction="forward", allow_exact_matches=True,
    ).dropna(subset=["hour_utc"])
    total_minted_hourly = (
        total_minted_h.groupby("hour_utc", as_index=False)["block_total_minted"]
        .sum()
        .rename(columns={"block_total_minted": "total_non_nfpm_minted"})
    )

    if jit.empty:
        empty = hb[["hour_utc"]].drop_duplicates().copy()
        empty["jit_liquidity"] = 0.0
        empty = empty.merge(total_minted_hourly, on="hour_utc", how="left")
        empty["total_non_nfpm_minted"] = empty["total_non_nfpm_minted"].fillna(0.0)
        return empty[["hour_utc", "jit_liquidity", "total_non_nfpm_minted"]], jit

    jit_assigned = pd.merge_asof(
        jit.sort_values("blockNumber"),
        right, left_on="blockNumber", right_on="block_number",
        direction="forward", allow_exact_matches=True,
    ).dropna(subset=["hour_utc"])

    # Merge block-level total minted for per-event share computation
    jit_assigned = jit_assigned.merge(block_total_minted, on="blockNumber", how="left")

    jit_hourly = (
        jit_assigned.groupby("hour_utc", as_index=False)["jit_liquidity"]
        .sum()
        .merge(total_minted_hourly, on="hour_utc", how="outer")
    )
    jit_hourly["jit_liquidity"] = jit_hourly["jit_liquidity"].fillna(0.0)
    jit_hourly["total_non_nfpm_minted"] = jit_hourly["total_non_nfpm_minted"].fillna(0.0)

    return jit_hourly, jit_assigned


def build_hourly_lp_features(hourly_blocks, pool, nfpm_positions, non_nfpm_events, top_ks=(1, 5, 10), whale_top_pct=0.05, rolling_window_hours=24):
    hb = _prepare_hourly_blocks(hourly_blocks)
    pool_h = _prepare_pool(pool)
    nfpm = _prepare_nfpm_positions(nfpm_positions)
    ev = _prepare_non_nfpm_events(non_nfpm_events)

    hb = hb.merge(pool_h, on="hour_utc", how="left").sort_values("hour_utc").reset_index(drop=True)

    non_nfpm_open = _build_non_nfpm_open_snapshots(hb[["hour_utc", "block_number"]], ev)
    assigned_flows = _assign_events_to_snapshot_hour(ev, hb[["hour_utc", "block_number"]])

    features = hb[["hour_utc", "block_number", "current_tick"]].copy()
    if "pool_liquidity" in hb.columns:
        features["pool_liquidity"] = hb["pool_liquidity"].values

    nfpm_tick = nfpm.merge(hb[["hour_utc", "current_tick"]], on="hour_utc", how="left")
    non_nfpm_tick = non_nfpm_open.merge(hb[["hour_utc", "current_tick"]], on="hour_utc", how="left")

    nfpm_tick["in_range"] = (nfpm_tick["tickLower"] <= nfpm_tick["current_tick"]) & (nfpm_tick["current_tick"] < nfpm_tick["tickUpper"])
    non_nfpm_tick["in_range"] = (non_nfpm_tick["tickLower"] <= non_nfpm_tick["current_tick"]) & (non_nfpm_tick["current_tick"] < non_nfpm_tick["tickUpper"])

    nfpm_in = nfpm_tick[nfpm_tick["in_range"]].copy()
    non_nfpm_in = non_nfpm_tick[non_nfpm_tick["in_range"]].copy()

    nfpm_in_liq = nfpm_in.groupby("hour_utc", as_index=False)["liquidity"].sum().rename(columns={"liquidity": "nfpm_in_range_liquidity"})
    non_nfpm_in_liq = non_nfpm_in.groupby("hour_utc", as_index=False)["liquidity"].sum().rename(columns={"liquidity": "non_nfpm_in_range_liquidity"})

    features = features.merge(nfpm_in_liq, on="hour_utc", how="left").merge(non_nfpm_in_liq, on="hour_utc", how="left")
    features[["nfpm_in_range_liquidity", "non_nfpm_in_range_liquidity"]] = features[["nfpm_in_range_liquidity", "non_nfpm_in_range_liquidity"]].fillna(0.0)
    features["total_in_range_liquidity"] = features["nfpm_in_range_liquidity"] + features["non_nfpm_in_range_liquidity"]

    denom = features["total_in_range_liquidity"].replace(0, np.nan)
    features["nfpm_in_range_share"] = features["nfpm_in_range_liquidity"] / denom
    features["non_nfpm_in_range_share"] = features["non_nfpm_in_range_liquidity"] / denom

    nfpm_owner = nfpm_in.groupby(["hour_utc", "owner"], as_index=False)["liquidity"].sum()
    if not nfpm_owner.empty:
        nfpm_owner["tot"] = nfpm_owner.groupby("hour_utc")["liquidity"].transform("sum")
        nfpm_owner["hhi_part"] = (nfpm_owner["liquidity"] / nfpm_owner["tot"]).pow(2)
        nfpm_owner_hhi = nfpm_owner.groupby("hour_utc", as_index=False)["hhi_part"].sum().rename(columns={"hhi_part": "nfpm_owner_hhi"})
        features = features.merge(nfpm_owner_hhi, on="hour_utc", how="left")
    else:
        features["nfpm_owner_hhi"] = np.nan

    nfpm_pos = nfpm_in.groupby(["hour_utc", "id"], as_index=False)["liquidity"].sum()
    if not nfpm_pos.empty:
        nfpm_pos["tot"] = nfpm_pos.groupby("hour_utc")["liquidity"].transform("sum")
        nfpm_pos["hhi_part"] = (nfpm_pos["liquidity"] / nfpm_pos["tot"]).pow(2)
        nfpm_pos_hhi = nfpm_pos.groupby("hour_utc", as_index=False)["hhi_part"].sum().rename(columns={"hhi_part": "nfpm_position_hhi"})
        features = features.merge(nfpm_pos_hhi, on="hour_utc", how="left")
    else:
        features["nfpm_position_hhi"] = np.nan

    top_input = nfpm_pos.sort_values(["hour_utc", "liquidity"], ascending=[True, False]).copy()
    top_input["rank"] = top_input.groupby("hour_utc").cumcount() + 1
    top_totals = top_input.groupby("hour_utc", as_index=False)["liquidity"].sum().rename(columns={"liquidity": "nfpm_in_range_total_for_topk"})

    for k in top_ks:
        topk = top_input[top_input["rank"] <= k].groupby("hour_utc", as_index=False)["liquidity"].sum()
        topk = topk.rename(columns={"liquidity": f"nfpm_top_{k}_liq"})
        features = features.merge(topk, on="hour_utc", how="left")
        features[f"nfpm_top_{k}_liq"] = features[f"nfpm_top_{k}_liq"].fillna(0.0)
        features = features.merge(top_totals, on="hour_utc", how="left")
        features[f"nfpm_top_{k}_share"] = features[f"nfpm_top_{k}_liq"] / features["nfpm_in_range_total_for_topk"].replace(0, np.nan)
        features = features.drop(columns=["nfpm_in_range_total_for_topk"])

    # --- JIT liquidity share ---
    jit_hourly, _ = _compute_jit_liquidity(ev, hb[["hour_utc", "block_number"]])
    features = features.merge(jit_hourly, on="hour_utc", how="left")
    features["jit_liquidity"] = features["jit_liquidity"].fillna(0.0)
    features["total_non_nfpm_minted"] = features["total_non_nfpm_minted"].fillna(0.0)
    # Share of minted volume that is JIT (avoids >1 issue from end-of-block snapshots)
    features["jit_mint_share"] = features["jit_liquidity"] / features["total_non_nfpm_minted"].replace(0, np.nan)
    features["jit_total_share"] = features["jit_liquidity"] / denom

    owner_flows = assigned_flows.groupby(["hour_utc", "owner"], as_index=False).agg(
        net_liquidity_flow=("delta_liquidity", "sum"),
        mint_flow=("delta_liquidity", lambda s: s[s > 0].sum()),
        burn_flow=("delta_liquidity", lambda s: -s[s < 0].sum()),
    )

    whale_flags = _compute_rolling_whale_flags(
        owner_flows=owner_flows,
        all_hours=hb[["hour_utc"]],
        whale_top_pct=whale_top_pct,
        rolling_window_hours=rolling_window_hours,
    )

    whale_flows = owner_flows.merge(whale_flags[["hour_utc", "owner", "is_whale"]], on=["hour_utc", "owner"], how="left")
    whale_flows["is_whale"] = whale_flows["is_whale"].fillna(False)

    whale_hourly = whale_flows[whale_flows["is_whale"]].groupby("hour_utc", as_index=False).agg(
        whale_net_liquidity_flow=("net_liquidity_flow", "sum"),
        whale_mint_flow=("mint_flow", "sum"),
        whale_burn_flow=("burn_flow", "sum"),
    )

    non_whale_hourly = whale_flows[~whale_flows["is_whale"]].groupby("hour_utc", as_index=False).agg(
        non_whale_net_liquidity_flow=("net_liquidity_flow", "sum"),
        non_whale_mint_flow=("mint_flow", "sum"),
        non_whale_burn_flow=("burn_flow", "sum"),
    )

    whale_counts = whale_flags.groupby("hour_utc", as_index=False).agg(
        whale_owner_count=("is_whale", "sum"),
        tracked_owner_count=("owner", "nunique"),
    )

    features = (
        features
        .merge(whale_hourly, on="hour_utc", how="left")
        .merge(non_whale_hourly, on="hour_utc", how="left")
        .merge(whale_counts, on="hour_utc", how="left")
        .sort_values("hour_utc")
        .reset_index(drop=True)
    )

    fill_zero = [
        "whale_net_liquidity_flow", "whale_mint_flow", "whale_burn_flow",
        "non_whale_net_liquidity_flow", "non_whale_mint_flow", "non_whale_burn_flow",
        "whale_owner_count", "tracked_owner_count",
    ]
    for c in fill_zero:
        if c in features.columns:
            features[c] = features[c].fillna(0.0)

    return features