"""Hourly backtest of concentrated-liquidity LP strategies on the Uniswap V3
USDC/USDT 0.01% pool. Numeraire: USDT. Price P = USDT per USDC = 1 + depeg_bps/1e4.

Timing convention:
  * bucket k covers [t_k, t_k+1h); its close price is P_k (= pool state at t_k+1h).
  * the strategy decides at the end of bucket k the position held during bucket k+1.
  * price-based rules see P_k immediately (keeper bots react within the block).
  * model signals stamped k (built from data up to the end of bucket k) are usable
    only from the end of bucket k+1 (1h data-collection lag): at decision k the
    latest usable model stamp is k-1.

Accounting per bucket:
  fee_k   = feesUSD_k * (1-jit) * f_in_range * L/(L_active + L)
  lvr_k   = Vpos(P1) - Vpos(P0) - xpos(P0)*(P1-P0)   (<= 0, convexity loss vs rebalancing)
  cost_k  = gas per burn/mint + (pool fee + slippage) * swapped notional
  hedged_k = fee_k + lvr_k - cost_k   (PnL of the strategy vs a delta-matched rebalancing portfolio)
"""
import numpy as np
import pandas as pd

LN = np.log(1.0001)


def tick_of(P):
    return int(np.floor(np.log(P) / LN))


def sp(t):
    return 1.0001 ** (t / 2.0)


def amounts(L, P, ta, tb):
    sa, sb = sp(ta), sp(tb)
    s = min(max(np.sqrt(P), sa), sb)
    return L * (1.0 / s - 1.0 / sb), L * (s - sa)


class Market:
    def __init__(self, panel, start, end):
        h = panel.loc[start:end].copy()
        self.idx = h.index
        P = 1.0 + h['depeg_bps'].values / 1e4
        prevP = np.r_[1.0 / h['open'].values[0], P[:-1]]
        with np.errstate(divide='ignore'):
            hi = 1.0 / h['low'].values
            lo = 1.0 / h['high'].values
        bad = (~np.isfinite(hi)) | (~np.isfinite(lo)) | (np.abs(hi - 1) > 0.05) | (np.abs(lo - 1) > 0.05)
        hi = np.where(bad, np.maximum(P, prevP), hi)
        lo = np.where(bad, np.minimum(P, prevP), lo)
        self.hi = np.maximum.reduce([hi, P, prevP])
        self.lo = np.minimum.reduce([lo, P, prevP])
        self.P, self.P0 = P, prevP
        self.fees = h['feesUSD'].values
        self.vol = h['volumeUSD'].values
        self.Lact = h['liquidity'].values.astype(float)
        self.depeg = h['depeg_bps'].values
        self.n = len(h)
        self.years = self.n / 8766.0


class Sim:
    def __init__(self, mkt, V0=1e6, gas=10.0, slip_bps=0.5, fee_bps=1.0, jit_haircut=0.0,
                 compound='none', compound_thr=1e-3):
        """compound: 'none'      fees accumulate as cash (simple interest);
                     'rebalance' accrued fees are reinvested whenever the vault re-mints (no extra gas);
                     'threshold' as 'rebalance', plus a collect-and-reinvest transaction whenever
                                 accrued fees exceed `compound_thr` x capital (pays gas + swap cost)."""
        self.m = mkt; self.V0 = V0; self.gas = gas
        self.swap_cost = (fee_bps + slip_bps) / 1e4
        self.jit = jit_haircut
        assert compound in ('none', 'rebalance', 'threshold')
        self.compound, self.compound_thr = compound, compound_thr

    # ---- execution helpers -------------------------------------------------
    def _execute(self, act, st, P):
        """Mutates st (dict with L, ta, tb, xi, yi, F). Returns cost."""
        if act[0] == 'keep':
            return (0.0, 0.0)
        if act[0] == 'compound':
            if st['L'] == 0:
                return (0.0, 0.0)
            act = ('lp', st['ta'], st['tb'], st.get('frac', 1.0))
        elif (act[0] == 'lp' and st['L'] > 0 and act[1] == st['ta'] and act[2] == st['tb']
              and (len(act) < 4 or abs(act[3] - st.get('frac', 1.0)) < 1e-12)):
            return (0.0, 0.0)
        if act[0] == 'hold' and st['L'] == 0 and act[1] is None:
            return (0.0, 0.0)
        c = 0.0
        if st['L'] > 0:  # burn
            x, y = amounts(st['L'], P, st['ta'], st['tb'])
            st['xi'] += x; st['yi'] += y; st['L'] = 0.0; st['ta'] = st['tb'] = None
            c += self.gas / 2
        if act[0] == 'lp' and self.compound != 'none':
            st['yi'] += st['F']; st['F'] = 0.0   # reinvest accrued fees (net of gas paid so far)
        V = st['xi'] * P + st['yi']
        if act[0] == 'lp':
            ta, tb = act[1], act[2]
            frac = act[3] if len(act) > 3 else 1.0   # share of capital deployed; the rest stays idle 50/50
            x1, y1 = amounts(1.0, P, ta, tb)
            vpl = x1 * P + y1
            xfrac = x1 * P / vpl
            Vp, Vi = frac * V, (1 - frac) * V
            swap = abs(xfrac * Vp + 0.5 * Vi - st['xi'] * P)
            sc = swap * self.swap_cost
            st['L'] = (Vp - sc) / vpl  # swap cost paid out of the minted capital
            st['ta'], st['tb'] = ta, tb
            st['xi'] = 0.5 * Vi / P; st['yi'] = 0.5 * Vi
            st['frac'] = frac
            c += self.gas / 2
            return (c, sc)
        else:
            frac = act[1]
            sc = 0.0
            if frac is not None:
                swap = abs(frac * V - st['xi'] * P)
                sc = swap * self.swap_cost
                V -= sc
                st['xi'] = frac * V / P; st['yi'] = (1 - frac) * V
            return (c, sc)

    def run(self, strat):
        m = self.m; n = m.n
        st = dict(L=0.0, ta=None, tb=None, xi=0.5 * self.V0 / m.P0[0], yi=0.5 * self.V0, F=0.0)
        out = {k: np.zeros(n) for k in ['V', 'X', 'fees', 'lvr', 'gas', 'swapcost', 'inrange', 'hedged',
                                        'ta', 'tb', 'mode', 'reb', 'share', 'compounds']}
        strat.reset(m)
        act = strat.decide(-1, m.P0[0], dict(L=0.0, ta=None, tb=None))
        g0, sc0 = self._execute(act, st, m.P0[0])
        st['F'] -= g0
        for k in range(n):
            P0, P1 = m.P0[k], m.P[k]
            fee = 0.0; lvr = 0.0; f = 0.0; share = 0.0
            if st['L'] > 0:
                ta, tb = st['ta'], st['tb']
                x0, y0 = amounts(st['L'], P0, ta, tb)
                x1, y1 = amounts(st['L'], P1, ta, tb)
                pa, pb = 1.0001 ** ta, 1.0001 ** tb
                lo, hi = m.lo[k], m.hi[k]
                if hi > lo:
                    f = max(0.0, min(hi, pb) - max(lo, pa)) / (hi - lo)
                else:
                    f = 1.0 if pa <= P1 <= pb else 0.0
                Lraw = st['L'] * 1e6
                share = Lraw / (m.Lact[k] + Lraw)
                fee = m.fees[k] * (1 - self.jit) * f * share
                lvr = (x1 * P1 + y1) - (x0 * P0 + y0) - x0 * (P1 - P0)
            st['F'] += fee
            act = strat.decide(k, P1, dict(L=st['L'], ta=st['ta'], tb=st['tb']))
            if (self.compound == 'threshold' and act[0] == 'keep' and st['L'] > 0
                    and st['F'] > self.compound_thr * self.V0):
                act = ('compound',)
            is_comp = act[0] == 'compound'
            g, sc = self._execute(act, st, P1)
            st['F'] -= g  # gas from fee cash; swap cost already removed from capital
            pos = amounts(st['L'], P1, st['ta'], st['tb']) if st['L'] > 0 else (0.0, 0.0)
            X = st['xi'] + pos[0]
            V = X * P1 + st['yi'] + pos[1] + st['F']
            out['V'][k] = V; out['X'][k] = X
            out['fees'][k] = fee; out['lvr'][k] = lvr; out['gas'][k] = g; out['swapcost'][k] = sc
            out['inrange'][k] = f; out['share'][k] = share
            if k == 0:  # book the initial mint (entry) cost in the first hour
                g += g0; sc += sc0
            out['hedged'][k] = fee + lvr - g - sc
            out['ta'][k] = st['ta'] if st['L'] > 0 else np.nan
            out['tb'][k] = st['tb'] if st['L'] > 0 else np.nan
            out['mode'][k] = getattr(strat, 'mode', 0)
            out['reb'][k] = 1.0 if ((g > 0 or sc > 0) and not is_comp) else 0.0
            out['compounds'][k] = 1.0 if is_comp else 0.0
        df = pd.DataFrame(out, index=m.idx)
        df['P'] = m.P
        df.attrs['V0'] = self.V0
        return df


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def hodl(mkt, V0=1e6):
    x = 0.5 * V0 / mkt.P0[0]; y = 0.5 * V0
    return pd.Series(x * mkt.P + y, index=mkt.idx)


def metrics(df, mkt, V0=None, label=None, event_mask=None):
    yrs = mkt.years
    V0 = V0 or df.attrs.get("V0", 1e6)
    h = df['hedged']
    cum_h = h.cumsum()
    dd_h = (cum_h - cum_h.cummax()).min()
    daily_h = h.resample('1D').sum()
    V = df['V']
    ret_daily = V.resample('1D').last().pct_change().dropna()
    dd_v = (V / V.cummax() - 1).min()
    hv = hodl(mkt, V0)
    r = dict(
        fee_apr=df['fees'].sum() / V0 / yrs * 1e4,          # bps / yr
        lvr_apr=df['lvr'].sum() / V0 / yrs * 1e4,
        cost_apr=-(df['gas'].sum() + df['swapcost'].sum()) / V0 / yrs * 1e4,
        net_vs_rebal_apr=h.sum() / V0 / yrs * 1e4,
        excess_vs_hodl_apr=(V.iloc[-1] - hv.iloc[-1]) / V0 / yrs * 1e4,
        total_ret_apr=(V.iloc[-1] / V0 - 1) / yrs * 1e4,
        hedged_cagr=((1 + h.sum() / V0) ** (1 / yrs) - 1) * 1e4,
        hedged_sharpe=daily_h.mean() / daily_h.std() * np.sqrt(365) if daily_h.std() > 0 else np.nan,
        # Sortino on hourly net PnL (target 0, annualised with sqrt(8766)): the vault's losses are intraday LVR
        # spikes that daily aggregation nets against same-day fees; the daily version is kept for reference.
        hedged_sortino=(h.mean() / np.sqrt(np.mean(np.minimum(h.values, 0.0) ** 2)) * np.sqrt(8766)
                        if (h < 0).any() else np.nan),
        hedged_sortino_daily=(daily_h.mean() / np.sqrt(np.mean(np.minimum(daily_h.values, 0.0) ** 2)) * np.sqrt(365)
                              if (daily_h < 0).any() else np.nan),
        hedged_maxdd_bps=dd_h / V0 * 1e4,
        hedged_cvar5_daily_bps=daily_h[daily_h <= daily_h.quantile(0.05)].mean() / V0 * 1e4,
        worst_day_bps=daily_h.min() / V0 * 1e4,
        mtm_maxdd_bps=dd_v * 1e4,
        time_in_range=df['inrange'].mean(),
        rebalances=int(df['reb'].sum()),
        rebal_per_month=df['reb'].sum() / (yrs * 12),
        compounds_per_month=df['compounds'].sum() / (yrs * 12) if 'compounds' in df else 0.0,
        avg_width_ticks=(df['tb'] - df['ta']).mean(),
    )
    if event_mask is not None:
        r['event_hedged_bps'] = h[event_mask].sum() / V0 * 1e4
        r['event_lvr_bps'] = df['lvr'][event_mask].sum() / V0 * 1e4
        r['event_fees_bps'] = df['fees'][event_mask].sum() / V0 * 1e4
        r['event_cost_bps'] = -(df['gas'] + df['swapcost'])[event_mask].sum() / V0 * 1e4
        r['calm_hedged_apr'] = h[~event_mask].sum() / V0 / yrs * 1e4
    return pd.Series(r, name=label)
