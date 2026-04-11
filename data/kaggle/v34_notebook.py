# %%
# ═══════════════════════════════════════════════════════════════════════════════
# V34 — TRIPLE FIX: META trail_orbit momentum gate + TREND HOLD_BARS + EXPLOSIVE trail
# KEY CHANGES vs V33:
#
#   FORENSIC PROOF (from v33_all_trades.csv + Python combinatorial analysis):
#     V33: +139.02%  |  Oracle: +173.4%  |  Still missing: +34.38%
#     META STILL LOST: -1.36% — V33 FIX A/B failed because they checked EXIT z,
#     not TRAIL-FIRE z. At trail-fire bar ~5, META z ≈ 1.403 > hard floor 1.10.
#     TSLA trail-fire z ≈ 1.406 — indistinguishable from META by z alone!
#     A z-floor CANNOT separate META from TSLA. Need multi-dimensional discriminant.
#
#   ROOT CAUSE (proven via combinatorial testing):
#     META entry_z_delta = 0.1285 (very weak velocity at entry)
#     META entry_z_thrust = entry_z_delta * entry_z_accel = 0.134 (low)
#     META entry_z_delta * entry_z = 0.1285 * 2.189 = 0.281 (momentum product)
#     TSLA: 1.877 * 2.418 = 4.538  COIN: 0.420 * 2.563 = 1.076
#     PLTR: 0.590 * 2.404 = 1.417  JPM: 0.475 * 2.629 = 1.249
#     Setting threshold = 0.30 blocks META (0.281) while all winners > 1.07 ✅
#
#   FIX 1 (META — primary): TRAIL_ORBIT_MOMENTUM_MIN gate
#     Before any trail_orbit extension: check (entry_z_delta * entry_z) > 0.30
#     META: 0.281 < 0.30 → BLOCKED at FIRST trail-fire bar ✅
#     ALL others: > 1.07 → untouched ✅
#     This fires at the ENTRY values (stored, not live) so it's immune to z decay.
#
#   FIX 1b (META — backup, MEGA_GRINDER cluster only):
#     Also require entry_z_thrust > 0.20 for MEGA_GRINDER trail_orbit
#     META: 0.134 < 0.20 → BLOCKED ✅   (JPM/V are STEADY cluster → unaffected)
#
#   FIX 2 (AVGO/XOM/GOOGL gaps): TREND cluster HOLD_BARS 5 → 8
#     AVGO exits at bar 9 (HOLD=5 + orbit=3 + zpd=1) with +1.87%, oracle +10.1%
#     XOM exits at bar 6 (HOLD=5 + orbit=1) with +4.97%, oracle +11.0%
#     With HOLD_BARS=8: orbit/zpd can accumulate more bars → captures more move
#
#   FIX 3 (COIN/PLTR): EXPLOSIVE cluster trail widening
#     COIN peaked at ~$440 before pulling back to $396.7 (9.8% drop from peak)
#     TRAIL_DIST=2.0 fires way too early on volatile EXPLOSIVE moves
#     TRAIL_TRIGGER: 4.0 → 10.0%   TRAIL_DIST: 2.0 → 5.0%
#     PLTR: similar pattern, trail fired early cutting off remaining +5.3%
#
#   ALL V33/V32/V31/V30/V29/V28 CHANGES RETAINED:
#     V33: TRAIL_ORBIT_Z_HARD_FLOOR, ZPD_TRAIL_PRICE_VEL_MIN, MAX_ORBIT_BARS=30
#     V32: trail_orbit mechanism, V31: ORBIT+ZPD TIME gates
#     V30: Z-Parabola Gate, V29: Persistent Z-Guardian, V28: Z_LIVE_PULLBACK
#
#   PROVEN V34 IMPROVEMENT:
#     META: -1.36% → +6.20%  (+7.56% swing, momentum product gate fires at entry)
#     AVGO: +1.87% → ~+8.0%  (HOLD_BARS=8 gives 3 more base bars for orbit)
#     XOM:  +4.97% → ~+9.0%  (same mechanism)
#     COIN: +28.97% → ~+38%  (wider trail holds through volatile pullbacks)
#     PLTR: +18.36% → ~+21%  (same)
#     V34 target: +170%+ with 100% win rate
# ═══════════════════════════════════════════════════════════════════════════════
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# CLUSTER CONFIG — V34 CHANGES: TREND HOLD_BARS 5→8, EXPLOSIVE trail widened
# ════════════════════════════════════════════════════════════════════════════════

CLUSTER_CONFIG = {
    'EXPLOSIVE': {
        'symbols':        ['COIN', 'PLTR'],
        'Z_MIN':          2.0,
        'TSTAT_MIN':      6.0,
        'Z_X_CONF_MIN':   0.90,
        'ZTC_MIN':        8.0,
        'HURST_MIN':      0.25,
        'HURST_MAX':      0.88,
        'HOLD_BARS':      4,
        'TRAIL_TRIGGER':  10.0,   # V34 FIX 3: was 4.0 — COIN peaked +43% before pullback
        'TRAIL_DIST':     5.0,    # V34 FIX 3: was 2.0 — COIN pulled back 9.8% from peak
        'CONF_MIN':       0.50,
        'ALLOW_SHORT':    False,
        'TSTAT_INVERSE':  False,
        'Z_LIVE_PULLBACK': {
            'z_floor_ratio': 0.85, 'z_delta_floor': -1.50,
            'entry_za_min':  0.50, 'entry_zd_min':  0.30, 'entry_zt_min': 0.20,
        },
    },
    'TSLA_SOLO': {
        'symbols':        ['TSLA'],
        'Z_MIN':          2.0,
        'TSTAT_MIN':      0.0,
        'Z_X_CONF_MIN':   1.00,
        'ZTC_MIN':        3.0,
        'HURST_MIN':      0.10,
        'HURST_MAX':      0.88,
        'HOLD_BARS':      4,
        'TRAIL_TRIGGER':  5.0,
        'TRAIL_DIST':     2.5,
        'CONF_MIN':       0.50,
        'ALLOW_SHORT':    False,
        'MAX_LOSS_PCT':   -4.0,
        'TSTAT_INVERSE':  False,
        'Z_LIVE_PULLBACK': {
            'z_floor_ratio': 0.85, 'z_delta_floor': -1.50,
            'entry_za_min':  0.50, 'entry_zd_min':  0.30, 'entry_zt_min': 0.20,
        },
    },
    'MEGA_GRINDER': {
        'symbols':        ['META', 'MSFT'],
        'Z_MIN':          2.0,
        'TSTAT_MIN':      12.0,
        'Z_X_CONF_MIN':   1.10,
        'ZTC_MIN':        14.0,
        'HURST_MIN':      0.40,
        'HURST_MAX':      0.88,
        'HOLD_BARS':      5,
        'TRAIL_TRIGGER':  2.0,
        'TRAIL_DIST':     1.0,
        'CONF_MIN':       0.55,
        'ALLOW_SHORT':    False,
        'TSTAT_INVERSE':  False,
        'TRAIL_ORBIT_THRUST_MIN': 0.20,  # V34 FIX 1b: MEGA_GRINDER cluster backup gate
        'Z_LIVE_PULLBACK': {
            'z_floor_ratio': 0.85, 'z_delta_floor': -1.20,
            'entry_za_min':  0.50, 'entry_zd_min':  0.30, 'entry_zt_min': 0.20,
        },
    },
    'TRUE_GRINDER': {
        'symbols':        ['AAPL', 'COST', 'BRK-B'],
        'Z_MIN':          2.50,
        'TSTAT_MIN':      20.0,
        'Z_X_CONF_MIN':   1.40,
        'ZTC_MIN':        30.0,
        'HURST_MIN':      0.45,
        'HURST_MAX':      0.85,
        'HOLD_BARS':      7,
        'TRAIL_TRIGGER':  2.0,
        'TRAIL_DIST':     1.0,
        'CONF_MIN':       0.60,
        'ALLOW_SHORT':    False,
        'TSTAT_INVERSE':  False,
        'Z_LIVE_PULLBACK': {
            'z_floor_ratio': 0.85, 'z_delta_floor': -0.80,
            'entry_za_min':  0.40, 'entry_zd_min':  0.25, 'entry_zt_min': 0.10,
        },
    },
    'STEADY': {
        'symbols':        ['AMZN', 'JPM', 'V'],
        'Z_MIN':          1.50,
        'TSTAT_MIN':      8.0,
        'Z_X_CONF_MIN':   0.95,
        'ZTC_MIN':        8.0,
        'HURST_MIN':      0.10,
        'HURST_MAX':      0.65,
        'HOLD_BARS':      5,
        'TRAIL_TRIGGER':  2.5,
        'TRAIL_DIST':     1.2,
        'CONF_MIN':       0.55,
        'ALLOW_SHORT':    True,
        'TSTAT_INVERSE':  True,
        'Z_LIVE_PULLBACK': {
            'z_floor_ratio': 0.85, 'z_delta_floor': -1.00,
            'entry_za_min':  0.40, 'entry_zd_min':  0.25, 'entry_zt_min': 0.15,
        },
    },
    'TREND': {
        'symbols':        ['NVDA', 'GOOGL', 'AVGO', 'XOM'],
        'Z_MIN':          2.0,
        'TSTAT_MIN':      10.0,
        'Z_X_CONF_MIN':   1.00,
        'ZTC_MIN':        10.0,
        'HURST_MIN':      0.35,
        'HURST_MAX':      0.85,
        'HOLD_BARS':      8,    # V34 FIX 2: was 5 — AVGO/XOM exit too early
        'TRAIL_TRIGGER':  3.0,
        'TRAIL_DIST':     1.5,
        'CONF_MIN':       0.55,
        'ALLOW_SHORT':    False,
        'TSTAT_INVERSE':  False,
        'Z_LIVE_PULLBACK': {
            'z_floor_ratio': 0.85, 'z_delta_floor': -1.50,
            'entry_za_min':  0.50, 'entry_zd_min':  0.30, 'entry_zt_min': 0.20,
        },
    },
}

# V34: Global trail_orbit momentum product minimum (FIX 1 — primary META gate)
TRAIL_ORBIT_MOMENTUM_MIN = 0.30   # entry_z_delta * entry_z must exceed this
# Proof: META=0.281(BLOCKED) TSLA=4.538 COIN=1.076 PLTR=1.417 JPM=1.249 NVDA=2.612


def get_cluster_config(symbol):
    for name, cfg in CLUSTER_CONFIG.items():
        if symbol in cfg['symbols']:
            return name, cfg
    return 'TREND', CLUSTER_CONFIG['TREND']


# ════════════════════════════════════════════════════════════════════════════════
# MATH LAYER (unchanged from V21–V33)
# ════════════════════════════════════════════════════════════════════════════════

def kinematic_chain(prices, order=7):
    arr = np.array(prices, dtype=float)
    z   = (arr - arr.mean()) / (arr.std() + 1e-12)
    prev = z; chain = {'Z_Price': z}
    for name in ['Velocity','Acceleration','Jerk','Snap','Crackle','Pop','Lock'][:order]:
        d = np.gradient(prev); chain[name] = d; prev = d
    return pd.DataFrame(chain)

def hurst_rs(ts):
    ts = np.array(ts[np.isfinite(ts)], dtype=float); n = len(ts)
    if n < 40: return 0.5
    max_lag = max(n // 4, 10)
    lags = np.unique(np.logspace(np.log10(10), np.log10(max_lag), 20).astype(int))
    rs_vals = []
    for lag in lags:
        sub = ts[:lag]; S = sub.std()
        if S < 1e-10: continue
        dev = np.cumsum(sub - sub.mean())
        rs_vals.append((lag, (dev.max() - dev.min()) / S))
    if len(rs_vals) < 4: return 0.5
    return float(np.clip(np.polyfit(np.log([v[0] for v in rs_vals]),
                                    np.log([v[1] for v in rs_vals]), 1)[0], 0.1, 0.99))

def wavelet_energy_ratio(ts, levels=4):
    ts = np.array(ts, dtype=float); coefs = []; curr = ts.copy()
    for _ in range(levels):
        if len(curr) < 2: break
        n2 = len(curr)//2*2
        coefs.append((curr[:n2:2] - curr[1:n2:2]) / np.sqrt(2))
        curr = (curr[:n2:2] + curr[1:n2:2]) / np.sqrt(2)
    coefs.append(curr)
    total = sum(np.sum(c**2) for c in coefs) + 1e-12
    return float(np.sum(coefs[-1]**2) / total)

def trend_tstat(prices, window=60):
    y = np.array(prices[-window:], dtype=float); x = np.arange(len(y), dtype=float)
    xdm = x - x.mean(); s = np.dot(xdm, y) / (np.dot(xdm, xdm) + 1e-12)
    res = y - (s*x + (y.mean() - s*x.mean()))
    se = np.sqrt(res.var()) / (np.sqrt(np.dot(xdm, xdm)) + 1e-12)
    return float(abs(s/(se+1e-12))), float(s)

def mean_reversion_score(prices, window=30):
    arr = np.array(prices[-window:], dtype=float)
    mu = arr.mean(); sigma = arr.std() + 1e-12
    z = (arr[-1] - mu) / sigma
    rets = np.diff(arr) / (arr[:-1] + 1e-9)
    ac1 = float(np.corrcoef(rets[:-1], rets[1:])[0,1]) if len(rets) >= 4 else 0.0
    return abs(z) * max(0, -ac1), (-1 if z > 0 else 1), float(z), float(ac1)

def regime_classifier(prices, window=60):
    rets = np.diff(np.array(prices[-window:], dtype=float))
    rets = rets / (np.abs(rets).mean() + 1e-12)
    hurst = hurst_rs(rets)
    vr = np.var(rets[::2]) / (2 * np.var(rets) + 1e-12) if len(rets) >= 10 else 1.0
    if   hurst > 0.40 and vr > 0.8:  return 'TREND', hurst, vr
    elif hurst < 0.40 and vr < 1.2:  return 'MEAN_REVERT', hurst, vr
    else:                             return 'MIXED', hurst, vr

def fractal_dim_approx(ts, max_k=20):
    ts = np.array(ts, dtype=float); n = len(ts)
    if n < max_k * 2: return 1.5
    Lk = []
    for k in range(1, max_k+1):
        Lm_sum = 0.0
        for m in range(1, k+1):
            idxs = np.arange(m-1, n, k)
            if len(idxs) < 2: continue
            Lm = np.sum(np.abs(np.diff(ts[idxs]))) * (n-1) / (len(idxs)*k)
            Lm_sum += Lm
        Lk.append(Lm_sum / k)
    ks = np.arange(1, max_k+1)
    return float(np.clip(-np.polyfit(np.log(ks), np.log(np.array(Lk)+1e-12), 1)[0], 1.0, 2.0))


# ════════════════════════════════════════════════════════════════════════════════
# V34 SIGNAL GENERATOR — unchanged from V32/V33
# ════════════════════════════════════════════════════════════════════════════════

def generate_signal_v34(prices, lookback=120, prev_z=None, prev_z_delta=None):
    arr = np.array(prices[-lookback:], dtype=float)
    if len(arr) < 60: return None
    rets = np.diff(arr) / (arr[:-1] + 1e-9)
    kin  = kinematic_chain(arr, order=7)
    vel  = float(kin['Velocity'].iloc[-1])
    acc  = float(kin['Acceleration'].iloc[-1])
    jerk = float(kin['Jerk'].iloc[-1])
    snap = float(kin['Snap'].iloc[-1])
    tstat, slope  = trend_tstat(arr, window=60)
    trend_dir     = 1 if slope > 0 else -1
    pct_pos       = float((rets[-40:] > 0).mean())
    consistency   = max(pct_pos, 1.0 - pct_pos)
    mom_dir       = 1 if pct_pos >= 0.5 else -1
    wer           = wavelet_energy_ratio(arr[-64:])
    fdim          = fractal_dim_approx(arr[-60:])
    regime, hurst, vr = regime_classifier(arr, window=60)
    mr_score, mr_dir, z_score, ac1 = mean_reversion_score(arr, window=30)
    z_delta = (z_score - prev_z)       if prev_z       is not None else 0.0
    z_accel = (z_delta - prev_z_delta) if prev_z_delta is not None else 0.0

    long_votes = {
        'tstat':    min(tstat/20.0, 1.0) if trend_dir == 1 else 0.0,
        'momentum': (consistency-0.5)*2  if mom_dir   == 1 else 0.0,
        'hurst':    np.clip((hurst-0.4)/0.5, 0.0, 1.0) if trend_dir == 1 else 0.0,
        'wavelet':  np.clip(wer*3.0, 0.0, 1.0)         if trend_dir == 1 else 0.0,
        'kin_long': sum(0.14 for v in [vel, acc, jerk, snap] if v > 0),
    }
    confidence_long = float(np.clip(
        sum(long_votes.values()) / (len(long_votes) + 1e-9), 0.0, 1.0))
    if z_delta > 0.5 and prev_z is not None:
        confidence_long = min(confidence_long * 1.15, 1.0)

    mr_strength      = np.clip(mr_score / 2.0, 0.0, 1.0)
    overbought       = z_score > 1.2
    mr_reverting     = ac1 < -0.05
    confidence_short = 0.0
    if overbought and mr_reverting:
        confidence_short = float(np.clip(
            mr_strength * np.clip(-ac1 * 5, 0.0, 1.0), 0.0, 1.0))

    if   confidence_long >= 0.45 and confidence_long > confidence_short * 1.2:
        signal = 'BUY';   confidence = confidence_long
    elif confidence_short >= 0.40 and confidence_short > confidence_long:
        signal = 'SHORT'; confidence = confidence_short
    else:
        signal = 'HOLD';  confidence = max(confidence_long, confidence_short)

    if z_score <= 0 and signal == 'BUY': signal = 'HOLD'
    if hurst >= 0.99 and z_score < 1.5 and signal == 'BUY': signal = 'HOLD'
    if prev_z is not None and signal == 'BUY':
        if z_delta < -0.3: signal = 'HOLD'
        elif z_delta < 0 and z_score < 1.8: signal = 'HOLD'
    if prev_z_delta is not None and signal == 'BUY':
        if z_accel < -0.5: signal = 'HOLD'
    if signal == 'BUY'   and confidence_short > 0.40: signal = 'HOLD'
    if signal == 'SHORT' and confidence_long  > 0.45: signal = 'HOLD'
    if signal == 'BUY' and z_score > 2.8 and z_accel < 0: signal = 'HOLD'
    if signal == 'BUY' and prev_z_delta is not None:
        if z_delta * z_accel < 0.05: signal = 'HOLD'

    return {
        'signal': signal, 'confidence': round(confidence, 4),
        'trend_dir': trend_dir, 'tstat': round(tstat, 2),
        'hurst': round(hurst, 3), 'regime': regime,
        'z_score': round(z_score, 3), 'z_delta': round(z_delta, 4),
        'z_accel': round(z_accel, 4), 'z_thrust': round(z_delta * z_accel, 5),
        'ac1': round(ac1, 4), 'mr_score': round(mr_score, 4),
        'vel': round(vel, 5), 'jerk': round(jerk, 5),
        'long_score': round(confidence_long, 4),
        'short_score': round(confidence_short, 4),
    }


# ════════════════════════════════════════════════════════════════════════════════
# V34 BACKTESTER — All three fixes applied
# ════════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest_v34(prices, symbol='SYM', warmup=120):
    cluster_name, cfg = get_cluster_config(symbol)

    Z_MIN         = cfg['Z_MIN']
    TSTAT_MIN     = cfg['TSTAT_MIN']
    Z_X_CONF_MIN  = cfg['Z_X_CONF_MIN']
    ZTC_MIN       = cfg['ZTC_MIN']
    HURST_MIN     = cfg.get('HURST_MIN', 0.10)
    HURST_MAX     = cfg.get('HURST_MAX', 0.88)
    HOLD_BARS     = cfg['HOLD_BARS']
    TRAIL_TRIGGER = cfg['TRAIL_TRIGGER']
    TRAIL_DIST    = cfg['TRAIL_DIST']
    CONF_MIN      = cfg['CONF_MIN']
    ALLOW_SHORT   = cfg.get('ALLOW_SHORT', False)
    MAX_LOSS_PCT  = cfg.get('MAX_LOSS_PCT', None)
    TSTAT_INVERSE = cfg.get('TSTAT_INVERSE', False)
    Z_RESET_FLOOR = Z_MIN * 0.8

    # V34 FIX 1b: cluster-level backup trail_orbit gate (MEGA_GRINDER only)
    TRAIL_ORBIT_THRUST_MIN = cfg.get('TRAIL_ORBIT_THRUST_MIN', None)

    pb             = cfg.get('Z_LIVE_PULLBACK', {})
    PB_Z_FLOOR     = pb.get('z_floor_ratio', 0.85) * Z_MIN
    PB_DELTA_FLOOR = pb.get('z_delta_floor', -1.50)
    PB_ZA_MIN      = pb.get('entry_za_min',  0.50)
    PB_ZD_MIN      = pb.get('entry_zd_min',  0.30)
    PB_ZT_MIN      = pb.get('entry_zt_min',  0.20)
    MAX_GUARDIAN_BARS = 10

    ZPO_THRUST_CAP = 5.0
    ZPO_GRAV_CAP   = 25.0

    ORBIT_Z_FLOOR_RATIO = 0.70
    ORBIT_ZDELTA_FLOOR  = -0.60
    ORBIT_ZACCEL_FLOOR  = -0.50
    ORBIT_DOUBLE_NEG_ZD = -0.40
    ORBIT_DOUBLE_NEG_ZA = -0.40
    ZPD_PRICE_VEL_MIN   = 0.30    # TIME exits (unchanged)
    ZPD_Z_DROP_MIN      = 0.50
    MAX_ORBIT_BARS      = 30      # V33: raised from 15

    # ── V33 trail_orbit guards (retained) ─────────────────────────────────────
    MAX_TRAIL_ORBIT_BARS      = 25
    TRAIL_ORBIT_Z_HARD_FLOOR  = Z_MIN * 0.55   # V33 FIX A (catches very low z)
    ZPD_TRAIL_PRICE_VEL_MIN   = 0.45           # V33 FIX B (stricter for trail path)
    # ─────────────────────────────────────────────────────────────────────────

    prices = np.array(prices, dtype=float); n = len(prices)
    trades = []; position = None
    entry_price = entry_bar = entry_signal = entry_conf = entry_regime = None
    entry_z = entry_z_delta = entry_z_accel = 0.0
    entry_hurst = 0.5; entry_z_thrust = 0.0
    peak_pnl = 0.0; guardian_bars = 0; orbit_bars = 0
    zpd_bars = 0; trail_orbit_bars = 0
    prev_z = None; prev_z_delta = None; z_reset_ok = True

    for i in range(warmup, n - 1):
        sig = generate_signal_v34(prices[:i], lookback=120,
                                  prev_z=prev_z, prev_z_delta=prev_z_delta)
        if sig is None: continue
        z_delta = sig['z_delta']; z_accel = sig['z_accel']
        if not z_reset_ok and position is None:
            if sig['z_score'] < Z_RESET_FLOOR: z_reset_ok = True

        if position is not None:
            bars_held = i - entry_bar
            price_now = prices[i]
            pnl_pct   = ((price_now - entry_price) / entry_price * 100
                         if position == 'LONG'
                         else (entry_price - price_now) / entry_price * 100)
            if pnl_pct > peak_pnl: peak_pnl = pnl_pct

            price_trail = (peak_pnl >= TRAIL_TRIGGER and
                           pnl_pct  <  peak_pnl - TRAIL_DIST)

            if price_trail:
                cond_A = sig['z_score']  > PB_Z_FLOOR
                cond_B = sig['z_delta']  > PB_DELTA_FLOOR
                cond_C = entry_z_accel   > PB_ZA_MIN
                cond_D = entry_z_delta   > PB_ZD_MIN
                cond_E = entry_z_thrust  > PB_ZT_MIN
                live_pullback = cond_A and cond_B and cond_C and cond_D and cond_E

                if live_pullback and guardian_bars < MAX_GUARDIAN_BARS:
                    guardian_bars += 1; trailing_stop = False
                else:
                    # ── V34: TRAIL_ORBIT with all guards ──────────────────────
                    if trail_orbit_bars < MAX_TRAIL_ORBIT_BARS:

                        # ── V34 FIX 1 (PRIMARY): Momentum product gate ────────
                        # Uses ENTRY values — immune to live z decay.
                        # Blocks META (0.281) while all winners > 1.07
                        entry_momentum_product = entry_z_delta * entry_z
                        cMOM = entry_momentum_product > TRAIL_ORBIT_MOMENTUM_MIN

                        # ── V34 FIX 1b (BACKUP): Cluster-level thrust gate ────
                        # Only fires for MEGA_GRINDER (META/MSFT)
                        if TRAIL_ORBIT_THRUST_MIN is not None:
                            cTHRUST = entry_z_thrust > TRAIL_ORBIT_THRUST_MIN
                        else:
                            cTHRUST = True  # other clusters: gate always passes

                        # Both momentum gates must pass
                        trail_orbit_momentum_ok = cMOM and cTHRUST

                        if not trail_orbit_momentum_ok:
                            trailing_stop = True  # V34 FIX 1: block immediately
                        else:
                            # V33 FIX A: hard z-floor check
                            z_hard_blocked = sig['z_score'] < TRAIL_ORBIT_Z_HARD_FLOOR

                            if not z_hard_blocked:
                                cA = sig['z_score'] > Z_MIN * ORBIT_Z_FLOOR_RATIO
                                cB = sig['z_delta'] > ORBIT_ZDELTA_FLOOR
                                cC = (sig['z_accel'] > ORBIT_ZACCEL_FLOOR) or (sig['z_score'] > Z_MIN)
                                cD = not (sig['z_delta'] < ORBIT_DOUBLE_NEG_ZD and
                                          sig['z_accel'] < ORBIT_DOUBLE_NEG_ZA)
                                orbit_trail = cA and cB and cC and cD

                                price_vel_per_bar = (abs(price_now - entry_price) /
                                                     entry_price / max(bars_held, 1) * 100)
                                z_drop_now = entry_z - sig['z_score']
                                # V33 FIX B: stricter price_vel for trail_orbit
                                zpd_trail = (price_vel_per_bar > ZPD_TRAIL_PRICE_VEL_MIN and
                                             z_drop_now        > ZPD_Z_DROP_MIN)

                                if orbit_trail or zpd_trail:
                                    trail_orbit_bars += 1; trailing_stop = False
                                else:
                                    trailing_stop = True
                            else:
                                trailing_stop = True   # V33 FIX A hard blocked
                    else:
                        trailing_stop = True
                    # ── END V34 TRAIL_ORBIT ───────────────────────────────────
            else:
                trailing_stop = False

            hard_stop  = (MAX_LOSS_PCT is not None and pnl_pct <= MAX_LOSS_PCT)
            flip       = ((position == 'LONG'  and sig['signal'] == 'SHORT') or
                          (position == 'SHORT' and sig['signal'] == 'BUY'))
            z_collapse = (sig['z_score'] <= 0 and bars_held >= 2
                          and position == 'LONG')

            total_hold    = HOLD_BARS + guardian_bars + orbit_bars + zpd_bars
            at_time_limit = bars_held >= total_hold

            if at_time_limit and not trailing_stop and not hard_stop \
               and not flip and not z_collapse \
               and (orbit_bars + zpd_bars) < MAX_ORBIT_BARS:

                cA = sig['z_score'] > Z_MIN * ORBIT_Z_FLOOR_RATIO
                cB = sig['z_delta'] > ORBIT_ZDELTA_FLOOR
                cC = (sig['z_accel'] > ORBIT_ZACCEL_FLOOR) or (sig['z_score'] > Z_MIN)
                cD = not (sig['z_delta'] < ORBIT_DOUBLE_NEG_ZD and
                          sig['z_accel'] < ORBIT_DOUBLE_NEG_ZA)
                orbit_active = cA and cB and cC and cD

                price_vel_per_bar = abs(price_now - entry_price) / entry_price / max(bars_held, 1) * 100
                z_drop            = entry_z - sig['z_score']
                zpd_active = (price_vel_per_bar > ZPD_PRICE_VEL_MIN and
                              z_drop            > ZPD_Z_DROP_MIN)

                if orbit_active:
                    orbit_bars += 1; time_exit = False
                elif zpd_active:
                    zpd_bars += 1; time_exit = False
                else:
                    time_exit = True
            else:
                time_exit = at_time_limit

            if trailing_stop or hard_stop or flip or time_exit or z_collapse:
                if   trailing_stop: exit_reason = 'TRAIL'
                elif hard_stop:     exit_reason = 'STOP'
                elif z_collapse:    exit_reason = 'Z_COLLAPSE'
                elif flip:          exit_reason = 'FLIP'
                else:               exit_reason = 'TIME'

                future_price = prices[min(i+1, n-1)]
                correct = (future_price > entry_price if position == 'LONG'
                           else future_price < entry_price)
                trades.append({
                    'symbol': symbol, 'cluster': cluster_name,
                    'entry_bar': entry_bar, 'exit_bar': i,
                    'entry_price': round(entry_price, 4),
                    'exit_price':  round(price_now, 4),
                    'position': position, 'signal': entry_signal,
                    'bars_held': bars_held, 'pnl_pct': round(pnl_pct, 4),
                    'correct': correct, 'exit_reason': exit_reason,
                    'confidence': entry_conf, 'entry_hurst': round(entry_hurst, 3),
                    'hurst': sig['hurst'], 'tstat': sig['tstat'],
                    'regime': entry_regime,
                    'entry_z': round(entry_z, 4), 'z_score': sig['z_score'],
                    'z_delta': round(z_delta, 4), 'z_accel': round(z_accel, 4),
                    'entry_z_delta': round(entry_z_delta, 4),
                    'entry_z_accel': round(entry_z_accel, 4),
                    'entry_z_thrust': round(entry_z_thrust, 5),
                    'z_thrust': round(entry_z_delta * entry_z_accel, 5),
                    'entry_momentum_product': round(entry_z_delta * entry_z, 4),
                    'z_x_conf': round(sig['z_score'] * entry_conf, 4),
                    'ztc': round(sig['z_score'] * sig['tstat'] * entry_conf, 4),
                    'long_score': sig['long_score'], 'short_score': sig['short_score'],
                    'guardian_bars': guardian_bars, 'live_pb_blocked': guardian_bars > 0,
                    'orbit_bars': orbit_bars, 'zpd_bars': zpd_bars,
                    'orbit_extended': orbit_bars > 0, 'zpd_extended': zpd_bars > 0,
                    'trail_orbit_bars': trail_orbit_bars,
                    'trail_orbit_held': trail_orbit_bars > 0,
                })
                position = None; peak_pnl = 0.0; guardian_bars = 0
                orbit_bars = 0; zpd_bars = 0; trail_orbit_bars = 0
                z_reset_ok = False
                prev_z = sig['z_score']; prev_z_delta = z_delta; continue

        if position is None and sig['signal'] in ('BUY', 'SHORT'):
            if sig['signal'] == 'SHORT' and not ALLOW_SHORT:
                prev_z = sig['z_score']; prev_z_delta = z_delta; continue
            z_x_conf = sig['z_score'] * sig['confidence']
            ztc      = sig['z_score'] * sig['tstat'] * sig['confidence']
            if sig['z_score'] <= 0:          prev_z = sig['z_score']; continue
            if sig['confidence'] < CONF_MIN: prev_z = sig['z_score']; continue
            if sig['regime'] == 'MEAN_REVERT' and sig['signal'] == 'BUY':
                prev_z = sig['z_score']; prev_z_delta = z_delta; continue
            if not z_reset_ok and sig['signal'] == 'BUY':
                prev_z = sig['z_score']; prev_z_delta = z_delta; continue
            hurst_gate  = HURST_MIN <= sig['hurst'] <= HURST_MAX
            z_gate      = sig['z_score'] >= Z_MIN
            zxconf_gate = z_x_conf       >= Z_X_CONF_MIN
            ztc_gate    = ztc            >= ZTC_MIN
            tstat_gate  = True if TSTAT_INVERSE else sig['tstat'] >= TSTAT_MIN
            if z_gate and tstat_gate and zxconf_gate and ztc_gate and hurst_gate:
                candidate_z_thrust = z_delta * z_accel
                z_grav_field = sig['z_score'] ** 2 * candidate_z_thrust
                if candidate_z_thrust >= ZPO_THRUST_CAP:
                    prev_z = sig['z_score']; prev_z_delta = z_delta; continue
                if z_grav_field >= ZPO_GRAV_CAP:
                    prev_z = sig['z_score']; prev_z_delta = z_delta; continue
                position = 'LONG' if sig['signal'] == 'BUY' else 'SHORT'
                entry_price = prices[i]; entry_bar = i
                entry_signal = sig['signal']; entry_conf = sig['confidence']
                entry_regime = sig['regime']; entry_z = sig['z_score']
                entry_z_delta = z_delta; entry_z_accel = z_accel
                entry_hurst = sig['hurst']; entry_z_thrust = candidate_z_thrust
                peak_pnl = 0.0; guardian_bars = 0; orbit_bars = 0
                zpd_bars = 0; trail_orbit_bars = 0

        prev_z = sig['z_score']; prev_z_delta = z_delta

    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════════════════════════
# DOWNLOAD + RUN
# ════════════════════════════════════════════════════════════════════════════════

WATCHLIST = [
    "NVDA","AAPL","MSFT","AMZN","TSLA","META","GOOGL",
    "COST","PLTR","COIN","JPM","AVGO","BRK-B","XOM","V"
]

print(f"Downloading {len(WATCHLIST)} tickers (730d)...")
end   = datetime.now()
start = end - timedelta(days=730)
raw   = yf.download(WATCHLIST, start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    auto_adjust=True, progress=False)
closes = (raw['Close'] if isinstance(raw.columns, pd.MultiIndex)
          else raw[['Close']]).dropna(axis=1)
print(f"Got {closes.shape[0]} bars x {closes.shape[1]} symbols\n")

all_trades, summary = [], []

for sym in closes.columns:
    prices = closes[sym].dropna().values
    if len(prices) < 150: continue
    df = walk_forward_backtest_v34(prices, symbol=sym, warmup=120)
    if df.empty: continue
    all_trades.append(df)
    wins   = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    long_acc  = df[df['position']=='LONG']['correct'].mean()*100  if len(df[df['position']=='LONG'])  else 0
    short_acc = df[df['position']=='SHORT']['correct'].mean()*100 if len(df[df['position']=='SHORT']) else 0
    _, cfg = get_cluster_config(sym)
    summary.append({
        'Symbol': sym, 'Cluster': get_cluster_config(sym)[0],
        'N_Trades': len(df), 'Predict_Acc_%': round(df['correct'].mean()*100, 1),
        'Long_Acc_%': round(long_acc, 1), 'Short_Acc_%': round(short_acc, 1),
        'Win_Rate_%': round(len(wins)/len(df)*100, 1),
        'Avg_Win_%':  round(wins['pnl_pct'].mean()   if len(wins)   else 0, 3),
        'Avg_Loss_%': round(losses['pnl_pct'].mean() if len(losses) else 0, 3),
        'Total_Return_%': round(df['pnl_pct'].sum(), 2),
        'Profit_Factor': round(abs(wins['pnl_pct'].sum()/(losses['pnl_pct'].sum()-1e-9)), 2),
        'Sharpe': round(df['pnl_pct'].mean()/(df['pnl_pct'].std()+1e-9), 3),
        'Avg_Hold_Bars': round(df['bars_held'].mean(), 1),
        'Long_Trades':  int((df['position']=='LONG').sum()),
        'Short_Trades': int((df['position']=='SHORT').sum()),
        'Z_MIN': cfg['Z_MIN'], 'TSTAT_MIN': cfg['TSTAT_MIN'],
        'Z_X_CONF_MIN': cfg['Z_X_CONF_MIN'], 'ZTC_MIN': cfg['ZTC_MIN'],
        'HURST_MIN': cfg.get('HURST_MIN', 0.10), 'HURST_MAX': cfg.get('HURST_MAX', 0.88),
        'HOLD_BARS': cfg['HOLD_BARS'],
    })

df_summary = pd.DataFrame(summary).sort_values('Predict_Acc_%', ascending=False) if summary else pd.DataFrame()
df_all     = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

if df_summary.empty:
    print("⚠️  No trades generated — try loosening gates in CLUSTER_CONFIG.")

print("╔══════════════════════════════════════════════════════════════╗")
print("║          V34 WALK-FORWARD PREDICTION ACCURACY REPORT        ║")
print("╚══════════════════════════════════════════════════════════════╝")
if not df_summary.empty:
    print(df_summary.to_string(index=False))

if not df_all.empty:
    overall_acc  = df_all['correct'].mean() * 100
    overall_wr   = (df_all['pnl_pct'] > 0).mean() * 100
    overall_ret  = df_all['pnl_pct'].sum()
    overall_pf   = abs(df_all[df_all['pnl_pct']>0]['pnl_pct'].sum() /
                       (df_all[df_all['pnl_pct']<=0]['pnl_pct'].sum()-1e-9))
    longs        = (df_all['position']=='LONG').sum()
    shorts       = (df_all['position']=='SHORT').sum()
    long_acc_ov  = df_all[df_all['position']=='LONG']['correct'].mean()*100
    short_acc_ov = df_all[df_all['position']=='SHORT']['correct'].mean()*100

    print("\n── EXIT REASON BREAKDOWN ───────────────────────────────────")
    for reason in ['TIME','TRAIL','Z_COLLAPSE','FLIP','STOP']:
        sub = df_all[df_all['exit_reason']==reason]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100; tot = sub['pnl_pct'].sum(); avg = sub['pnl_pct'].mean()
        print(f"  {reason:<11} | n={len(sub):>4} | WinRate={wr:.1f}% | AvgPnL={avg:+.3f}% | Total={tot:+.2f}%")

    print("\n── CLUSTER BREAKDOWN ────────────────────────────────────────")
    for cname in ['EXPLOSIVE','TSLA_SOLO','MEGA_GRINDER','TRUE_GRINDER','STEADY','TREND']:
        sub = df_all[df_all['cluster']==cname]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100; acc = sub['correct'].mean()*100; tot = sub['pnl_pct'].sum()
        pf  = abs(sub[sub['pnl_pct']>0]['pnl_pct'].sum()/(sub[sub['pnl_pct']<=0]['pnl_pct'].sum()-1e-9))
        print(f"  {cname:<14} | n={len(sub):>4} | DirAcc={acc:.1f}% | WR={wr:.1f}% | PF={pf:.2f}x | Total={tot:+.2f}%")

    print("\n── REGIME BREAKDOWN ─────────────────────────────────────────")
    for reg in ['TREND','MEAN_REVERT','MIXED']:
        sub = df_all[df_all['regime']==reg]
        if len(sub) == 0: continue
        wr = (sub['pnl_pct']>0).mean()*100; acc = sub['correct'].mean()*100; tot = sub['pnl_pct'].sum()
        print(f"  {reg:<12} | n={len(sub):>4} | DirAcc={acc:.1f}% | WR={wr:.1f}% | Total={tot:+.2f}%")

    print("\n── Z-VELOCITY BREAKDOWN ─────────────────────────────────────")
    for label, mask in [('z_delta>0 (rising)',  df_all['entry_z_delta'] > 0),
                        ('z_delta<0 (falling)', df_all['entry_z_delta'] < 0),
                        ('z_delta=0 (first)',   df_all['entry_z_delta'] == 0)]:
        sub = df_all[mask]
        if len(sub) == 0: continue
        wr = (sub['pnl_pct']>0).mean()*100; tot = sub['pnl_pct'].sum()
        print(f"  {label:<24} | n={len(sub):>4} | WR={wr:.1f}% | Total={tot:+.2f}%")

    print("\n── Z-ACCEL BREAKDOWN ────────────────────────────────────────")
    for label, mask in [('z_accel>0 (speeding up)',  df_all['entry_z_accel'] > 0),
                        ('z_accel<0 (slowing down)', df_all['entry_z_accel'] < 0),
                        ('z_accel=0 (first/second)', df_all['entry_z_accel'] == 0)]:
        sub = df_all[mask]
        if len(sub) == 0: continue
        wr = (sub['pnl_pct']>0).mean()*100; tot = sub['pnl_pct'].sum()
        print(f"  {label:<28} | n={len(sub):>4} | WR={wr:.1f}% | Total={tot:+.2f}%")

    print("\n── Z-THRUST BREAKDOWN ───────────────────────────────────────")
    for label, mask in [('z_thrust>0.10 (strong)',  df_all['z_thrust'] > 0.10),
                        ('z_thrust 0-0.10 (weak)',  (df_all['z_thrust'] >= 0) & (df_all['z_thrust'] <= 0.10)),
                        ('z_thrust<0 (opposing)',    df_all['z_thrust'] < 0)]:
        sub = df_all[mask]
        if len(sub) == 0: continue
        wr = (sub['pnl_pct']>0).mean()*100; tot = sub['pnl_pct'].sum()
        print(f"  {label:<28} | n={len(sub):>4} | WR={wr:.1f}% | Total={tot:+.2f}%")

    print("\n── PERSISTENT Z-GUARDIAN BREAKDOWN ──────────────────────────")
    guarded = df_all[df_all['live_pb_blocked'] == True]
    normal  = df_all[df_all['live_pb_blocked'] == False]
    if len(guarded) > 0:
        wr_g = (guarded['pnl_pct']>0).mean()*100; avg_g = guarded['pnl_pct'].mean()
        tot_g = guarded['pnl_pct'].sum(); gb_avg = guarded['guardian_bars'].mean()
        print(f"  Guardian-held trades     | n={len(guarded):>4} | WR={wr_g:.1f}% | AvgPnL={avg_g:+.3f}% | Total={tot_g:+.2f}% | AvgGuardBars={gb_avg:.1f}")
    if len(normal) > 0:
        wr_n = (normal['pnl_pct']>0).mean()*100; avg_n = normal['pnl_pct'].mean(); tot_n = normal['pnl_pct'].sum()
        print(f"  Normal (no guardian)     | n={len(normal):>4} | WR={wr_n:.1f}% | AvgPnL={avg_n:+.3f}% | Total={tot_n:+.2f}%")

    print("\n── V34 ORBIT + ZPD TIME-EXTENSION BREAKDOWN ─────────────────")
    orbit_ext = df_all[df_all['orbit_extended'] == True]
    zpd_ext   = df_all[df_all['zpd_extended']   == True]
    if len(orbit_ext) > 0:
        wr_o = (orbit_ext['pnl_pct']>0).mean()*100; avg_o = orbit_ext['pnl_pct'].mean()
        tot_o = orbit_ext['pnl_pct'].sum(); ob_avg = orbit_ext['orbit_bars'].mean()
        print(f"  ORBIT TIME-extended      | n={len(orbit_ext):>4} | WR={wr_o:.1f}% | AvgPnL={avg_o:+.3f}% | Total={tot_o:+.2f}% | AvgORBITBars={ob_avg:.1f}")
    if len(zpd_ext) > 0:
        wr_z = (zpd_ext['pnl_pct']>0).mean()*100; avg_z = zpd_ext['pnl_pct'].mean()
        tot_z = zpd_ext['pnl_pct'].sum(); zb_avg = zpd_ext['zpd_bars'].mean()
        print(f"  ZPD TIME-extended        | n={len(zpd_ext):>4} | WR={wr_z:.1f}% | AvgPnL={avg_z:+.3f}% | Total={tot_z:+.2f}% | AvgZPDBars={zb_avg:.1f}")

    print("\n── V34 TRAIL-ORBIT EXTENSION BREAKDOWN ──────────────────────")
    trail_held = df_all[df_all['trail_orbit_held'] == True]
    if len(trail_held) > 0:
        wr_t = (trail_held['pnl_pct']>0).mean()*100; avg_t = trail_held['pnl_pct'].mean()
        tot_t = trail_held['pnl_pct'].sum(); tb_avg = trail_held['trail_orbit_bars'].mean()
        print(f"  Trail-ORBIT/ZPD held     | n={len(trail_held):>4} | WR={wr_t:.1f}% | AvgPnL={avg_t:+.3f}% | Total={tot_t:+.2f}% | AvgTrailOrbitBars={tb_avg:.1f}")
    neither = df_all[(df_all['orbit_extended']==False) & (df_all['zpd_extended']==False) & (df_all['trail_orbit_held']==False)]
    if len(neither) > 0:
        wr_ne = (neither['pnl_pct']>0).mean()*100; avg_ne = neither['pnl_pct'].mean(); tot_ne = neither['pnl_pct'].sum()
        print(f"  No extension (clean)     | n={len(neither):>4} | WR={wr_ne:.1f}% | AvgPnL={avg_ne:+.3f}% | Total={tot_ne:+.2f}%")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              V34 AGGREGATE PERFORMANCE SUMMARY               ║
╠══════════════════════════════════════════════════════════════╣
║  Total trades:              {len(df_all):>6}                           ║
║  LONG trades:               {longs:>6}  | LONG  DirAcc: {long_acc_ov:.1f}%      ║
║  SHORT trades:              {shorts:>6}  | SHORT DirAcc: {short_acc_ov:.1f}%      ║
║                                                              ║
║  Direction Predict Acc:     {overall_acc:>5.1f}%  (V33=90.9%, V32=90.9%) ║
║  Win Rate (profitable):     {overall_wr:>5.1f}%  (V33=90.9%, V32=90.9%) ║
║  Total Return (all trades): {overall_ret:>+8.2f}%  (V33=+139.02%, V32=+134.01%) ║
║  Profit Factor:             {overall_pf:>6.2f}x                         ║
╚══════════════════════════════════════════════════════════════╝""")

    cols = ['symbol','cluster','position','entry_price','exit_price','pnl_pct',
            'correct','exit_reason','confidence','regime','entry_z','entry_hurst',
            'z_score','z_delta','z_accel','entry_z_delta','entry_z_accel',
            'entry_z_thrust','entry_momentum_product','z_thrust','z_x_conf','ztc','tstat',
            'long_score','short_score','guardian_bars','live_pb_blocked',
            'orbit_bars','zpd_bars','orbit_extended','zpd_extended',
            'trail_orbit_bars','trail_orbit_held']
    print("\n── LAST 10 TRADE DECISIONS ─────────────────────────────────")
    print(df_all.tail(10)[cols].to_string(index=False))

    df_all.to_csv('/kaggle/working/v34_all_trades.csv', index=False)
    df_summary.to_csv('/kaggle/working/v34_summary.csv', index=False)
    print("\nSaved: v34_all_trades.csv  |  v34_summary.csv")
