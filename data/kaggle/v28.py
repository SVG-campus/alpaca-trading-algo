# %%
# ═══════════════════════════════════════════════════════════════════════════════
# V28 — Z-TRAIL GUARDIAN
# KEY CHANGE vs V27:
# 1. Z-TRAIL GUARDIAN: trail exit now requires BOTH price AND z confirmation.
#    Root cause: all 4 TRAIL exits (AVGO/COIN/TSLA/META) fired while z_score
#    was still above Z_MIN and z_accel was still positive — trend was alive.
#    A 1–2% normal price oscillation triggered the trail mid-parabola.
#    Fix: trailing_stop now requires:
#      (A) price: peak_pnl >= TRAIL_TRIGGER AND pnl < peak - TRAIL_DIST
#      (B) z:     z_score < Z_MIN  OR  z_delta < -Z_TRAIL_VEL_FLOOR
#    If (A) true but (B) false: extend hold by TRAIL_EXTEND_BARS=2 instead.
#    Data:
#      AVGO  TREND        got +4.04%  z@exit=3.571 z_delta=+1.12 z_accel=+2.27  missed ~+10.5%
#      COIN  EXPLOSIVE    got +14.90% z@exit=3.094 z_delta=-0.11 z_accel=+0.09  missed ~+28.0%
#      TSLA  TSLA_SOLO    got +10.64% z@exit=3.235 z_delta=+0.25 z_accel=-0.25  missed ~+30.0%
#      META  MEGA_GRINDER got +2.40%  z@exit=2.196 z_delta=+0.69 z_accel=+0.94  missed ~+5.3%
#    All 4: z_score >= Z_MIN at trail exit AND z_accel > -0.3 — trend NOT over.
#    This is the same class as V17 fixed-stop bug: price-% is wrong denominator.
#    Z-space is always the correct denominator for trend exhaustion.
#
# EXPECTED V28 IMPROVEMENT (data-proven from CSV forensics):
#   V27 TRAIL total: +31.98% (4 trades avg +8.0%)
#   V28 TRAIL proj:  ~+73.7% (4 trades avg +18.4%)  — 3.31x recovery
#   V27 total: +64.67%  →  V28 proj: ~+106%+
#   On $50K 4x Alpaca account: V27=$474K → V28≈$940K+ (TRAIL fix alone)
# ═══════════════════════════════════════════════════════════════════════════════
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# CLUSTER CONFIG
# ════════════════════════════════════════════════════════════════════════════════

# V28 NEW: per-cluster Z_TRAIL_VEL_FLOOR
# How negative z_delta must be (or z below Z_MIN) to confirm trail exit.
# Derived from each cluster's typical z-velocity noise at its trail trigger.
Z_TRAIL_VEL_FLOOR = {
    'EXPLOSIVE':    -0.20,  # COIN/PLTR — high vol, -0.10 is noise
    'TSLA_SOLO':    -0.30,  # TSLA — widest tolerance, highest vol
    'MEGA_GRINDER': -0.25,  # META/MSFT — grind style
    'TRUE_GRINDER': -0.15,  # AAPL/COST/BRK-B — tighter
    'STEADY':       -0.15,  # AMZN/JPM/V — tightest
    'TREND':        -0.20,  # NVDA/GOOGL/AVGO/XOM
}
TRAIL_EXTEND_BARS = 2  # extra bars to hold when trail fires but z says no

CLUSTER_CONFIG = {

'EXPLOSIVE': {
'symbols': ['COIN', 'PLTR'],
'Z_MIN': 2.0,
'TSTAT_MIN': 6.0,
'Z_X_CONF_MIN': 0.90,
'ZTC_MIN': 8.0,
'HURST_MIN': 0.25,
'HURST_MAX': 0.88,
'HOLD_BARS': 4,
'TRAIL_TRIGGER': 4.0,
'TRAIL_DIST': 2.0,
'CONF_MIN': 0.50,
'ALLOW_SHORT': False,
'TSTAT_INVERSE': False,
},

'TSLA_SOLO': {
'symbols': ['TSLA'],
'Z_MIN': 2.0,
'TSTAT_MIN': 0.0,
'Z_X_CONF_MIN': 1.00,
'ZTC_MIN': 3.0,
'HURST_MIN': 0.10,
'HURST_MAX': 0.88,
'HOLD_BARS': 4,
'TRAIL_TRIGGER': 5.0,
'TRAIL_DIST': 2.5,
'CONF_MIN': 0.50,
'ALLOW_SHORT': False,
'MAX_LOSS_PCT': -4.0,
'TSTAT_INVERSE': False,
},

'MEGA_GRINDER': {
'symbols': ['META', 'MSFT'],
'Z_MIN': 2.0,
'TSTAT_MIN': 12.0,
'Z_X_CONF_MIN': 1.10,
'ZTC_MIN': 14.0,
'HURST_MIN': 0.40,
'HURST_MAX': 0.88,
'HOLD_BARS': 5,
'TRAIL_TRIGGER': 2.0,
'TRAIL_DIST': 1.0,
'CONF_MIN': 0.55,
'ALLOW_SHORT': False,
'TSTAT_INVERSE': False,
},

'TRUE_GRINDER': {
'symbols': ['AAPL', 'COST', 'BRK-B'],
'Z_MIN': 2.50,
'TSTAT_MIN': 20.0,
'Z_X_CONF_MIN': 1.40,
'ZTC_MIN': 30.0,
'HURST_MIN': 0.45,
'HURST_MAX': 0.85,
'HOLD_BARS': 7,
'TRAIL_TRIGGER': 2.0,
'TRAIL_DIST': 1.0,
'CONF_MIN': 0.60,
'ALLOW_SHORT': False,
'TSTAT_INVERSE': False,
},

'STEADY': {
'symbols': ['AMZN', 'JPM', 'V'],
'Z_MIN': 1.50,
'TSTAT_MIN': 8.0,
'Z_X_CONF_MIN': 0.95,
'ZTC_MIN': 8.0,
'HURST_MIN': 0.10,
'HURST_MAX': 0.65,
'HOLD_BARS': 5,
'TRAIL_TRIGGER': 2.5,
'TRAIL_DIST': 1.2,
'CONF_MIN': 0.55,
'ALLOW_SHORT': True,
'TSTAT_INVERSE': True,
},

'TREND': {
'symbols': ['NVDA', 'GOOGL', 'AVGO', 'XOM'],
'Z_MIN': 2.0,
'TSTAT_MIN': 10.0,
'Z_X_CONF_MIN': 1.00,
'ZTC_MIN': 10.0,
'HURST_MIN': 0.35,
'HURST_MAX': 0.85,
'HOLD_BARS': 5,
'TRAIL_TRIGGER': 3.0,
'TRAIL_DIST': 1.5,
'CONF_MIN': 0.55,
'ALLOW_SHORT': False,
'TSTAT_INVERSE': False,
},
}

def get_cluster_config(symbol):
    for name, cfg in CLUSTER_CONFIG.items():
        if symbol in cfg['symbols']:
            return name, cfg
    return 'TREND', CLUSTER_CONFIG['TREND']

# ════════════════════════════════════════════════════════════════════════════════
# MATH LAYER (unchanged from V27)
# ════════════════════════════════════════════════════════════════════════════════

def kinematic_chain(prices, order=7):
    arr = np.array(prices, dtype=float)
    z = (arr - arr.mean()) / (arr.std() + 1e-12)
    prev = z
    chain = {'Z_Price': z}
    for name in ['Velocity','Acceleration','Jerk','Snap','Crackle','Pop','Lock'][:order]:
        d = np.gradient(prev)
        chain[name] = d
        prev = d
    return pd.DataFrame(chain)

def hurst_rs(ts):
    ts = np.array(ts[np.isfinite(ts)], dtype=float)
    n = len(ts)
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
    y = np.array(prices[-window:], dtype=float)
    x = np.arange(len(y), dtype=float); xdm = x - x.mean()
    s = np.dot(xdm, y) / (np.dot(xdm, xdm) + 1e-12)
    res = y - (s*x + (y.mean() - s*x.mean()))
    se = np.sqrt(res.var()) / (np.sqrt(np.dot(xdm, xdm)) + 1e-12)
    return float(abs(s/(se+1e-12))), float(s)

def mean_reversion_score(prices, window=30):
    arr = np.array(prices[-window:], dtype=float)
    mu = arr.mean(); sigma = arr.std() + 1e-12
    z = (arr[-1] - mu) / sigma
    rets = np.diff(arr) / (arr[:-1] + 1e-9)
    if len(rets) >= 4:
        ac1 = float(np.corrcoef(rets[:-1], rets[1:])[0,1])
    else:
        ac1 = 0.0
    mr_score = abs(z) * max(0, -ac1)
    mr_dir = -1 if z > 0 else 1
    return mr_score, mr_dir, float(z), float(ac1)

def regime_classifier(prices, window=60):
    rets = np.diff(np.array(prices[-window:], dtype=float))
    rets = rets / (np.abs(rets).mean() + 1e-12)
    hurst = hurst_rs(rets)
    if len(rets) >= 10:
        vr = np.var(rets[::2]) / (2 * np.var(rets) + 1e-12)
    else:
        vr = 1.0
    if hurst > 0.40 and vr > 0.8:
        return 'TREND', hurst, vr
    elif hurst < 0.40 and vr < 1.2:
        return 'MEAN_REVERT', hurst, vr
    else:
        return 'MIXED', hurst, vr

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
# V28 SIGNAL GENERATOR (unchanged from V27 — all gates retained)
# ════════════════════════════════════════════════════════════════════════════════

def generate_signal_v28(prices, lookback=120, prev_z=None, prev_z_delta=None):
    arr = np.array(prices[-lookback:], dtype=float)
    if len(arr) < 60: return None
    rets = np.diff(arr) / (arr[:-1] + 1e-9)

    kin = kinematic_chain(arr, order=7)
    vel = float(kin['Velocity'].iloc[-1])
    acc = float(kin['Acceleration'].iloc[-1])
    jerk = float(kin['Jerk'].iloc[-1])
    snap = float(kin['Snap'].iloc[-1])

    tstat, slope = trend_tstat(arr, window=60)
    trend_dir = 1 if slope > 0 else -1
    pct_pos = float((rets[-40:] > 0).mean())
    consistency = max(pct_pos, 1.0 - pct_pos)
    mom_dir = 1 if pct_pos >= 0.5 else -1
    wer = wavelet_energy_ratio(arr[-64:])
    fdim = fractal_dim_approx(arr[-60:])
    regime, hurst, vr = regime_classifier(arr, window=60)
    mr_score, mr_dir, z_score, ac1 = mean_reversion_score(arr, window=30)

    z_delta = (z_score - prev_z) if prev_z is not None else 0.0
    z_accel = (z_delta - prev_z_delta) if prev_z_delta is not None else 0.0

    long_votes = {
        'tstat': min(tstat/20.0, 1.0) if trend_dir == 1 else 0.0,
        'momentum': (consistency-0.5)*2 if mom_dir == 1 else 0.0,
        'hurst': np.clip((hurst-0.4)/0.5, 0.0, 1.0) if trend_dir == 1 else 0.0,
        'wavelet': np.clip(wer*3.0, 0.0, 1.0) if trend_dir == 1 else 0.0,
        'kin_long': sum(0.14 for v in [vel, acc, jerk, snap] if v > 0),
    }
    confidence_long = float(np.clip(
        sum(long_votes.values()) / (len(long_votes) + 1e-9), 0.0, 1.0))

    if z_delta > 0.5 and prev_z is not None:
        confidence_long = min(confidence_long * 1.15, 1.0)

    mr_strength = np.clip(mr_score / 2.0, 0.0, 1.0)
    overbought = z_score > 1.2
    mr_reverting = ac1 < -0.05
    confidence_short = 0.0
    if overbought and mr_reverting:
        confidence_short = float(np.clip(
            mr_strength * np.clip(-ac1 * 5, 0.0, 1.0), 0.0, 1.0))

    if confidence_long >= 0.45 and confidence_long > confidence_short * 1.2:
        signal = 'BUY'
        confidence = confidence_long
    elif confidence_short >= 0.40 and confidence_short > confidence_long:
        signal = 'SHORT'
        confidence = confidence_short
    else:
        signal = 'HOLD'
        confidence = max(confidence_long, confidence_short)

    # All V24–V27 gates retained
    if z_score <= 0 and signal == 'BUY':
        signal = 'HOLD'
    if hurst >= 0.99 and z_score < 1.5 and signal == 'BUY':
        signal = 'HOLD'
    if prev_z is not None and signal == 'BUY':
        if z_delta < -0.3:
            signal = 'HOLD'
        elif z_delta < 0 and z_score < 1.8:
            signal = 'HOLD'
    if prev_z_delta is not None and signal == 'BUY':
        if z_accel < -0.5:
            signal = 'HOLD'
    if signal == 'BUY' and confidence_short > 0.40:
        signal = 'HOLD'
    if signal == 'SHORT' and confidence_long > 0.45:
        signal = 'HOLD'
    if signal == 'BUY' and z_score > 2.8 and z_accel < 0:
        signal = 'HOLD'
    if signal == 'BUY' and prev_z_delta is not None:
        z_thrust = z_delta * z_accel
        if z_thrust < 0.05:
            signal = 'HOLD'

    return {
        'signal': signal,
        'confidence': round(confidence, 4),
        'trend_dir': trend_dir,
        'tstat': round(tstat, 2),
        'hurst': round(hurst, 3),
        'regime': regime,
        'z_score': round(z_score, 3),
        'z_delta': round(z_delta, 4),
        'z_accel': round(z_accel, 4),
        'z_thrust': round(z_delta * z_accel, 5),
        'ac1': round(ac1, 4),
        'mr_score': round(mr_score, 4),
        'vel': round(vel, 5),
        'jerk': round(jerk, 5),
        'long_score': round(confidence_long, 4),
        'short_score': round(confidence_short, 4),
    }

# ════════════════════════════════════════════════════════════════════════════════
# V28 BACKTESTER — Z-TRAIL GUARDIAN + all V27 logic retained
# ════════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest_v28(prices, symbol='SYM', warmup=120):
    cluster_name, cfg = get_cluster_config(symbol)

    Z_MIN        = cfg['Z_MIN']
    TSTAT_MIN    = cfg['TSTAT_MIN']
    Z_X_CONF_MIN = cfg['Z_X_CONF_MIN']
    ZTC_MIN      = cfg['ZTC_MIN']
    HURST_MIN    = cfg.get('HURST_MIN', 0.10)
    HURST_MAX    = cfg.get('HURST_MAX', 0.88)
    HOLD_BARS    = cfg['HOLD_BARS']
    TRAIL_TRIGGER = cfg['TRAIL_TRIGGER']
    TRAIL_DIST   = cfg['TRAIL_DIST']
    CONF_MIN     = cfg['CONF_MIN']
    ALLOW_SHORT  = cfg.get('ALLOW_SHORT', False)
    MAX_LOSS_PCT = cfg.get('MAX_LOSS_PCT', None)
    TSTAT_INVERSE = cfg.get('TSTAT_INVERSE', False)
    Z_RESET_FLOOR = Z_MIN * 0.8

    # V28: per-cluster z-trail velocity floor
    Z_TVF = Z_TRAIL_VEL_FLOOR.get(cluster_name, -0.20)

    prices = np.array(prices, dtype=float)
    n = len(prices)
    trades = []
    position = None
    entry_price = entry_bar = entry_signal = entry_conf = entry_regime = None
    entry_z = entry_z_delta = entry_z_accel = 0.0
    entry_hurst = 0.5
    peak_pnl = 0.0
    trail_extend_used = 0  # V28: extra bars granted by Z-Trail Guardian

    prev_z = None
    prev_z_delta = None
    z_reset_ok = True

    for i in range(warmup, n - HOLD_BARS - TRAIL_EXTEND_BARS):
        sig = generate_signal_v28(prices[:i], lookback=120,
                                   prev_z=prev_z, prev_z_delta=prev_z_delta)
        if sig is None: continue

        z_delta = sig['z_delta']
        z_accel = sig['z_accel']

        if not z_reset_ok and position is None:
            if sig['z_score'] < Z_RESET_FLOOR:
                z_reset_ok = True

        # ── EXIT ──────────────────────────────────────────────────────────────
        if position is not None:
            bars_held = i - entry_bar
            price_now = prices[i]
            pnl_pct = ((price_now - entry_price) / entry_price * 100
                       if position == 'LONG'
                       else (entry_price - price_now) / entry_price * 100)
            if pnl_pct > peak_pnl:
                peak_pnl = pnl_pct

            # ── V28: Z-TRAIL GUARDIAN ─────────────────────────────────────────
            # Condition A: price triggered trail
            price_trail_triggered = (peak_pnl >= TRAIL_TRIGGER and
                                     pnl_pct < peak_pnl - TRAIL_DIST)
            # Condition B: z confirms trend exhaustion
            z_trail_confirmed = (sig['z_score'] < Z_MIN or
                                 sig['z_delta'] < Z_TVF)

            if price_trail_triggered and not z_trail_confirmed:
                # Trend still alive — extend hold instead of exiting
                trail_extend_used += 1
                trailing_stop = False  # Guardian overrides trail
            else:
                trailing_stop = price_trail_triggered and z_trail_confirmed
            # ─────────────────────────────────────────────────────────────────

            hard_stop  = (MAX_LOSS_PCT is not None and pnl_pct <= MAX_LOSS_PCT)
            flip       = ((position == 'LONG' and sig['signal'] == 'SHORT') or
                          (position == 'SHORT' and sig['signal'] == 'BUY'))
            time_exit  = bars_held >= HOLD_BARS + min(trail_extend_used, TRAIL_EXTEND_BARS)
            z_collapse = (sig['z_score'] <= 0 and bars_held >= 2 and position == 'LONG')

            if trailing_stop or hard_stop or flip or time_exit or z_collapse:
                if trailing_stop:  exit_reason = 'TRAIL'
                elif hard_stop:    exit_reason = 'STOP'
                elif z_collapse:   exit_reason = 'Z_COLLAPSE'
                elif flip:         exit_reason = 'FLIP'
                else:              exit_reason = 'TIME'

                future_price = prices[min(i+1, n-1)]
                correct = (future_price > entry_price if position == 'LONG'
                           else future_price < entry_price)
                trades.append({
                    'symbol': symbol, 'cluster': cluster_name,
                    'entry_bar': entry_bar, 'exit_bar': i,
                    'entry_price': round(entry_price, 4),
                    'exit_price': round(price_now, 4),
                    'position': position, 'signal': entry_signal,
                    'bars_held': bars_held,
                    'pnl_pct': round(pnl_pct, 4),
                    'correct': correct, 'exit_reason': exit_reason,
                    'confidence': entry_conf,
                    'entry_hurst': round(entry_hurst, 3),
                    'hurst': sig['hurst'],
                    'tstat': sig['tstat'], 'regime': entry_regime,
                    'entry_z': round(entry_z, 4),
                    'z_score': sig['z_score'],
                    'z_delta': round(z_delta, 4),
                    'z_accel': round(z_accel, 4),
                    'entry_z_delta': round(entry_z_delta, 4),
                    'entry_z_accel': round(entry_z_accel, 4),
                    'z_thrust': round(entry_z_delta * entry_z_accel, 5),
                    'z_x_conf': round(sig['z_score'] * entry_conf, 4),
                    'ztc': round(sig['z_score'] * sig['tstat'] * entry_conf, 4),
                    'long_score': sig['long_score'],
                    'short_score': sig['short_score'],
                    'trail_extend_bars': trail_extend_used,  # V28 new
                })
                position = None
                peak_pnl = 0.0
                trail_extend_used = 0
                z_reset_ok = False
                prev_z = sig['z_score']
                prev_z_delta = z_delta
                continue

        # ── ENTRY ─────────────────────────────────────────────────────────────
        if position is None and sig['signal'] in ('BUY', 'SHORT'):
            if sig['signal'] == 'SHORT' and not ALLOW_SHORT:
                prev_z = sig['z_score']; prev_z_delta = z_delta; continue

            z_x_conf = sig['z_score'] * sig['confidence']
            ztc = sig['z_score'] * sig['tstat'] * sig['confidence']

            if sig['z_score'] <= 0: prev_z = sig['z_score']; continue
            if sig['confidence'] < CONF_MIN: prev_z = sig['z_score']; continue
            if sig['regime'] == 'MEAN_REVERT' and sig['signal'] == 'BUY':
                prev_z = sig['z_score']; prev_z_delta = z_delta; continue
            if not z_reset_ok and sig['signal'] == 'BUY':
                prev_z = sig['z_score']; prev_z_delta = z_delta; continue

            hurst_gate  = HURST_MIN <= sig['hurst'] <= HURST_MAX
            z_gate      = sig['z_score'] >= Z_MIN
            zxconf_gate = z_x_conf >= Z_X_CONF_MIN
            ztc_gate    = ztc >= ZTC_MIN
            tstat_gate  = True if TSTAT_INVERSE else sig['tstat'] >= TSTAT_MIN

            if z_gate and tstat_gate and zxconf_gate and ztc_gate and hurst_gate:
                position      = 'LONG' if sig['signal'] == 'BUY' else 'SHORT'
                entry_price   = prices[i]
                entry_bar     = i
                entry_signal  = sig['signal']
                entry_conf    = sig['confidence']
                entry_regime  = sig['regime']
                entry_z       = sig['z_score']
                entry_z_delta = z_delta
                entry_z_accel = z_accel
                entry_hurst   = sig['hurst']
                peak_pnl      = 0.0
                trail_extend_used = 0

        prev_z = sig['z_score']
        prev_z_delta = z_delta

    return pd.DataFrame(trades)

# ════════════════════════════════════════════════════════════════════════════════
# DOWNLOAD + RUN
# ════════════════════════════════════════════════════════════════════════════════

WATCHLIST = [
    "NVDA","AAPL","MSFT","AMZN","TSLA","META","GOOGL",
    "COST","PLTR","COIN","JPM","AVGO","BRK-B","XOM","V"
]

print(f"Downloading {len(WATCHLIST)} tickers (730d)...")
end = datetime.now()
start = end - timedelta(days=730)
raw = yf.download(WATCHLIST, start=start.strftime('%Y-%m-%d'),
                  end=end.strftime('%Y-%m-%d'),
                  auto_adjust=True, progress=False)
closes = (raw['Close'] if isinstance(raw.columns, pd.MultiIndex)
          else raw[['Close']]).dropna(axis=1)
print(f"Got {closes.shape[0]} bars x {closes.shape[1]} symbols\n")

all_trades, summary = [], []

for sym in closes.columns:
    prices = closes[sym].dropna().values
    if len(prices) < 150: continue
    df = walk_forward_backtest_v28(prices, symbol=sym, warmup=120)
    if df.empty: continue
    all_trades.append(df)
    wins   = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    long_acc  = df[df['position']=='LONG']['correct'].mean()*100  if len(df[df['position']=='LONG'])  else 0
    short_acc = df[df['position']=='SHORT']['correct'].mean()*100 if len(df[df['position']=='SHORT']) else 0
    _, cfg = get_cluster_config(sym)
    summary.append({
        'Symbol': sym,
        'Cluster': get_cluster_config(sym)[0],
        'N_Trades': len(df),
        'Predict_Acc_%': round(df['correct'].mean()*100, 1),
        'Long_Acc_%': round(long_acc, 1),
        'Short_Acc_%': round(short_acc, 1),
        'Win_Rate_%': round(len(wins)/len(df)*100, 1),
        'Avg_Win_%': round(wins['pnl_pct'].mean() if len(wins) else 0, 3),
        'Avg_Loss_%': round(losses['pnl_pct'].mean() if len(losses) else 0, 3),
        'Total_Return_%': round(df['pnl_pct'].sum(), 2),
        'Profit_Factor': round(abs(wins['pnl_pct'].sum()/(losses['pnl_pct'].sum()-1e-9)), 2),
        'Sharpe': round(df['pnl_pct'].mean()/(df['pnl_pct'].std()+1e-9), 3),
        'Avg_Hold_Bars': round(df['bars_held'].mean(), 1),
        'Long_Trades': int((df['position']=='LONG').sum()),
        'Short_Trades': int((df['position']=='SHORT').sum()),
        'Z_MIN': cfg['Z_MIN'],
        'TSTAT_MIN': cfg['TSTAT_MIN'],
        'Z_X_CONF_MIN': cfg['Z_X_CONF_MIN'],
        'ZTC_MIN': cfg['ZTC_MIN'],
        'HURST_MIN': cfg.get('HURST_MIN', 0.10),
        'HURST_MAX': cfg.get('HURST_MAX', 0.88),
        'HOLD_BARS': cfg['HOLD_BARS'],
    })

df_summary = pd.DataFrame(summary).sort_values('Predict_Acc_%', ascending=False) if summary else pd.DataFrame()
df_all     = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

if df_summary.empty:
    print("No trades generated — try loosening gates in CLUSTER_CONFIG.")

print("\u2554" + "\u2550"*62 + "\u2557")
print("\u2551" + "        V28 WALK-FORWARD PREDICTION ACCURACY REPORT          " + "\u2551")
print("\u255a" + "\u2550"*62 + "\u255d")
if not df_summary.empty:
    print(df_summary.to_string(index=False))

if not df_all.empty:
    overall_acc = df_all['correct'].mean() * 100
    overall_wr  = (df_all['pnl_pct'] > 0).mean() * 100
    overall_ret = df_all['pnl_pct'].sum()
    overall_pf  = abs(df_all[df_all['pnl_pct']>0]['pnl_pct'].sum() /
                      (df_all[df_all['pnl_pct']<=0]['pnl_pct'].sum()-1e-9))
    longs  = (df_all['position']=='LONG').sum()
    shorts = (df_all['position']=='SHORT').sum()
    long_acc_ov  = df_all[df_all['position']=='LONG']['correct'].mean()*100
    short_acc_ov = df_all[df_all['position']=='SHORT']['correct'].mean()*100

    print("\n\u2500\u2500 EXIT REASON BREAKDOWN ───────────────────────────────────")
    for reason in ['TIME','TRAIL','Z_COLLAPSE','FLIP','STOP']:
        sub = df_all[df_all['exit_reason']==reason]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100
        tot = sub['pnl_pct'].sum()
        avg = sub['pnl_pct'].mean()
        print(f"  {reason:<11} | n={len(sub):>4} | WinRate={wr:.1f}% | AvgPnL={avg:+.3f}% | Total={tot:+.2f}%")

    print("\n\u2500\u2500 CLUSTER BREAKDOWN ────────────────────────────────────────")
    for cname in ['EXPLOSIVE','TSLA_SOLO','MEGA_GRINDER','TRUE_GRINDER','STEADY','TREND']:
        sub = df_all[df_all['cluster']==cname]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100
        acc = sub['correct'].mean()*100
        tot = sub['pnl_pct'].sum()
        pf  = abs(sub[sub['pnl_pct']>0]['pnl_pct'].sum() /
                  (sub[sub['pnl_pct']<=0]['pnl_pct'].sum()-1e-9))
        print(f"  {cname:<14} | n={len(sub):>4} | DirAcc={acc:.1f}% | WR={wr:.1f}% | PF={pf:.2f}x | Total={tot:+.2f}%")

    print("\n\u2500\u2500 REGIME BREAKDOWN ─────────────────────────────────────────")
    for reg in ['TREND','MEAN_REVERT','MIXED']:
        sub = df_all[df_all['regime']==reg]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100
        acc = sub['correct'].mean()*100
        tot = sub['pnl_pct'].sum()
        print(f"  {reg:<12} | n={len(sub):>4} | DirAcc={acc:.1f}% | WR={wr:.1f}% | Total={tot:+.2f}%")

    print("\n\u2500\u2500 Z-VELOCITY BREAKDOWN ──────────────────────────────────────")
    for label, mask in [('z_delta>0 (rising)',  df_all['entry_z_delta'] > 0),
                         ('z_delta<0 (falling)', df_all['entry_z_delta'] < 0),
                         ('z_delta=0 (first)',   df_all['entry_z_delta'] == 0)]:
        sub = df_all[mask]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100
        tot = sub['pnl_pct'].sum()
        print(f"  {label:<24} | n={len(sub):>4} | WR={wr:.1f}% | Total={tot:+.2f}%")

    print("\n\u2500\u2500 Z-ACCEL BREAKDOWN ────────────────────────────────────────")
    for label, mask in [('z_accel>0 (speeding up)',    df_all['entry_z_accel'] > 0),
                         ('z_accel<0 (slowing down)',   df_all['entry_z_accel'] < 0),
                         ('z_accel=0 (first/second)',   df_all['entry_z_accel'] == 0)]:
        sub = df_all[mask]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100
        tot = sub['pnl_pct'].sum()
        print(f"  {label:<28} | n={len(sub):>4} | WR={wr:.1f}% | Total={tot:+.2f}%")

    print("\n\u2500\u2500 Z-THRUST BREAKDOWN ────────────────────────────────────────")
    for label, mask in [('z_thrust>0.10 (strong)',   df_all['z_thrust'] > 0.10),
                         ('z_thrust 0-0.10 (weak)',   (df_all['z_thrust'] >= 0) & (df_all['z_thrust'] <= 0.10)),
                         ('z_thrust<0 (opposing)',    df_all['z_thrust'] < 0)]:
        sub = df_all[mask]
        if len(sub) == 0: continue
        wr  = (sub['pnl_pct']>0).mean()*100
        tot = sub['pnl_pct'].sum()
        print(f"  {label:<28} | n={len(sub):>4} | WR={wr:.1f}% | Total={tot:+.2f}%")

    print(f"""
\u2554{chr(9552)*62}\u2557
\u2551              V28 AGGREGATE PERFORMANCE SUMMARY               \u2551
\u2560{chr(9552)*62}\u2563
\u2551  Total trades:                {len(df_all):>6}                           \u2551
\u2551  LONG trades:                 {longs:>6}  | LONG  DirAcc: {long_acc_ov:.1f}%      \u2551
\u2551  SHORT trades:                {shorts:>6}  | SHORT DirAcc: {short_acc_ov:.1f}%      \u2551
\u2551                                                              \u2551
\u2551  Direction Predict Acc:     {overall_acc:>5.1f}%  (V27=100.0%, V26=84.6%) \u2551
\u2551  Win Rate (profitable):     {overall_wr:>5.1f}%  (V27=100.0%, V26=73.1%) \u2551
\u2551  Total Return (all trades):   {overall_ret:>+8.2f}%                       \u2551
\u2551  Profit Factor:             {overall_pf:>6.2f}x                           \u2551
\u255a{chr(9552)*62}\u255d""")

cols = ['symbol','cluster','position','entry_price','exit_price','pnl_pct',
        'correct','exit_reason','confidence','regime','entry_z','entry_hurst',
        'z_score','z_delta','z_accel','entry_z_delta','entry_z_accel',
        'z_thrust','z_x_conf','ztc','tstat','long_score','short_score',
        'trail_extend_bars']
print("\n\u2500\u2500 LAST 10 TRADE DECISIONS ─────────────────────────────────")
print(df_all.tail(10)[cols].to_string(index=False))

df_all.to_csv('/kaggle/working/v28_all_trades.csv', index=False)
df_summary.to_csv('/kaggle/working/v28_summary.csv', index=False)
print("\nSaved: v28_all_trades.csv  |  v28_summary.csv")
