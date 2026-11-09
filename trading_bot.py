"""
Self-improving stock trading bot (paper trading first).

How the "self-improving" part works (walk-forward optimization):
  1. Take a rolling TRAIN window of history; grid-search strategy parameters on it.
  2. Apply the best parameters to the NEXT unseen TEST window; record that result.
  3. Roll forward and repeat. Only out-of-sample results count.
  4. Strategies whose out-of-sample Sharpe is poor get dropped; the best one trades.

Strategies: moving-average crossover, momentum, mean reversion, plus two
ICT-inspired ones (fair value gap fill, liquidity sweep) on daily bars.

Usage:
  pip install pandas numpy yfinance alpaca-py
  python trading_bot.py backtest --symbol SPY            # walk-forward backtest
  python trading_bot.py backtest --synthetic             # offline test, fake data
  python trading_bot.py paper --symbol SPY               # one live PAPER trading step

For paper mode set env vars ALPACA_KEY and ALPACA_SECRET (paper account keys).
Not financial advice. Most retail bots lose to buy-and-hold. Paper trade for weeks first.
"""
import argparse
import itertools
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
COST_PER_TRADE = 0.0005      # 5 bps per position change (fees + slippage)
TRAIN_DAYS = 504             # ~2 years
TEST_DAYS = 63               # ~1 quarter
WARMUP_DAYS = 250            # history before each test window so indicators are defined
MIN_OOS_SHARPE = 0.3         # strategies below this are not allowed to trade
RISK_PER_TRADE = 0.01        # hitting the stop loss should lose about 1% of equity
MAX_POSITION_PCT = 0.15      # but never put more than 15% of equity in one stock
STOP_LOSS = 0.08             # exit if price falls 8% from entry
DAILY_KILL_SWITCH = -0.03    # stop trading for the day if equity drops 3%
TRAIL_OPTIONS = [None, 0.05, 0.08, 0.12]    # trailing stop distances the bot may try (None = off)
# Scaled take-profit ladder: (gain from entry, fraction of the ORIGINAL position to sell).
# +10% sell half, +15% sell another quarter, +20% sell the final quarter.
TAKE_PROFIT_LADDER = [(0.10, 0.50), (0.15, 0.25), (0.20, 0.25)]
TAKE_PROFIT_OPTIONS = [None, TAKE_PROFIT_LADDER]   # bot tests with and without the ladder
# Stock discovery ("--auto" mode)
CORE_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
             "XLI", "XLU", "XLB", "XLC", "GLD", "TLT"]
MIN_PRICE = 10.0             # ignore penny stocks
MIN_DOLLAR_VOLUME = 30e6     # ignore stocks that trade less than $30M a day
SCAN_TOP_N = 60              # how many of the most-traded stocks get fully tested each run
MAX_HOLDINGS = 5             # hold at most this many stocks at once
TOTAL_EXPOSURE = 0.65        # at most this share of equity invested across ALL picks
# A stock is only picked if its best strategy clears ALL of these on unseen data.
# They are deliberately strict: testing many stocks means some pass by pure luck.
PICK_MIN_SHARPE = 0.8        # risk-adjusted return, and it must also beat buy-and-hold's Sharpe
PICK_MIN_T = 3.0             # statistical confidence (Sharpe x sqrt(years)); ~3 is a common bar
                             # when ~60 stocks x 5 strategies are being tested at once
PICK_MIN_WIN_RATE = 0.60     # share of invested quarters that made money
PICK_MIN_ACTIVE_QUARTERS = 12  # needs at least this many invested quarters of evidence
PICK_MAX_WORST_QUARTER = -0.15 # no single quarter may lose more than 15%
PICK_DD_VS_HOLD = 0.80       # max drawdown must be at most 80% of buy-and-hold's
MIN_EXIT_GAIN = 0.10         # an exit rule must beat the baseline Sharpe by this much in training


# ----------------------------------------------------------------------------
# Data: a DataFrame with Open, High, Low, Close columns, indexed by date
# ----------------------------------------------------------------------------
def load_prices(symbol, years=10):
    import yfinance as yf
    df = yf.download(symbol, period=f"{years}y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def synthetic_prices(n=2500, seed=0):
    """Random walk with regime changes, for offline testing."""
    rng = np.random.default_rng(seed)
    drift = np.where((np.arange(n) // 250) % 2 == 0, 0.0006, -0.0002)
    rets = drift + rng.normal(0, 0.011, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


# ----------------------------------------------------------------------------
# Strategies: each takes the OHLC DataFrame and returns a position series in {0, 1}.
# Signals use data up to day t and are applied on day t+1 (no lookahead).
# ----------------------------------------------------------------------------
def ma_crossover(df, fast, slow):
    c = df["Close"]
    if fast >= slow:
        return pd.Series(0, index=df.index)
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(int)


def momentum(df, lookback, _unused=0):
    return (df["Close"].pct_change(lookback) > 0).astype(int)


def mean_reversion(df, window, z_entry):
    c = df["Close"]
    z = (c - c.rolling(window).mean()) / c.rolling(window).std()
    pos = pd.Series(np.nan, index=df.index)
    pos[z < -z_entry] = 1      # enter when oversold
    pos[z > 0] = 0             # exit back at the mean
    return pos.ffill().fillna(0).astype(int)


def ict_fvg(df, min_gap, hold):
    """
    ICT-inspired Fair Value Gap, long side only, daily bars.
      - Bullish FVG: today's low is above the high from two days ago, and the middle
        candle was bullish. The gap (high[t-2] .. low[t]) is the zone.
      - Entry: price later trades back down into the zone but closes above its bottom.
      - Exit: close below the zone bottom (invalidated) or after `hold` days.
    min_gap is the minimum gap size as a fraction of price (filters tiny gaps).
    """
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    n = len(df)
    pos = np.zeros(n, dtype=int)
    zones = []                 # each zone: [bottom, top, created_index]
    in_pos, stop, entry_i = False, 0.0, 0
    for i in range(2, n):
        # Manage open position first.
        if in_pos:
            if c[i] < stop or i - entry_i >= hold:
                in_pos = False
            else:
                pos[i] = 1
        # Detect a new bullish FVG.
        if l[i] > h[i - 2] and c[i - 1] > o[i - 1] and (l[i] - h[i - 2]) / c[i] >= min_gap:
            zones.append([h[i - 2], l[i], i])
        # Check zones created before today.
        keep = []
        for z in zones:
            bottom, top, born = z
            if born == i:
                keep.append(z)
            elif c[i] < bottom or i - born > 60:   # invalidated or too old
                continue
            elif l[i] <= top and not in_pos:       # price tapped the zone and held
                in_pos, stop, entry_i = True, bottom, i
                pos[i] = 1
            else:
                keep.append(z)
        zones = keep
    return pd.Series(pos, index=df.index)


def ict_liquidity_sweep(df, lookback, hold):
    """
    ICT-inspired sell-side liquidity sweep, long side only, daily bars.
      - Today's low takes out the lowest low of the previous `lookback` days (the stops
        resting below it), but the day closes back above that old low.
      - Enter long; exit if the close falls below today's low or after `hold` days.
    """
    h, l, c = (df[k].to_numpy() for k in ("High", "Low", "Close"))
    n = len(df)
    pos = np.zeros(n, dtype=int)
    in_pos, stop, entry_i = False, 0.0, 0
    for i in range(lookback, n):
        if in_pos:
            if c[i] < stop or i - entry_i >= hold:
                in_pos = False
            else:
                pos[i] = 1
        if not in_pos:
            prior_low = l[i - lookback:i].min()
            if l[i] < prior_low and c[i] > prior_low:
                in_pos, stop, entry_i = True, l[i], i
                pos[i] = 1
    return pd.Series(pos, index=df.index)


STRATEGIES = {
    "ma_crossover": (ma_crossover, {"fast": [10, 20, 50], "slow": [50, 100, 200]}),
    "momentum": (momentum, {"lookback": [20, 60, 120, 200], "_unused": [0]}),
    "mean_reversion": (mean_reversion, {"window": [10, 20, 40], "z_entry": [1.0, 1.5, 2.0]}),
    "ict_fvg": (ict_fvg, {"min_gap": [0.002, 0.005, 0.01], "hold": [5, 10, 20]}),
    "ict_liq_sweep": (ict_liquidity_sweep, {"lookback": [10, 20, 40], "hold": [3, 5, 10]}),
}


# ----------------------------------------------------------------------------
# Exit rules, layered on top of any strategy's signal
# ----------------------------------------------------------------------------
def apply_exits(df, signal, stop=STOP_LOSS, trail=None, take_profit=None):
    """
    Exit rules layered on any strategy's signal. Returns the fraction of the full
    position held each day (0 to 1). Measured from the entry close:
      - stop: sell everything if price falls `stop` below entry (on by default),
      - trail: sell everything if price falls `trail` below its highest close since entry,
      - take_profit: a ladder [(gain, fraction), ...]; each time price first reaches
        entry * (1 + gain), sell that fraction of the original position.
    After a full forced exit the bot stays out until the strategy's own signal goes flat
    and then fires again, so it doesn't instantly re-buy the same trade.
    Pass None to turn the trail or take-profit off.
    """
    c = df["Close"].to_numpy()
    sig = signal.to_numpy()
    out = np.zeros(len(c))
    in_trade, blocked = False, False
    entry = peak = 0.0
    remaining = 1.0
    taken = set()
    for i in range(len(c)):
        if sig[i] == 0:
            in_trade, blocked = False, False
            continue
        if blocked:
            continue
        if not in_trade:
            in_trade, entry, peak, remaining, taken = True, c[i], c[i], 1.0, set()
        else:
            peak = max(peak, c[i])
        if ((stop is not None and c[i] <= entry * (1 - stop))
                or (trail is not None and c[i] <= peak * (1 - trail))):
            in_trade, blocked = False, True
            continue
        if take_profit:
            for k, (gain, fraction) in enumerate(take_profit):
                if k not in taken and c[i] >= entry * (1 + gain) - 1e-9:
                    taken.add(k)
                    remaining = max(0.0, remaining - fraction)
        if remaining <= 1e-9:
            in_trade, blocked = False, True
        else:
            out[i] = remaining
    return pd.Series(out, index=signal.index)


# ----------------------------------------------------------------------------
# Backtester
# ----------------------------------------------------------------------------
def backtest(df, position):
    """Daily strategy returns with transaction costs. Position is lagged 1 day."""
    pos = position.shift(1).fillna(0).clip(0, 1)
    rets = df["Close"].pct_change().fillna(0)
    trades = pos.diff().abs().fillna(0)
    return pos * rets - trades * COST_PER_TRADE


def sharpe(returns):
    if returns.std() == 0 or len(returns) < 20:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std())


def max_drawdown(returns):
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def param_grid(grid):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def optimize(df, fn, grid):
    """Best parameters by in-sample Sharpe."""
    best, best_score = None, -np.inf
    for params in param_grid(grid):
        score = sharpe(backtest(df, apply_exits(df, fn(df, **params))))
        if score > best_score:
            best, best_score = params, score
    return best, best_score


def tune(df, fn, grid):
    """
    Step 1: pick strategy settings (with only the fixed stop loss).
    Step 2: try trailing-stop / take-profit combos on those settings. An exit rule is
    adopted only if it beats the baseline Sharpe by MIN_EXIT_GAIN; otherwise none is used.
    """
    params, base_score = optimize(df, fn, grid)
    sig = fn(df, **params)
    best_exits, best_score = {}, base_score
    for trail, tp in itertools.product(TRAIL_OPTIONS, TAKE_PROFIT_OPTIONS):
        if trail is None and tp is None:
            continue
        score = sharpe(backtest(df, apply_exits(df, sig, trail=trail, take_profit=tp)))
        if score > best_score:
            best_exits, best_score = {"trail": trail, "take_profit": tp}, score
    if best_score - base_score < MIN_EXIT_GAIN:
        best_exits = {}
    return params, best_exits


# ----------------------------------------------------------------------------
# Walk-forward: the self-improvement loop
# ----------------------------------------------------------------------------
def walk_forward(df, verbose=True):
    results = {name: [] for name in STRATEGIES}       # with the exit rules the bot chose
    base_results = {name: [] for name in STRATEGIES}  # same settings, fixed stop loss only
    adopted = {name: [] for name in STRATEGIES}
    log = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= len(df):
        train = df.iloc[start - TRAIN_DAYS:start]
        # Test slice includes warm-up history so indicators are defined at the start.
        test_full = df.iloc[start - WARMUP_DAYS:start + TEST_DAYS]
        for name, (fn, grid) in STRATEGIES.items():
            params, exits = tune(train, fn, grid)
            sig = fn(test_full, **params)
            oos = backtest(test_full, apply_exits(test_full, sig, **exits)).iloc[-TEST_DAYS:]
            base = backtest(test_full, apply_exits(test_full, sig)).iloc[-TEST_DAYS:]
            results[name].append(oos)
            base_results[name].append(base)
            adopted[name].append(bool(exits))
            log.append((df.index[start], name, params, exits, sharpe(oos)))
        start += TEST_DAYS

    summary = {}
    for name in STRATEGIES:
        r, b = pd.concat(results[name]), pd.concat(base_results[name])
        # The self-removal step: keep the exit rules only if they helped out-of-sample.
        use_exits = sharpe(r) > sharpe(b)
        chosen = r if use_exits else b
        summary[name] = {"oos_sharpe": sharpe(chosen),
                         "oos_return": float((1 + chosen).prod() - 1),
                         "max_dd": max_drawdown(chosen),
                         "use_exits": use_exits,
                         "exit_rate": float(np.mean(adopted[name]))}
        # Consistency stats, quarter by quarter, for the stricter stock filter.
        chunks = results[name] if use_exits else base_results[name]
        qret = np.array([float((1 + c).prod() - 1) for c in chunks])
        active = qret[np.abs(qret) > 1e-9]          # quarters where the strategy was invested
        summary[name].update(
            win_rate=float((active > 0).mean()) if len(active) else 0.0,
            active_quarters=int(len(active)),
            worst_q=float(qret.min()),
            t_stat=float(sharpe(chosen) * np.sqrt(len(chosen) / 252)))
    bh = df["Close"].pct_change().fillna(0).iloc[TRAIN_DAYS:]
    summary["buy_and_hold"] = {"oos_sharpe": sharpe(bh),
                               "oos_return": float((1 + bh).prod() - 1),
                               "max_dd": max_drawdown(bh)}
    if verbose:
        print(f"{'strategy':16}{'OOS Sharpe':>12}{'OOS return':>12}{'Max DD':>10}{'Exit rules':>12}")
        for name, s in summary.items():
            exits_col = ("kept" if s["use_exits"] else "removed") if "use_exits" in s else "-"
            print(f"{name:16}{s['oos_sharpe']:12.2f}{s['oos_return']:12.1%}{s['max_dd']:10.1%}{exits_col:>12}")
        print("('Exit rules' = trailing stop / take profit. 'removed' means they didn't help "
              "on unseen data, so the bot turned them off for that strategy.)")
    return summary, log


def pick_strategy(summary):
    """Demote strategies that fail out-of-sample. Returns best survivor or None."""
    ok = {k: v for k, v in summary.items()
          if k != "buy_and_hold" and v["oos_sharpe"] >= MIN_OOS_SHARPE}
    return max(ok, key=lambda k: ok[k]["oos_sharpe"]) if ok else None


# ----------------------------------------------------------------------------
# Stock discovery: find candidates, test each one, keep only those that pass
# ----------------------------------------------------------------------------
def get_universe():
    """S&P 500 members (from Wikipedia) plus a few broad ETFs."""
    import io
    import requests
    syms = []
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        table = pd.read_html(io.StringIO(html))[0]
        syms = [s for s in table["Symbol"].astype(str) if "." not in s]  # skip BRK.B style
    except Exception as e:
        print(f"Couldn't fetch the S&P 500 list ({e}). Using the core ETFs only.")
    return sorted(set(syms) | set(CORE_ETFS))


def load_many(symbols, years=10):
    """Download daily bars for many symbols at once. Returns {symbol: DataFrame}."""
    import yfinance as yf
    raw = yf.download(symbols, period=f"{years}y", auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s][["Open", "High", "Low", "Close", "Volume"]].dropna()
        except KeyError:
            continue
        if len(d) >= TRAIN_DAYS + 4 * TEST_DAYS:   # need enough history to validate
            out[s] = d
    return out


def screen(data):
    """Keep liquid, non-penny stocks; test the most heavily traded ones."""
    rows = []
    for s, d in data.items():
        dollar_vol = float((d["Close"] * d["Volume"]).iloc[-60:].mean())
        if d["Close"].iloc[-1] >= MIN_PRICE and dollar_vol >= MIN_DOLLAR_VOLUME:
            rows.append((s, dollar_vol))
    rows.sort(key=lambda r: -r[1])
    return [s for s, _ in rows[:SCAN_TOP_N]]


def fail_reasons(summary, name):
    """Which strict-filter tests this strategy fails (empty list = it passes all)."""
    s, bh = summary[name], summary["buy_and_hold"]
    checks = [
        ("Sharpe too low", s["oos_sharpe"] >= PICK_MIN_SHARPE and s["oos_return"] > 0),
        ("doesn't beat holding the stock on Sharpe", s["oos_sharpe"] > bh["oos_sharpe"]),
        ("not statistically convincing", s["t_stat"] >= PICK_MIN_T),
        ("too few invested quarters", s["active_quarters"] >= PICK_MIN_ACTIVE_QUARTERS),
        ("loses money in too many quarters", s["win_rate"] >= PICK_MIN_WIN_RATE),
        ("has a quarter that lost too much", s["worst_q"] >= PICK_MAX_WORST_QUARTER),
        ("drawdown not better than holding", abs(s["max_dd"]) <= PICK_DD_VS_HOLD * abs(bh["max_dd"])),
    ]
    return [label for label, ok in checks if not ok]


def research(data=None, verbose=True):
    """
    The bot's own research loop: screen the universe, run the full walk-forward test on each
    candidate, and keep only stocks where a strategy passed on unseen data.
    Returns (picks, data, universe) where picks is a ranked list of dicts.
    """
    universe = get_universe() if data is None else sorted(data)
    if data is None:
        data = load_many(universe)
    candidates = screen(data)
    if verbose:
        print(f"Universe {len(universe)} -> {len(data)} with enough history "
              f"-> {len(candidates)} candidates to test")
    passed, why_not = [], {}
    for s in candidates:
        summary, _ = walk_forward(data[s], verbose=False)
        ok = [n for n in STRATEGIES if not fail_reasons(summary, n)]
        if not ok:
            best = max(STRATEGIES, key=lambda n: summary[n]["oos_sharpe"])
            for r in fail_reasons(summary, best):
                why_not[r] = why_not.get(r, 0) + 1
            continue
        winner = max(ok, key=lambda n: summary[n]["oos_sharpe"])
        w = summary[winner]
        passed.append({"symbol": s, "strategy": winner, "summary": summary,
                       "sharpe": w["oos_sharpe"], "ret": w["oos_return"],
                       "bh_ret": summary["buy_and_hold"]["oos_return"],
                       "win_rate": w["win_rate"], "worst_q": w["worst_q"], "t": w["t_stat"],
                       "use_exits": w["use_exits"]})
    passed.sort(key=lambda p: -p["sharpe"])
    picks = passed[:MAX_HOLDINGS]
    if verbose:
        print(f"{len(passed)} of {len(candidates)} candidates passed the strict validation.")
        if why_not:
            top = sorted(why_not.items(), key=lambda kv: -kv[1])[:3]
            print("Most common reasons stocks failed: " + "; ".join(f"{k} ({v})" for k, v in top))
        if picks:
            print("Expect some picks to still be luck, not skill.")
            print(f"{'symbol':8}{'strategy':16}{'Sharpe':>8}{'Return':>10}{'Hold':>10}"
                  f"{'Win%':>7}{'WorstQ':>8}{'t-stat':>8}")
            for p in picks:
                print(f"{p['symbol']:8}{p['strategy']:16}{p['sharpe']:8.2f}{p['ret']:10.1%}"
                      f"{p['bh_ret']:10.1%}{p['win_rate']:7.0%}{p['worst_q']:8.1%}{p['t']:8.1f}")
        else:
            print("No stock passed. The bot will hold cash.")
    return picks, data, universe


# ----------------------------------------------------------------------------
# Paper trading via Alpaca
# ----------------------------------------------------------------------------
def _val(x):
    return str(getattr(x, "value", x)).lower()


def is_stop_order(o):
    kind = getattr(o, "order_type", None) or getattr(o, "type", None)
    return _val(o.side) == "sell" and _val(kind) == "stop"


def risk_size(equity, price, frac=1.0):
    """
    Shares to hold so that hitting the stop loss costs about RISK_PER_TRADE of equity,
    capped at MAX_POSITION_PCT of equity. `frac` scales it down as the profit ladder sells.
    """
    by_risk = equity * RISK_PER_TRADE / (price * STOP_LOSS)
    by_cap = equity * MAX_POSITION_PCT / price
    return int(min(by_risk, by_cap) * frac)


def manage_position(client, symbol, target_qty, held_qty, basis_price, last_close,
                    open_orders, dry_run=False):
    """
    Move one stock to its target share count AND keep a real GTC stop-loss order at the
    broker for every share held, so the stop works during the trading day.
      - new position: market buy bundled with a stop-loss leg (OTO order)
      - reduce/sell:  cancel the old stop, sell, then re-place a stop for what remains
      - add to position: plain buy; next run extends the stop to cover the new total
      - no change:    make sure the stop quantity matches the shares held
    """
    import time
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    stops = [o for o in open_orders if is_stop_order(o)]
    if any(not is_stop_order(o) for o in open_orders):
        print(f"{symbol}: a buy/sell order is still pending, skipping")
        return
    delta = target_qty - held_qty
    stop_price = lambda price: round(price * (1 - STOP_LOSS), 2)

    def cancel_stops():
        for o in stops:
            client.cancel_order_by_id(o.id)
        if stops:
            time.sleep(2)        # let Alpaca free the shares the old stop reserved

    def place_stop(qty):
        client.submit_order(StopOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
            stop_price=stop_price(basis_price)))
        print(f"{symbol}: stop-loss order for {qty} shares at ${stop_price(basis_price):.2f}")

    if dry_run:
        if delta > 0 and held_qty == 0:
            print(f"DRY RUN, would buy {delta} {symbol} with a stop at ${stop_price(last_close):.2f}")
        elif delta != 0:
            print(f"DRY RUN, would {'buy' if delta > 0 else 'sell'} {abs(delta)} {symbol}")
        elif held_qty > 0 and sum(int(float(o.qty)) for o in stops) != held_qty:
            print(f"DRY RUN, would set the stop-loss on {symbol} to cover {held_qty} shares")
        return

    if delta > 0 and held_qty == 0:
        try:
            client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=delta, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                order_class=OrderClass.OTO,
                stop_loss=StopLossRequest(stop_price=stop_price(last_close))))
            print(f"Submitted PAPER order: buy {delta} {symbol} with stop at "
                  f"${stop_price(last_close):.2f}")
        except Exception as e:
            print(f"{symbol}: buy-with-stop was rejected ({e}). Placing a plain buy; "
                  f"the stop gets added on the next run.")
            client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=delta, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
    elif delta > 0:
        client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=delta, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
        print(f"Submitted PAPER order: buy {delta} {symbol} (stop is extended on the next run)")
    elif delta < 0:
        cancel_stops()
        try:
            client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=abs(delta), side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            print(f"Submitted PAPER order: sell {abs(delta)} {symbol}")
        except Exception as e:
            print(f"{symbol}: sell failed ({e}). Restoring the stop on the full position.")
            place_stop(held_qty)
            return
        if target_qty > 0:
            place_stop(target_qty)
    elif held_qty > 0 and sum(int(float(o.qty)) for o in stops) != held_qty:
        cancel_stops()
        place_stop(held_qty)


def _open_orders_by_symbol(client):
    from collections import defaultdict
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    by = defaultdict(list)
    for o in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
        by[o.symbol].append(o)
    return by


def paper_step(symbol, dry_run=False):
    from alpaca.trading.client import TradingClient

    client = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
    df = load_prices(symbol)

    # 1. Re-evaluate everything on the latest data (the self-improving step).
    summary, _ = walk_forward(df, verbose=True)
    winner = pick_strategy(summary)
    if winner is None:
        print("No strategy passed out-of-sample validation. Staying flat.")
        target_frac = 0.0
    else:
        fn, grid = STRATEGIES[winner]
        params, exits = tune(df.iloc[-TRAIN_DAYS:], fn, grid)
        if not summary[winner]["use_exits"]:
            exits = {}      # the walk-forward test showed the extra exits didn't help
        target_frac = float(apply_exits(df, fn(df, **params), **exits).iloc[-1])
        print(f"Using {winner} {params} exits={exits or 'fixed stop only'} "
              f"-> hold {target_frac:.0%} of full position")

    # 2. Risk checks.
    account = client.get_account()
    equity, last_equity = float(account.equity), float(account.last_equity)
    if equity / last_equity - 1 <= DAILY_KILL_SWITCH:
        print("Daily kill switch hit. No new trading today.")
        return

    try:
        pos = client.get_open_position(symbol)
        held_qty, entry = int(float(pos.qty)), float(pos.avg_entry_price)
    except Exception:
        held_qty, entry = 0, 0.0

    last_price = float(df["Close"].iloc[-1])
    if held_qty > 0 and last_price < entry * (1 - STOP_LOSS):
        target_frac = 0.0
        print("Stop loss level breached. Selling.")

    # 3. Size by risk, then move to target (this also maintains the broker-side stop).
    target_qty = risk_size(equity, last_price, target_frac)
    print(f"Target {target_qty} shares (risking about {RISK_PER_TRADE:.0%} of equity per trade)")
    manage_position(client, symbol, target_qty, held_qty, entry or last_price, last_price,
                    _open_orders_by_symbol(client).get(symbol, []), dry_run)


def paper_auto(dry_run=False):
    """Research the whole universe, then hold the best few stocks on the PAPER account."""
    from alpaca.trading.client import TradingClient

    client = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
    picks, data, universe = research()

    account = client.get_account()
    equity, last_equity = float(account.equity), float(account.last_equity)
    if equity / last_equity - 1 <= DAILY_KILL_SWITCH:
        print("Daily kill switch hit. No new trading today.")
        return

    positions = {p.symbol: p for p in client.get_all_positions()}
    orders = _open_orders_by_symbol(client)

    # Target share count for every stock we care about.
    targets, prices = {}, {}
    for p in picks:
        s, df = p["symbol"], data[p["symbol"]]
        fn, grid = STRATEGIES[p["strategy"]]
        params, exits = tune(df.iloc[-TRAIN_DAYS:], fn, grid)
        if not p["use_exits"]:
            exits = {}
        frac = float(apply_exits(df, fn(df, **params), **exits).iloc[-1])
        prices[s] = float(df["Close"].iloc[-1])
        targets[s] = risk_size(equity, prices[s], frac)
        print(f"{s}: {p['strategy']} {params} exits={exits or 'fixed stop only'} "
              f"-> hold {frac:.0%} ({targets[s]} shares)")
    # Keep total money invested under TOTAL_EXPOSURE by scaling everything down if needed.
    invested = sum(targets[s] * prices[s] for s in targets)
    if invested > equity * TOTAL_EXPOSURE:
        scale = equity * TOTAL_EXPOSURE / invested
        targets = {s: int(q * scale) for s, q in targets.items()}
        print(f"Scaled positions by {scale:.0%} to stay under {TOTAL_EXPOSURE:.0%} total exposure")
    # Stocks we hold (within our universe) that are no longer picked get sold.
    for s in positions:
        if s in universe and s not in targets:
            targets[s] = 0
            print(f"{s}: no longer passes validation -> sell")
    # Backup stop check from your actual fill price, in case a stop order didn't trigger.
    for s, pos in positions.items():
        if s in targets and float(pos.current_price) < float(pos.avg_entry_price) * (1 - STOP_LOSS):
            targets[s] = 0
            print(f"{s}: stop loss level breached -> sell")

    for s, target_qty in targets.items():
        held = int(float(positions[s].qty)) if s in positions else 0
        basis = float(positions[s].avg_entry_price) if s in positions else prices.get(s, 0.0)
        last = prices.get(s) or (float(positions[s].current_price) if s in positions else basis)
        manage_position(client, s, target_qty, held, basis, last, orders.get(s, []), dry_run)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["backtest", "paper", "scan"])
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="paper mode: let the bot find and trade its own stocks")
    ap.add_argument("--dry-run", action="store_true",
                    help="paper mode: print the orders instead of sending them")
    args = ap.parse_args()

    if args.mode == "backtest":
        px = synthetic_prices() if args.synthetic else load_prices(args.symbol)
        summary, _ = walk_forward(px)
        print("\nWould trade:", pick_strategy(summary) or "nothing (no strategy passed validation)")
    elif args.mode == "scan":
        research()          # no Alpaca keys needed; prints what the bot would pick
    elif args.auto:
        paper_auto(dry_run=args.dry_run)
    else:
        paper_step(args.symbol, dry_run=args.dry_run)
