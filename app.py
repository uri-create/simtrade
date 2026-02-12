from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os
import uuid

app = Flask(__name__)

# Suppress noisy yfinance/urllib3 logs
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

@app.errorhandler(Exception)
def handle_error(e):
    """Catch all errors so the server never crashes."""
    return jsonify({'error': str(e)}), 500

# Available instruments organized by category
# All tickers are real Yahoo Finance symbols for accurate data
STOCK_CATEGORIES = {
    'Tech': {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'NVDA': 'NVIDIA', 'META': 'Meta',
        'TSLA': 'Tesla', 'NFLX': 'Netflix', 'AMD': 'AMD',
        'INTC': 'Intel', 'CRM': 'Salesforce', 'ORCL': 'Oracle',
        'ADBE': 'Adobe', 'AVGO': 'Broadcom', 'QCOM': 'Qualcomm',
        'UBER': 'Uber', 'SHOP': 'Shopify', 'SQ': 'Block',
        'SNAP': 'Snap', 'PLTR': 'Palantir', 'NET': 'Cloudflare',
        'SNOW': 'Snowflake', 'COIN': 'Coinbase', 'MSTR': 'MicroStrategy',
    },
    'Finance': {
        'JPM': 'JPMorgan', 'BAC': 'Bank of America', 'GS': 'Goldman Sachs',
        'MS': 'Morgan Stanley', 'V': 'Visa', 'MA': 'Mastercard',
        'BRK-B': 'Berkshire Hathaway', 'C': 'Citigroup', 'WFC': 'Wells Fargo',
        'AXP': 'American Express', 'PYPL': 'PayPal', 'SCHW': 'Charles Schwab',
    },
    'Healthcare': {
        'JNJ': 'Johnson & Johnson', 'UNH': 'UnitedHealth', 'PFE': 'Pfizer',
        'ABBV': 'AbbVie', 'MRK': 'Merck', 'LLY': 'Eli Lilly',
        'TMO': 'Thermo Fisher', 'ABT': 'Abbott Labs', 'MRNA': 'Moderna',
    },
    'Consumer': {
        'WMT': 'Walmart', 'COST': 'Costco', 'HD': 'Home Depot',
        'NKE': 'Nike', 'SBUX': 'Starbucks', 'MCD': 'McDonald\'s',
        'DIS': 'Disney', 'KO': 'Coca-Cola', 'PEP': 'PepsiCo',
        'PG': 'Procter & Gamble',
    },
    'Industrial': {
        'BA': 'Boeing', 'CAT': 'Caterpillar', 'GE': 'GE Aerospace',
        'LMT': 'Lockheed Martin', 'RTX': 'RTX Corp', 'DE': 'John Deere',
        'UPS': 'UPS', 'FDX': 'FedEx', 'HON': 'Honeywell',
    },
    'Energy': {
        'XOM': 'ExxonMobil', 'CVX': 'Chevron', 'COP': 'ConocoPhillips',
        'SLB': 'Schlumberger', 'OXY': 'Occidental', 'EOG': 'EOG Resources',
    },
    'Crypto': {
        'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'BNB-USD': 'BNB',
        'SOL-USD': 'Solana', 'XRP-USD': 'XRP', 'ADA-USD': 'Cardano',
        'DOGE-USD': 'Dogecoin', 'AVAX-USD': 'Avalanche', 'DOT-USD': 'Polkadot',
        'MATIC-USD': 'Polygon', 'LINK-USD': 'Chainlink', 'UNI-USD': 'Uniswap',
        'LTC-USD': 'Litecoin', 'SHIB-USD': 'Shiba Inu', 'ATOM-USD': 'Cosmos',
    },
    'Precious Metals': {
        'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum',
        'PA=F': 'Palladium',
    },
    'Commodities': {
        'CL=F': 'Crude Oil WTI', 'BZ=F': 'Brent Crude', 'NG=F': 'Natural Gas',
        'HG=F': 'Copper', 'ZC=F': 'Corn', 'ZW=F': 'Wheat',
        'ZS=F': 'Soybeans', 'KC=F': 'Coffee', 'CT=F': 'Cotton',
    },
    'Indices': {
        '^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000', '^VIX': 'VIX Volatility',
        '^FTSE': 'FTSE 100', '^N225': 'Nikkei 225',
    },
    'Forex': {
        'EURUSD=X': 'EUR/USD', 'GBPUSD=X': 'GBP/USD', 'USDJPY=X': 'USD/JPY',
        'AUDUSD=X': 'AUD/USD', 'USDCAD=X': 'USD/CAD', 'USDCHF=X': 'USD/CHF',
    },
    'ETFs': {
        'SPY': 'S&P 500 ETF', 'QQQ': 'NASDAQ 100 ETF', 'IWM': 'Russell 2000 ETF',
        'DIA': 'Dow Jones ETF', 'GLD': 'Gold ETF', 'SLV': 'Silver ETF',
        'USO': 'Oil ETF', 'TLT': 'Treasury Bond ETF', 'VXX': 'VIX ETF',
        'ARKK': 'ARK Innovation', 'XLF': 'Financial ETF', 'XLE': 'Energy ETF',
    },
}

# Flat lookup: symbol -> name (for validation)
STOCKS = {}
for cat, items in STOCK_CATEGORIES.items():
    STOCKS.update(items)

# Timeframe config: yfinance interval and period
# Extended ranges so charts go back to 2024+
TIMEFRAMES = {
    '1s':  {'interval': '1m',  'period': '5d'},     # simulate 1s from 1m data
    '1m':  {'interval': '1m',  'period': '5d'},
    '1h':  {'interval': '1h',  'period': '2y'},
    '1d':  {'interval': '1d',  'period': '2y'},
    '1mo': {'interval': '1d',  'period': '5y'},
    '1y':  {'interval': '1wk', 'period': 'max'},
}

# In-memory wallet
wallet = {
    'cash': 100000.0,
    'positions': {},   # symbol -> {shares, avg_price}
    'history': [],     # list of trade records
    'initial_balance': 100000.0,
}


WALLET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wallet_data.json')


def save_wallet():
    """Persist wallet state to JSON file."""
    try:
        with open(WALLET_FILE, 'w') as f:
            json.dump({
                'cash': wallet['cash'],
                'positions': wallet['positions'],
                'history': wallet['history'],
                'initial_balance': wallet['initial_balance'],
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save wallet: {e}")


def load_wallet():
    """Load wallet state from JSON file if it exists."""
    if os.path.exists(WALLET_FILE):
        try:
            with open(WALLET_FILE, 'r') as f:
                data = json.load(f)
            wallet['cash'] = data.get('cash', 100000.0)
            wallet['positions'] = data.get('positions', {})
            wallet['history'] = data.get('history', [])
            wallet['initial_balance'] = data.get('initial_balance', 100000.0)
            print(f"Loaded wallet: ${wallet['cash']:.2f} cash, {len(wallet['positions'])} positions, {len(wallet['history'])} trades")
        except Exception as e:
            print(f"Warning: Could not load wallet, starting fresh: {e}")


def reset_wallet():
    wallet['cash'] = 100000.0
    wallet['positions'] = {}
    wallet['history'] = []


def calc_sma(closes, period):
    s = pd.Series(closes)
    return s.rolling(window=period, min_periods=1).mean().tolist()


def calc_ema(closes, period):
    s = pd.Series(closes)
    return s.ewm(span=period, adjust=False).mean().tolist()


def calc_rsi(closes, period=14):
    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).tolist()


def calc_macd(closes, fast=12, slow=26, signal=9):
    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        'macd': macd_line.tolist(),
        'signal': signal_line.tolist(),
        'histogram': histogram.tolist(),
    }


def calc_bollinger(closes, period=20, std_dev=2):
    s = pd.Series(closes)
    sma = s.rolling(window=period, min_periods=1).mean()
    std = s.rolling(window=period, min_periods=1).std().fillna(0)
    return {
        'upper': (sma + std_dev * std).tolist(),
        'middle': sma.tolist(),
        'lower': (sma - std_dev * std).tolist(),
    }


def fetch_ohlcv(symbol, timeframe):
    tf = TIMEFRAMES.get(timeframe, TIMEFRAMES['1d'])
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=tf['period'], interval=tf['interval'])
    if df.empty:
        return []

    candles = []
    for idx, row in df.iterrows():
        ts = int(idx.timestamp())
        candles.append({
            'time': ts,
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume']),
        })
    return candles


# ─── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stocks')
def get_stocks():
    return jsonify(STOCK_CATEGORIES)


@app.route('/api/chart/<symbol>')
def get_chart(symbol):
    symbol = symbol.upper()
    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 404
    timeframe = request.args.get('timeframe', '1d')
    candles = fetch_ohlcv(symbol, timeframe)
    return jsonify(candles)


@app.route('/api/indicators/<symbol>')
def get_indicators(symbol):
    symbol = symbol.upper()
    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 404

    timeframe = request.args.get('timeframe', '1d')
    candles = fetch_ohlcv(symbol, timeframe)
    if not candles:
        return jsonify({'error': 'No data'}), 404

    closes = [c['close'] for c in candles]
    times = [c['time'] for c in candles]

    result = {
        'times': times,
        'sma20': calc_sma(closes, 20),
        'sma50': calc_sma(closes, 50),
        'ema12': calc_ema(closes, 12),
        'ema26': calc_ema(closes, 26),
        'rsi': calc_rsi(closes),
        'macd': calc_macd(closes),
        'bollinger': calc_bollinger(closes),
    }
    return jsonify(result)


@app.route('/api/simulation/<symbol>')
def get_simulation(symbol):
    symbol = symbol.upper()
    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 404
    timeframe = request.args.get('timeframe', '1d')
    candles = fetch_ohlcv(symbol, timeframe)
    return jsonify(candles)


# ─── Smart Simulation Engine ──────────────────────────────────────
# Stores per-session simulation context so candles respect TA patterns
sim_sessions = {}


def analyze_chart_for_simulation(candles):
    """Analyze historical candles to extract trend, key levels, momentum, and volatility
    so the simulation can generate TA-aware candles that help users learn."""

    if not candles or len(candles) < 10:
        return {
            'trend': 'sideways', 'trend_strength': 0.5,
            'momentum': 0, 'volatility': 0.015,
            'support_levels': [], 'resistance_levels': [],
            'ath': 0, 'atl': 0, 'avg_volume': 1000000,
            'recent_range_high': 0, 'recent_range_low': 0,
            'sma20': 0, 'sma50': 0,
            'rsi': 50, 'trend_age': 0,
        }

    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    volumes = [c['volume'] for c in candles]

    # ── Trend Detection ──
    # Use SMA20 vs SMA50 crossover + price position relative to SMAs
    sma20_vals = calc_sma(closes, 20)
    sma50_vals = calc_sma(closes, 50)
    current_sma20 = sma20_vals[-1] if sma20_vals else closes[-1]
    current_sma50 = sma50_vals[-1] if sma50_vals else closes[-1]
    current_price = closes[-1]

    # Trend direction
    if current_sma20 > current_sma50 and current_price > current_sma20:
        trend = 'uptrend'
    elif current_sma20 < current_sma50 and current_price < current_sma20:
        trend = 'downtrend'
    else:
        trend = 'sideways'

    # Trend strength (0-1): how far apart the SMAs are relative to price
    sma_spread = abs(current_sma20 - current_sma50) / current_price if current_price > 0 else 0
    trend_strength = min(sma_spread * 20, 1.0)  # normalize: 5% spread = max strength

    # How long has the trend been going? Count consecutive candles in trend direction
    trend_age = 0
    if len(closes) > 1:
        for i in range(len(closes) - 1, 0, -1):
            if trend == 'uptrend' and closes[i] > closes[i - 1]:
                trend_age += 1
            elif trend == 'downtrend' and closes[i] < closes[i - 1]:
                trend_age += 1
            else:
                break

    # ── Momentum (RSI) ──
    rsi_vals = calc_rsi(closes, 14)
    current_rsi = rsi_vals[-1] if rsi_vals else 50

    # Momentum score: -1 (oversold) to +1 (overbought)
    momentum = (current_rsi - 50) / 50

    # ── Volatility ──
    if len(closes) > 20:
        recent_closes = closes[-20:]
        returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1]
                   for i in range(1, len(recent_closes))]
        volatility = float(np.std(returns)) if returns else 0.015
    else:
        volatility = 0.015

    # ── Key Levels: Support & Resistance ──
    # Find swing highs and swing lows (local maxima/minima)
    support_levels = []
    resistance_levels = []
    lookback = min(5, len(candles) // 4)  # how many candles on each side to check

    if lookback >= 2:
        for i in range(lookback, len(candles) - lookback):
            # Swing high: higher than `lookback` candles on each side
            is_swing_high = all(highs[i] >= highs[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            # Swing low: lower than `lookback` candles on each side
            is_swing_low = all(lows[i] <= lows[j] for j in range(i - lookback, i + lookback + 1) if j != i)

            if is_swing_high:
                resistance_levels.append(highs[i])
            if is_swing_low:
                support_levels.append(lows[i])

    # Deduplicate nearby levels (within 0.5% of each other)
    def dedup_levels(levels, threshold_pct=0.005):
        if not levels:
            return []
        levels = sorted(levels)
        result = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - result[-1]) / result[-1] > threshold_pct:
                result.append(lvl)
            else:
                # Average them
                result[-1] = (result[-1] + lvl) / 2
        return result

    support_levels = dedup_levels(support_levels)
    resistance_levels = dedup_levels(resistance_levels)

    # Only keep levels that are relevant to current price (within 15%)
    price = closes[-1]
    support_levels = [s for s in support_levels if s < price and s > price * 0.85]
    resistance_levels = [r for r in resistance_levels if r > price and r < price * 1.15]

    # Sort: nearest supports/resistances first
    support_levels.sort(reverse=True)    # highest support first (nearest)
    resistance_levels.sort()             # lowest resistance first (nearest)

    # Keep top 5 each
    support_levels = support_levels[:5]
    resistance_levels = resistance_levels[:5]

    # ── ATH / ATL ──
    ath = max(highs) if highs else price
    atl = min(lows) if lows else price

    # ── Recent range ──
    recent = candles[-20:] if len(candles) >= 20 else candles
    recent_range_high = max(c['high'] for c in recent)
    recent_range_low = min(c['low'] for c in recent)

    # ── Average volume ──
    recent_vols = volumes[-20:] if len(volumes) >= 20 else volumes
    avg_volume = int(np.mean(recent_vols)) if recent_vols else 1000000

    return {
        'trend': trend,
        'trend_strength': round(trend_strength, 3),
        'momentum': round(momentum, 3),
        'volatility': round(volatility, 5),
        'support_levels': [round(s, 2) for s in support_levels],
        'resistance_levels': [round(r, 2) for r in resistance_levels],
        'ath': round(ath, 2),
        'atl': round(atl, 2),
        'avg_volume': avg_volume,
        'recent_range_high': round(recent_range_high, 2),
        'recent_range_low': round(recent_range_low, 2),
        'sma20': round(current_sma20, 2),
        'sma50': round(current_sma50, 2),
        'rsi': round(current_rsi, 1),
        'trend_age': trend_age,
    }


@app.route('/api/simulation/init', methods=['POST'])
def init_simulation():
    """Analyze historical candles and create a simulation session with market context."""
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    timeframe = data.get('timeframe', '1d')

    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 404

    candles = fetch_ohlcv(symbol, timeframe)
    if not candles:
        return jsonify({'error': 'No chart data'}), 404

    context = analyze_chart_for_simulation(candles)

    # Create session with a unique ID
    session_id = str(uuid.uuid4())[:8]

    sim_sessions[session_id] = {
        'symbol': symbol,
        'timeframe': timeframe,
        'context': context,
        'candle_count': 0,
        'last_close': candles[-1]['close'],
        'last_time': candles[-1]['time'],
        # Track recent sim closes for evolving momentum/trends
        'recent_closes': [c['close'] for c in candles[-20:]],
        # Current simulation state (can evolve from the initial analysis)
        'sim_trend': context['trend'],
        'sim_trend_candles': 0,      # how many candles current sim trend has lasted
        'reversal_zone': False,       # are we near a key level?
        'consolidating': False,       # in a consolidation/range phase?
        'consol_candles': 0,          # how long in consolidation
    }

    return jsonify({'session_id': session_id, 'context': context})


@app.route('/api/simulation/next/<symbol>')
def get_next_candle(symbol):
    """Generate the next synthetic candle using TA-aware smart simulation."""
    symbol = symbol.upper()
    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 404

    timeframe = request.args.get('timeframe', '1d')
    last_close = float(request.args.get('last_close', 100))
    last_time = int(request.args.get('last_time', 0))
    session_id = request.args.get('session_id', '')

    # Timeframe intervals in seconds
    intervals = {
        '1s': 1, '1m': 60, '1h': 3600,
        '1d': 86400, '1mo': 2592000, '1y': 31536000,
    }
    dt = intervals.get(timeframe, 86400)

    # Get simulation session or fall back to basic mode
    session = sim_sessions.get(session_id)
    if not session:
        # Fallback: no context, use basic random walk (backwards compat)
        vol = 0.015
        tf_scale = {'1s': 0.05, '1m': 0.15, '1h': 0.5, '1d': 1.0, '1mo': 3.0, '1y': 5.0}
        vol *= tf_scale.get(timeframe, 1.0)
        drift = np.random.normal(0.0001, vol)
        new_close = round(last_close * (1 + drift), 2)
        new_open = round(last_close * (1 + np.random.normal(0, vol * 0.3)), 2)
        intra_vol = abs(drift) + vol * abs(np.random.normal(0, 0.5))
        new_high = round(max(new_open, new_close) * (1 + abs(np.random.normal(0, intra_vol * 0.5))), 2)
        new_low = round(min(new_open, new_close) * (1 - abs(np.random.normal(0, intra_vol * 0.5))), 2)
        new_high = max(new_high, new_open, new_close)
        new_low = min(new_low, new_open, new_close)
        return jsonify({
            'time': last_time + dt, 'open': new_open, 'high': new_high,
            'low': new_low, 'close': new_close, 'volume': 1000000,
        })

    ctx = session['context']
    session['candle_count'] += 1
    session['last_close'] = last_close
    session['last_time'] = last_time

    # ── Base volatility scaled to timeframe ──
    base_vol = ctx['volatility']
    tf_scale = {'1s': 0.05, '1m': 0.15, '1h': 0.5, '1d': 1.0, '1mo': 3.0, '1y': 5.0}
    vol = base_vol * tf_scale.get(timeframe, 1.0)

    # ── Determine the drift (directional bias) based on TA context ──
    sim_trend = session['sim_trend']
    sim_trend_candles = session['sim_trend_candles']
    price = last_close

    # 1. Base drift from current trend
    if sim_trend == 'uptrend':
        base_drift = vol * 0.35   # Positive bias (trend up)
    elif sim_trend == 'downtrend':
        base_drift = -vol * 0.35  # Negative bias (trend down)
    else:
        base_drift = 0            # No bias (sideways/consolidation)

    # 2. Trend exhaustion: the longer a trend runs, the more likely a reversal
    #    After ~30 candles, start weakening. After ~60, likely reversal.
    exhaustion_factor = 1.0
    if sim_trend_candles > 20:
        exhaustion_factor = max(0.1, 1.0 - (sim_trend_candles - 20) / 50)
    base_drift *= exhaustion_factor

    # 3. Support / Resistance interaction
    nearest_support = ctx['support_levels'][0] if ctx['support_levels'] else 0
    nearest_resistance = ctx['resistance_levels'][0] if ctx['resistance_levels'] else float('inf')

    # Check proximity to key levels (within 1% of price)
    near_support = nearest_support > 0 and abs(price - nearest_support) / price < 0.01
    near_resistance = nearest_resistance < float('inf') and abs(price - nearest_resistance) / price < 0.01
    near_ath = ctx['ath'] > 0 and abs(price - ctx['ath']) / price < 0.015

    # Support bounce: if near support in uptrend, strong bounce probability
    if near_support and sim_trend != 'downtrend':
        bounce_chance = 0.75
        if np.random.random() < bounce_chance:
            base_drift = abs(vol * 0.5)  # Strong upward push
            session['reversal_zone'] = False
        else:
            # Break support → trend reversal
            base_drift = -abs(vol * 0.6)
            session['sim_trend'] = 'downtrend'
            session['sim_trend_candles'] = 0
            session['reversal_zone'] = True

    # Resistance rejection: if near resistance, may reject or break through
    elif near_resistance and sim_trend != 'uptrend':
        reject_chance = 0.70
        if np.random.random() < reject_chance:
            base_drift = -abs(vol * 0.4)  # Rejection downward
            session['reversal_zone'] = False
        else:
            # Break resistance → breakout
            base_drift = abs(vol * 0.7)  # Strong breakout
            session['sim_trend'] = 'uptrend'
            session['sim_trend_candles'] = 0
            session['reversal_zone'] = True

    # Near ATH: dramatic behavior
    elif near_ath:
        if sim_trend == 'uptrend' and ctx['trend_strength'] > 0.4:
            # Strong trend → 40% chance to break ATH
            if np.random.random() < 0.40:
                base_drift = abs(vol * 0.8)  # ATH breakout
            else:
                base_drift = -abs(vol * 0.5)  # Rejection from ATH
        else:
            # Weak trend → more likely to reject
            if np.random.random() < 0.25:
                base_drift = abs(vol * 0.6)
            else:
                base_drift = -abs(vol * 0.6)

    # 4. In uptrend approaching resistance from below → may consolidate
    if sim_trend == 'uptrend' and nearest_resistance < float('inf'):
        dist_to_res = (nearest_resistance - price) / price
        if 0.01 < dist_to_res < 0.03:
            # Getting close — reduce drift, potential consolidation
            if np.random.random() < 0.3:
                session['consolidating'] = True
                session['consol_candles'] = 0

    # In downtrend approaching support from above → may consolidate
    if sim_trend == 'downtrend' and nearest_support > 0:
        dist_to_sup = (price - nearest_support) / price
        if 0.01 < dist_to_sup < 0.03:
            if np.random.random() < 0.3:
                session['consolidating'] = True
                session['consol_candles'] = 0

    # 5. Consolidation phase: small random moves, builds up for breakout
    if session.get('consolidating'):
        session['consol_candles'] += 1
        base_drift = np.random.normal(0, vol * 0.15)  # Very small moves
        vol *= 0.5  # Tighter candles during consolidation

        # Break out of consolidation after 5-15 candles
        if session['consol_candles'] > np.random.randint(5, 16):
            session['consolidating'] = False
            session['consol_candles'] = 0
            # Breakout direction: usually continues trend (65%) or reverses (35%)
            if np.random.random() < 0.65:
                if sim_trend == 'uptrend':
                    base_drift = abs(vol * 1.5)  # Bullish breakout
                else:
                    base_drift = -abs(vol * 1.5)  # Bearish breakdown
                vol *= 2.0  # Wide breakout candle
            else:
                # Reversal breakout
                if sim_trend == 'uptrend':
                    base_drift = -abs(vol * 1.2)
                    session['sim_trend'] = 'downtrend'
                else:
                    base_drift = abs(vol * 1.2)
                    session['sim_trend'] = 'uptrend'
                session['sim_trend_candles'] = 0
                vol *= 1.8

    # 6. Random trend reversal chance (even without key levels)
    #    ~3% per candle, increasing with trend age
    reversal_chance = 0.03 + sim_trend_candles * 0.001
    if not session.get('consolidating') and np.random.random() < reversal_chance:
        if sim_trend == 'uptrend':
            session['sim_trend'] = 'downtrend'
            base_drift = -abs(vol * 0.5)
        elif sim_trend == 'downtrend':
            session['sim_trend'] = 'uptrend'
            base_drift = abs(vol * 0.5)
        session['sim_trend_candles'] = 0

    # 7. Pullbacks in trends: ~25% of candles go against the trend briefly
    if sim_trend in ('uptrend', 'downtrend') and not session.get('consolidating'):
        if np.random.random() < 0.25:
            # Pullback: counter-trend candle (smaller magnitude)
            base_drift *= -0.5

    # ── Generate the candle ──
    # Add noise to the drift
    noise = np.random.normal(0, vol * 0.5)
    final_drift = base_drift + noise

    new_close = round(price * (1 + final_drift), 2)
    # Ensure price doesn't go negative
    new_close = max(new_close, price * 0.9)

    # Open: gap from previous close (small)
    gap = np.random.normal(0, vol * 0.15)
    new_open = round(price * (1 + gap), 2)

    # Intra-candle wicks
    body_size = abs(new_close - new_open)
    wick_factor = vol * price
    new_high = max(new_open, new_close) + abs(np.random.normal(0, wick_factor * 0.4))
    new_low = min(new_open, new_close) - abs(np.random.normal(0, wick_factor * 0.4))

    # Ensure high/low bounds
    new_high = round(max(new_high, new_open, new_close), 2)
    new_low = round(min(new_low, new_open, new_close), 2)
    new_low = max(new_low, 0.01)

    # ── Volume: contextual ──
    avg_vol = ctx['avg_volume']
    if session.get('consolidating'):
        # Low volume during consolidation
        volume = int(avg_vol * (0.3 + np.random.random() * 0.4))
    elif abs(final_drift) > vol * 1.2:
        # High volume on big moves / breakouts
        volume = int(avg_vol * (1.3 + np.random.random() * 0.8))
    elif near_support or near_resistance or near_ath:
        # Higher volume near key levels
        volume = int(avg_vol * (1.1 + np.random.random() * 0.5))
    else:
        # Normal volume
        volume = int(avg_vol * (0.5 + np.random.random()))

    new_time = last_time + dt

    # ── Update session state ──
    session['sim_trend_candles'] += 1
    session['recent_closes'].append(new_close)
    if len(session['recent_closes']) > 30:
        session['recent_closes'] = session['recent_closes'][-30:]

    # Dynamically update support/resistance as price moves beyond old levels
    if new_close > nearest_resistance and nearest_resistance < float('inf'):
        # Price broke resistance — remove it, old resistance becomes support
        if ctx['resistance_levels']:
            broken = ctx['resistance_levels'].pop(0)
            ctx['support_levels'].insert(0, broken)
            ctx['support_levels'] = ctx['support_levels'][:5]
    if new_close < nearest_support and nearest_support > 0:
        # Price broke support — remove it, old support becomes resistance
        if ctx['support_levels']:
            broken = ctx['support_levels'].pop(0)
            ctx['resistance_levels'].insert(0, broken)
            ctx['resistance_levels'] = ctx['resistance_levels'][:5]

    # Update ATH if broken
    if new_high > ctx['ath']:
        ctx['ath'] = round(new_high, 2)

    candle = {
        'time': new_time,
        'open': new_open,
        'high': new_high,
        'low': new_low,
        'close': new_close,
        'volume': volume,
    }
    return jsonify(candle)


@app.route('/api/trade', methods=['POST'])
def trade():
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    action = data.get('action', '').lower()  # 'buy', 'sell', 'short', 'cover'
    quantity = int(data.get('quantity', 0))
    price = float(data.get('price', 0))

    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 400
    if quantity <= 0:
        return jsonify({'error': 'Quantity must be positive'}), 400
    if price <= 0:
        return jsonify({'error': 'Invalid price'}), 400

    total_cost = price * quantity

    if action == 'buy':
        # Normal long buy
        if total_cost > wallet['cash']:
            return jsonify({'error': 'Insufficient funds'}), 400
        wallet['cash'] -= total_cost
        pos = wallet['positions'].get(symbol, {'shares': 0, 'avg_price': 0, 'type': 'long'})
        total_shares = pos['shares'] + quantity
        if total_shares > 0:
            pos['avg_price'] = ((pos['avg_price'] * pos['shares']) + total_cost) / total_shares
        pos['shares'] = total_shares
        pos['type'] = 'long'
        wallet['positions'][symbol] = pos

    elif action == 'sell':
        # Sell long position
        pos = wallet['positions'].get(symbol, {'shares': 0, 'avg_price': 0, 'type': 'long'})
        if pos.get('type') == 'short':
            return jsonify({'error': 'Use cover to close a short position'}), 400
        if quantity > pos['shares']:
            return jsonify({'error': f'Not enough shares. You have {pos["shares"]}'}), 400
        sell_pnl = (price - pos['avg_price']) * quantity
        wallet['cash'] += total_cost
        pos['shares'] -= quantity
        if pos['shares'] == 0:
            del wallet['positions'][symbol]
        else:
            wallet['positions'][symbol] = pos

    elif action == 'short':
        # Short sell: borrow shares and sell at current price
        # Requires margin (collateral) = total_cost held from cash
        if total_cost > wallet['cash']:
            return jsonify({'error': 'Insufficient margin for short'}), 400
        wallet['cash'] -= total_cost  # hold as margin collateral
        pos = wallet['positions'].get(symbol)
        if pos and pos.get('type') == 'long':
            return jsonify({'error': 'Already have a long position. Sell it first.'}), 400
        if pos and pos.get('type') == 'short':
            # Add to existing short
            total_shares = pos['shares'] + quantity
            pos['avg_price'] = ((pos['avg_price'] * pos['shares']) + total_cost) / total_shares
            pos['shares'] = total_shares
        else:
            pos = {'shares': quantity, 'avg_price': price, 'type': 'short'}
        wallet['positions'][symbol] = pos

    elif action == 'cover':
        # Cover short: buy back shares to close short position
        pos = wallet['positions'].get(symbol)
        if not pos or pos.get('type') != 'short':
            return jsonify({'error': f'No short position in {symbol}'}), 400
        if quantity > pos['shares']:
            return jsonify({'error': f'Only {pos["shares"]} shares shorted'}), 400
        # Profit/loss = (entry_price - cover_price) * quantity
        # Return the original margin + profit (or - loss)
        margin_returned = pos['avg_price'] * quantity  # original collateral back
        cover_pnl = (pos['avg_price'] - price) * quantity    # short profit if price dropped
        wallet['cash'] += margin_returned + cover_pnl
        pos['shares'] -= quantity
        if pos['shares'] == 0:
            del wallet['positions'][symbol]
        else:
            wallet['positions'][symbol] = pos
    else:
        return jsonify({'error': 'Action must be buy, sell, short, or cover'}), 400

    # Calculate P&L for closing trades (sell/cover)
    trade_pnl = None
    if action == 'sell':
        trade_pnl = round(sell_pnl, 2)
    elif action == 'cover':
        trade_pnl = round(cover_pnl, 2)

    trade_record = {
        'time': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'price': price,
        'total': round(total_cost, 2),
    }
    if trade_pnl is not None:
        trade_record['pnl'] = trade_pnl

    wallet['history'].append(trade_record)

    save_wallet()
    result = {'success': True, 'wallet': get_wallet_state()}
    if trade_pnl is not None:
        result['pnl'] = trade_pnl
    return jsonify(result)


def get_wallet_state():
    return {
        'cash': round(wallet['cash'], 2),
        'positions': wallet['positions'],
        'history': wallet['history'],  # all trades
        'initial_balance': wallet['initial_balance'],
    }


@app.route('/api/wallet')
def get_wallet():
    return jsonify(get_wallet_state())


@app.route('/api/wallet/reset', methods=['POST'])
def wallet_reset():
    reset_wallet()
    save_wallet()
    return jsonify({'success': True, 'wallet': get_wallet_state()})


@app.route('/api/quote/<symbol>')
def get_quote(symbol):
    """Get latest price for a symbol."""
    symbol = symbol.upper()
    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 404
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d', interval='1m')
        if hist.empty:
            hist = ticker.history(period='5d', interval='1d')
        if hist.empty:
            return jsonify({'error': 'No price data'}), 404
        last = hist.iloc[-1]
        return jsonify({
            'symbol': symbol,
            'price': round(float(last['Close']), 2),
            'name': STOCKS.get(symbol, symbol),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trade/close', methods=['POST'])
def close_position():
    """Close an entire position at market price (works for both long and short)."""
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    price = float(data.get('price', 0))

    pos = wallet['positions'].get(symbol)
    if not pos or pos['shares'] <= 0:
        return jsonify({'error': f'No position in {symbol}'}), 400
    if price <= 0:
        return jsonify({'error': 'Invalid price'}), 400

    quantity = pos['shares']
    pos_type = pos.get('type', 'long')

    if pos_type == 'short':
        # Cover short: return margin + pnl
        margin_returned = pos['avg_price'] * quantity
        pnl = (pos['avg_price'] - price) * quantity
        wallet['cash'] += margin_returned + pnl
        action = 'cover'
    else:
        # Sell long position
        total = price * quantity
        pnl = (price - pos['avg_price']) * quantity
        wallet['cash'] += total
        action = 'sell'

    del wallet['positions'][symbol]

    wallet['history'].append({
        'time': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'price': price,
        'total': round(price * quantity, 2),
        'pnl': round(pnl, 2),
    })

    save_wallet()
    return jsonify({'success': True, 'wallet': get_wallet_state(), 'pnl': round(pnl, 2)})


# Load saved wallet state on startup
load_wallet()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("  SIMTRADE — Paper Trading Simulator")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 50)
    # Use waitress (production server) - much more stable than Flask dev server
    # Won't crash on errors, handles requests properly
    from waitress import serve
    serve(app, host='0.0.0.0', port=port, threads=4)
