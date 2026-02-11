from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os

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


@app.route('/api/simulation/next/<symbol>')
def get_next_candle(symbol):
    """Generate the next synthetic candle continuing from a given price."""
    symbol = symbol.upper()
    if symbol not in STOCKS:
        return jsonify({'error': 'Unknown symbol'}), 404

    timeframe = request.args.get('timeframe', '1d')
    last_close = float(request.args.get('last_close', 100))
    last_time = int(request.args.get('last_time', 0))

    # Timeframe intervals in seconds
    intervals = {
        '1s': 1, '1m': 60, '1h': 3600,
        '1d': 86400, '1mo': 2592000, '1y': 31536000,
    }
    dt = intervals.get(timeframe, 86400)

    # Calculate volatility from recent real data to make it realistic
    candles = fetch_ohlcv(symbol, '1d')
    if candles and len(candles) > 20:
        recent = candles[-20:]
        returns = []
        for i in range(1, len(recent)):
            r = (recent[i]['close'] - recent[i-1]['close']) / recent[i-1]['close']
            returns.append(r)
        volatility = float(np.std(returns)) if returns else 0.015
    else:
        volatility = 0.015

    # Scale volatility to timeframe
    tf_scale = {'1s': 0.05, '1m': 0.15, '1h': 0.5, '1d': 1.0, '1mo': 3.0, '1y': 5.0}
    vol = volatility * tf_scale.get(timeframe, 1.0)

    # Generate a realistic candle using random walk
    drift = np.random.normal(0.0001, vol)
    new_close = round(last_close * (1 + drift), 2)

    intra_vol = abs(drift) + vol * abs(np.random.normal(0, 0.5))
    new_open = round(last_close * (1 + np.random.normal(0, vol * 0.3)), 2)
    new_high = round(max(new_open, new_close) * (1 + abs(np.random.normal(0, intra_vol * 0.5))), 2)
    new_low = round(min(new_open, new_close) * (1 - abs(np.random.normal(0, intra_vol * 0.5))), 2)

    # Ensure high >= open,close and low <= open,close
    new_high = max(new_high, new_open, new_close)
    new_low = min(new_low, new_open, new_close)

    # Volume: random around recent average
    if candles and len(candles) > 5:
        avg_vol = int(np.mean([c['volume'] for c in candles[-5:]]))
        volume = int(avg_vol * (0.5 + np.random.random()))
    else:
        volume = int(1000000 * (0.5 + np.random.random()))

    new_time = last_time + dt

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
        pnl = (pos['avg_price'] - price) * quantity    # short profit if price dropped
        wallet['cash'] += margin_returned + pnl
        pos['shares'] -= quantity
        if pos['shares'] == 0:
            del wallet['positions'][symbol]
        else:
            wallet['positions'][symbol] = pos
    else:
        return jsonify({'error': 'Action must be buy, sell, short, or cover'}), 400

    wallet['history'].append({
        'time': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'price': price,
        'total': round(total_cost, 2),
    })

    save_wallet()
    return jsonify({'success': True, 'wallet': get_wallet_state()})


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
    })

    save_wallet()
    return jsonify({'success': True, 'wallet': get_wallet_state()})


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
