// ─── State ────────────────────────────────────────────────────────
const state = {
    symbol: 'AAPL',
    timeframe: '1h',
    currentPrice: 0,
    candles: [],
    wallet: null,
    // Multi-window
    windows: [],
    activeWindowId: 0,
    layout: '1',
    // Simulation (runs on active window)
    simulation: {
        running: false,
        paused: false,
        timer: null,
        lastClose: 0,
        lastTime: 0,
        candleCount: 0,
    },
    // Modal
    modalPrice: 0,
    stockCategories: null,
};

// Default symbols for layouts
const DEFAULT_SYMBOLS = ['AAPL', 'BTC-USD', 'GC=F', '^GSPC'];

// Timeframe intervals in seconds (for live candle countdown)
const TIMEFRAME_SECONDS = {
    '1s': 1, '1m': 60, '1h': 3600,
    '1d': 86400, '1mo': 2592000, '1y': 31536000,
};

// ─── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    await loadStocks();
    createWindow('AAPL', '1h');
    setLayout('1');
    await loadWallet();
    bindEvents();
    updateTradeTotal();
    initPaint();
});

// ─── API Helpers ──────────────────────────────────────────────────
async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    return res.json();
}

// ─── Load Stocks ──────────────────────────────────────────────────
async function loadStocks() {
    state.stockCategories = await api('/api/stocks');
    populateStockSelector(document.getElementById('stock-selector'), state.symbol);
}

function populateStockSelector(sel, selectedSymbol) {
    sel.innerHTML = '';
    if (!state.stockCategories) return;
    for (const [category, stocks] of Object.entries(state.stockCategories)) {
        const group = document.createElement('optgroup');
        group.label = category;
        for (const [sym, name] of Object.entries(stocks)) {
            const opt = document.createElement('option');
            opt.value = sym;
            opt.textContent = `${sym} — ${name}`;
            if (sym === selectedSymbol) opt.selected = true;
            group.appendChild(opt);
        }
        sel.appendChild(group);
    }
}

// ═══════════════════════════════════════════════════════════════════
// ─── MULTI-WINDOW CHART SYSTEM ───────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

function getActiveWindow() {
    return state.windows[state.activeWindowId] || state.windows[0];
}

function createWindow(symbol = 'AAPL', timeframe = '1h') {
    const id = state.windows.length;
    const win = {
        id,
        symbol,
        timeframe,
        candles: [],
        chart: null,
        candleSeries: null,
        volumeSeries: null,
        indicatorSeries: {},
        currentPrice: 0,
        container: null,
        // Live candle updates
        liveInterval: null,
        countdownInterval: null,
        nextCandleTime: 0,
    };

    // Create DOM
    const div = document.createElement('div');
    div.className = 'chart-window' + (id === 0 ? ' active' : '');
    div.dataset.windowId = id;
    div.innerHTML = `
        <div class="window-header">
            <select class="window-symbol-selector"></select>
            <div class="window-timeframes">
                <button class="tf-btn${timeframe === '1m' ? ' active' : ''}" data-tf="1m">1M</button>
                <button class="tf-btn${timeframe === '1h' ? ' active' : ''}" data-tf="1h">1H</button>
                <button class="tf-btn${timeframe === '1d' ? ' active' : ''}" data-tf="1d">1D</button>
            </div>
            <span class="window-live-badge" style="display:none;"><span class="live-dot"></span>LIVE</span>
            <span class="window-price">--</span>
        </div>
        <div class="window-chart-wrapper">
            <div class="window-chart-container"></div>
            <div class="chart-countdown-overlay" style="display:none;">
                <span class="countdown-label">Next candle</span>
                <span class="countdown-time">--:--</span>
            </div>
            <canvas class="window-drawing-canvas"></canvas>
        </div>
    `;

    // Populate the window's symbol selector
    const winSel = div.querySelector('.window-symbol-selector');
    populateStockSelector(winSel, symbol);

    win.container = div;
    state.windows.push(win);
    document.getElementById('chart-grid').appendChild(div);

    // Click to focus
    div.addEventListener('mousedown', () => setActiveWindow(id));

    // Window symbol change
    winSel.addEventListener('change', async () => {
        win.symbol = winSel.value;
        await loadWindowChart(win);
        if (win.id === state.activeWindowId) syncLeftPanel(win);
    });

    // Window timeframe buttons
    div.querySelectorAll('.window-timeframes .tf-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            div.querySelectorAll('.window-timeframes .tf-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            win.timeframe = btn.dataset.tf;
            await loadWindowChart(win);
            if (win.id === state.activeWindowId) syncLeftPanel(win);
        });
    });

    // Create chart
    createWindowChart(win);
    loadWindowChart(win);

    return win;
}

function createWindowChart(win) {
    const container = win.container.querySelector('.window-chart-container');

    win.chart = LightweightCharts.createChart(container, {
        layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
        grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#30363d' },
        timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
    });

    win.candleSeries = win.chart.addCandlestickSeries({
        upColor: '#3fb950', downColor: '#f85149',
        borderDownColor: '#f85149', borderUpColor: '#3fb950',
        wickDownColor: '#f85149', wickUpColor: '#3fb950',
    });

    win.volumeSeries = win.chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: '',
    });
    win.volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
    });

    // ResizeObserver
    const ro = new ResizeObserver(() => {
        if (container.clientWidth > 0 && container.clientHeight > 0) {
            win.chart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight,
            });
        }
    });
    ro.observe(container);

    // Crosshair for price
    win.chart.subscribeCrosshairMove((param) => {
        if (param.seriesData && param.seriesData.has(win.candleSeries)) {
            const data = param.seriesData.get(win.candleSeries);
            if (data && data.close !== undefined) {
                win.currentPrice = data.close;
                const priceEl = win.container.querySelector('.window-price');
                if (priceEl) priceEl.textContent = '$' + data.close.toFixed(2);
                if (win.id === state.activeWindowId) {
                    state.currentPrice = data.close;
                }
            }
        }
    });
}

async function loadWindowChart(win) {
    // Stop live updates while reloading
    stopLiveUpdates(win);

    try {
        const candles = await api(`/api/chart/${win.symbol}?timeframe=${win.timeframe}`);
        win.candles = candles;

        win.candleSeries.setData(candles.map(c => ({
            time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
        })));

        win.volumeSeries.setData(candles.map(c => ({
            time: c.time, value: c.volume,
            color: c.close >= c.open ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)',
        })));

        if (candles.length > 0) {
            const last = candles[candles.length - 1];
            win.currentPrice = last.close;
            const priceEl = win.container.querySelector('.window-price');
            if (priceEl) priceEl.textContent = '$' + last.close.toFixed(2);
        }

        win.chart.timeScale().fitContent();

        // If this is the active window, update left panel
        if (win.id === state.activeWindowId) {
            syncLeftPanel(win);
            await loadIndicators();
        }

        // Start live candle updates (only if simulation is NOT running on this window)
        if (!state.simulation.running || win.id !== state.activeWindowId) {
            startLiveUpdates(win);
        }
    } catch (e) {
        console.error(`Failed to load chart for ${win.symbol}:`, e);
    }
}

function setActiveWindow(id) {
    if (id === state.activeWindowId) return;
    state.activeWindowId = id;

    document.querySelectorAll('.chart-window').forEach(el => {
        el.classList.toggle('active', parseInt(el.dataset.windowId) === id);
    });

    const win = getActiveWindow();
    if (win) {
        syncLeftPanel(win);
        // Update paint canvas reference
        updatePaintCanvas();
    }
}

function syncLeftPanel(win) {
    state.symbol = win.symbol;
    state.timeframe = win.timeframe;
    state.currentPrice = win.currentPrice;
    state.candles = win.candles;

    document.getElementById('stock-selector').value = win.symbol;
    updatePriceDisplay(win.currentPrice);
    document.getElementById('trade-price').value = win.currentPrice.toFixed(2);
    updateTradeTotal();

    // Sync timeframe buttons in left panel
    document.querySelectorAll('#controls-panel .tf-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.tf === win.timeframe);
    });
}

function setLayout(layout) {
    state.layout = layout;
    const grid = document.getElementById('chart-grid');

    switch (layout) {
        case '1':
            grid.style.gridTemplateColumns = '1fr';
            grid.style.gridTemplateRows = '1fr';
            ensureWindowCount(1);
            break;
        case '2h':
            grid.style.gridTemplateColumns = '1fr 1fr';
            grid.style.gridTemplateRows = '1fr';
            ensureWindowCount(2);
            break;
        case '4':
            grid.style.gridTemplateColumns = '1fr 1fr';
            grid.style.gridTemplateRows = '1fr 1fr';
            ensureWindowCount(4);
            break;
    }

    // Update layout buttons
    document.querySelectorAll('.layout-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.layout === layout);
    });

    // Trigger resize on all visible charts after a small delay
    setTimeout(() => {
        state.windows.forEach((win, i) => {
            if (win.chart && win.container.style.display !== 'none') {
                const container = win.container.querySelector('.window-chart-container');
                if (container.clientWidth > 0 && container.clientHeight > 0) {
                    win.chart.applyOptions({
                        width: container.clientWidth,
                        height: container.clientHeight,
                    });
                    win.chart.timeScale().fitContent();
                }
            }
        });
    }, 100);
}

function ensureWindowCount(count) {
    // Create new windows if needed
    while (state.windows.length < count) {
        const sym = DEFAULT_SYMBOLS[state.windows.length] || 'AAPL';
        createWindow(sym, '1h');
    }

    // Show/hide windows and manage live updates
    state.windows.forEach((win, i) => {
        const wasHidden = win.container.style.display === 'none';
        const shouldShow = i < count;
        win.container.style.display = shouldShow ? '' : 'none';

        if (!shouldShow) {
            // Stop live updates for hidden windows to save API calls
            stopLiveUpdates(win);
        } else if (wasHidden && shouldShow && win.candles && win.candles.length > 0) {
            // Restart live updates for newly visible windows
            startLiveUpdates(win);
        }
    });

    // Make sure active window is visible
    if (state.activeWindowId >= count) {
        setActiveWindow(0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// ─── LIVE CANDLE UPDATES + COUNTDOWN TIMER ──────────────────────
// ═══════════════════════════════════════════════════════════════════

function startLiveUpdates(win) {
    // Stop any existing live updates first
    stopLiveUpdates(win);

    if (!win.candles || win.candles.length === 0) return;

    const tfSeconds = TIMEFRAME_SECONDS[win.timeframe] || 86400;
    const lastCandle = win.candles[win.candles.length - 1];

    // Calculate when the next candle should start
    win.nextCandleTime = lastCandle.time + tfSeconds;

    // Show the LIVE badge in header
    const liveBadge = win.container.querySelector('.window-live-badge');
    if (liveBadge) liveBadge.style.display = 'flex';

    // Show the countdown overlay on the chart
    const countdownOverlay = win.container.querySelector('.chart-countdown-overlay');
    if (countdownOverlay) countdownOverlay.style.display = 'flex';

    // Track previous price for flash effect
    let prevPrice = win.currentPrice;

    // --- Quote polling (every 3 seconds for responsive candle movement) ---
    async function pollQuote() {
        if (!win.candles || win.candles.length === 0) return;
        try {
            const quote = await api(`/api/quote/${win.symbol}`);
            if (quote.error) return;

            const price = quote.price;
            const now = Math.floor(Date.now() / 1000);

            // Check if we need to roll over to a new candle
            if (now >= win.nextCandleTime) {
                // Close the current candle as-is, start a new one
                const prevCandle = win.candles[win.candles.length - 1];

                // Create the new candle
                const newCandle = {
                    time: win.nextCandleTime,
                    open: prevCandle.close,
                    high: Math.max(prevCandle.close, price),
                    low: Math.min(prevCandle.close, price),
                    close: price,
                    volume: 0,
                };
                win.candles.push(newCandle);

                win.candleSeries.update({
                    time: newCandle.time, open: newCandle.open,
                    high: newCandle.high, low: newCandle.low, close: newCandle.close,
                });
                win.volumeSeries.update({
                    time: newCandle.time, value: 0,
                    color: newCandle.close >= newCandle.open ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)',
                });

                // Advance the next candle time (skip forward if multiple intervals passed)
                while (win.nextCandleTime <= now) {
                    win.nextCandleTime += tfSeconds;
                }
            } else {
                // Update the current (last) candle in-place — THIS is what makes the candle move
                const lastIdx = win.candles.length - 1;
                const candle = win.candles[lastIdx];
                candle.close = price;
                candle.high = Math.max(candle.high, price);
                candle.low = Math.min(candle.low, price);

                // Update chart with same timestamp = candle body + wicks move visually
                win.candleSeries.update({
                    time: candle.time, open: candle.open,
                    high: candle.high, low: candle.low, close: candle.close,
                });
            }

            // Flash the price green/red based on direction
            const priceEl = win.container.querySelector('.window-price');
            if (priceEl) {
                priceEl.textContent = '$' + price.toFixed(2);
                if (price > prevPrice) {
                    priceEl.classList.remove('flash-down');
                    priceEl.classList.add('flash-up');
                } else if (price < prevPrice) {
                    priceEl.classList.remove('flash-up');
                    priceEl.classList.add('flash-down');
                }
                // Remove flash class after 1.5s so it can flash again next update
                setTimeout(() => {
                    priceEl.classList.remove('flash-up', 'flash-down');
                }, 1500);
            }

            // Update price state
            win.currentPrice = price;
            prevPrice = price;

            if (win.id === state.activeWindowId) {
                state.currentPrice = price;
                updatePriceDisplay(price);
                document.getElementById('trade-price').value = price.toFixed(2);
                updateTradeTotal();
            }
        } catch (e) {
            // Silently ignore polling errors
        }
    }

    // Poll immediately, then every 3 seconds
    pollQuote();
    win.liveInterval = setInterval(pollQuote, 3000);

    // --- Countdown timer (ticks every second, displayed on chart overlay) ---
    function tickCountdown() {
        const now = Math.floor(Date.now() / 1000);
        let remaining = win.nextCandleTime - now;
        if (remaining < 0) remaining = 0;

        const overlay = win.container.querySelector('.chart-countdown-overlay');
        if (!overlay) return;

        const timeEl = overlay.querySelector('.countdown-time');
        if (!timeEl) return;

        const formatted = formatCountdown(remaining, tfSeconds);
        timeEl.textContent = formatted;

        // Make it flash orange when less than 10% of the interval remains
        const urgentThreshold = Math.max(Math.floor(tfSeconds * 0.1), 5);
        if (remaining <= urgentThreshold && remaining > 0) {
            timeEl.classList.add('urgent');
        } else {
            timeEl.classList.remove('urgent');
        }
    }

    tickCountdown();
    win.countdownInterval = setInterval(tickCountdown, 1000);
}

function stopLiveUpdates(win) {
    if (win.liveInterval) {
        clearInterval(win.liveInterval);
        win.liveInterval = null;
    }
    if (win.countdownInterval) {
        clearInterval(win.countdownInterval);
        win.countdownInterval = null;
    }
    if (!win.container) return;

    const liveBadge = win.container.querySelector('.window-live-badge');
    if (liveBadge) liveBadge.style.display = 'none';

    const overlay = win.container.querySelector('.chart-countdown-overlay');
    if (overlay) overlay.style.display = 'none';
}

function formatCountdown(seconds, tfSeconds) {
    if (seconds <= 0) return '00:00';

    if (tfSeconds >= 86400) {
        // Daily or longer: show Xd HH:MM
        const days = Math.floor(seconds / 86400);
        const hrs = Math.floor((seconds % 86400) / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        if (days > 0) return `${days}d ${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}`;
        return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}`;
    }
    if (tfSeconds >= 3600) {
        // Hourly: show HH:MM:SS
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
    if (tfSeconds >= 60) {
        // Minute: show MM:SS
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
    // Seconds: show SS
    return String(seconds).padStart(2, '0');
}

// ═══════════════════════════════════════════════════════════════════
// ─── INDICATORS (on active window) ───────────────────────────────
// ═══════════════════════════════════════════════════════════════════

async function loadIndicators() {
    const win = getActiveWindow();
    if (!win) return;

    // Remove old indicator series
    for (const [key, series] of Object.entries(win.indicatorSeries)) {
        try { win.chart.removeSeries(series); } catch {}
    }
    win.indicatorSeries = {};

    const checks = {
        sma20: document.getElementById('ind-sma20').checked,
        sma50: document.getElementById('ind-sma50').checked,
        ema12: document.getElementById('ind-ema12').checked,
        ema26: document.getElementById('ind-ema26').checked,
        bollinger: document.getElementById('ind-bollinger').checked,
        rsi: document.getElementById('ind-rsi').checked,
        macd: document.getElementById('ind-macd').checked,
    };

    const anyChecked = Object.values(checks).some(v => v);
    if (!anyChecked) return;

    try {
        const ind = await api(`/api/indicators/${win.symbol}?timeframe=${win.timeframe}`);
        const times = ind.times;

        if (checks.sma20) {
            const s = win.chart.addLineSeries({ color: '#d29922', lineWidth: 1, title: 'SMA20' });
            s.setData(times.map((t, i) => ({ time: t, value: ind.sma20[i] })));
            win.indicatorSeries.sma20 = s;
        }
        if (checks.sma50) {
            const s = win.chart.addLineSeries({ color: '#f0883e', lineWidth: 1, title: 'SMA50' });
            s.setData(times.map((t, i) => ({ time: t, value: ind.sma50[i] })));
            win.indicatorSeries.sma50 = s;
        }
        if (checks.ema12) {
            const s = win.chart.addLineSeries({ color: '#58a6ff', lineWidth: 1, title: 'EMA12' });
            s.setData(times.map((t, i) => ({ time: t, value: ind.ema12[i] })));
            win.indicatorSeries.ema12 = s;
        }
        if (checks.ema26) {
            const s = win.chart.addLineSeries({ color: '#bc8cff', lineWidth: 1, title: 'EMA26' });
            s.setData(times.map((t, i) => ({ time: t, value: ind.ema26[i] })));
            win.indicatorSeries.ema26 = s;
        }
        if (checks.bollinger) {
            const upper = win.chart.addLineSeries({ color: '#8b949e', lineWidth: 1, lineStyle: 2, title: 'BB Upper' });
            upper.setData(times.map((t, i) => ({ time: t, value: ind.bollinger.upper[i] })));
            win.indicatorSeries.bb_upper = upper;
            const middle = win.chart.addLineSeries({ color: '#8b949e', lineWidth: 1, title: 'BB Mid' });
            middle.setData(times.map((t, i) => ({ time: t, value: ind.bollinger.middle[i] })));
            win.indicatorSeries.bb_middle = middle;
            const lower = win.chart.addLineSeries({ color: '#8b949e', lineWidth: 1, lineStyle: 2, title: 'BB Lower' });
            lower.setData(times.map((t, i) => ({ time: t, value: ind.bollinger.lower[i] })));
            win.indicatorSeries.bb_lower = lower;
        }
    } catch (e) {
        console.error('Indicator error:', e);
    }
}

// ═══════════════════════════════════════════════════════════════════
// ─── SIMULATION (on active window) ───────────────────────────────
// ═══════════════════════════════════════════════════════════════════

async function startSimulation() {
    if (state.simulation.running) return;
    const win = getActiveWindow();
    if (!win || !win.candles || win.candles.length === 0) {
        showToast('Load a chart first', 'error');
        return;
    }

    // Stop live updates on this window — simulation generates its own candles
    stopLiveUpdates(win);

    const lastCandle = win.candles[win.candles.length - 1];
    state.simulation.lastClose = lastCandle.close;
    state.simulation.lastTime = lastCandle.time;
    state.simulation.candleCount = 0;
    state.simulation.running = true;
    state.simulation.paused = false;

    document.getElementById('sim-progress-wrap').style.display = 'block';
    updateSimButtons();
    updateSimStatus('Running');
    showToast('Simulation started', 'info');
    runSimTick();
}

async function runSimTick() {
    if (!state.simulation.running || state.simulation.paused) return;
    const speed = parseInt(document.getElementById('sim-speed').value);
    const win = getActiveWindow();

    state.simulation.timer = setTimeout(async () => {
        const sim = state.simulation;
        try {
            const candle = await api(
                `/api/simulation/next/${win.symbol}?timeframe=${win.timeframe}` +
                `&last_close=${sim.lastClose}&last_time=${sim.lastTime}`
            );

            if (candle.error) {
                showToast(candle.error, 'error');
                stopSimulation();
                return;
            }

            win.candleSeries.update({
                time: candle.time, open: candle.open, high: candle.high,
                low: candle.low, close: candle.close,
            });
            win.volumeSeries.update({
                time: candle.time, value: candle.volume,
                color: candle.close >= candle.open ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)',
            });

            sim.lastClose = candle.close;
            sim.lastTime = candle.time;
            sim.candleCount++;
            win.candles.push(candle);
            win.currentPrice = candle.close;

            const priceEl = win.container.querySelector('.window-price');
            if (priceEl) priceEl.textContent = '$' + candle.close.toFixed(2);

            if (win.id === state.activeWindowId) {
                state.currentPrice = candle.close;
                updatePriceDisplay(candle.close);
                document.getElementById('trade-price').value = candle.close.toFixed(2);
                updateTradeTotal();
                updateWalletUI();
            }

            document.getElementById('sim-progress').style.width = '100%';
            document.getElementById('sim-candle-count').textContent = `${sim.candleCount} new candles`;

            runSimTick();
        } catch (e) {
            console.error('Sim tick error:', e);
            runSimTick();
        }
    }, speed);
}

function pauseSimulation() {
    state.simulation.paused = !state.simulation.paused;
    if (state.simulation.paused) {
        clearTimeout(state.simulation.timer);
        updateSimStatus('Paused');
    } else {
        updateSimStatus('Running');
        runSimTick();
    }
    updateSimButtons();
}

function stopSimulation() {
    state.simulation.running = false;
    state.simulation.paused = false;
    clearTimeout(state.simulation.timer);
    document.getElementById('sim-progress-wrap').style.display = 'none';
    updateSimButtons();
    updateSimStatus('Stopped');
}

function updateSimButtons() {
    const sim = state.simulation;
    document.getElementById('btn-sim-start').disabled = sim.running;
    document.getElementById('btn-sim-pause').disabled = !sim.running;
    document.getElementById('btn-sim-pause').textContent = sim.paused ? 'Resume' : 'Pause';
    document.getElementById('btn-sim-stop').disabled = !sim.running;
}

function updateSimStatus(text) {
    const badge = document.getElementById('sim-status');
    badge.textContent = text;
    badge.className = 'sim-badge';
    if (text === 'Running') badge.classList.add('running');
    else if (text === 'Paused') badge.classList.add('paused');
}

// ═══════════════════════════════════════════════════════════════════
// ─── TRADING ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

async function executeTrade(action) {
    const qty = parseInt(document.getElementById('trade-qty').value);
    const price = parseFloat(document.getElementById('trade-price').value);

    if (!qty || qty <= 0) { showToast('Enter a valid quantity', 'error'); return; }

    try {
        const res = await api('/api/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: state.symbol, action, quantity: qty, price }),
        });

        if (res.error) { showToast(res.error, 'error'); return; }

        state.wallet = res.wallet;
        updateWalletUI();
        showToast(`${action.toUpperCase()} ${qty} ${state.symbol} @ $${price.toFixed(2)}`, 'success');
    } catch (e) {
        showToast('Trade failed', 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════
// ─── TRADE MODAL (2-step: Long/Short → Details) ────────────────
// ═══════════════════════════════════════════════════════════════════

// Track modal state
let modalDirection = 'long'; // 'long' or 'short'

async function openTradeModal(preselectedSymbol = null) {
    const modal = document.getElementById('trade-modal');
    modal.classList.remove('hidden');

    // Always show step 1 first
    document.getElementById('modal-step-direction').style.display = '';
    document.getElementById('modal-step-details').style.display = 'none';

    const sel = document.getElementById('modal-symbol-selector');
    if (sel.children.length === 0) {
        populateStockSelector(sel, preselectedSymbol || state.symbol);
    }

    if (preselectedSymbol) sel.value = preselectedSymbol;

    await updateModalPrice();
}

async function updateModalPrice() {
    const symbol = document.getElementById('modal-symbol-selector').value;
    const nameEl = document.getElementById('modal-symbol-name');
    const priceEl = document.getElementById('modal-current-price');

    nameEl.textContent = 'Loading...';
    priceEl.textContent = '--';

    try {
        const quote = await api(`/api/quote/${symbol}`);
        if (quote.error) {
            nameEl.textContent = symbol;
            priceEl.textContent = 'N/A';
            return;
        }
        nameEl.textContent = quote.name;
        priceEl.textContent = '$' + quote.price.toFixed(2);
        state.modalPrice = quote.price;
        // Update total on step 2
        const qty = parseInt(document.getElementById('modal-trade-qty').value) || 0;
        document.getElementById('modal-trade-total').textContent = formatMoney(qty * quote.price);
    } catch (e) {
        nameEl.textContent = symbol;
        priceEl.textContent = 'Error';
    }
}

function selectDirection(direction) {
    modalDirection = direction;

    // Hide step 1, show step 2
    document.getElementById('modal-step-direction').style.display = 'none';
    document.getElementById('modal-step-details').style.display = '';

    // Update direction badge
    const badge = document.getElementById('modal-direction-badge');
    badge.textContent = direction === 'long' ? 'LONG — Buy to profit when price rises' : 'SHORT — Sell to profit when price drops';
    badge.className = 'modal-direction-badge ' + direction;

    // Update detail price display
    const symbol = document.getElementById('modal-symbol-selector').value;
    document.getElementById('modal-detail-name').textContent =
        document.getElementById('modal-symbol-name').textContent;
    document.getElementById('modal-detail-price').textContent =
        document.getElementById('modal-current-price').textContent;

    // Update execute button style
    const execBtn = document.getElementById('btn-modal-execute');
    if (direction === 'long') {
        execBtn.className = 'btn btn-buy';
        execBtn.textContent = 'Execute Long Trade';
    } else {
        execBtn.className = 'btn btn-sell';
        execBtn.textContent = 'Execute Short Trade';
    }
    execBtn.style.cssText = 'flex:1; padding:12px; font-size:14px;';

    // Update total
    const qty = parseInt(document.getElementById('modal-trade-qty').value) || 0;
    document.getElementById('modal-trade-total').textContent = formatMoney(qty * state.modalPrice);
}

function modalGoBack() {
    document.getElementById('modal-step-direction').style.display = '';
    document.getElementById('modal-step-details').style.display = 'none';
}

async function executeModalTrade() {
    const symbol = document.getElementById('modal-symbol-selector').value;
    const qty = parseInt(document.getElementById('modal-trade-qty').value);
    const price = state.modalPrice;

    if (!qty || qty <= 0 || !price) {
        showToast('Invalid trade parameters', 'error');
        return;
    }

    // Determine action based on direction
    const action = modalDirection === 'long' ? 'buy' : 'short';

    try {
        const res = await api('/api/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, action, quantity: qty, price }),
        });

        if (res.error) { showToast(res.error, 'error'); return; }

        state.wallet = res.wallet;
        updateWalletUI();
        const label = modalDirection === 'long' ? 'LONG BUY' : 'SHORT SELL';
        showToast(`${label} ${qty} ${symbol} @ $${price.toFixed(2)}`, 'success');
        document.getElementById('trade-modal').classList.add('hidden');
    } catch (e) {
        showToast('Trade failed', 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════
// ─── WALLET ──────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

async function loadWallet() {
    state.wallet = await api('/api/wallet');
    updateWalletUI();
}

function updateWalletUI() {
    if (!state.wallet) return;
    const w = state.wallet;

    // Backup to localStorage
    try { localStorage.setItem('simtrade_wallet_backup', JSON.stringify(w)); } catch (e) {}

    // Top bar
    document.getElementById('topbar-cash').textContent = formatMoney(w.cash);

    let posValue = 0;
    for (const [sym, pos] of Object.entries(w.positions)) {
        posValue += pos.shares * pos.avg_price;
    }
    const totalValue = w.cash + posValue;
    document.getElementById('topbar-portfolio').textContent = formatMoney(totalValue);

    const pnl = totalValue - w.initial_balance;
    const pnlEl = document.getElementById('topbar-pnl');
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + formatMoney(pnl);
    pnlEl.className = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral';

    // Positions with quick actions
    const posEl = document.getElementById('positions-list');
    if (Object.keys(w.positions).length === 0) {
        posEl.innerHTML = '<p class="empty-msg">No open positions</p>';
    } else {
        posEl.innerHTML = '';
        for (const [sym, pos] of Object.entries(w.positions)) {
            const card = document.createElement('div');
            card.className = 'position-card';
            const isShort = pos.type === 'short';
            const cost = pos.shares * pos.avg_price;
            const marketVal = pos.shares * state.currentPrice;
            // For longs: profit when price goes up. For shorts: profit when price goes down.
            let posPnl = 0;
            if (sym === state.symbol) {
                posPnl = isShort ? (pos.avg_price - state.currentPrice) * pos.shares : marketVal - cost;
            }
            const typeBadge = isShort
                ? '<span style="font-size:9px;font-weight:700;background:var(--red-bg);color:var(--red);padding:1px 5px;border-radius:3px;margin-left:6px;">SHORT</span>'
                : '<span style="font-size:9px;font-weight:700;background:var(--green-bg);color:var(--green);padding:1px 5px;border-radius:3px;margin-left:6px;">LONG</span>';
            const closeLabel = isShort ? 'Cover' : 'Close';
            card.innerHTML = `
                <div class="pos-header">
                    <div class="symbol">${sym}${typeBadge}</div>
                    <div class="pos-actions">
                        <button class="btn-pos-add" data-symbol="${sym}" title="Add to position">+</button>
                        <button class="btn-pos-close" data-symbol="${sym}" title="${closeLabel} position">${closeLabel}</button>
                    </div>
                </div>
                <div class="details">
                    ${pos.shares} shares @ $${pos.avg_price.toFixed(2)}<br>
                    ${isShort ? 'Margin' : 'Cost'}: ${formatMoney(cost)}
                </div>
                ${sym === state.symbol ? `<div class="pnl ${posPnl >= 0 ? 'positive' : 'negative'}">${posPnl >= 0 ? '+' : '-'}${formatMoney(Math.abs(posPnl))}</div>` : ''}
            `;
            posEl.appendChild(card);
        }

        // Quick action buttons
        posEl.querySelectorAll('.btn-pos-add').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                openTradeModal(btn.dataset.symbol);
            });
        });
        posEl.querySelectorAll('.btn-pos-close').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const sym = btn.dataset.symbol;
                if (confirm(`Close entire ${sym} position?`)) {
                    try {
                        const quote = await api(`/api/quote/${sym}`);
                        if (quote.error) { showToast(quote.error, 'error'); return; }
                        const res = await api('/api/trade/close', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ symbol: sym, price: quote.price }),
                        });
                        if (res.error) { showToast(res.error, 'error'); return; }
                        state.wallet = res.wallet;
                        updateWalletUI();
                        showToast(`Closed ${sym} position`, 'success');
                    } catch (e) { showToast('Failed to close position', 'error'); }
                }
            });
        });
    }

    // Trade history
    const histEl = document.getElementById('trade-history');
    if (!w.history || w.history.length === 0) {
        histEl.innerHTML = '<p class="empty-msg">No trades yet</p>';
    } else {
        histEl.innerHTML = '';
        const reversed = [...w.history].reverse();
        for (const trade of reversed) {
            const rec = document.createElement('div');
            rec.className = 'trade-record';
            rec.innerHTML = `
                <span class="trade-tag ${trade.action}">${trade.action}</span>
                <span class="trade-info">${trade.quantity} ${trade.symbol} @ $${trade.price.toFixed(2)}</span>
                <span class="trade-amount">${formatMoney(trade.total)}</span>
            `;
            histEl.appendChild(rec);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// ─── UI HELPERS ──────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

function updatePriceDisplay(price) {
    state.currentPrice = price;
    document.getElementById('stock-price').textContent = '$' + price.toFixed(2);
}

function updateTradeTotal() {
    const qty = parseInt(document.getElementById('trade-qty').value) || 0;
    const price = parseFloat(document.getElementById('trade-price').value) || 0;
    document.getElementById('trade-total').textContent = formatMoney(qty * price);
}

function formatMoney(val) {
    return '$' + Math.abs(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function showToast(msg, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = `toast ${type} show`;
    setTimeout(() => { toast.className = 'toast hidden'; }, 3000);
}

// ═══════════════════════════════════════════════════════════════════
// ─── EVENT BINDINGS ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

function bindEvents() {
    // Left panel stock selector -> changes active window's symbol
    document.getElementById('stock-selector').addEventListener('change', async (e) => {
        const win = getActiveWindow();
        if (!win) return;
        win.symbol = e.target.value;
        // Also update the window's own selector
        const winSel = win.container.querySelector('.window-symbol-selector');
        if (winSel) winSel.value = e.target.value;
        stopSimulation();
        await loadWindowChart(win);
        syncLeftPanel(win);
        await loadWallet();
    });

    // Left panel timeframe buttons -> changes active window's timeframe
    document.querySelectorAll('#controls-panel .tf-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const win = getActiveWindow();
            if (!win) return;
            document.querySelectorAll('#controls-panel .tf-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            win.timeframe = btn.dataset.tf;
            // Also update the window's own timeframe buttons
            win.container.querySelectorAll('.window-timeframes .tf-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.tf === btn.dataset.tf);
            });
            stopSimulation();
            await loadWindowChart(win);
        });
    });

    // Indicator checkboxes
    document.querySelectorAll('.indicator-toggles input').forEach(cb => {
        cb.addEventListener('change', () => loadIndicators());
    });

    // Simulation
    document.getElementById('btn-sim-start').addEventListener('click', startSimulation);
    document.getElementById('btn-sim-pause').addEventListener('click', pauseSimulation);
    document.getElementById('btn-sim-stop').addEventListener('click', () => {
        stopSimulation();
        const win = getActiveWindow();
        if (win) loadWindowChart(win);
    });

    // Trading - left panel buttons open the modal for Long/Short selection
    document.getElementById('btn-buy').addEventListener('click', () => openTradeModal());
    document.getElementById('btn-sell').addEventListener('click', () => openTradeModal());
    document.getElementById('trade-qty').addEventListener('input', updateTradeTotal);

    // Reset wallet
    document.getElementById('btn-reset-wallet').addEventListener('click', async () => {
        if (confirm('Reset wallet to $100,000? All positions and history will be cleared.')) {
            await api('/api/wallet/reset', { method: 'POST' });
            await loadWallet();
            showToast('Wallet reset to $100,000', 'info');
        }
    });

    // Layout buttons
    document.querySelectorAll('.layout-btn').forEach(btn => {
        btn.addEventListener('click', () => setLayout(btn.dataset.layout));
    });

    // New Trade modal
    document.getElementById('btn-new-trade').addEventListener('click', () => openTradeModal());
    document.getElementById('btn-close-modal').addEventListener('click', () => {
        document.getElementById('trade-modal').classList.add('hidden');
    });
    document.getElementById('modal-symbol-selector').addEventListener('change', updateModalPrice);
    document.getElementById('btn-modal-execute').addEventListener('click', executeModalTrade);

    // Long/Short direction buttons (step 1)
    document.getElementById('btn-go-long').addEventListener('click', () => selectDirection('long'));
    document.getElementById('btn-go-short').addEventListener('click', () => selectDirection('short'));

    // Back button (step 2 → step 1)
    document.getElementById('btn-modal-back').addEventListener('click', modalGoBack);

    // Modal qty input
    document.getElementById('modal-trade-qty').addEventListener('input', () => {
        const qty = parseInt(document.getElementById('modal-trade-qty').value) || 0;
        const price = state.modalPrice || 0;
        document.getElementById('modal-trade-total').textContent = formatMoney(qty * price);
    });

    // Close modal on overlay click
    document.getElementById('trade-modal').addEventListener('click', (e) => {
        if (e.target.id === 'trade-modal') {
            document.getElementById('trade-modal').classList.add('hidden');
        }
    });

    // Close modal on Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            document.getElementById('trade-modal').classList.add('hidden');
        }
    });
}

// ═══════════════════════════════════════════════════════════════════
// ─── SIMPLE PAINTBRUSH ───────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

const paint = {
    canvas: null,
    ctx: null,
    on: false,
    drawing: false,
    color: '#f85149',
    size: 4,
    strokes: [],
    current: [],
};

function initPaint() {
    updatePaintCanvas();

    // Toggle button
    document.getElementById('btn-paint-toggle').addEventListener('click', togglePaint);

    // Colour buttons
    document.querySelectorAll('.paint-color').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.paint-color').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            paint.color = btn.dataset.color;
        });
    });

    // Brush size
    document.getElementById('paint-size').addEventListener('change', (e) => {
        paint.size = parseInt(e.target.value);
    });

    // Undo / Clear
    document.getElementById('btn-paint-undo').addEventListener('click', () => {
        if (paint.strokes.length === 0) return;
        paint.strokes.pop();
        repaint();
    });
    document.getElementById('btn-paint-clear').addEventListener('click', () => {
        if (paint.strokes.length === 0) return;
        paint.strokes = [];
        repaint();
    });

    // Keyboard: P to toggle, Escape to turn off
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key.toLowerCase() === 'p') togglePaint();
        if (e.key === 'Escape' && paint.on) togglePaint();
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
            e.preventDefault();
            if (paint.strokes.length > 0) { paint.strokes.pop(); repaint(); }
        }
    });
}

function updatePaintCanvas() {
    const win = getActiveWindow();
    if (!win || !win.container) return;

    const canvas = win.container.querySelector('.window-drawing-canvas');
    if (!canvas) return;

    // Remove old event listeners by replacing canvas reference
    if (paint.canvas && paint.canvas !== canvas) {
        paint.canvas.removeEventListener('mousedown', paintStart);
        paint.canvas.removeEventListener('mousemove', paintMove);
        paint.canvas.removeEventListener('mouseup', paintEnd);
        paint.canvas.removeEventListener('mouseleave', paintEnd);
        paint.canvas.classList.remove('active');
    }

    paint.canvas = canvas;
    paint.ctx = canvas.getContext('2d');

    // Size canvas
    sizeCanvas();
    // Attach new observer
    const wrapper = win.container.querySelector('.window-chart-wrapper');
    if (wrapper) {
        new ResizeObserver(sizeCanvas).observe(wrapper);
    }

    // Canvas mouse events
    canvas.addEventListener('mousedown', paintStart);
    canvas.addEventListener('mousemove', paintMove);
    canvas.addEventListener('mouseup', paintEnd);
    canvas.addEventListener('mouseleave', paintEnd);

    // Apply current paint state
    canvas.classList.toggle('active', paint.on);
}

function sizeCanvas() {
    if (!paint.canvas) return;
    const wrapper = paint.canvas.parentElement;
    if (!wrapper) return;
    const dpr = window.devicePixelRatio || 1;
    paint.canvas.width = wrapper.clientWidth * dpr;
    paint.canvas.height = wrapper.clientHeight * dpr;
    paint.canvas.style.width = wrapper.clientWidth + 'px';
    paint.canvas.style.height = wrapper.clientHeight + 'px';
    paint.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    repaint();
}

function togglePaint() {
    paint.on = !paint.on;
    const btn = document.getElementById('btn-paint-toggle');
    btn.classList.toggle('on', paint.on);
    if (paint.canvas) paint.canvas.classList.toggle('active', paint.on);
}

function getXY(e) {
    const r = paint.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function paintStart(e) {
    if (!paint.on) return;
    paint.drawing = true;
    const p = getXY(e);
    paint.current = [p];
    paint.ctx.beginPath();
    paint.ctx.arc(p.x, p.y, paint.size / 2, 0, Math.PI * 2);
    paint.ctx.fillStyle = paint.color;
    paint.ctx.fill();
}

function paintMove(e) {
    if (!paint.drawing) return;
    const p = getXY(e);
    paint.current.push(p);
    const pts = paint.current;
    const ctx = paint.ctx;
    ctx.strokeStyle = paint.color;
    ctx.lineWidth = paint.size;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (pts.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(pts[pts.length - 2].x, pts[pts.length - 2].y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
    }
}

function paintEnd() {
    if (!paint.drawing) return;
    paint.drawing = false;
    if (paint.current.length > 0) {
        paint.strokes.push({
            color: paint.color,
            size: paint.size,
            points: [...paint.current],
        });
    }
    paint.current = [];
}

function repaint() {
    if (!paint.ctx || !paint.canvas) return;
    const ctx = paint.ctx;
    const w = paint.canvas.clientWidth;
    const h = paint.canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    for (const stroke of paint.strokes) {
        if (stroke.points.length === 0) continue;
        ctx.strokeStyle = stroke.color;
        ctx.fillStyle = stroke.color;
        ctx.lineWidth = stroke.size;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        if (stroke.points.length === 1) {
            ctx.beginPath();
            ctx.arc(stroke.points[0].x, stroke.points[0].y, stroke.size / 2, 0, Math.PI * 2);
            ctx.fill();
        } else {
            ctx.beginPath();
            ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
            for (let i = 1; i < stroke.points.length; i++) {
                ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
            }
            ctx.stroke();
        }
    }
}
