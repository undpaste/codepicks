import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
STRATEGY_WEIGHTS = {
    "Balanced": {'value': 0.25, 'quality': 0.30, 'momentum': 0.20, 'sentiment': 0.15, 'low_vol': 0.10},
    "Quality Compounder": {'value': 0.20, 'quality': 0.50, 'momentum': 0.10, 'sentiment': 0.10, 'low_vol': 0.10},
    "High-Growth Momentum": {'value': 0.10, 'quality': 0.20, 'momentum': 0.45, 'sentiment': 0.20, 'low_vol': 0.05},
    "Deep Value": {'value': 0.50, 'quality': 0.25, 'momentum': 0.10, 'sentiment': 0.10, 'low_vol': 0.05},
}

MARKETS = {
    "🇺🇸 US Market": {"suffix": "", "min_mcap": 2_000_000_000, "currency": "$"},
    "🇮🇩 Indonesia (IDX)": {"suffix": ".JK", "min_mcap": 1_000_000_000, "currency": "Rp"},
}

MAX_WORKERS = 3
CACHE_TTL = 3600
RETRY_ATTEMPTS = 3
Z_SCORE_EXEMPT_SECTORS = {'Financial Services', 'Real Estate', 'Financials'}
EXTREME_RETURN_12M = 200
EXTREME_RETURN_6M = 150

# IDX sector mapping (yfinance sometimes returns 'Unknown' for IDX)
IDX_SECTOR_FALLBACK = {
    'BBCA.JK': 'Financials', 'BBRI.JK': 'Financials', 'BMRI.JK': 'Financials',
    'BBNI.JK': 'Financials', 'TLKM.JK': 'Communication Services',
    'ASII.JK': 'Consumer Cyclical', 'UNVR.JK': 'Consumer Defensive',
    'ICBP.JK': 'Consumer Defensive', 'INDF.JK': 'Consumer Defensive',
    'ADRO.JK': 'Energy', 'PTBA.JK': 'Energy', 'ITMG.JK': 'Energy',
    'ANTM.JK': 'Basic Materials', 'INCO.JK': 'Basic Materials',
    'SMGR.JK': 'Basic Materials', 'INTP.JK': 'Basic Materials',
}

# ============================================================
# UNIVERSE
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_us_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        raw = tables[0]['Symbol'].tolist()
        seen, tickers = set(), []
        for t in raw:
            t = str(t).strip().replace('.', '-')
            if t and t not in seen:
                tickers.append(t); seen.add(t)
        return sorted(tickers)
    except Exception as e:
        st.warning(f"Wikipedia failed ({e}). Using fallback US list.")
        return ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','UNH','XOM',
                'JNJ','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP','KO',
                'AVGO','COST','WMT','TMO','MCD','CSCO','ACN','ABT','CRM','ADBE','DHR',
                'LIN','NKE','TXN','AMD','PM','NEE','WFC','DIS','UPS','RTX','BMY','ORCL']

@st.cache_data(ttl=86400, show_spinner=False)
def get_idx_tickers():
    """Fetch IDX ticker list from Wikipedia or fallback to LQ45 + IDX80."""
    try:
        url = 'https://en.wikipedia.org/wiki/Indonesia_Stock_Exchange'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        # Wikipedia doesn't have a clean table, so use curated LQ45 + IDX80 list
        return get_idx_fallback()
    except Exception:
        return get_idx_fallback()

def get_idx_fallback():
    """Curated list of liquid IDX stocks (LQ45 + IDX80 components)."""
    return [
        'BBCA.JK','BBRI.JK','BMRI.JK','BBNI.JK','TLKM.JK','ASII.JK','UNVR.JK',
        'ICBP.JK','INDF.JK','ADRO.JK','PTBA.JK','ITMG.JK','ANTM.JK','INCO.JK',
        'SMGR.JK','INTP.JK','UNTR.JK','HEAL.JK','MTCN.JK','GOTO.JK','BUKA.JK',
        'ARTO.JK','BRIS.JK','BTPS.JK','EXCL.JK','ISAT.JK','TOWR.JK','MTEL.JK',
        'AMRT.JK','ACES.JK','MAPI.JK','ERAA.JK','CPIN.JK','JPFA.JK','MAIN.JK',
        'BRPT.JK','TPIA.JK','INKP.JK','TKIM.JK','SRIL.JK','PTPP.JK','WIKA.JK',
        'ADHI.JK','WSKT.JK','JSMR.JK','PGAS.JK','AKRA.JK','MEDC.JK','ELSA.JK',
        'HRUM.JK','BUMI.JK','DOID.JK','HRTA.JK','MDKA.JK','NCKL.JK','TINS.JK',
        'SCMA.JK','MNCN.JK','EMTK.JK','FILM.JK','SIDO.JK','KAEF.JK','INAF.JK',
        'MIKA.JK','SILO.JK','PRDA.JK','SAME.JK','BFIN.JK','ADMF.JK','PNLF.JK',
        'TUGU.JK','ASBI.JK','LPGI.JK','MEGA.JK','BJTM.JK','BJBR.JK','BABP.JK',
        'AGRO.JK','PNBN.JK','NISP.JK','MCOR.JK'
    ]

def sample_universe(all_tickers, size, use_full):
    if use_full or size >= len(all_tickers):
        return all_tickers
    rng = random.Random(42)
    return rng.sample(all_tickers, size)

# ============================================================
# HELPERS
# ============================================================
def calculate_altman_z(info, sector):
    if sector in Z_SCORE_EXEMPT_SECTORS:
        return np.nan
    try:
        wc = info.get('totalCurrentAssets', 0) - info.get('totalCurrentLiabilities', 0)
        ta = info.get('totalAssets', 1) or 1
        re = info.get('retainedEarnings', 0)
        ebit = info.get('ebit', 0)
        mcap = info.get('marketCap', 0)
        tl = info.get('totalLiabilities', 1) or 1
        sales = info.get('totalRevenue', 0)
        if ta <= 0 or tl <= 0: return np.nan
        return (1.2*(wc/ta) + 1.4*(re/ta) + 3.3*(ebit/ta) + 0.6*(mcap/tl) + 1.0*(sales/ta))
    except Exception:
        return np.nan

def fetch_with_retry(fn, attempts=RETRY_ATTEMPTS):
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep((1.5 ** i) + random.uniform(0, 0.4))
    raise last_err

# ============================================================
# FETCH PER STOCK (cached)
# ============================================================
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_stock_data(ticker, min_mcap):
    try:
        time.sleep(random.uniform(0.05, 0.2))
        stock = yf.Ticker(ticker)
        info = fetch_with_retry(lambda: stock.info)
        hist = fetch_with_retry(lambda: stock.history(period='2y'))

        if hist.empty or len(hist) < 200:
            return None

        market_cap = info.get('marketCap', np.nan)
        if market_cap and market_cap < min_mcap:
            return None

        sector = info.get('sector') or 'Unknown'
        if sector in ('Unknown', '', None):
            sector = IDX_SECTOR_FALLBACK.get(ticker, 'Other')

        # Fundamentals
        forward_pe = info.get('forwardPE', np.nan)
        trailing_pe = info.get('trailingPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        ps = info.get('priceToSalesTrailing12Months', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        roa = info.get('returnOnAssets', np.nan)
        margin = info.get('operatingMargins', np.nan)

        net_income = info.get('netIncomeToCommon', np.nan)
        cfo = info.get('operatingCashflow', np.nan)
        total_assets = info.get('totalAssets', np.nan)
        accruals = (net_income - cfo) / total_assets if total_assets and total_assets > 0 else np.nan

        altman_z = calculate_altman_z(info, sector)
        debt_to_equity = info.get('debtToEquity', np.nan)
        rev_growth = info.get('revenueGrowth', np.nan)

        # Price & Momentum
        close = hist['Close']
        current_price = close.iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        above_sma200 = bool(current_price > sma_200) if pd.notna(sma_200) else True

        return_12_1 = (close.iloc[-22] / close.iloc[-273] - 1) * 100 if len(close) >= 273 else np.nan
        return_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) >= 126 else np.nan

        extreme_move = bool(
            (pd.notna(return_12_1) and abs(return_12_1) > EXTREME_RETURN_12M) or
            (pd.notna(return_6m) and abs(return_6m) > EXTREME_RETURN_6M)
        )

        daily_returns = close.pct_change().dropna()
        volatility_1y = daily_returns.tail(252).std() * np.sqrt(252) * 100

        # Sentiment & Short
        target_mean = info.get('targetMeanPrice', np.nan)
        analyst_upside = ((target_mean / current_price) - 1) * 100 if target_mean and current_price else np.nan
        short_ratio = info.get('shortRatio', np.nan)
        short_pct_float = info.get('shortPercentOfFloat', np.nan)
        name = info.get('shortName', ticker)

        return {
            'ticker': ticker, 'name': name, 'sector': sector,
            'price': current_price, 'market_cap': market_cap,
            'forward_pe': forward_pe, 'trailing_pe': trailing_pe,
            'pb': pb, 'ps': ps, 'roe': roe, 'roa': roa, 'margin': margin,
            'accruals': accruals, 'altman_z': altman_z,
            'debt_to_equity': debt_to_equity,
            'return_12_1': return_12_1, 'return_6m': return_6m,
            'above_sma200': above_sma200, 'volatility': volatility_1y,
            'analyst_upside': analyst_upside,
            'short_ratio': short_ratio, 'short_percent_float': short_pct_float,
            'rev_growth': rev_growth, 'extreme_move': extreme_move,
        }
    except Exception:
        return None

# ============================================================
# SCORING
# ============================================================
def winsorize(series, lower=0.01, upper=0.99):
    return series.clip(series.quantile(lower), series.quantile(upper))

def sector_neutral_score(df, column, ascending=True):
    return df.groupby('sector')[column].rank(pct=True, ascending=ascending) * 100

def calculate_scores(df, weights):
    df = df.copy()

    if 'rev_growth' in df.columns:
        df['rev_growth'] = df['rev_growth'].clip(upper=2.0)

    numeric_cols = ['forward_pe', 'trailing_pe', 'pb', 'ps', 'roe', 'roa',
                    'margin', 'accruals', 'return_12_1', 'volatility',
                    'analyst_upside', 'short_ratio', 'short_percent_float']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = winsorize(df[col])

    df['fpe_score'] = sector_neutral_score(df, 'forward_pe', ascending=True)
    df['pb_score'] = sector_neutral_score(df, 'pb', ascending=True)
    df['ps_score'] = sector_neutral_score(df, 'ps', ascending=True)
    df['roe_score'] = sector_neutral_score(df, 'roe', ascending=False)
    df['roa_score'] = sector_neutral_score(df, 'roa', ascending=False)
    df['margin_score'] = sector_neutral_score(df, 'margin', ascending=False)
    df['accruals_score'] = sector_neutral_score(df, 'accruals', ascending=True)
    df['momentum_score'] = sector_neutral_score(df, 'return_12_1', ascending=False)
    df['low_vol_score'] = sector_neutral_score(df, 'volatility', ascending=True)
    df['sentiment_score'] = sector_neutral_score(df, 'analyst_upside', ascending=False)
    df['short_score'] = sector_neutral_score(df, 'short_percent_float', ascending=True)

    score_cols = ['fpe_score', 'pb_score', 'ps_score', 'roe_score', 'roa_score',
                  'margin_score', 'accruals_score', 'momentum_score', 'low_vol_score',
                  'sentiment_score', 'short_score']

    for col in score_cols:
        df[col] = df[col].fillna(50)

    mask_other = df['sector'] == 'Other'
    for col in score_cols:
        df.loc[mask_other, col] = 50

    df['value_score'] = df['fpe_score']*0.5 + df['pb_score']*0.25 + df['ps_score']*0.25
    df['quality_score'] = (df['roe_score']*0.3 + df['roa_score']*0.2 +
                           df['margin_score']*0.3 + df['accruals_score']*0.2)
    df['momentum_score_final'] = df['momentum_score']
    df['sentiment_score_final'] = df['sentiment_score']*0.7 + df['short_score']*0.3
    df['low_vol_score_final'] = df['low_vol_score']

    df['composite_score'] = (
        df['value_score']*weights['value'] +
        df['quality_score']*weights['quality'] +
        df['momentum_score_final']*weights['momentum'] +
        df['sentiment_score_final']*weights['sentiment'] +
        df['low_vol_score_final']*weights['low_vol']
    )
    df['rating'] = df['composite_score'].round(1)
    return df

# ============================================================
# PIPELINE
# ============================================================
def run_screener(tickers, weights, strategy_name, min_mcap, progress_callback=None):
    results = []
    total = len(tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_stock_data, t, min_mcap): t for t in tickers}
        for future in as_completed(futures):
            completed += 1
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                pass
            if progress_callback:
                progress_callback(completed, total, futures[future])

    df = pd.DataFrame(results)
    n_failed = total - len(df) if not df.empty else total

    if df.empty:
        return df, n_failed, 0, 0

    n_before = len(df)

    # Filter 1: Financial health
    df = df[
        (df['altman_z'].isna() | (df['altman_z'] >= 1.8)) &
        (df['debt_to_equity'].isna() | (df['debt_to_equity'] <= 300))
    ].reset_index(drop=True)

    n_health = len(df)
    if df.empty:
        return df, n_failed, n_before - n_health, 0

    # Filter 2: Falling Knife Guard (relaxed for Deep Value)
    if strategy_name == "Deep Value":
        limit, require_trend = -50, False
    else:
        limit, require_trend = -25, True

    if require_trend:
        df = df[
            (df['return_6m'].isna() | (df['return_6m'] > limit)) &
            ((df['above_sma200'] == True) | df['return_6m'].isna() | (df['return_6m'] > -10))
        ].reset_index(drop=True)
    else:
        df = df[(df['return_6m'].isna() | (df['return_6m'] > limit))].reset_index(drop=True)

    n_knife = len(df)
    if df.empty:
        return df, n_failed, n_before - n_health, n_health - n_knife

    df = calculate_scores(df, weights)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df, n_failed, n_before - n_health, n_health - n_knife

# ============================================================
# UI COMPONENTS
# ============================================================
def render_funnel(n_requested, n_failed, n_health, n_knife, n_final):
    with st.expander("🔍 Screening Funnel — why this many stocks?", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Requested", n_requested)
        col2.metric("Fetch Failed", n_failed)
        col3.metric("Health Filter", f"-{n_health}")
        col4.metric("Falling Knife", f"-{n_knife}")
        col5.metric("✅ Passed", n_final)

def render_stock_table(df, top_n, currency):
    display = df.head(top_n)[[
        'rank', 'ticker', 'name', 'sector', 'price', 'market_cap',
        'rating', 'forward_pe', 'roe', 'return_12_1', 'return_6m',
        'value_score', 'quality_score', 'momentum_score_final',
        'sentiment_score_final', 'low_vol_score_final', 'extreme_move'
    ]].copy()

    display['ticker'] = display.apply(
        lambda r: f"⚠️ {r['ticker']}" if r['extreme_move'] else r['ticker'], axis=1
    )
    display = display.drop(columns=['extreme_move'])

    # Format
    display['price'] = display['price'].apply(lambda x: f"{currency}{x:,.2f}" if pd.notna(x) else "—")
    display['market_cap'] = display['market_cap'].apply(
        lambda x: f"{currency}{x/1e9:.1f}B" if pd.notna(x) else "—"
    )
    display['forward_pe'] = display['forward_pe'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    display['roe'] = display['roe'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    display['return_12_1'] = display['return_12_1'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    display['return_6m'] = display['return_6m'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    for col in ['rating', 'value_score', 'quality_score', 'momentum_score_final',
                'sentiment_score_final', 'low_vol_score_final']:
        display[col] = display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    display.columns = ['Rank', 'Ticker', 'Name', 'Sector', 'Price', 'Mkt Cap',
                       'Rating', 'Fwd P/E', 'ROE', 'Ret 12-1', 'Ret 6M',
                       'Value', 'Quality', 'Momentum', 'Sentiment', 'Low Vol']

    st.dataframe(display, use_container_width=True, hide_index=True)

    if df.head(top_n)['extreme_move'].any():
        st.caption("⚠️ = Return >200% (12-1M) or >150% (6M). Verify manually before acting.")

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="Global Stock Screener", page_icon="📊", layout="wide")

st.title("📊 Global Stock Screener")
st.caption(f"⚡ {MAX_WORKERS} parallel threads (+retry) | 💾 Cache {CACHE_TTL//60} min | 🏦 Multi-market quant screening")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    market = st.selectbox("Market:", list(MARKETS.keys()))
    market_cfg = MARKETS[market]

    strategy = st.selectbox("Strategy:", list(STRATEGY_WEIGHTS.keys()))
    weights = STRATEGY_WEIGHTS[strategy]

    use_full = st.checkbox(
        "Screen entire index (slower, complete)",
        value=False,
        help="If unchecked, only a random fixed sample is screened."
    )

    universe_size = st.slider(
        "Universe size:", min_value=30, max_value=500, value=200, step=10,
        disabled=use_full
    )

    top_n = st.slider("Show Top N:", 5, 50, 20)

    st.divider()
    st.caption("**Strategy weights:**")
    st.caption(f"Value {weights['value']*100:.0f}% | Quality {weights['quality']*100:.0f}% | "
               f"Momentum {weights['momentum']*100:.0f}% | Sentiment {weights['sentiment']*100:.0f}% | "
               f"Low Vol {weights['low_vol']*100:.0f}%")

    if strategy == "Deep Value":
        st.info("💡 Falling Knife Guard relaxed for Deep Value (-50% threshold, no SMA200 requirement).")

    st.divider()
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Data will be re-fetched.")
        st.rerun()

# Main content
if st.button("🚀 Run Screener", type="primary", use_container_width=True):
    with st.spinner(f"Loading {market} universe..."):
        if "US" in market:
            all_tickers = get_us_tickers()
        else:
            all_tickers = get_idx_tickers()

    tickers = sample_universe(all_tickers, universe_size, use_full)

    st.info(f"Screening **{len(tickers)} stocks** from **{market}** using **{strategy}** strategy...")

    progress_bar = st.progress(0, text="Initializing...")

    def update(completed, total, ticker):
        progress_bar.progress(completed / total, text=f"[{completed}/{total}] {ticker}")

    df, n_failed, n_health, n_knife = run_screener(
        tickers, weights, strategy, market_cfg["min_mcap"], progress_callback=update
    )
    progress_bar.empty()

    if df.empty:
        st.error("No data retrieved. Reduce universe size or wait a few minutes (Yahoo Finance rate limit).")
    else:
        st.success(f"✅ **{len(df)} stocks** passed all filters from {len(tickers)} requested.")

        render_funnel(len(tickers), n_failed, n_health, n_knife, len(df))

        # Top picks table
        st.subheader(f"🏆 Top {top_n} — {strategy}")
        render_stock_table(df, top_n, market_cfg["currency"])

        # Sector distribution
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🏭 Sector Distribution")
            st.bar_chart(df.head(top_n)['sector'].value_counts())
        with col_b:
            st.subheader("📊 Factor Breakdown")
            chart = df.head(top_n).set_index('ticker')[
                ['value_score', 'quality_score', 'momentum_score_final',
                 'sentiment_score_final', 'low_vol_score_final']
            ]
            st.bar_chart(chart)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full CSV", csv, "screener_results.csv", "text/csv")