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
# KONFIGURASI STRATEGI
# ============================================================
STRATEGY_WEIGHTS = {
    "⚖️ Balanced (default)": {
        'value': 0.25, 'quality': 0.30, 'momentum': 0.20, 'sentiment': 0.15, 'low_vol': 0.10
    },
    "🏆 Quality Compounder": {
        'value': 0.20, 'quality': 0.50, 'momentum': 0.10, 'sentiment': 0.10, 'low_vol': 0.10
    },
    "🚀 High-Growth Momentum": {
        'value': 0.10, 'quality': 0.20, 'momentum': 0.45, 'sentiment': 0.20, 'low_vol': 0.05
    },
    "💰 Deep Value": {
        'value': 0.50, 'quality': 0.25, 'momentum': 0.10, 'sentiment': 0.10, 'low_vol': 0.05
    },
}

MARKETS = {
    "🇺🇸 US Market (S&P 500)": {
        "suffix": "",
        "min_mcap": 2_000_000_000,      # $2B
        "currency": "$",
        "mcap_divisor": 1e9,
        "mcap_unit": "B",
        "universe_label": "S&P 500",
        "universe_max": 500,
        "default_size": 100,
    },
    "🇮🇩 Indonesia (IDX)": {
        "suffix": ".JK",
        "min_mcap": 500_000_000_000,    # Rp 500B
        "currency": "Rp ",
        "mcap_divisor": 1e12,
        "mcap_unit": "T",
        "universe_label": "IDX Likuid",
        "universe_max": 220,
        "default_size": 100,
    },
}

MAX_WORKERS = 2          # Diturunin biar aman rate limit
CACHE_TTL = 3600
RETRY_ATTEMPTS = 3

Z_SCORE_EXEMPT_SECTORS = {'Financial Services', 'Real Estate', 'Financials'}
EXTREME_RETURN_12M_THRESHOLD = 200
EXTREME_RETURN_6M_THRESHOLD = 150

# Liquidity minimum untuk IDX: Rp 1 miliar transaksi harian
IDX_MIN_DAILY_VALUE = 1_000_000_000

# ============================================================
# IDX SECTOR FALLBACK (yfinance sering return Unknown untuk IDX)
# ============================================================
IDX_SECTOR_FALLBACK = {
    'BBCA': 'Financials', 'BBRI': 'Financials', 'BMRI': 'Financials',
    'BBNI': 'Financials', 'BRIS': 'Financials', 'BTPS': 'Financials',
    'ARTO': 'Financials', 'BBTN': 'Financials', 'BJBR': 'Financials',
    'BJTM': 'Financials', 'BNGA': 'Financials', 'BNLI': 'Financials',
    'PNBN': 'Financials', 'MEGA': 'Financials', 'NISP': 'Financials',
    'BFIN': 'Financials', 'ADMF': 'Financials', 'PNLF': 'Financials',
    'TUGU': 'Financials', 'ASBI': 'Financials', 'LPGI': 'Financials',
    'MCOR': 'Financials', 'BABP': 'Financials', 'AGRO': 'Financials',
    'TLKM': 'Communication Services', 'EXCL': 'Communication Services',
    'ISAT': 'Communication Services', 'TOWR': 'Communication Services',
    'MTEL': 'Communication Services', 'MNCN': 'Communication Services',
    'SCMA': 'Communication Services',
    'ASII': 'Consumer Cyclical', 'AUTO': 'Consumer Cyclical',
    'SMSM': 'Consumer Cyclical', 'MAPI': 'Consumer Cyclical',
    'ACES': 'Consumer Cyclical', 'ERAA': 'Consumer Cyclical',
    'UNTR': 'Industrials', 'HEXA': 'Industrials',
    'UNVR': 'Consumer Defensive', 'ICBP': 'Consumer Defensive',
    'INDF': 'Consumer Defensive', 'MYOR': 'Consumer Defensive',
    'SIDO': 'Consumer Defensive', 'AMRT': 'Consumer Defensive',
    'CPIN': 'Consumer Defensive', 'JPFA': 'Consumer Defensive',
    'MAIN': 'Consumer Defensive', 'HMSP': 'Consumer Defensive',
    'GGRM': 'Consumer Defensive', 'WIIM': 'Consumer Defensive',
    'DMND': 'Consumer Defensive',
    'ADRO': 'Energy', 'PTBA': 'Energy', 'ITMG': 'Energy',
    'MEDC': 'Energy', 'PGAS': 'Energy', 'HRUM': 'Energy',
    'AKRA': 'Energy', 'ELSA': 'Energy', 'BUMI': 'Energy',
    'DOID': 'Energy', 'HRTA': 'Energy',
    'ANTM': 'Basic Materials', 'INCO': 'Basic Materials', 'TINS': 'Basic Materials',
    'SMGR': 'Basic Materials', 'INTP': 'Basic Materials', 'BRPT': 'Basic Materials',
    'TPIA': 'Basic Materials', 'INKP': 'Basic Materials', 'TKIM': 'Basic Materials',
    'MDKA': 'Basic Materials', 'NCKL': 'Basic Materials',
    'GOTO': 'Technology', 'BUKA': 'Technology', 'EMTK': 'Technology',
    'MIKA': 'Healthcare', 'SILO': 'Healthcare', 'HEAL': 'Healthcare',
    'PRDA': 'Healthcare', 'KAEF': 'Healthcare', 'INAF': 'Healthcare',
    'SAME': 'Healthcare', 'MTCN': 'Healthcare',
    'PTPP': 'Industrials', 'WIKA': 'Industrials', 'ADHI': 'Industrials',
    'WSKT': 'Industrials', 'JSMR': 'Industrials',
    'BSDE': 'Real Estate', 'CTRA': 'Real Estate', 'PWON': 'Real Estate',
    'SMRA': 'Real Estate', 'ASRI': 'Real Estate', 'LPKR': 'Real Estate',
}

# ============================================================
# UNIVERSE
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        raw_tickers = tables[0]['Symbol'].tolist()
        tickers, seen = [], set()
        for t in raw_tickers:
            t = str(t).strip().replace('.', '-')
            if t and t not in seen:
                tickers.append(t); seen.add(t)
        return sorted(tickers)
    except Exception as e:
        st.warning(f"Wikipedia gagal ({e}). Pakai fallback S&P 100.")
        return get_us_fallback()

def get_us_fallback():
    return [
        'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','UNH','XOM',
        'JNJ','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP','KO',
        'AVGO','COST','WMT','TMO','MCD','CSCO','ACN','ABT','CRM','ADBE','DHR',
        'LIN','NKE','TXN','AMD','PM','NEE','WFC','DIS','UPS','RTX','BMY','ORCL',
        'QCOM','INTC','HON','T','UNP','BA','LOW','SBUX','GS','INTU','AMAT','DE',
        'BLK','ISRG','ADI','MDLZ','GILD','ADP','VRTX','REGN','MMC','LRCX','ZTS',
        'CI','CB','SO','DUK','PGR','BDX','ITW','SYK','BSX','MO','APD','EQIX',
        'CCI','PLD','NSC','AON','MU','EL','KLAC','SNPS','CDNS','MRVL','FTNT',
        'PANW','CRWD','DDOG','SNOW','ZS','NET'
    ]

@st.cache_data(ttl=86400, show_spinner=False)
def get_idx_tickers():
    """IDX universe ~200 saham likuid (LQ45 + IDX80 + Kompas100 + additional)."""
    return [
        # === BANKING & FINANCIALS ===
        'BBCA','BBRI','BMRI','BBNI','BRIS','BTPS','ARTO','BBTN','BJBR','BJTM',
        'BNGA','BNLI','PNBN','MEGA','NISP','BFIN','ADMF','PNLF','TUGU','ASBI',
        'LPGI','MCOR','BABP','AGRO','BNII','BBHI','BCIC','AMAR','MFIN','WOMF',
        # === TELECOM ===
        'TLKM','EXCL','ISAT','TOWR','MTEL','TBIG','CENT',
        # === CONSUMER CYCLICAL ===
        'ASII','AUTO','SMSM','MAPI','ACES','ERAA','LPPF','RALS','SCCO',
        'MAPA','CSAP','RANC','DIGI','FAST','RDTX','KIJA',
        # === CONSUMER DEFENSIVE ===
        'UNVR','ICBP','INDF','MYOR','SIDO','AMRT','CPIN','JPFA','MAIN',
        'HMSP','GGRM','WIIM','DMND','CAMP','ULTJ','STTP','TBLA','AISA','DLTA',
        'MLBI','INDR','KEJU','CEKA','GOOD','PSDN','SKBM','ICBP',
        # === ENERGY ===
        'ADRO','PTBA','ITMG','MEDC','PGAS','HRUM','AKRA','ELSA','BUMI','DOID',
        'HRTA','TOBA','PTRO','KKGI','MYOH','DEWA','TGRA',
        # === BASIC MATERIALS ===
        'ANTM','INCO','TINS','SMGR','INTP','BRPT','TPIA','INKP','TKIM','MDKA',
        'NCKL','MBMA','NIKL','PSAB','IFSH','KRAS','ISSP','BAJA','JKSW','LION',
        # === TECHNOLOGY ===
        'GOTO','BUKA','EMTK','DMMX','MTDL','WIFI','AWAN','MLPT','TECH','LMAS',
        # === HEALTHCARE ===
        'MIKA','SILO','HEAL','PRDA','KAEF','INAF','SAME','MTCN','DVLA','TSPC',
        'PYFA','PEHA','SRAJ','RSGK','BIMA',
        # === INDUSTRIALS ===
        'UNTR','HEXA','PTPP','WIKA','ADHI','WSKT','JSMR','IPCM','TMAS','BULL',
        'SOCI','ASSA','SMDR','HITS','PSSI','BLTA','MIRA','LEAD','IATA',
        # === COMMUNICATION SERVICES / MEDIA ===
        'MNCN','SCMA','FILM','BMTR','MSKY','IPTV','NETV',
        # === REAL ESTATE ===
        'BSDE','CTRA','PWON','SMRA','ASRI','LPKR','DILD','DART','APLN','KIJA',
        'MTLA','BEST','RDTX','MKPI',
        # === ADDITIONAL LIQUID NAMES ===
        'TAPG','DSNG','SSMS','LSIP','AALI','SGRO','TBLA','CSRA','BWPT',
        'INDS','SRIL','TRIS','ESTI','PBRX','POLY','UNIT','TRAM',
        'MPMX','SDPC','IMAS','BRAM','GDYR','SSTM',
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
        if ta <= 0 or tl <= 0:
            return np.nan
        return (1.2*(wc/ta) + 1.4*(re/ta) + 3.3*(ebit/ta) +
                0.6*(mcap/tl) + 1.0*(sales/ta))
    except Exception:
        return np.nan

def fetch_with_retry(fn, *args, attempts=RETRY_ATTEMPTS, **kwargs):
    last_err = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep((1.5 ** i) + random.uniform(0, 0.4))
    raise last_err

# ============================================================
# FETCH PER SAHAM (cached)
# ============================================================
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_stock_data(ticker_base, market_suffix="", min_mcap=2_000_000_000):
    try:
        time.sleep(random.uniform(0.15, 0.4))

        full_ticker = f"{ticker_base}{market_suffix}" if market_suffix else ticker_base

        stock = yf.Ticker(full_ticker)
        info = fetch_with_retry(lambda: stock.info)
        hist = fetch_with_retry(lambda: stock.history(period='2y'))

        if hist.empty or len(hist) < 200:
            return None

        market_cap = info.get('marketCap', np.nan)
        if market_cap and market_cap < min_mcap:
            return None

        sector = info.get('sector') or 'Unknown'
        if sector in ('Unknown', '', None):
            sector = IDX_SECTOR_FALLBACK.get(ticker_base, 'Other')

        # Fundamental
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

        # Dividend yield
        div_yield = info.get('dividendYield', np.nan)
        if pd.notna(div_yield) and div_yield > 1:
            div_yield = div_yield / 100

        # Harga & Momentum
        close = hist['Close']
        current_price = close.iloc[-1]

        # Liquidity (butuh current_price dulu)
        avg_volume = info.get('averageVolume', np.nan)
        avg_daily_value = (avg_volume * current_price) if pd.notna(avg_volume) and pd.notna(current_price) else np.nan

        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        above_sma200 = bool(current_price > sma_200) if pd.notna(sma_200) else True

        if len(close) >= 273:
            return_12_1 = (close.iloc[-22] / close.iloc[-273] - 1) * 100
        else:
            return_12_1 = np.nan

        if len(close) >= 126:
            return_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100
        else:
            return_6m = np.nan

        extreme_move = bool(
            (pd.notna(return_12_1) and abs(return_12_1) > EXTREME_RETURN_12M_THRESHOLD) or
            (pd.notna(return_6m) and abs(return_6m) > EXTREME_RETURN_6M_THRESHOLD)
        )

        daily_returns = close.pct_change().dropna()
        volatility_1y = daily_returns.tail(252).std() * np.sqrt(252) * 100

        target_mean = info.get('targetMeanPrice', np.nan)
        analyst_upside = ((target_mean / current_price) - 1) * 100 if target_mean and current_price else np.nan
        short_ratio = info.get('shortRatio', np.nan)
        short_percent_float = info.get('shortPercentOfFloat', np.nan)

        name = info.get('shortName', ticker_base)

        # Data completeness score
        fundamental_fields = [forward_pe, pb, roe, roa, margin, rev_growth, div_yield]
        data_completeness = sum(pd.notna(x) for x in fundamental_fields) / len(fundamental_fields) * 100

        return {
            'ticker': ticker_base, 'name': name, 'sector': sector,
            'price': current_price, 'market_cap': market_cap,
            'forward_pe': forward_pe, 'trailing_pe': trailing_pe,
            'pb': pb, 'ps': ps, 'roe': roe, 'roa': roa, 'margin': margin,
            'accruals': accruals, 'altman_z': altman_z,
            'debt_to_equity': debt_to_equity,
            'return_12_1': return_12_1, 'return_6m': return_6m,
            'above_sma200': above_sma200, 'volatility': volatility_1y,
            'analyst_upside': analyst_upside,
            'short_ratio': short_ratio, 'short_percent_float': short_percent_float,
            'rev_growth': rev_growth, 'div_yield': div_yield,
            'avg_daily_value': avg_daily_value,
            'data_completeness': data_completeness,
            'extreme_move': extreme_move,
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

def calculate_scores(df, weights, is_idx=False):
    df = df.copy()

    if 'rev_growth' in df.columns:
        df['rev_growth'] = df['rev_growth'].clip(upper=2.0)

    numeric_cols = ['forward_pe', 'trailing_pe', 'pb', 'ps', 'roe', 'roa',
                    'margin', 'accruals', 'return_12_1', 'volatility',
                    'analyst_upside', 'short_ratio', 'short_percent_float',
                    'div_yield']
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
    df['div_score'] = sector_neutral_score(df, 'div_yield', ascending=False)

    score_cols = ['fpe_score', 'pb_score', 'ps_score', 'roe_score', 'roa_score',
                  'margin_score', 'accruals_score', 'momentum_score', 'low_vol_score',
                  'sentiment_score', 'short_score', 'div_score']

    for col in score_cols:
        df[col] = df[col].fillna(50)

    mask_other = df['sector'] == 'Other'
    for col in score_cols:
        df.loc[mask_other, col] = 50

    # Value score: IDX pakai bobot PBV lebih besar (value investing culture)
    if is_idx:
        df['value_score'] = df['fpe_score']*0.35 + df['pb_score']*0.45 + df['ps_score']*0.20
    else:
        df['value_score'] = df['fpe_score']*0.50 + df['pb_score']*0.25 + df['ps_score']*0.25

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
def run_full_screener(tickers, weights, strategy_name, market_suffix="",
                      min_mcap=2_000_000_000, is_idx=False, progress_callback=None):
    results = []
    failed_tickers = []
    total = len(tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_stock_data, t, market_suffix, min_mcap): t
            for t in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                data = future.result()
                if data:
                    results.append(data)
                else:
                    failed_tickers.append(ticker)
            except Exception:
                failed_tickers.append(ticker)
            if progress_callback:
                progress_callback(completed, total, ticker)

    df = pd.DataFrame(results)
    n_failed = total - len(df)

    if df.empty:
        return df, n_failed, 0, 0, failed_tickers

    n_before_filters = len(df)

    # Filter 0: Likuiditas (IDX only)
    if is_idx:
        df = df[
            (df['avg_daily_value'].isna()) | (df['avg_daily_value'] >= IDX_MIN_DAILY_VALUE)
        ].reset_index(drop=True)
        if df.empty:
            return df, n_failed, n_before_filters, 0, failed_tickers

    n_after_liquidity = len(df)

    # Filter 1: Kesehatan keuangan
    df = df[
        (df['altman_z'].isna() | (df['altman_z'] >= 1.8)) &
        (df['debt_to_equity'].isna() | (df['debt_to_equity'] <= 300))
    ].reset_index(drop=True)

    n_after_health = len(df)

    if df.empty:
        return df, n_failed, n_before_filters - n_after_liquidity, 0, failed_tickers

    # Filter 2: Falling Knife Guard
    if strategy_name == "💰 Deep Value":
        drawdown_limit, require_trend = -50, False
    else:
        drawdown_limit, require_trend = -25, True

    if require_trend:
        df = df[
            (df['return_6m'].isna() | (df['return_6m'] > drawdown_limit)) &
            ((df['above_sma200'] == True) | df['return_6m'].isna() | (df['return_6m'] > -10))
        ].reset_index(drop=True)
    else:
        df = df[
            (df['return_6m'].isna() | (df['return_6m'] > drawdown_limit))
        ].reset_index(drop=True)

    n_after_knife = len(df)

    if df.empty:
        return df, n_failed, n_before_filters - n_after_health, n_after_health - n_after_knife, failed_tickers

    df = calculate_scores(df, weights, is_idx=is_idx)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    n_filtered_health = n_before_filters - n_after_liquidity - n_after_health + (n_after_liquidity - n_after_health)
    # Simplify: total filtered dari filter 0+1 = n_before_filters - n_after_health
    n_filtered_pre = n_before_filters - n_after_health
    n_filtered_knife = n_after_health - n_after_knife

    return df, n_failed, n_filtered_pre, n_filtered_knife, failed_tickers

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Global Stock Screener Pro", layout="wide")
st.title("📊 Global Stock Screener Pro")
st.caption(f"⚡ {MAX_WORKERS} threads (+retry) | 💾 Cache {CACHE_TTL//60} min | 🌏 US + Indonesia")

with st.sidebar:
    st.header("⚙️ Settings")

    market = st.selectbox("Market:", list(MARKETS.keys()))
    market_cfg = MARKETS[market]
    is_idx = "IDX" in market or "Indonesia" in market

    strategy = st.selectbox("Strategy:", list(STRATEGY_WEIGHTS.keys()))
    weights = STRATEGY_WEIGHTS[strategy]

    use_full_universe = st.checkbox(
        f"Screen ALL {market_cfg['universe_label']} stocks",
        value=False,
        help="Kalau off, cuma random sample yang di-screen."
    )

    universe_size = st.slider(
        "Universe size (kalau bukan full):",
        min_value=30,
        max_value=market_cfg["universe_max"],
        value=market_cfg["default_size"],
        step=10,
        disabled=use_full_universe,
    )

    top_n = st.slider("Show Top N:", 5, 50, 20)

    st.divider()
    st.caption(f"**Weights:** Value {weights['value']*100:.0f}% | Quality {weights['quality']*100:.0f}% | "
               f"Momentum {weights['momentum']*100:.0f}% | Sentiment {weights['sentiment']*100:.0f}% | "
               f"Low Vol {weights['low_vol']*100:.0f}%")

    if strategy == "💰 Deep Value":
        st.caption("ℹ️ Falling Knife Guard dilonggarkan (-50%, no SMA200 requirement).")

    st.divider()
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Re-run screener.")
        st.rerun()

if st.button("🚀 Run Screener", type="primary"):
    if "US" in market:
        all_tickers = get_sp500_tickers()
    else:
        all_tickers = get_idx_tickers()

    st.caption(f"📋 {len(all_tickers)} tickers dari {market_cfg['universe_label']}.")

    tickers = sample_universe(all_tickers, universe_size, use_full_universe)

    st.info(f"Screening **{len(tickers)} saham {market}** dengan strategi **{strategy}**...")

    progress_bar = st.progress(0, text="Starting...")

    def update_progress(completed, total, ticker):
        progress_bar.progress(completed / total, text=f"[{completed}/{total}] {ticker}")

    df_results, n_failed, n_filtered_pre, n_filtered_knife, failed_tickers = run_full_screener(
        tickers, weights, strategy,
        market_suffix=market_cfg["suffix"],
        min_mcap=market_cfg["min_mcap"],
        is_idx=is_idx,
        progress_callback=update_progress
    )
    progress_bar.empty()

    if df_results.empty:
        st.error("Tidak ada data. Coba kurangi jumlah saham atau tunggu beberapa menit.")
    else:
        st.success(f"✅ **{len(df_results)} saham** lolos semua filter, dari {len(tickers)} diminta.")

        with st.expander("ℹ️ Funnel detail"):
            st.write(f"- Diminta: {len(tickers)} ticker")
            st.write(f"- Gagal fetch (rate limit / data kosong / mcap kecil): {n_failed}")
            st.write(f"- Tersaring filter kesehatan + likuiditas: {n_filtered_pre}")
            st.write(f"- Tersaring Falling Knife Guard: {n_filtered_knife}")
            st.write(f"- **Lolos semua filter: {len(df_results)}**")

        if failed_tickers:
            with st.expander(f"⚠️ {len(failed_tickers)} ticker gagal fetch"):
                st.write(", ".join(failed_tickers))
                st.caption("Kemungkinan rate limit Yahoo, data kosong, atau mcap di bawah threshold.")

        st.subheader(f"🏆 Top {top_n} — {strategy}")

        # Kolom dinamis: ROE untuk US, Dividend Yield untuk IDX
        base_cols = ['rank', 'ticker', 'name', 'sector', 'price', 'market_cap',
                     'rating', 'forward_pe', 'pb']
        if is_idx:
            base_cols.append('div_yield')
        else:
            base_cols.append('roe')
        base_cols.extend(['return_12_1', 'return_6m',
                          'value_score', 'quality_score', 'momentum_score_final',
                          'sentiment_score_final', 'low_vol_score_final', 'extreme_move'])

        display_df = df_results.head(top_n)[base_cols].copy()

        display_df['ticker'] = display_df.apply(
            lambda r: f"⚠️ {r['ticker']}" if r['extreme_move'] else r['ticker'], axis=1
        )
        display_df = display_df.drop(columns=['extreme_move'])

        currency = market_cfg["currency"]
        mcap_div = market_cfg["mcap_divisor"]
        mcap_unit = market_cfg["mcap_unit"]

        display_df['price'] = display_df['price'].apply(lambda x: f"{currency}{x:,.2f}" if pd.notna(x) else "—")
        display_df['market_cap'] = display_df['market_cap'].apply(
            lambda x: f"{currency}{x/mcap_div:.1f}{mcap_unit}" if pd.notna(x) else "—"
        )
        display_df['forward_pe'] = display_df['forward_pe'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        display_df['pb'] = display_df['pb'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        display_df['return_12_1'] = display_df['return_12_1'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        display_df['return_6m'] = display_df['return_6m'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

        if is_idx:
            display_df['div_yield'] = display_df['div_yield'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
            )
        else:
            display_df['roe'] = display_df['roe'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

        for col in ['rating', 'value_score', 'quality_score', 'momentum_score_final',
                    'sentiment_score_final', 'low_vol_score_final']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

        if is_idx:
            display_df.columns = ['Rank', 'Ticker', 'Nama', 'Sektor', 'Harga', 'Mkt Cap',
                                  'Rating', 'Fwd P/E', 'PBV', 'Div Yield', 'Ret 12-1', 'Ret 6M',
                                  'Value', 'Quality', 'Momentum', 'Sentiment', 'Low Vol']
        else:
            display_df.columns = ['Rank', 'Ticker', 'Nama', 'Sektor', 'Harga', 'Mkt Cap',
                                  'Rating', 'Fwd P/E', 'PBV', 'ROE', 'Ret 12-1', 'Ret 6M',
                                  'Value', 'Quality', 'Momentum', 'Sentiment', 'Low Vol']

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if df_results.head(top_n)['extreme_move'].any():
            st.caption("⚠️ = return ekstrem. Worth cross-check manual.")

        st.subheader("🏭 Sector Distribution")
        st.bar_chart(df_results.head(top_n)['sector'].value_counts())

        st.subheader("📊 Factor Breakdown")
        chart_data = df_results.head(top_n).set_index('ticker')[
            ['value_score', 'quality_score', 'momentum_score_final',
             'sentiment_score_final', 'low_vol_score_final']
        ]
        st.bar_chart(chart_data)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")