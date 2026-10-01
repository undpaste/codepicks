import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
FACTOR_WEIGHTS = {
    'value': 0.25,
    'quality': 0.30,
    'momentum': 0.20,
    'sentiment': 0.15,
    'low_vol': 0.10
}

# ============================================================
# STEP 1: UNIVERSE
# ============================================================
def get_all_us_tickers(limit=None):
    """Mengambil daftar saham AS dari NASDAQ Stock Screener."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    api_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&offset=0"
    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        data = response.json()
        tickers = [row['symbol'] for row in data['data']['rows']]
        tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
        if limit:
            tickers = tickers[:limit]
        return tickers
    except Exception as e:
        st.warning(f"Gagal ambil universe dari NASDAQ ({e}). Pakai fallback S&P 100.")
        return get_fallback_tickers()

def get_fallback_tickers():
    """Fallback: 100 saham besar jika API NASDAQ gagal."""
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

# ============================================================
# STEP 2: HELPER — ALTman Z-Score
# ============================================================
def calculate_altman_z(info):
    """Hitung Altman Z-Score untuk risiko kebangkrutan."""
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

        z = (1.2 * (wc/ta) + 1.4 * (re/ta) + 3.3 * (ebit/ta) +
             0.6 * (mcap/tl) + 1.0 * (sales/ta))
        return z
    except:
        return np.nan

# ============================================================
# STEP 3: FETCH DATA PER SAHAM
# ============================================================
def fetch_stock_data(ticker):
    """Fetch data fundamental, harga, dan sentimen untuk satu saham."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='2y')

        if hist.empty or len(hist) < 300:
            return None

        # --- Value (forward-looking) ---
        forward_pe = info.get('forwardPE', np.nan)
        trailing_pe = info.get('trailingPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        ps = info.get('priceToSalesTrailing12Months', np.nan)

        # --- Quality ---
        roe = info.get('returnOnEquity', np.nan)
        roa = info.get('returnOnAssets', np.nan)
        margin = info.get('operatingMargins', np.nan)

        net_income = info.get('netIncomeToCommon', np.nan)
        cfo = info.get('operatingCashflow', np.nan)
        total_assets = info.get('totalAssets', np.nan)
        accruals = (net_income - cfo) / total_assets if total_assets and total_assets > 0 else np.nan

        # --- Financial Health Filters ---
        altman_z = calculate_altman_z(info)
        debt_to_equity = info.get('debtToEquity', np.nan)

        # --- Momentum (12-1) ---
        close = hist['Close']
        if len(close) >= 273:
            return_12_1 = (close.iloc[-22] / close.iloc[-273] - 1) * 100
        else:
            return_12_1 = np.nan

        # --- Low Volatility ---
        daily_returns = close.pct_change().dropna()
        volatility_1y = daily_returns.tail(252).std() * np.sqrt(252) * 100

        # --- Analyst Sentiment ---
        target_mean = info.get('targetMeanPrice', np.nan)
        current_price = close.iloc[-1]
        analyst_upside = ((target_mean / current_price) - 1) * 100 if target_mean and current_price else np.nan

        # --- Short Interest ---
        short_ratio = info.get('shortRatio', np.nan)
        short_percent_float = info.get('shortPercentOfFloat', np.nan)

        # --- Sector ---
        sector = info.get('sector', 'Unknown')

        return {
            'ticker': ticker,
            'sector': sector,
            'price': current_price,
            'forward_pe': forward_pe,
            'trailing_pe': trailing_pe,
            'pb': pb, 'ps': ps,
            'roe': roe, 'roa': roa, 'margin': margin,
            'accruals': accruals,
            'altman_z': altman_z,
            'debt_to_equity': debt_to_equity,
            'return_12_1': return_12_1,
            'volatility': volatility_1y,
            'analyst_upside': analyst_upside,
            'short_ratio': short_ratio,
            'short_percent_float': short_percent_float,
        }
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return None

# ============================================================
# STEP 4: SCORING
# ============================================================
def winsorize(series, lower=0.01, upper=0.99):
    return series.clip(series.quantile(lower), series.quantile(upper))

def sector_neutral_score(df, column, ascending=True):
    return df.groupby('sector')[column].rank(pct=True, ascending=ascending) * 100

def calculate_scores(df):
    df = df.copy()

    # 1. Winsorize
    numeric_cols = ['forward_pe', 'trailing_pe', 'pb', 'ps', 'roe', 'roa',
                    'margin', 'accruals', 'return_12_1', 'volatility',
                    'analyst_upside', 'short_ratio', 'short_percent_float']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = winsorize(df[col])

    # 2. Sector-neutral scores
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

    # 3. Agregasi per faktor
    df['value_score'] = df['fpe_score'] * 0.5 + df['pb_score'] * 0.25 + df['ps_score'] * 0.25
    df['quality_score'] = (df['roe_score'] * 0.3 + df['roa_score'] * 0.2 +
                           df['margin_score'] * 0.3 + df['accruals_score'] * 0.2)
    df['momentum_score_final'] = df['momentum_score']
    df['sentiment_score_final'] = df['sentiment_score'] * 0.7 + df['short_score'] * 0.3
    df['low_vol_score_final'] = df['low_vol_score']

    # 4. Composite
    df['composite_score'] = (
        df['value_score'] * FACTOR_WEIGHTS['value'] +
        df['quality_score'] * FACTOR_WEIGHTS['quality'] +
        df['momentum_score_final'] * FACTOR_WEIGHTS['momentum'] +
        df['sentiment_score_final'] * FACTOR_WEIGHTS['sentiment'] +
        df['low_vol_score_final'] * FACTOR_WEIGHTS['low_vol']
    )
    df['rating'] = df['composite_score'].round(1)
    return df

# ============================================================
# STEP 5: PIPELINE
# ============================================================
def run_full_screener(tickers, progress_callback=None):
    results = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        data = fetch_stock_data(ticker)
        if data:
            results.append(data)
        time.sleep(0.3)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Filter kesehatan keuangan SEBELUM scoring
    df = df[
        (df['altman_z'].isna() | (df['altman_z'] >= 1.8)) &
        (df['debt_to_equity'].isna() | (df['debt_to_equity'] <= 300))
    ].reset_index(drop=True)

    df = calculate_scores(df)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="US Stock Screener Pro", layout="wide")
st.title("📊 US Stock Screener Pro")

with st.sidebar:
    st.header("Pengaturan")
    universe_size = st.slider("Jumlah saham di universe:", 50, 500, 100, step=50)
    top_n = st.slider("Tampilkan Top N:", 5, 50, 20)

if st.button("🚀 Jalankan Screener"):
    tickers = get_all_us_tickers(limit=universe_size)
    st.info(f"Screening {len(tickers)} saham. Ini bisa memakan waktu beberapa menit...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(i, total, ticker):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"[{i+1}/{total}] Fetching {ticker}...")

    df_results = run_full_screener(tickers, progress_callback=update_progress)

    progress_bar.empty()
    status_text.empty()

    if df_results.empty:
        st.error("Tidak ada data yang berhasil diambil.")
    else:
        st.success(f"Berhasil screening {len(df_results)} saham.")

        cols = ['rank', 'ticker', 'sector', 'price', 'rating',
                'value_score', 'quality_score', 'momentum_score_final',
                'sentiment_score_final', 'low_vol_score_final']
        st.subheader(f"Top {top_n} Saham")
        st.dataframe(df_results[cols].head(top_n), use_container_width=True)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")