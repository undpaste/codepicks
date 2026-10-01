import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_WORKERS = 5  # Jumlah thread paralel (sesuaikan: 5-8 aman)
CACHE_TTL = 3600  # Cache 1 jam

# ============================================================
# STEP 1: UNIVERSE (Wikipedia S&P 500, stabil)
# ============================================================
@st.cache_data(ttl=86400)  # Cache daftar ticker selama 1 hari
def get_sp500_tickers():
    """Ambil S&P 500 dari Wikipedia, fallback ke daftar hardcoded."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        st.warning(f"Wikipedia gagal ({e}). Pakai fallback S&P 100.")
        return get_fallback_tickers()

def get_fallback_tickers():
    """Fallback: 100 saham besar jika Wikipedia gagal."""
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
# STEP 2: HELPER — Altman Z-Score
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
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_stock_data(ticker):
    """Fetch data fundamental, harga, dan sentimen untuk satu saham."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='2y')

        if hist.empty or len(hist) < 300:
            return None

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

        altman_z = calculate_altman_z(info)
        debt_to_equity = info.get('debtToEquity', np.nan)

        close = hist['Close']
        if len(close) >= 273:
            return_12_1 = (close.iloc[-22] / close.iloc[-273] - 1) * 100
        else:
            return_12_1 = np.nan

        daily_returns = close.pct_change().dropna()
        volatility_1y = daily_returns.tail(252).std() * np.sqrt(252) * 100

        target_mean = info.get('targetMeanPrice', np.nan)
        current_price = close.iloc[-1]
        analyst_upside = ((target_mean / current_price) - 1) * 100 if target_mean and current_price else np.nan

        short_ratio = info.get('shortRatio', np.nan)
        short_percent_float = info.get('shortPercentOfFloat', np.nan)

        sector = info.get('sector', 'Unknown')

        return {
            'ticker': ticker, 'sector': sector, 'price': current_price,
            'forward_pe': forward_pe, 'trailing_pe': trailing_pe,
            'pb': pb, 'ps': ps, 'roe': roe, 'roa': roa, 'margin': margin,
            'accruals': accruals, 'altman_z': altman_z,
            'debt_to_equity': debt_to_equity, 'return_12_1': return_12_1,
            'volatility': volatility_1y, 'analyst_upside': analyst_upside,
            'short_ratio': short_ratio, 'short_percent_float': short_percent_float,
        }
    except Exception as e:
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

    df['value_score'] = df['fpe_score'] * 0.5 + df['pb_score'] * 0.25 + df['ps_score'] * 0.25
    df['quality_score'] = (df['roe_score'] * 0.3 + df['roa_score'] * 0.2 +
                           df['margin_score'] * 0.3 + df['accruals_score'] * 0.2)
    df['momentum_score_final'] = df['momentum_score']
    df['sentiment_score_final'] = df['sentiment_score'] * 0.7 + df['short_score'] * 0.3
    df['low_vol_score_final'] = df['low_vol_score']

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
# STEP 5: PIPELINE DENGAN MULTITHREADING
# ============================================================
def run_full_screener(tickers, progress_callback=None):
    """
    Jalankan screener dengan multithreading (tanpa caching di level ini).
    Progress dilaporkan via callback.
    """
    results = []
    total = len(tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_stock_data, ticker): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                pass

            # Update progress via callback
            if progress_callback:
                progress_callback(completed, total, ticker)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Filter kesehatan keuangan SEBELUM scoring
    df = df[
        (df['altman_z'].isna() | (df['altman_z'] >= 1.8)) &
        (df['debt_to_equity'].isna() | (df['debt_to_equity'] <= 300))
    ].reset_index(drop=True)

    if df.empty:
        return df

    df = calculate_scores(df)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="US Stock Screener Pro", layout="wide")
st.title("📊 US Stock Screener Pro")
st.caption(f"⚡ Multithreading aktif ({MAX_WORKERS} thread) | 💾 Cache {CACHE_TTL//60} menit")

with st.sidebar:
    st.header("Pengaturan")
    universe_size = st.slider(
        "Jumlah saham di universe (S&P 500):",
        min_value=50, max_value=500, value=100, step=50,
        help="Semakin banyak saham, semakin lama prosesnya."
    )
    top_n = st.slider("Tampilkan Top N:", 5, 50, 20)

if st.button("🚀 Jalankan Screener", type="primary"):
    tickers = get_sp500_tickers()
    tickers = tickers[:universe_size]

    st.info(f"Screening {len(tickers)} saham dengan {MAX_WORKERS} thread paralel...")

    progress_bar = st.progress(0, text="Memulai...")

    def update_progress(completed, total, ticker):
        progress_bar.progress(
            completed / total,
            text=f"[{completed}/{total}] Selesai: {ticker}"
        )

    df_results = run_full_screener(tickers, progress_callback=update_progress)

    progress_bar.empty()

    if df_results.empty:
        st.error("Tidak ada data yang berhasil diambil. Coba kurangi jumlah saham.")
    else:
        st.success(f"✅ Berhasil screening {len(df_results)} saham dari {len(tickers)} yang diminta.")

        cols = ['rank', 'ticker', 'sector', 'price', 'rating',
                'value_score', 'quality_score', 'momentum_score_final',
                'sentiment_score_final', 'low_vol_score_final']
        st.subheader(f"Top {top_n} Saham")
        st.dataframe(df_results[cols].head(top_n), use_container_width=True)

        # Detail skor
        st.subheader("📊 Breakdown Skor per Faktor")
        chart_data = df_results.head(top_n).set_index('ticker')[
            ['value_score', 'quality_score', 'momentum_score_final',
             'sentiment_score_final', 'low_vol_score_final']
        ]
        st.bar_chart(chart_data)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "screener_results.csv", "text/csv")