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

MAX_WORKERS = 3  # FIX: diturunin dari 5 -> 3, lebih aman dari rate limit Yahoo
CACHE_TTL = 3600
MIN_MARKET_CAP = 2_000_000_000
RETRY_ATTEMPTS = 3  # FIX: retry kalau fetch gagal (bisa jadi rate limit, bukan beneran gak ada data)

# FIX: sektor yang gak cocok dievaluasi pakai Altman Z-score klasik
# (formula aslinya buat manufaktur; current assets/liabilities bank & REIT strukturnya beda)
Z_SCORE_EXEMPT_SECTORS = {'Financial Services', 'Real Estate'}

# FIX: ambang batas buat nandain return yang ekstrem di UI (bukan di-exclude, cuma di-flag)
EXTREME_RETURN_12M_THRESHOLD = 200   # %
EXTREME_RETURN_6M_THRESHOLD = 150    # %

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
        # FIX: strip whitespace & drop kosong/duplikat biar gak ada entri aneh
        tickers = []
        seen = set()
        for t in raw_tickers:
            t = str(t).strip().replace('.', '-')
            if t and t not in seen:
                tickers.append(t)
                seen.add(t)
        return sorted(tickers)
    except Exception as e:
        st.warning(f"Wikipedia gagal ({e}). Pakai fallback S&P 100.")
        return get_fallback_tickers()

def get_fallback_tickers():
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

def sample_universe(all_tickers, size, use_full):
    # FIX: opsi screening penuh, bukan selalu random sample
    if use_full or size >= len(all_tickers):
        return all_tickers
    rng = random.Random(42)
    return rng.sample(all_tickers, size)

# ============================================================
# HELPERS
# ============================================================
def calculate_altman_z(info, sector):
    # FIX: skip Altman Z buat sektor yang gak cocok sama modelnya (return NaN = otomatis lolos filter)
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
    # FIX: retry dengan backoff + jitter, biar rate-limit sementara gak langsung dianggap "no data"
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
def fetch_stock_data(ticker):
    try:
        # FIX: delay kecil acak sebelum tiap request, biar burst request ke Yahoo lebih halus
        time.sleep(random.uniform(0.05, 0.2))

        stock = yf.Ticker(ticker)
        info = fetch_with_retry(lambda: stock.info)
        hist = fetch_with_retry(lambda: stock.history(period='2y'))

        if hist.empty or len(hist) < 300:
            return None

        market_cap = info.get('marketCap', np.nan)
        if market_cap and market_cap < MIN_MARKET_CAP:
            return None

        sector = info.get('sector') or 'Unknown'
        if sector in ('Unknown', '', None):
            sector = 'Other'

        # --- Fundamental ---
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

        altman_z = calculate_altman_z(info, sector)  # FIX: pass sector buat exempt check
        debt_to_equity = info.get('debtToEquity', np.nan)

        # --- Revenue Growth ---
        rev_growth = info.get('revenueGrowth', np.nan)

        # --- Harga & Momentum ---
        close = hist['Close']
        current_price = close.iloc[-1]

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

        # FIX: flag return ekstrem (bukan dibuang, cuma ditandai di UI buat sanity-check manual)
        extreme_move = bool(
            (pd.notna(return_12_1) and abs(return_12_1) > EXTREME_RETURN_12M_THRESHOLD) or
            (pd.notna(return_6m) and abs(return_6m) > EXTREME_RETURN_6M_THRESHOLD)
        )

        daily_returns = close.pct_change().dropna()
        volatility_1y = daily_returns.tail(252).std() * np.sqrt(252) * 100

        # --- Sentiment & Short ---
        target_mean = info.get('targetMeanPrice', np.nan)
        analyst_upside = ((target_mean / current_price) - 1) * 100 if target_mean and current_price else np.nan
        short_ratio = info.get('shortRatio', np.nan)
        short_percent_float = info.get('shortPercentOfFloat', np.nan)

        name = info.get('shortName', ticker)

        return {
            'ticker': ticker, 'name': name, 'sector': sector,
            'price': current_price, 'market_cap': market_cap,
            'forward_pe': forward_pe, 'trailing_pe': trailing_pe,
            'pb': pb, 'ps': ps, 'roe': roe, 'roa': roa, 'margin': margin,
            'accruals': accruals, 'altman_z': altman_z,
            'debt_to_equity': debt_to_equity,
            'return_12_1': return_12_1, 'return_6m': return_6m,
            'above_sma200': above_sma200,
            'volatility': volatility_1y,
            'analyst_upside': analyst_upside,
            'short_ratio': short_ratio, 'short_percent_float': short_percent_float,
            'rev_growth': rev_growth,
            'extreme_move': extreme_move,  # FIX
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
def run_full_screener(tickers, weights, strategy_name, progress_callback=None):
    results = []
    total = len(tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(fetch_stock_data, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                pass
            if progress_callback:
                progress_callback(completed, total, ticker)

    df = pd.DataFrame(results)
    # FIX: hitung berapa ticker yang gagal total (bukan cuma kefilter), biar transparan ke user
    n_fetched = len(df)
    n_failed = total - n_fetched

    if df.empty:
        return df, n_failed, 0, 0

    n_before_filters = len(df)

    # --- Filter 1: Kesehatan keuangan ---
    df = df[
        (df['altman_z'].isna() | (df['altman_z'] >= 1.8)) &
        (df['debt_to_equity'].isna() | (df['debt_to_equity'] <= 300))
    ].reset_index(drop=True)

    n_after_health = len(df)

    if df.empty:
        return df, n_failed, n_before_filters - n_after_health, 0

    # --- Filter 2: Falling Knife Guard ---
    # FIX: strategi Deep Value butuh filter yang lebih longgar - justru nyari saham yang udah jatuh
    if strategy_name == "💰 Deep Value":
        drawdown_limit = -50
        require_trend = False
    else:
        drawdown_limit = -25
        require_trend = True

    if require_trend:
        df = df[
            (df['return_6m'].isna() | (df['return_6m'] > drawdown_limit)) &
            (
                (df['above_sma200'] == True) |
                (df['return_6m'].isna()) |
                (df['return_6m'] > -10)
            )
        ].reset_index(drop=True)
    else:
        df = df[
            (df['return_6m'].isna() | (df['return_6m'] > drawdown_limit))
        ].reset_index(drop=True)

    n_after_knife = len(df)

    if df.empty:
        return df, n_failed, n_before_filters - n_after_health, n_after_health - n_after_knife

    df = calculate_scores(df, weights)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df, n_failed, n_before_filters - n_after_health, n_after_health - n_after_knife

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="US Stock Screener Pro", layout="wide")
st.title("📊 US Stock Screener Pro")
st.caption(f"⚡ {MAX_WORKERS} thread paralel (+retry) | 💾 Cache {CACHE_TTL//60} menit | 🏦 Min. Market Cap ${MIN_MARKET_CAP/1e9:.0f}B")

with st.sidebar:
    st.header("⚙️ Pengaturan")

    strategy = st.selectbox("Pilih strategi:", list(STRATEGY_WEIGHTS.keys()))
    weights = STRATEGY_WEIGHTS[strategy]

    # FIX: opsi eksplisit buat screening SEMUA saham, bukan cuma random sample
    use_full_universe = st.checkbox(
        "Screening semua ~500 saham S&P 500 (lebih lambat & lengkap)",
        value=False,
        help="Kalau dimatikan, hasil cuma diambil dari random sample di bawah (sample-nya tetap/fixed, bukan seluruh index)."
    )

    universe_size = st.slider(
        "Jumlah saham di universe (kalau bukan full screening):",
        min_value=50, max_value=500, value=200, step=50,
        disabled=use_full_universe,
        help="Random sample dari S&P 500 dengan seed tetap (hasil konsisten tiap run, tapi BUKAN keseluruhan index)."
    )
    if not use_full_universe:
        st.caption("⚠️ Ini sample, bukan screening keseluruhan S&P 500. Centang opsi di atas buat hasil yang mencakup semua saham.")

    top_n = st.slider("Tampilkan Top N:", 5, 50, 20)

    st.divider()
    st.caption(f"**Bobot strategi:**")
    st.caption(f"Value {weights['value']*100:.0f}% | Quality {weights['quality']*100:.0f}% | Momentum {weights['momentum']*100:.0f}% | Sentiment {weights['sentiment']*100:.0f}% | Low Vol {weights['low_vol']*100:.0f}%")
    if strategy == "💰 Deep Value":
        st.caption("ℹ️ Falling Knife Guard dilonggarkan buat strategi ini (threshold -50% & gak wajib above SMA200), karena Deep Value memang nyari saham yang udah terkoreksi.")

    st.divider()
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache dibersihkan. Data akan di-fetch ulang.")
        st.rerun()

if st.button("🚀 Jalankan Screener", type="primary"):
    all_tickers = get_sp500_tickers()

    # FIX: sanity check ticker list - flag ticker 1 karakter buat di-review manual
    # (bukan di-exclude otomatis, karena beberapa ticker 1 huruf itu valid: F, T, V, C, dll)
    single_char = [t for t in all_tickers if len(t) == 1]
    st.caption(f"📋 {len(all_tickers)} ticker dimuat dari S&P 500.")
    if single_char:
        st.caption(f"🔍 Ticker 1-huruf yang ke-load: {', '.join(single_char)} — kalau ada yang gak familiar, cek manual, kemungkinan sisa parsing tabel Wikipedia yang keliru.")

    tickers = sample_universe(all_tickers, universe_size, use_full_universe)

    st.info(f"Screening **{len(tickers)} saham** dengan strategi **{strategy}**...")

    progress_bar = st.progress(0, text="Memulai...")

    def update_progress(completed, total, ticker):
        progress_bar.progress(
            completed / total,
            text=f"[{completed}/{total}] Selesai: {ticker}"
        )

    df_results, n_failed, n_filtered_health, n_filtered_knife = run_full_screener(
        tickers, weights, strategy, progress_callback=update_progress
    )
    progress_bar.empty()

    if df_results.empty:
        st.error("Tidak ada data yang berhasil diambil. Coba kurangi jumlah saham atau tunggu beberapa menit (kemungkinan kena rate limit Yahoo Finance).")
    else:
        st.success(f"✅ Berhasil screening **{len(df_results)} saham** lolos semua filter, dari {len(tickers)} yang diminta.")

        # FIX: transparansi funnel - biar user tau kenapa hasil akhir sedikit
        with st.expander("ℹ️ Detail funnel screening (kenapa jumlahnya segini)"):
            st.write(f"- Diminta: {len(tickers)} ticker")
            st.write(f"- Gagal di-fetch (rate limit / data kosong / market cap kekecilan): {n_failed}")
            st.write(f"- Tersaring filter kesehatan keuangan (Altman Z / Debt-to-Equity): {n_filtered_health}")
            st.write(f"- Tersaring Falling Knife Guard: {n_filtered_knife}")
            st.write(f"- **Lolos semua filter: {len(df_results)}**")

        # ============ TABEL UTAMA ============
        st.subheader(f"🏆 Top {top_n} Saham — {strategy}")

        display_df = df_results.head(top_n)[[
            'rank', 'ticker', 'name', 'sector', 'price', 'market_cap',
            'rating', 'forward_pe', 'pb', 'roe', 'return_12_1', 'return_6m',
            'value_score', 'quality_score', 'momentum_score_final',
            'sentiment_score_final', 'low_vol_score_final', 'extreme_move'
        ]].copy()

        # FIX: tandain ticker dengan return ekstrem biar user sadar buat sanity-check manual
        display_df['ticker'] = display_df.apply(
            lambda r: f"⚠️ {r['ticker']}" if r['extreme_move'] else r['ticker'], axis=1
        )
        display_df = display_df.drop(columns=['extreme_move'])

        display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "-")
        display_df['forward_pe'] = display_df['forward_pe'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        display_df['pb'] = display_df['pb'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        display_df['roe'] = display_df['roe'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
        display_df['return_12_1'] = display_df['return_12_1'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        display_df['return_6m'] = display_df['return_6m'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        for col in ['rating', 'value_score', 'quality_score', 'momentum_score_final',
                    'sentiment_score_final', 'low_vol_score_final']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

        display_df.columns = ['Rank', 'Ticker', 'Nama', 'Sektor', 'Harga', 'Mkt Cap',
                              'Rating', 'Fwd P/E', 'PBV', 'ROE', 'Ret 12-1', 'Ret 6M',
                              'Value', 'Quality', 'Momentum', 'Sentiment', 'Low Vol']

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        if df_results.head(top_n)['extreme_move'].any():
            st.caption("⚠️ = return 12-1 bulan >200% atau return 6 bulan >150%. Bukan otomatis salah data, tapi worth di-cross-check manual ke sumber lain sebelum dipakai (bisa jadi rally beneran, bisa juga ada korporat aksi yang gak ke-handle sempurna).")

        # ============ BREAKDOWN PER SEKTOR ============
        st.subheader("🏭 Distribusi Sektor di Top Picks")
        sector_counts = df_results.head(top_n)['sector'].value_counts()
        st.bar_chart(sector_counts)

        # ============ BREAKDOWN SKOR ============
        st.subheader("📊 Breakdown Skor per Faktor")
        chart_data = df_results.head(top_n).set_index('ticker')[
            ['value_score', 'quality_score', 'momentum_score_final',
             'sentiment_score_final', 'low_vol_score_final']
        ]
        st.bar_chart(chart_data)

        # ============ DOWNLOAD ============
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV (semua hasil)", csv, "screener_results.csv", "text/csv")