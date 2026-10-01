import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
# Bobot faktor (contoh, idealnya dari backtest)
FACTOR_WEIGHTS = {
    'value': 0.25,
    'quality': 0.30,
    'momentum': 0.20,
    'sentiment': 0.15,
    'low_vol': 0.10
}

# ============================================================
# STEP 1: FETCH UNIVERSE (S&P 500 dari Wikipedia)
# ============================================================
def get_all_us_tickers():
    """Mengambil daftar seluruh saham AS dari NASDAQ Stock Screener."""
    url = "https://www.nasdaq.com/market-activity/stocks/screener"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    # NASDAQ menyediakan endpoint JSON internal yang bisa kita akses
    api_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&offset=0"
    response = requests.get(api_url, headers=headers)
    data = response.json()
    
    tickers = [row['symbol'] for row in data['data']['rows']]
    # Bersihkan ticker yang mengandung karakter aneh (misal: warrant, right)
    tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
    return tickers

# ============================================================
# STEP 2: FETCH DATA PER SAHAM (BANYAK FAKTOR)
# ============================================================
def fetch_stock_data(ticker):
    """Fetch data fundamental, harga, dan sentimen untuk satu saham."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='2y') # 2 tahun untuk momentum 12-1
        
        if hist.empty or len(hist) < 300:
            return None
        
        # --- Fundamental (Value & Quality) ---
        pe = info.get('trailingPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        ps = info.get('priceToSalesTrailing12Months', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        roa = info.get('returnOnAssets', np.nan)
        margin = info.get('operatingMargins', np.nan)
        
        # --- Quality Lanjutan (Accruals & F-Score) ---
        # Accruals: (Net Income - CFO) / Total Assets. Makin rendah makin baik.
        net_income = info.get('netIncomeToCommon', np.nan)
        cfo = info.get('operatingCashflow', np.nan)
        total_assets = info.get('totalAssets', np.nan)
        accruals = (net_income - cfo) / total_assets if total_assets and total_assets > 0 else np.nan
        
        # --- Momentum (12-1) ---
        close = hist['Close']
        # Return 12 bulan (252 hari) sampai 1 bulan lalu (21 hari)
        if len(close) >= 273:
            return_12_1 = (close.iloc[-22] / close.iloc[-273] - 1) * 100
        else:
            return_12_1 = np.nan
        
        # --- Low Volatility ---
        daily_returns = close.pct_change().dropna()
        volatility_1y = daily_returns.tail(252).std() * np.sqrt(252) * 100
        
        # --- Analyst Sentiment (dari yfinance) ---
        # yfinance menyediakan targetMeanPrice dan numberOfAnalystOpinions
        target_mean = info.get('targetMeanPrice', np.nan)
        current_price = close.iloc[-1]
        analyst_upside = ((target_mean / current_price) - 1) * 100 if target_mean and current_price else np.nan
        
        # --- Short Interest (dari info) ---
        short_ratio = info.get('shortRatio', np.nan) # Days to cover
        short_percent_float = info.get('shortPercentOfFloat', np.nan)
        
        # --- Sector (untuk sector-neutral scoring) ---
        sector = info.get('sector', 'Unknown')
        
        return {
            'ticker': ticker,
            'sector': sector,
            'price': current_price,
            'pe': pe, 'pb': pb, 'ps': ps,
            'roe': roe, 'roa': roa, 'margin': margin,
            'accruals': accruals,
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
# STEP 3: WINSORIZATION & SECTOR-NEUTRAL SCORING
# ============================================================
def winsorize(series, lower=0.01, upper=0.99):
    """Potong outlier pada persentil tertentu."""
    return series.clip(series.quantile(lower), series.quantile(upper))

def sector_neutral_score(df, column, ascending=True):
    """
    Hitung percentile rank DI DALAM SEKTOR masing-masing.
    ascending=True: nilai rendah = skor tinggi (untuk P/E, accruals, short interest)
    ascending=False: nilai tinggi = skor tinggi (untuk ROE, momentum, analyst upside)
    """
    scores = df.groupby('sector')[column].rank(pct=True, ascending=ascending) * 100
    return scores

def calculate_scores(df):
    """Hitung skor komposit dengan normalisasi sector-neutral."""
    df = df.copy()
    
    # 1. Winsorize semua kolom numerik
    numeric_cols = ['pe', 'pb', 'ps', 'roe', 'roa', 'margin', 'accruals', 
                    'return_12_1', 'volatility', 'analyst_upside', 
                    'short_ratio', 'short_percent_float']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = winsorize(df[col])
    
    # 2. Hitung skor sector-neutral untuk setiap metrik
    # Value: makin rendah makin baik
    df['pe_score'] = sector_neutral_score(df, 'pe', ascending=True)
    df['pb_score'] = sector_neutral_score(df, 'pb', ascending=True)
    df['ps_score'] = sector_neutral_score(df, 'ps', ascending=True)
    
    # Quality: makin tinggi makin baik, tapi accruals makin rendah makin baik
    df['roe_score'] = sector_neutral_score(df, 'roe', ascending=False)
    df['roa_score'] = sector_neutral_score(df, 'roa', ascending=False)
    df['margin_score'] = sector_neutral_score(df, 'margin', ascending=False)
    df['accruals_score'] = sector_neutral_score(df, 'accruals', ascending=True) # rendah = baik
    
    # Momentum: 12-1 return, makin tinggi makin baik
    df['momentum_score'] = sector_neutral_score(df, 'return_12_1', ascending=False)
    
    # Low Volatility: makin rendah makin baik
    df['low_vol_score'] = sector_neutral_score(df, 'volatility', ascending=True)
    
    # Analyst Sentiment: makin tinggi upside makin baik
    df['sentiment_score'] = sector_neutral_score(df, 'analyst_upside', ascending=False)
    
    # Short Interest: makin rendah makin baik (hindari saham yang di-short)
    df['short_score'] = sector_neutral_score(df, 'short_percent_float', ascending=True)
    
    # 3. Agregasi per faktor
    df['value_score'] = (df['pe_score'] * 0.4 + df['pb_score'] * 0.3 + df['ps_score'] * 0.3)
    df['quality_score'] = (df['roe_score'] * 0.3 + df['roa_score'] * 0.2 + 
                           df['margin_score'] * 0.3 + df['accruals_score'] * 0.2)
    df['momentum_score_final'] = df['momentum_score']
    df['sentiment_score_final'] = df['sentiment_score'] * 0.7 + df['short_score'] * 0.3
    df['low_vol_score_final'] = df['low_vol_score']
    
    # 4. Skor komposit dengan bobot
    df['composite_score'] = (
        df['value_score'] * FACTOR_WEIGHTS['value'] +
        df['quality_score'] * FACTOR_WEIGHTS['quality'] +
        df['momentum_score_final'] * FACTOR_WEIGHTS['momentum'] +
        df['sentiment_score_final'] * FACTOR_WEIGHTS['sentiment'] +
        df['low_vol_score_final'] * FACTOR_WEIGHTS['low_vol']
    )
    
    # 5. Rating 1-100
    df['rating'] = df['composite_score'].round(1)
    
    return df

# ============================================================
# STEP 4: MAIN PIPELINE (OTOMATIS, TANPA INPUT MANUAL)
# ============================================================
def run_full_screener(top_n=20):
    """Jalankan screener untuk seluruh S&P 500."""
    print("Mengambil daftar S&P 500...")
    tickers = get_sp500_tickers()
    print(f"Ditemukan {len(tickers)} saham. Mengambil data...")
    
    results = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}...", end='\r')
        data = fetch_stock_data(ticker)
        if data:
            results.append(data)
    
    print(f"\nBerhasil mengambil {len(results)} saham.")
    df = pd.DataFrame(results)
    
    print("Menghitung skor...")
    df = calculate_scores(df)
    
    # Ranking
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    # Output
    cols = ['rank', 'ticker', 'sector', 'price', 'rating', 
            'value_score', 'quality_score', 'momentum_score_final', 
            'sentiment_score_final', 'low_vol_score_final']
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} SAHAM BERDASARKAN SKOR KOMPOSIT (SECTOR-NEUTRAL)")
    print(f"{'='*80}")
    print(df[cols].head(top_n).to_string(index=False))
    
    return df

# Jalankan
if __name__ == "__main__":
    df_results = run_full_screener(top_n=20)
    df_results.to_csv('stock_screener_pro.csv', index=False)
    print("\nHasil disimpan ke stock_screener_pro.csv")
    def calculate_altman_z(info):
    """Hitung Altman Z-Score untuk risiko kebangkrutan."""
    try:
        wc = info.get('totalCurrentAssets', 0) - info.get('totalCurrentLiabilities', 0)
        ta = info.get('totalAssets', 1)
        re = info.get('retainedEarnings', 0)
        ebit = info.get('ebit', 0)
        mcap = info.get('marketCap', 0)
        tl = info.get('totalLiabilities', 1)
        sales = info.get('totalRevenue', 0)
        
        if ta <= 0 or tl <= 0:
            return np.nan
        
        z = (1.2 * (wc/ta) + 1.4 * (re/ta) + 3.3 * (ebit/ta) + 
             0.6 * (mcap/tl) + 1.0 * (sales/ta))
        return z
    except:
        return np.nan
    # Sebelum scoring, buang saham yang:
# - Altman Z-Score < 1.8 (risiko bangkrut tinggi)
# - Piotroski F-Score < 4 (fundamental memburuk)
# - Debt/Equity > 3.0 (utang terlalu besar)
df = df[df['altman_z'] >= 1.8]
df = df[df['f_score'] >= 4]
df = df[df['debt_to_equity'] <= 3.0]
# Ganti ini:
pe = info.get('trailingPE', np.nan)

# Dengan ini:
forward_pe = info.get('forwardPE', np.nan)
trailing_pe = info.get('trailingPE', np.nan)

# Earnings Revisions (data ini sangat berharga)
eps_revisions = stock.eps_revisions  # DataFrame dengan kolom 'upLast7days', 'upLast30days', dll.
if not eps_revisions.empty:
    rev_30d = eps_revisions.iloc[0].get('upLast30days', 0) - eps_revisions.iloc[0].get('downLast30days', 0)
else:
    rev_30d = 0