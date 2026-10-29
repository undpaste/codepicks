"""
Global Stock Screener Pro v5.0
===============================
Changelog v5.0 (Codepicks integration):
 [NEW] Tab UI: Stock Screener | Crypto | Macro AI | Portfolio | AI Fund
 [NEW] AI Macro Analysis (OpenAI gpt-4o-mini) — regime, risks, allocation
 [NEW] Crypto Screener (30 top coins, live-cached 60s, refresh button)
 [NEW] Portfolio Builder + Sharpe/Sortino/Calmar/MaxDD + max-Sharpe optimizer
 [NEW] AI Fund (paper trading, auto-rebalance dari Top-N screener, benchmark)
 [IMPROVED] Custom CSS, metric cards, expander, status indicators
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import random
import warnings
from collections import Counter
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# ---------- Optional deps (graceful fallback) ----------
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================
# KONFIGURASI (existing)
# ============================================================
NEGLECTED = "🔎 Neglected Value (small-cap inefisien)"
DEEP_VALUE = "💰 Deep Value"

STRATEGY_WEIGHTS = {
    "⚖️ Balanced (default)":   {'value': 0.25, 'quality': 0.30, 'momentum': 0.20, 'sentiment': 0.15, 'low_vol': 0.10, 'neglect': 0.00},
    "🏆 Quality Compounder":   {'value': 0.20, 'quality': 0.50, 'momentum': 0.10, 'sentiment': 0.10, 'low_vol': 0.10, 'neglect': 0.00},
    "🚀 High-Growth Momentum": {'value': 0.10, 'quality': 0.20, 'momentum': 0.45, 'sentiment': 0.20, 'low_vol': 0.05, 'neglect': 0.00},
    DEEP_VALUE:               {'value': 0.50, 'quality': 0.25, 'momentum': 0.10, 'sentiment': 0.10, 'low_vol': 0.05, 'neglect': 0.00},
    NEGLECTED:                {'value': 0.40, 'quality': 0.25, 'momentum': 0.10, 'sentiment': 0.00, 'low_vol': 0.05, 'neglect': 0.20},
}

DEFAULT_PRESET = {
    'IDX': {'mcap_min': 1.0, 'mcap_max': 0.0, 'max_runup': 0.0, 'require_profit': False, 'knife': (-40, -20, True)},
    'US':  {'mcap_min': 2.0, 'mcap_max': 0.0, 'max_runup': 0.0, 'require_profit': False, 'knife': (-25, -10, True)},
}
STRATEGY_PRESETS = {
    DEEP_VALUE: {'IDX': {'knife': (-60, None, False)}, 'US': {'knife': (-50, None, False)}},
    NEGLECTED: {
        'IDX': {'mcap_min': 0.3, 'mcap_max': 5.0,  'max_runup': 80.0, 'require_profit': True, 'knife': (-35, None, False)},
        'US':  {'mcap_min': 0.5, 'mcap_max': 10.0, 'max_runup': 80.0, 'require_profit': True, 'knife': (-30, None, False)},
    },
}

def get_preset(strategy, code):
    p = dict(DEFAULT_PRESET[code])
    p.update(STRATEGY_PRESETS.get(strategy, {}).get(code, {}))
    return p

MARKETS = {
    "🇺🇸 US Market (S&P 500)": {
        'code': 'US', 'suffix': '', 'currency': '$',
        'mcap_mult': 1e9, 'mcap_unit': 'B',
        'pos_label': 'Target posisi per saham ($ ribu)', 'pos_mult': 1e3, 'pos_default': 5.0,
        'adv_mult': 1e6, 'adv_unit': '$M', 'fetch_min_adv': 1e6,
        'universe_label': 'S&P 500', 'universe_max': 500, 'default_size': 100, 'default_full': False,
        'min_price': 5.0, 'max_pb': 20.0, 'max_vol': 150.0,
        'benchmark': '^GSPC',
    },
    "🇮🇩 Indonesia (IDX)": {
        'code': 'IDX', 'suffix': '.JK', 'currency': 'Rp ',
        'mcap_mult': 1e12, 'mcap_unit': 'T',
        'pos_label': 'Target posisi per saham (Rp juta)', 'pos_mult': 1e6, 'pos_default': 50.0,
        'adv_mult': 1e9, 'adv_unit': 'Rp B', 'fetch_min_adv': 1e8,
        'universe_label': 'IDX', 'universe_max': 950, 'default_size': 300, 'default_full': True,
        'min_price': 100.0, 'max_pb': 10.0, 'max_vol': 120.0,
        'benchmark': '^JKSE',
    },
}

MAX_WORKERS = 2
CACHE_TTL = 6 * 3600
RETRY_ATTEMPTS = 4
MIN_LISTING_DAYS = 250
MIN_SECTOR_SIZE = 8
MIN_FACTOR_COVERAGE = 0.30
MISSING_PENALTY = 35
PARTICIPATION = 0.10
FINANCIAL_SECTORS = {'Financial Services'}
LEVERAGE_EXEMPT = {'Financial Services', 'Real Estate', 'Utilities'}
EXTREME_RETURN_12M_THRESHOLD = 200
EXTREME_RETURN_6M_THRESHOLD = 150
SECTOR_ALIASES = {'Financials': 'Financial Services'}

IDX_NAME_MAP = {
    'BBCA':'Bank Central Asia Tbk','BBRI':'Bank Rakyat Indonesia Tbk',
    'BMRI':'Bank Mandiri Tbk','BBNI':'Bank Negara Indonesia Tbk',
    'BRIS':'Bank Syariah Indonesia Tbk','BTPS':'Bank BTPN Syariah Tbk',
    'ARTO':'Bank Jago Tbk','BBTN':'Bank Tabungan Negara Tbk',
    'BJBR':'Bank BJB Tbk','BJTM':'Bank Jatim Tbk',
    'BNGA':'Bank CIMB Niaga Tbk','BNLI':'Bank Permata Tbk',
    'TLKM':'Telkom Indonesia Tbk','EXCL':'XL Axiata Tbk',
    'ISAT':'Indosat Tbk','TOWR':'Sarana Menara Nusantara Tbk',
    'MTEL':'Dayamitra Telekomunikasi Tbk',
    'ASII':'Astra International Tbk','UNTR':'United Tractors Tbk',
    'UNVR':'Unilever Indonesia Tbk','ICBP':'Indofood CBP Tbk',
    'INDF':'Indofood Sukses Makmur Tbk','MYOR':'Mayora Indah Tbk',
    'SIDO':'Sido Muncul Tbk','AMRT':'Sumber Alfaria Trijaya Tbk',
    'CPIN':'Charoen Pokphand Indonesia Tbk','JPFA':'Japfa Comfeed Tbk',
    'HMSP':'HM Sampoerna Tbk','GGRM':'Gudang Garam Tbk',
    'ADRO':'Alamtri Resources Indonesia Tbk','PTBA':'Bukit Asam Tbk',
    'ITMG':'Indo Tambangraya Megah Tbk','MEDC':'Medco Energi Tbk',
    'PGAS':'Perusahaan Gas Negara Tbk','HRUM':'Harum Energy Tbk',
    'AKRA':'AKR Corporindo Tbk','ELSA':'Elnusa Tbk',
    'ANTM':'Aneka Tambang Tbk','INCO':'Vale Indonesia Tbk',
    'TINS':'Timah Tbk','SMGR':'Semen Indonesia Tbk',
    'INTP':'Indocement Tunggal Prakarsa Tbk','BRPT':'Barito Pacific Tbk',
    'TPIA':'Chandra Asri Pacific Tbk','INKP':'Indah Kiat Pulp & Paper Tbk',
    'MDKA':'Merdeka Copper Gold Tbk','GOTO':'GoTo Gojek Tokopedia Tbk',
    'BUKA':'Bukalapak.com Tbk','EMTK':'Elang Mahkota Teknologi Tbk',
    'MIKA':'Mitra Keluarga Karyasehat Tbk','SILO':'Siloam International Hospitals Tbk',
    'HEAL':'Medikaloka Hermina Tbk','KAEF':'Kimia Farma Tbk',
    'PTPP':'PP (Persero) Tbk','WIKA':'Wijaya Karya Tbk',
    'ADHI':'Adhi Karya Tbk','WSKT':'Waskita Karya Tbk',
    'JSMR':'Jasa Marga Tbk','BSDE':'Bumi Serpong Damai Tbk',
    'CTRA':'Ciputra Development Tbk','PWON':'Pakuwon Jati Tbk',
    'SMRA':'Summarecon Agung Tbk','ASRI':'Alam Sutera Realty Tbk',
    'LPKR':'Lippo Karawaci Tbk','MNCN':'Media Nusantara Citra Tbk',
    'SCMA':'Surya Citra Media Tbk',
}

IDX_SECTOR_FALLBACK = {
    'BBCA':'Financials','BBRI':'Financials','BMRI':'Financials','BBNI':'Financials',
    'BRIS':'Financials','BTPS':'Financials','ARTO':'Financials','BBTN':'Financials',
    'BJBR':'Financials','BJTM':'Financials','BNGA':'Financials','BNLI':'Financials',
    'PNBN':'Financials','MEGA':'Financials','NISP':'Financials','BFIN':'Financials',
    'ADMF':'Financials','PNLF':'Financials','TUGU':'Financials','ASBI':'Financials',
    'LPGI':'Financials','MCOR':'Financials','BABP':'Financials','AGRO':'Financials',
    'TLKM':'Communication Services','EXCL':'Communication Services','ISAT':'Communication Services',
    'TOWR':'Communication Services','MTEL':'Communication Services','MNCN':'Communication Services',
    'SCMA':'Communication Services',
    'ASII':'Consumer Cyclical','AUTO':'Consumer Cyclical','SMSM':'Consumer Cyclical',
    'MAPI':'Consumer Cyclical','ACES':'Consumer Cyclical','ERAA':'Consumer Cyclical',
    'UNTR':'Industrials','HEXA':'Industrials',
    'UNVR':'Consumer Defensive','ICBP':'Consumer Defensive','INDF':'Consumer Defensive',
    'MYOR':'Consumer Defensive','SIDO':'Consumer Defensive','AMRT':'Consumer Defensive',
    'CPIN':'Consumer Defensive','JPFA':'Consumer Defensive','MAIN':'Consumer Defensive',
    'HMSP':'Consumer Defensive','GGRM':'Consumer Defensive','WIIM':'Consumer Defensive',
    'DMND':'Consumer Defensive',
    'ADRO':'Energy','PTBA':'Energy','ITMG':'Energy','MEDC':'Energy','PGAS':'Energy',
    'HRUM':'Energy','AKRA':'Energy','ELSA':'Energy','BUMI':'Energy','DOID':'Energy',
    'HRTA':'Energy',
    'ANTM':'Basic Materials','INCO':'Basic Materials','TINS':'Basic Materials',
    'SMGR':'Basic Materials','INTP':'Basic Materials','BRPT':'Basic Materials',
    'TPIA':'Basic Materials','INKP':'Basic Materials','TKIM':'Basic Materials',
    'MDKA':'Basic Materials','NCKL':'Basic Materials',
    'GOTO':'Technology','BUKA':'Technology','EMTK':'Technology',
    'MIKA':'Healthcare','SILO':'Healthcare','HEAL':'Healthcare','PRDA':'Healthcare',
    'KAEF':'Healthcare','INAF':'Healthcare','SAME':'Healthcare','MTCN':'Healthcare',
    'PTPP':'Industrials','WIKA':'Industrials','ADHI':'Industrials','WSKT':'Industrials',
    'JSMR':'Industrials',
    'BSDE':'Real Estate','CTRA':'Real Estate','PWON':'Real Estate','SMRA':'Real Estate',
    'ASRI':'Real Estate','LPKR':'Real Estate',
}

def get_idx_fallback_comprehensive():
    raw = ['BBCA','BBRI','BMRI','BBNI','BRIS','BTPS','ARTO','BBTN','BJBR','BJTM',
           'BNGA','BNLI','PNBN','MEGA','NISP','BFIN','ADMF','PNLF','TUGU','ASBI',
           'LPGI','MCOR','BABP','AGRO','ANTM','INCO','TINS','SMGR','INTP','BRPT',
           'TPIA','INKP','TKIM','MDKA','NCKL','MBMA','ADRO','PTBA','ITMG','MEDC',
           'PGAS','HRUM','AKRA','ELSA','BUMI','DOID','UNVR','ICBP','INDF','MYOR',
           'SIDO','AMRT','CPIN','JPFA','MAIN','HMSP','GGRM','ASII','AUTO','SMSM',
           'MAPI','ACES','ERAA','TLKM','EXCL','ISAT','TOWR','MTEL','MNCN','SCMA',
           'GOTO','BUKA','EMTK','MIKA','SILO','HEAL','PRDA','KAEF','INAF','PTPP',
           'WIKA','ADHI','WSKT','JSMR','BSDE','CTRA','PWON','SMRA','ASRI','LPKR',
           'UNTR','HEXA']
    return list(dict.fromkeys(raw))


# ============================================================
# UNIVERSE (existing)
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        raw = tables[0]['Symbol'].tolist()
        tickers, seen = [], set()
        for t in raw:
            t = str(t).strip().replace('.', '-')
            if t and t not in seen:
                tickers.append(t); seen.add(t)
        return sorted(tickers)
    except Exception as e:
        st.warning(f"Wikipedia gagal ({e}). Pakai fallback.")
        return get_us_fallback()

def get_us_fallback():
    return ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','UNH','XOM',
            'JNJ','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP','KO',
            'AVGO','COST','WMT','TMO','MCD','CSCO','ACN','ABT','CRM','ADBE','DHR',
            'LIN','NKE','TXN','AMD','PM','NEE','WFC','DIS','UPS','RTX','BMY','ORCL',
            'QCOM','INTC','HON','T','UNP','BA','LOW','SBUX','GS','INTU','AMAT','DE']

@st.cache_data(ttl=86400, show_spinner=False)
def get_idx_tickers():
    try:
        url = "https://www.idx.co.id/primary/StockData/GetSecuritiesStock"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.idx.co.id/en/market-data/stocks-data/stock-list/",
            "Accept": "application/json, text/plain, */*",
        }
        params = {"start": 0, "length": 9999, "code": "", "sector": "", "board": "", "language": "en-us"}
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            rows = r.json().get("data", [])
            tickers, seen = [], set()
            for row in rows:
                code = str(row.get("Code", "")).strip().upper()
                if code and code.isalpha() and 3 <= len(code) <= 5 and code not in seen:
                    tickers.append(code); seen.add(code)
            if len(tickers) > 500:
                return sorted(tickers)
    except Exception:
        pass
    try:
        url = "https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        tickers, seen = [], set()
        for tbl in tables:
            for col_name in tbl.columns:
                if "Kode" in str(col_name) or "kode" in str(col_name):
                    for raw in tbl[col_name].astype(str):
                        code = raw.split(":")[-1].strip().upper().replace("[", "").replace("]", "").strip()
                        if code.isalpha() and 3 <= len(code) <= 5 and code not in seen:
                            tickers.append(code); seen.add(code)
                    break
        if len(tickers) > 300:
            return sorted(tickers)
    except Exception:
        pass
    return get_idx_fallback_comprehensive()

def sample_universe(all_tickers, size, use_full):
    shuffled = list(all_tickers)
    random.Random(42).shuffle(shuffled)
    if use_full or size >= len(shuffled):
        return shuffled
    return shuffled[:size]


# ============================================================
# HELPERS (existing)
# ============================================================
def _num(x):
    try:
        if x is None or isinstance(x, bool):
            return np.nan
        v = float(x)
        return v if np.isfinite(v) else np.nan
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

def calculate_rsi(close, period=14):
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else np.nan
    except Exception:
        return np.nan

def normalize_sector(sector, ticker_base):
    if not sector or sector in ('Unknown', ''):
        sector = IDX_SECTOR_FALLBACK.get(ticker_base, 'Other')
    return SECTOR_ALIASES.get(sector, sector)


# ============================================================
# FETCH (existing) — 2-phase batch
# ============================================================
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_batch_histories(tickers_tuple, suffix):
    full = [f"{t}{suffix}" for t in tickers_tuple]
    try:
        return yf.download(full, period='2y', progress=False, auto_adjust=True,
                          threads=True, group_by='ticker', timeout=60)
    except Exception:
        return None

def _process_batch_history(batch, ticker_base, suffix, fetch_min_adv):
    full = f"{ticker_base}{suffix}"
    try:
        if batch is None or batch.empty:
            return ('no_data', None)
        if full not in batch.columns.get_level_values(0):
            return ('no_data', None)
        sub = batch[full].dropna(subset=['Close'])
        if len(sub) < MIN_LISTING_DAYS:
            return ('short_history', None)
        close = sub['Close']
        volume = sub['Volume'].fillna(0)
        price = float(close.iloc[-1])
        if not np.isfinite(price) or price <= 0:
            return ('no_data', None)
        adv_median = float((close * volume).tail(60).median())
        zero_vol_ratio = float((volume.tail(60) == 0).mean())
        if not np.isfinite(adv_median) or adv_median < fetch_min_adv:
            return ('illiquid', None)
        return ('ok', {'price': price, 'close': close, 'adv_median': adv_median,
                       'zero_vol_ratio': zero_vol_ratio, 'hist_len': len(sub)})
    except Exception:
        return ('error', None)

def _fetch_info_and_build(ticker_base, full_ticker, price, adv_median,
                         zero_vol_ratio, close):
    try:
        time.sleep(random.uniform(0.2, 0.5))
        stock = yf.Ticker(full_ticker)
        info = fetch_with_retry(lambda: stock.info) or {}
        if len(info) < 5:
            return ('no_info', None)
        ccy, fin_ccy = info.get('currency'), info.get('financialCurrency')
        fin_ok = not (ccy and fin_ccy and ccy != fin_ccy)
        market_cap = _num(info.get('marketCap'))
        shares = _num(info.get('sharesOutstanding'))
        if np.isnan(market_cap) and np.isfinite(shares):
            market_cap = shares * price
        trailing_pe = _num(info.get('trailingPE'))
        forward_pe = _num(info.get('forwardPE'))
        pb = _num(info.get('priceToBook'))
        ps = _num(info.get('priceToSalesTrailing12Months'))
        ev_ebitda = _num(info.get('enterpriseToEbitda'))
        roe = _num(info.get('returnOnEquity'))
        roa = _num(info.get('returnOnAssets'))
        margin = _num(info.get('operatingMargins'))
        profit_margin = _num(info.get('profitMargins'))
        rev_growth = _num(info.get('revenueGrowth'))
        debt_to_equity = _num(info.get('debtToEquity'))
        eps = _num(info.get('trailingEps')) if fin_ok else np.nan
        net_income = _num(info.get('netIncomeToCommon')) if fin_ok else np.nan
        ocf = _num(info.get('operatingCashflow')) if fin_ok else np.nan
        fcf = _num(info.get('freeCashflow')) if fin_ok else np.nan
        revenue = _num(info.get('totalRevenue')) if fin_ok else np.nan
        total_debt = _num(info.get('totalDebt')) if fin_ok else np.nan
        total_cash = _num(info.get('totalCash')) if fin_ok else np.nan
        ebitda = _num(info.get('ebitda')) if fin_ok else np.nan
        analysts_n = _num(info.get('numberOfAnalystOpinions'))
        inst_pct = _num(info.get('heldPercentInstitutions'))
        float_sh = _num(info.get('floatShares'))
        target_mean = _num(info.get('targetMeanPrice'))
        short_pct_float = _num(info.get('shortPercentOfFloat'))
        if np.isfinite(trailing_pe) and trailing_pe > 0:
            earnings_yield = 1.0 / trailing_pe
        elif np.isfinite(eps):
            earnings_yield = eps / price
        else:
            earnings_yield = np.nan
        if np.isfinite(pb):
            book_yield = (1.0 / pb) if pb > 0 else -1.0
        else:
            book_yield = np.nan
        fcf_yield = (fcf / market_cap) if (np.isfinite(fcf) and np.isfinite(market_cap) and market_cap > 0) else np.nan
        ebitda_yield = (1.0 / ev_ebitda) if (np.isfinite(ev_ebitda) and ev_ebitda > 0) else (np.nan if not np.isfinite(ev_ebitda) else -1.0)
        sales_yield = (1.0 / ps) if (np.isfinite(ps) and ps > 0) else np.nan
        div_rate = _num(info.get('dividendRate'))
        if np.isnan(div_rate):
            div_rate = _num(info.get('trailingAnnualDividendRate'))
        if np.isfinite(div_rate) and div_rate > 0:
            div_yield = div_rate / price
            if div_yield > 0.30:
                div_yield = np.nan
        else:
            div_yield = 0.0
        accruals = ((net_income - ocf) / revenue) if (np.isfinite(net_income) and np.isfinite(ocf)
                                                       and np.isfinite(revenue) and revenue > 0) else np.nan
        net_debt_ebitda = np.nan
        distress_flag = False
        if np.isfinite(ebitda) and np.isfinite(total_debt) and np.isfinite(total_cash):
            if ebitda > 0:
                net_debt_ebitda = (total_debt - total_cash) / ebitda
            elif total_debt > total_cash:
                distress_flag = True
        float_pct = np.nan
        if np.isfinite(float_sh) and np.isfinite(shares) and shares > 0:
            float_pct = min(100.0, float_sh / shares * 100)
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        above_sma200 = bool(price > sma_200) if pd.notna(sma_200) else True
        lb = min(252, len(close) - 1)
        return_12m = (close.iloc[-1] / close.iloc[-1 - lb] - 1) * 100
        return_12_1 = (close.iloc[-22] / close.iloc[-273] - 1) * 100 if len(close) >= 273 else np.nan
        return_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) >= 126 else np.nan
        rsi = calculate_rsi(close)
        volatility = close.pct_change().dropna().tail(252).std() * np.sqrt(252) * 100
        extreme_move = bool(
            (pd.notna(return_12_1) and abs(return_12_1) > EXTREME_RETURN_12M_THRESHOLD) or
            (pd.notna(return_6m) and abs(return_6m) > EXTREME_RETURN_6M_THRESHOLD)
        )
        analyst_upside = ((target_mean / price) - 1) * 100 if np.isfinite(target_mean) else np.nan
        core = [earnings_yield, book_yield, roe, roa, margin, rev_growth, debt_to_equity, ocf]
        data_completeness = sum(pd.notna(x) for x in core) / len(core) * 100
        sector = normalize_sector(info.get('sector'), ticker_base)
        name = info.get('longName') or info.get('shortName')
        if not name or name == full_ticker:
            name = IDX_NAME_MAP.get(ticker_base, ticker_base)
        return ('ok', {
            'ticker': ticker_base, 'name': name, 'sector': sector,
            'price': price, 'market_cap': market_cap,
            'trailing_pe': trailing_pe, 'forward_pe': forward_pe, 'pb': pb,
            'earnings_yield': earnings_yield, 'book_yield': book_yield, 'fcf_yield': fcf_yield,
            'ebitda_yield': ebitda_yield, 'sales_yield': sales_yield, 'div_yield': div_yield,
            'roe': roe, 'roa': roa, 'margin': margin, 'profit_margin': profit_margin,
            'rev_growth': rev_growth, 'debt_to_equity': debt_to_equity,
            'accruals': accruals, 'net_debt_ebitda': net_debt_ebitda, 'distress_flag': distress_flag,
            'net_income': net_income, 'ocf': ocf,
            'analysts_n': analysts_n, 'inst_pct': inst_pct, 'float_pct': float_pct,
            'analyst_upside': analyst_upside, 'short_pct_float': short_pct_float,
            'return_12m': return_12m, 'return_12_1': return_12_1, 'return_6m': return_6m,
            'above_sma200': above_sma200, 'rsi': rsi, 'volatility': volatility,
            'adv_median': adv_median, 'zero_vol_ratio': zero_vol_ratio,
            'data_completeness': data_completeness, 'extreme_move': extreme_move,
            'fin_currency_ok': fin_ok,
        })
    except Exception:
        return ('error', None)

def fetch_universe(tickers, suffix, fetch_min_adv, progress_callback=None):
    results, failed = [], {}
    counts = Counter()
    total = len(tickers)
    if progress_callback:
        progress_callback(0, total, "Phase 1: batch download history...")
    batch = fetch_batch_histories(tuple(tickers), suffix)
    candidates = []
    for tb in tickers:
        status, payload = _process_batch_history(batch, tb, suffix, fetch_min_adv)
        counts[status] += 1
        if status == 'ok':
            candidates.append((tb, f"{tb}{suffix}", payload))
        else:
            failed.setdefault(status, []).append(tb)
    n_candidates = len(candidates)
    if n_candidates == 0:
        return pd.DataFrame(results), dict(counts), failed
    if progress_callback:
        progress_callback(0, n_candidates, f"Phase 2: fetch info ({n_candidates} kandidat)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for tb, full, payload in candidates:
            futures[ex.submit(
                _fetch_info_and_build, tb, full, payload['price'],
                payload['adv_median'], payload['zero_vol_ratio'], payload['close']
            )] = tb
        done = 0
        for fut in as_completed(futures):
            tb = futures[fut]
            done += 1
            try:
                status, data = fut.result()
            except Exception:
                status, data = 'error', None
            counts[status] += 1
            if status == 'ok' and data:
                results.append(data)
            else:
                failed.setdefault(status, []).append(tb)
            if progress_callback:
                progress_callback(done, n_candidates, f"info: {tb}")
    return pd.DataFrame(results), dict(counts), failed


# ============================================================
# SCORING (existing)
# ============================================================
def pct_score(df, col, higher_is_better=True):
    asc = bool(higher_is_better)
    s = df[col]
    global_r = s.rank(pct=True, ascending=asc) * 100
    sector_r = df.groupby('sector')[col].rank(pct=True, ascending=asc) * 100
    sector_n = df.groupby('sector')[col].transform('count')
    return sector_r.where(sector_n >= MIN_SECTOR_SIZE, global_r)

def weighted_available(score_df, weights, min_components=2):
    cols = list(weights.keys())
    w = pd.Series(weights, dtype=float)
    vals = score_df[cols]
    avail = vals.notna().astype(float)
    num = (vals.fillna(0) * w).sum(axis=1)
    den = (avail * w).sum(axis=1)
    out = num / den.replace(0, np.nan)
    return out.where(avail.sum(axis=1) >= min_components)

def calculate_scores(df, weights):
    df = df.copy()
    fin = df['sector'].isin(FINANCIAL_SECTORS)
    for c in ['fcf_yield', 'ebitda_yield', 'sales_yield', 'net_debt_ebitda', 'accruals', 'debt_to_equity']:
        df.loc[fin, c] = np.nan
    val_scores = pd.DataFrame({
        'ey': pct_score(df, 'earnings_yield', True),
        'by': pct_score(df, 'book_yield', True),
        'fy': pct_score(df, 'fcf_yield', True),
        'eb': pct_score(df, 'ebitda_yield', True),
        'sy': pct_score(df, 'sales_yield', True),
        'dy': pct_score(df, 'div_yield', True),
    })
    value = weighted_available(val_scores, {'ey': .30, 'by': .25, 'fy': .20, 'eb': .10, 'sy': .05, 'dy': .10}, 2)
    qual_scores = pd.DataFrame({
        'roe': pct_score(df, 'roe', True),
        'roa': pct_score(df, 'roa', True),
        'mar': pct_score(df, 'margin', True),
        'acc': pct_score(df, 'accruals', False),
        'de':  pct_score(df, 'debt_to_equity', False),
        'gr':  pct_score(df.assign(rev_growth=df['rev_growth'].clip(-0.5, 1.0)), 'rev_growth', True),
    })
    quality = weighted_available(qual_scores, {'roe': .25, 'roa': .15, 'mar': .25, 'acc': .15, 'de': .10, 'gr': .10}, 3)
    an = df['analysts_n'].fillna(0)
    an_score = pd.Series(np.select([an == 0, an <= 2, an <= 5], [100, 70, 40], default=10), index=df.index)
    neg_scores = pd.DataFrame({
        'an': an_score,
        'sz': pct_score(df, 'market_cap', False),
        'in': pct_score(df, 'inst_pct', False),
    })
    neglect = weighted_available(neg_scores, {'an': .5, 'sz': .3, 'in': .2}, 1)
    r12, r6 = df['return_12_1'], df['return_6m']
    mom = pct_score(df, 'return_12_1', True)
    mom = mom.where(~(r12 < 0), mom.clip(upper=60))
    mom = mom.where(~(r12 < -20), mom.clip(upper=30))
    mom = mom.where(~((r12 > 100) & (r6 < -20)), mom * 0.3)
    mom = mom.where(~((r6 > 150) & (df['volatility'] > 80)), mom * 0.5)
    rsi_s = df['rsi'].apply(lambda x: max(0.0, 100 - abs(x - 55) * 2) if pd.notna(x) else np.nan)
    momentum = (mom * 0.7 + rsi_s * 0.3).where(rsi_s.notna(), mom)
    sent_scores = pd.DataFrame({
        'up': pct_score(df, 'analyst_upside', True),
        'sh': pct_score(df, 'short_pct_float', False),
    })
    sentiment = weighted_available(sent_scores, {'up': .7, 'sh': .3}, 1)
    low_vol = pct_score(df, 'volatility', False)
    factor_scores = {'value': value, 'quality': quality, 'momentum': momentum,
                     'sentiment': sentiment, 'low_vol': low_vol, 'neglect': neglect}
    eff_w = {}
    for k, sc in factor_scores.items():
        coverage = float(sc.notna().mean()) if len(sc) else 0.0
        eff_w[k] = weights.get(k, 0.0) if coverage >= MIN_FACTOR_COVERAGE else 0.0
    total_w = sum(eff_w.values())
    eff_w = {k: (v / total_w if total_w > 0 else 0.0) for k, v in eff_w.items()}
    composite = sum(eff_w[k] * factor_scores[k].fillna(MISSING_PENALTY) for k in factor_scores)
    df['value_score'] = value
    df['quality_score'] = quality
    df['neglect_score'] = neglect
    df['momentum_score_final'] = momentum
    df['sentiment_score_final'] = sentiment
    df['low_vol_score_final'] = low_vol
    df['composite_score'] = composite
    df['rating'] = composite.round(1)
    return df, eff_w


# ============================================================
# FILTER (existing)
# ============================================================
def _step(df, funnel, label, cond):
    before = len(df)
    out = df[cond].reset_index(drop=True)
    funnel.append((label, before - len(out)))
    return out

def screen(raw_df, p, strategy, weights):
    funnel = []
    df = raw_df.copy()
    if df.empty:
        return df, funnel, {}
    df = _step(df, funnel, "Market cap di luar rentang",
               df['market_cap'].notna() & (df['market_cap'] >= p['mcap_min']) &
               ((df['market_cap'] <= p['mcap_max']) if p['mcap_max'] > 0 else True))
    df = _step(df, funnel, "Harga di bawah minimum", df['price'] >= p['min_price'])
    df = _step(df, funnel, "Terlalu tidak likuid untuk ukuran posisi kamu", df['adv_median'] >= p['min_adv_required'])
    df = _step(df, funnel, "Terlalu banyak hari tanpa transaksi", df['zero_vol_ratio'] <= p['max_zero_vol'])
    df = _step(df, funnel, "Data fundamental terlalu bolong", df['data_completeness'] >= p['min_completeness'])
    pb_ok = df['pb'].isna() | (df['pb'] <= p['max_pb'])
    if p['drop_negative_book']:
        pb_ok &= df['pb'].isna() | (df['pb'] > 0)
    df = _step(df, funnel, "PBV terlalu tinggi / ekuitas negatif", pb_ok)
    df = _step(df, funnel, "Volatilitas terlalu tinggi", df['volatility'].isna() | (df['volatility'] <= p['max_vol']))
    df = _step(df, funnel, "Free float terlalu kecil (jika datanya ada)",
               df['float_pct'].isna() | (df['float_pct'] >= p['min_float_pct']))
    if p['require_profit']:
        fin = df['sector'].isin(FINANCIAL_SECTORS)
        ni, pm, roe = df['net_income'], df['profit_margin'], df['roe']
        profit_ok = pd.Series(
            np.where(ni.notna(), ni > 0, np.where(pm.notna(), pm > 0, np.where(roe.notna(), roe > 0, False))),
            index=df.index)
        ocf_ok = df['ocf'].isna() | (df['ocf'] > 0) | fin
        df = _step(df, funnel, "Tidak laba / arus kas operasi negatif", profit_ok & ocf_ok)
    exempt = df['sector'].isin(LEVERAGE_EXEMPT)
    fin = df['sector'].isin(FINANCIAL_SECTORS)
    health_ok = (
        (df['debt_to_equity'].isna() | (df['debt_to_equity'] <= 300) | fin) &
        (df['net_debt_ebitda'].isna() | (df['net_debt_ebitda'] <= 6) | exempt) &
        (~df['distress_flag'].astype(bool) | exempt)
    )
    df = _step(df, funnel, "Kesehatan keuangan (utang / net debt-EBITDA / EBITDA negatif)", health_ok)
    if p['max_runup'] and p['max_runup'] > 0:
        df = _step(df, funnel, "Sudah naik terlalu tinggi 12 bulan (anti-sudah-lari)",
                   df['return_12m'].isna() | (df['return_12m'] <= p['max_runup']))
    limit, sma_exc, require_trend = p['knife']
    knife_ok = df['return_6m'].isna() | (df['return_6m'] > limit)
    if require_trend:
        knife_ok &= (df['above_sma200'] == True) | df['return_6m'].isna() | (df['return_6m'] > sma_exc)
    df = _step(df, funnel, "Falling knife guard", knife_ok)
    if df.empty:
        return df, funnel, {}
    df, eff_w = calculate_scores(df, weights)
    def _flags(r):
        f = []
        if r['extreme_move']: f.append("⚠️ret-ekstrem")
        if pd.notna(r['float_pct']) and r['float_pct'] < 15: f.append("💧float-kecil")
        if r['zero_vol_ratio'] > 0.10: f.append("🔸hari-sepi")
        if pd.notna(r['return_12m']) and r['return_12m'] > 60: f.append("🚀sudah-lari")
        if not r['fin_currency_ok']: f.append("💱lapor-valas")
        return " ".join(f)
    df['flag'] = df.apply(_flags, axis=1)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df['days_to_exit'] = p['pos_value'] / (PARTICIPATION * df['adv_median'])
    return df, funnel, eff_w


def fmt_pct(x, mult=1.0, nd=1):
    return f"{x*mult:.{nd}f}%" if pd.notna(x) else "—"


# ============================================================
# [NEW v5.0] CRYPTO MODULE
# ============================================================
CRYPTO_UNIVERSE = {
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'BNB-USD': 'BNB',
    'SOL-USD': 'Solana', 'XRP-USD': 'XRP', 'ADA-USD': 'Cardano',
    'DOGE-USD': 'Dogecoin', 'AVAX-USD': 'Avalanche', 'DOT-USD': 'Polkadot',
    'LINK-USD': 'Chainlink', 'MATIC-USD': 'Polygon', 'TRX-USD': 'TRON',
    'LTC-USD': 'Litecoin', 'ATOM-USD': 'Cosmos', 'UNI-USD': 'Uniswap',
    'NEAR-USD': 'NEAR', 'APT-USD': 'Aptos', 'ARB-USD': 'Arbitrum',
    'OP-USD': 'Optimism', 'INJ-USD': 'Injective', 'SUI-USD': 'Sui',
    'TIA-USD': 'Celestia', 'SEI-USD': 'Sei', 'RNDR-USD': 'Render',
    'FIL-USD': 'Filecoin', 'HBAR-USD': 'Hedera', 'XLM-USD': 'Stellar',
    'ICP-USD': 'Internet Computer', 'AAVE-USD': 'Aave', 'MKR-USD': 'Maker',
}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_crypto_histories(tickers_tuple):
    try:
        return yf.download(list(tickers_tuple), period='1y', interval='1d',
                          progress=False, auto_adjust=True, threads=True,
                          group_by='ticker', timeout=30)
    except Exception:
        return None

def score_crypto(raw_df):
    rows = []
    for ticker in raw_df.columns.get_level_values(0).unique():
        try:
            sub = raw_df[ticker].dropna(subset=['Close'])
            if len(sub) < 30:
                continue
            close = sub['Close']
            volume = sub['Volume'].fillna(0)
            price = float(close.iloc[-1])
            if not np.isfinite(price) or price <= 0:
                continue
            ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else np.nan
            ret_7d = (close.iloc[-1] / close.iloc[-8] - 1) * 100 if len(close) >= 8 else np.nan
            ret_30d = (close.iloc[-1] / close.iloc[-31] - 1) * 100 if len(close) >= 31 else np.nan
            ret_90d = (close.iloc[-1] / close.iloc[-91] - 1) * 100 if len(close) >= 91 else np.nan
            vol_30d = close.pct_change().tail(30).std() * np.sqrt(365) * 100
            sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
            rsi = calculate_rsi(close)
            adv_usd = float((close * volume).tail(30).median())
            rows.append({
                'ticker': ticker, 'name': CRYPTO_UNIVERSE.get(ticker, ticker.split('-')[0]),
                'price': price, 'ret_1d': ret_1d, 'ret_7d': ret_7d,
                'ret_30d': ret_30d, 'ret_90d': ret_90d,
                'vol_30d': vol_30d,
                'above_sma20': bool(price > sma_20) if pd.notna(sma_20) else False,
                'above_sma50': bool(price > sma_50) if pd.notna(sma_50) else False,
                'rsi': rsi, 'adv_usd': adv_usd,
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    def pct(col, hb=True):
        return df[col].rank(pct=True, ascending=hb) * 100
    df['momentum_score'] = pct('ret_30d') * 0.4 + pct('ret_90d') * 0.3 + pct('ret_7d') * 0.3
    df['trend_score'] = df['above_sma20'].astype(float) * 50 + df['above_sma50'].astype(float) * 50
    df['low_vol_score'] = pct('vol_30d', False)
    df['liquidity_score'] = pct('adv_usd', True)
    df['rsi_score'] = df['rsi'].apply(lambda x: max(0, 100 - abs(x - 55) * 2) if pd.notna(x) else 50)
    df['composite'] = (df['momentum_score'] * 0.35 + df['trend_score'] * 0.20 +
                       df['low_vol_score'] * 0.15 + df['liquidity_score'] * 0.15 +
                       df['rsi_score'] * 0.15)
    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df


# ============================================================
# [NEW v5.0] MACRO MODULE
# ============================================================
MACRO_TICKERS = {
    'VIX (Volatility)': '^VIX',
    'DXY (USD Index)': 'DX-Y.NYB',
    'US 10Y Yield': '^TNX',
    'S&P 500': '^GSPC',
    'Nasdaq': '^IXIC',
    'Gold': 'GC=F',
    'Crude Oil': 'CL=F',
    'Bitcoin': 'BTC-USD',
    'IDX Composite': '^JKSE',
    'EM ETF (EEM)': 'EEM',
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_data():
    results = {}
    for label, ticker in MACRO_TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            h = t.history(period='6mo', interval='1d', auto_adjust=True)
            if h.empty:
                continue
            close = h['Close']
            current = float(close.iloc[-1])
            ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0.0
            ret_1w = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0.0
            ret_1m = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) >= 22 else 0.0
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
            results[label] = {
                'ticker': ticker, 'current': current,
                'ret_1d': ret_1d, 'ret_1w': ret_1w, 'ret_1m': ret_1m,
                'sma_50': sma_50,
                'trend': 'bullish' if pd.notna(sma_50) and current > sma_50 else 'bearish',
                'history': close,
            }
        except Exception:
            continue
    return results

def build_macro_prompt(macro_data):
    lines = []
    for label, d in macro_data.items():
        lines.append(f"- {label}: {d['current']:.2f} | 1d {d['ret_1d']:+.2f}% | "
                     f"1w {d['ret_1w']:+.2f}% | 1m {d['ret_1m']:+.2f}% | "
                     f"trend vs SMA50: {d['trend']}")
    return "\n".join(lines)

def run_ai_macro_analysis(api_key, macro_data, extra_context=""):
    if not OPENAI_AVAILABLE:
        return ("❌ **Library `openai` tidak terinstall.**\n\n"
                "Jalankan: `pip install openai` lalu restart Streamlit.")
    if not api_key:
        return "🔑 Masukkan OpenAI API key di sidebar."
    try:
        client = openai.OpenAI(api_key=api_key)
        data_str = build_macro_prompt(macro_data)
        prompt = f"""You are a senior macro strategist advising a retail investor who runs a global equity + crypto screener.

CURRENT MARKET DATA (latest close):
{data_str}

{f"USER CONTEXT: {extra_context}" if extra_context else ""}

Provide CONCISE analysis in this exact structure (markdown):

## 🌡️ Regime Assessment
1-2 sentences: risk-on or risk-off? Key driver.

## 🚨 Top 3 Risks
- Specific, data-driven bullets

## 🎯 Top 3 Opportunities
- Specific sectors/assets bullets

## 📊 Asset Allocation Guidance
- Equities: X% | Bonds/Cash: X% | Crypto: X% | Commodities: X% (must sum to 100%)

## 🧭 Actionable Plays
- 3-5 concrete bullets (sectors, tickers, positioning)

## ⚡ Crypto-Specific View
- 1-2 sentences on BTC/ETH regime

Reference actual numbers. Avoid generic statements."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error: `{e}`"


# ============================================================
# [NEW v5.0] PORTFOLIO MODULE
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_portfolio_prices(tickers_tuple, period='1y'):
    try:
        data = yf.download(list(tickers_tuple), period=period, progress=False,
                          auto_adjust=True, threads=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.dropna(how='all')
    except Exception:
        return None

def compute_portfolio_stats(prices, weights, rf=0.04):
    if prices is None or prices.empty or len(prices) < 30:
        return {}
    returns = prices.pct_change().dropna()
    if returns.empty:
        return {}
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    mean_daily = returns.mean()
    cov_daily = returns.cov()
    port_ret = float((mean_daily.values * w).sum() * 252)
    port_vol = float(np.sqrt(w @ (cov_daily.values * 252) @ w))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    port_returns_daily = (returns * w).sum(axis=1)
    cum = (1 + port_returns_daily).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = float(dd.min())
    downside = port_returns_daily[port_returns_daily < 0].std() * np.sqrt(252)
    sortino = (port_ret - rf) / downside if downside > 0 else np.nan
    calmar = port_ret / abs(max_dd) if max_dd < 0 else np.nan
    rolling_sharpe = (
        port_returns_daily.rolling(30).mean() / port_returns_daily.rolling(30).std()
    ) * np.sqrt(252)
    return {
        'annual_return': port_ret, 'annual_vol': port_vol,
        'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': max_dd, 'calmar': calmar,
        'equity_curve': cum, 'drawdown': dd,
        'daily_returns': port_returns_daily,
        'rolling_sharpe': rolling_sharpe,
        'weights': dict(zip(prices.columns, w)),
    }

def optimize_max_sharpe(prices, rf=0.04):
    if not SCIPY_AVAILABLE or prices is None or prices.empty:
        return None
    returns = prices.pct_change().dropna()
    if returns.empty or returns.shape[1] < 2:
        return None
    n = returns.shape[1]
    mean_ret = returns.mean().values * 252
    cov = returns.cov().values * 252
    def neg_sharpe(w):
        r = w @ mean_ret
        s = np.sqrt(w @ cov @ w)
        return -(r - rf) / s if s > 0 else 1e6
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    try:
        res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 500})
        if res.success:
            return dict(zip(prices.columns, res.x))
    except Exception:
        pass
    return None


# ============================================================
# [NEW v5.0] AI FUND MODULE
# ============================================================
def init_fund_holdings(fund, raw_df, top_n, strategy_label, capital, benchmark):
    picks = raw_df.head(top_n).copy()
    if picks.empty:
        return False
    if strategy_label == "Equal Weight":
        weights = np.ones(len(picks)) / len(picks)
    elif strategy_label == "Score Weighted":
        weights = (picks['composite_score'] / picks['composite_score'].sum()).values
    else:  # Inverse Vol
        inv_vol = 1.0 / picks['volatility'].fillna(picks['volatility'].median())
        weights = (inv_vol / inv_vol.sum()).values
    fund['holdings'] = {}
    for i, (_, row) in enumerate(picks.iterrows()):
        alloc = capital * weights[i]
        shares = alloc / row['price']
        fund['holdings'][row['ticker']] = {
            'shares': float(shares),
            'entry_price': float(row['price']),
            'allocation': float(alloc),
            'entry_date': pd.Timestamp.now().normalize(),
        }
    fund['cash'] = 0.0
    fund['initial_capital'] = float(capital)
    fund['start_date'] = pd.Timestamp.now().normalize()
    fund['benchmark'] = benchmark
    fund['strategy_label'] = strategy_label
    fund['top_n'] = top_n
    return True

def compute_fund_equity_curve(fund):
    if not fund.get('holdings'):
        return None, None
    start = pd.Timestamp(fund['start_date']) - pd.Timedelta(days=2)
    tickers = list(fund['holdings'].keys()) + [fund['benchmark']]
    try:
        hist = yf.download(tickers, start=start, progress=False,
                          auto_adjust=True, threads=True)['Close']
        if isinstance(hist, pd.Series):
            hist = hist.to_frame()
    except Exception:
        return None, None
    if hist.empty:
        return None, None
    port = pd.Series(0.0, index=hist.index)
    for t, h in fund['holdings'].items():
        if t in hist.columns:
            port += hist[t].fillna(h['entry_price']) * h['shares']
    port += fund['cash']
    if fund['benchmark'] in hist.columns:
        bench_series = hist[fund['benchmark']].dropna()
        if not bench_series.empty:
            bench = bench_series / bench_series.iloc[0] * fund['initial_capital']
        else:
            bench = None
    else:
        bench = None
    return port, bench

def run_ai_fund_review(api_key, fund, macro_summary=""):
    if not OPENAI_AVAILABLE:
        return "❌ Install `openai` dulu: `pip install openai`"
    if not api_key:
        return "🔑 Masukkan OpenAI API key di sidebar."
    try:
        client = openai.OpenAI(api_key=api_key)
        holdings_str = "\n".join(
            f"- {t}: {h['shares']:.2f} shares @ ${h['entry_price']:.2f} "
            f"(alloc ${h['allocation']:.0f})"
            for t, h in fund['holdings'].items()
        )
        prompt = f"""You are an AI portfolio manager reviewing a small paper-trading fund.

FUND STATE:
- Initial capital: ${fund['initial_capital']:.0f}
- Current value: ${fund.get('current_value', fund['initial_capital']):.0f}
- PnL: {((fund.get('current_value', fund['initial_capital'])/fund['initial_capital'])-1)*100:+.2f}%
- Strategy: {fund.get('strategy_label', 'Equal Weight')}
- Benchmark: {fund.get('benchmark', 'SPY')}

HOLDINGS:
{holdings_str}

{f"MACRO CONTEXT:{chr(10)}{macro_summary}" if macro_summary else ""}

Provide a concise review (markdown, max 250 words):

## 🎯 Verdict
One of: **HOLD** / **REBALANCE** / **EXIT** (bold, first line)

## ✅ What's Working
- 2-3 bullets

## ⚠️ What to Watch
- 2-3 bullets

## 🔄 Suggested Actions
- 2-3 concrete bullets (add/trim/exit which ticker, why)

Be specific and data-driven."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error: `{e}`"


# ============================================================
# [NEW v5.0] UI HELPERS
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #eee; }
    .stTabs [data-baseweb="tab"] {
        height: 46px; padding: 0 18px; background-color: #f7f9fc;
        border-radius: 8px 8px 0 0; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f77b4 0%, #2a9d8f 100%);
        color: white !important;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #666; }
    .stButton > button { border-radius: 8px; font-weight: 500; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stDataFrame { border-radius: 8px; }
    div[data-testid="stExpander"] { border-radius: 8px; border: 1px solid #eee; }
    .stAlert { border-radius: 8px; }
    h1 { background: linear-gradient(90deg, #1f77b4, #2a9d8f);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 800; }
    .stCaption { color: #666; }
    </style>
    """, unsafe_allow_html=True)


def render_stock_screener(settings):
    cfg = settings['cfg']
    code = settings['code']
    is_idx = settings['is_idx']
    strategy = settings['strategy']
    weights = settings['weights']
    preset = settings['preset']
    use_full = settings['use_full']
    size = settings['size']
    top_n = settings['top_n']

    if st.button("🚀 Fetch data & Run", type="primary", key="run_screener"):
        all_tickers = get_sp500_tickers() if code == 'US' else get_idx_tickers()
        tickers = sample_universe(all_tickers, size, use_full)
        st.info(f"Fetch **{len(tickers)}** dari {len(all_tickers)} ticker {cfg['universe_label']} (batch mode, cache {CACHE_TTL//3600} jam).")
        bar = st.progress(0, text="Mulai...")
        def cb(c, t, tk):
            bar.progress(min(c / max(t, 1), 1.0), text=f"[{c}/{t}] {tk}")
        raw_df, counts, failed = fetch_universe(tickers, cfg['suffix'], cfg['fetch_min_adv'], cb)
        bar.empty()
        st.session_state['raw'] = {
            'code': code, 'df': raw_df, 'counts': counts, 'failed': failed,
            'n_requested': len(tickers), 'n_universe': len(all_tickers)
        }

    raw = st.session_state.get('raw')
    if not raw or raw['code'] != code:
        st.info("Klik **🚀 Fetch data & Run** dulu. Setelah data masuk, ubah slider di sidebar → hasil auto-update.")
        return

    raw_df = raw['df']
    if raw_df.empty:
        st.error(f"Tidak ada data. Status: {raw['counts']}. Coba lagi (kemungkinan rate limit Yahoo).")
        return

    pos_value = settings['pos_value_u'] * cfg['pos_mult']
    params = {
        'mcap_min': settings['mcap_min'] * cfg['mcap_mult'],
        'mcap_max': settings['mcap_max'] * cfg['mcap_mult'],
        'min_price': settings['min_price'], 'pos_value': pos_value,
        'min_adv_required': pos_value / (PARTICIPATION * max(settings['max_exit_days'], 1)),
        'max_zero_vol': settings['max_zero_vol'] / 100.0,
        'min_completeness': float(settings['min_completeness']),
        'max_pb': settings['max_pb'], 'drop_negative_book': is_idx,
        'max_vol': settings['max_vol'], 'min_float_pct': float(settings['min_float']),
        'max_runup': settings['max_runup'],
        'require_profit': settings['require_profit'], 'knife': preset['knife'],
    }
    df, funnel, eff_w = screen(raw_df, params, strategy, weights)

    st.success(f"✅ **{len(df)} saham** lolos semua filter (dari {len(raw_df)} berhasil di-fetch).")
    with st.expander("ℹ️ Funnel: kenapa jumlahnya segini"):
        st.write(f"- Diminta: {raw['n_requested']} | Data OK: {len(raw_df)}")
        st.write(f"- Status fetch: {raw['counts']}")
        for label, n in funnel:
            st.write(f"- {label}: **-{n}**")
        st.write(f"- **Lolos: {len(df)}**")
        if eff_w:
            st.write("- Bobot efektif: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in eff_w.items() if v > 0))
    if raw['failed']:
        with st.expander("⚠️ Ticker gagal / dibuang"):
            for status, lst in raw['failed'].items():
                st.write(f"**{status}** ({len(lst)}): " + ", ".join(lst[:80]) + (" ..." if len(lst) > 80 else ""))

    if df.empty:
        st.warning("Tidak ada yang lolos. Longgarkan filter di sidebar.")
        return

    st.subheader(f"🏆 Top {top_n} — {strategy}")
    cur, mdiv, mu = cfg['currency'], cfg['mcap_mult'], cfg['mcap_unit']
    t = df.head(top_n)
    out = pd.DataFrame({
        'Rank': t['rank'], 'Ticker': t['ticker'], 'Nama': t['name'], 'Sektor': t['sector'],
        'Harga': t['price'].apply(lambda x: f"{cur}{x:,.2f}"),
        'Mkt Cap': t['market_cap'].apply(lambda x: f"{cur}{x/mdiv:.2f}{mu}" if pd.notna(x) else "—"),
        'Rating': t['rating'].round(1),
        'P/E': t['trailing_pe'].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "—"),
        'PBV': t['pb'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        'Earn Yld': t['earnings_yield'].apply(lambda x: fmt_pct(x, 100)),
        'Div Yld': t['div_yield'].apply(lambda x: fmt_pct(x, 100, 2)),
        'ROE': t['roe'].apply(lambda x: fmt_pct(x, 100)),
        'Ret 12M': t['return_12m'].apply(lambda x: fmt_pct(x)),
        'Ret 6M': t['return_6m'].apply(lambda x: fmt_pct(x)),
        'ADV': t['adv_median'].apply(lambda x: f"{x/cfg['adv_mult']:.2f} {cfg['adv_unit']}"),
        'Hari Exit': t['days_to_exit'].round(1),
        'Analis': t['analysts_n'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "0"),
        'Value': t['value_score'].round(0), 'Quality': t['quality_score'].round(0),
        'Neglect': t['neglect_score'].round(0), 'Momentum': t['momentum_score_final'].round(0),
        'Flag': t['flag'],
    })
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.caption("Skor 0-100 = persentil dalam sektor. Flag ⚠️/💧/🔸/🚀 = cek manual. Ini KANDIDAT, bukan sinyal beli.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏭 Sector Distribution")
        st.bar_chart(t['sector'].value_counts())
    with c2:
        st.subheader("📊 Factor Breakdown")
        st.bar_chart(t.set_index('ticker')[['value_score', 'quality_score', 'neglect_score',
                                            'momentum_score_final', 'low_vol_score_final']])
    st.download_button("📥 Download CSV (semua hasil)", df.to_csv(index=False).encode('utf-8'),
                       "screener_results.csv", "text/csv")


def render_crypto_screener():
    st.subheader("🪙 Crypto Screener")
    st.caption(f"Live data (cache 60s) · {len(CRYPTO_UNIVERSE)} top coins · "
               "Scoring: momentum (35%) + trend (20%) + low-vol (15%) + liquidity (15%) + RSI (15%)")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Refresh (live)", type="primary", key="crypto_refresh"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        top_n = st.number_input("Top N", 5, len(CRYPTO_UNIVERSE), 15, key="crypto_topn")
    with c3:
        st.caption(f"⏱️ Last fetch: {pd.Timestamp.now().strftime('%H:%M:%S')} (WIB)")

    with st.spinner("Fetch crypto histories..."):
        raw = fetch_crypto_histories(tuple(CRYPTO_UNIVERSE.keys()))
    if raw is None or raw.empty:
        st.error("Gagal fetch crypto data. Coba refresh.")
        return

    df = score_crypto(raw)
    if df.empty:
        st.warning("Tidak ada crypto yang berhasil di-score.")
        return

    top = df.head(int(top_n))
    st.subheader(f"🏆 Top {top_n} Crypto")
    display = pd.DataFrame({
        'Rank': top['rank'], 'Ticker': top['ticker'], 'Name': top['name'],
        'Price': top['price'].apply(lambda x: f"${x:,.4f}" if x < 1 else f"${x:,.2f}"),
        '1D %': top['ret_1d'].apply(lambda x: fmt_pct(x)),
        '7D %': top['ret_7d'].apply(lambda x: fmt_pct(x)),
        '30D %': top['ret_30d'].apply(lambda x: fmt_pct(x)),
        '90D %': top['ret_90d'].apply(lambda x: fmt_pct(x)),
        'Vol 30D': top['vol_30d'].apply(lambda x: fmt_pct(x, 1)),
        'RSI': top['rsi'].round(1),
        'SMA20': top['above_sma20'].map({True: '✅', False: '❌'}),
        'SMA50': top['above_sma50'].map({True: '✅', False: '❌'}),
        'Score': top['composite'].round(1),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Price chart for top 5
    st.subheader("📈 Top 5 — Price (90D, normalized)")
    top5 = df.head(5)['ticker'].tolist()
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for tk in top5:
            try:
                series = raw[tk]['Close'].dropna().tail(90)
                if not series.empty:
                    norm = series / series.iloc[0] * 100
                    fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=tk, mode='lines'))
            except Exception:
                continue
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation='h', y=1.1),
                          yaxis_title="Indexed (100 = 90d ago)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        norm_data = pd.DataFrame({
            tk: (raw[tk]['Close'].dropna().tail(90) / raw[tk]['Close'].dropna().tail(90).iloc[0] * 100)
            for tk in top5
        })
        st.line_chart(norm_data)

    # Volume
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 30D Return")
        st.bar_chart(df.set_index('ticker')['ret_30d'].head(15))
    with c2:
        st.subheader("📊 Volatility (30D annualized)")
        st.bar_chart(df.set_index('ticker')['vol_30d'].head(15))


def render_macro_ai(openai_key):
    st.subheader("🌍 Macro Overview + AI Analysis")
    st.caption("Data: yfinance (10 indikator) · AI: OpenAI gpt-4o-mini (opsional)")

    with st.spinner("Fetch macro data..."):
        macro = fetch_macro_data()
    if not macro:
        st.error("Gagal fetch macro data.")
        return

    # Metric cards
    cols = st.columns(5)
    items = list(macro.items())
    for i, (label, d) in enumerate(items):
        with cols[i % 5]:
            delta_color = "normal" if d['ret_1d'] >= 0 else "inverse"
            st.metric(label=label, value=f"{d['current']:.2f}",
                      delta=f"{d['ret_1d']:+.2f}% (1d)")

    st.divider()

    # Snapshot table
    with st.expander("📋 Full Snapshot", expanded=False):
        snap = pd.DataFrame([{
            'Indicator': label, 'Ticker': d['ticker'],
            'Current': f"{d['current']:.2f}",
            '1D %': f"{d['ret_1d']:+.2f}",
            '1W %': f"{d['ret_1w']:+.2f}",
            '1M %': f"{d['ret_1m']:+.2f}",
            'Trend vs SMA50': d['trend'],
        } for label, d in items])
        st.dataframe(snap, use_container_width=True, hide_index=True)

    # Charts
    st.subheader("📈 Price Charts (6M, normalized)")
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for label, d in items:
            series = d['history'].dropna()
            if not series.empty:
                norm = series / series.iloc[0] * 100
                fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=label, mode='lines'))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation='h', y=1.1),
                          yaxis_title="Indexed (100 = 6mo ago)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame({label: d['history'] for label, d in items}))

    # AI Analysis
    st.divider()
    st.subheader("🤖 AI Macro Analysis")
    extra_ctx = st.text_area("Context / pertanyaan tambahan (opsional):",
                             placeholder="Misal: 'Saya 70% US equity, 20% crypto, 10% cash. Perlu rebalance?'",
                             key="macro_context")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("🧠 Run AI Analysis", type="primary", key="run_macro_ai")
    with col2:
        if not OPENAI_AVAILABLE:
            st.warning("⚠️ Library `openai` belum terinstall. `pip install openai`")
        elif not openai_key:
            st.info("🔑 Masukkan OpenAI API key di sidebar untuk mengaktifkan AI.")

    if run_btn:
        if not openai_key:
            st.error("Butuh OpenAI API key.")
        else:
            with st.spinner("AI menganalisis macro..."):
                result = run_ai_macro_analysis(openai_key, macro, extra_ctx)
            st.markdown(result)
            st.session_state['macro_ai_result'] = result
    elif 'macro_ai_result' in st.session_state:
        st.markdown(st.session_state['macro_ai_result'])


def render_portfolio(openai_key, code):
    st.subheader("💼 Portfolio Builder + Sharpe Ratio")
    st.caption("Analisis risk-adjusted return dari portofolio custom.")

    raw = st.session_state.get('raw')
    screener_tickers = []
    if raw and not raw['df'].empty:
        screener_tickers = raw['df']['ticker'].tolist()

    # Init default SEBELUM widget dibuat
    if "port_tickers" not in st.session_state:
        st.session_state["port_tickers"] = (
            ", ".join(screener_tickers[:5]) if screener_tickers
            else "AAPL, MSFT, GOOGL, AMZN, NVDA"
        )

    # ⚠️ IMPORTANT: tombol HARUS dirender SEBELUM text_input
    c1, c2 = st.columns([3, 1])
    with c2:
        if screener_tickers and st.button("📥 Import Top-10", key="port_import"):
            # aman: text_input belum dibuat di run ini
            st.session_state["port_tickers"] = ", ".join(screener_tickers[:10])
    with c1:
        tickers_input = st.text_input("Tickers (comma-separated)", key="port_tickers")

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.warning("Minimal 2 ticker.")
        return

    with st.spinner(f"Fetch prices untuk {len(tickers)} ticker..."):
        prices = fetch_portfolio_prices(tuple(tickers))
    if prices is None or prices.empty:
        st.error("Gagal fetch harga.")
        return

    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]
    if len(available) < 2:
        st.error(f"Hanya {len(available)} ticker valid.")
        return

    st.subheader("⚖️ Bobot")
    weight_mode = st.radio("Metode bobot:",
                          ["Equal Weight", "Custom", "Max Sharpe (optimizer)"],
                          horizontal=True, key="weight_mode")

    n = len(available)
    weights = None
    if weight_mode == "Equal Weight":
        weights = np.ones(n) / n
    elif weight_mode == "Custom":
        cols = st.columns(min(n, 5))
        w_list = []
        for i, t in enumerate(available):
            with cols[i % min(n, 5)]:
                w = st.number_input(t, 0.0, 100.0, float(100.0/n), step=1.0, key=f"w_{t}")
                w_list.append(w)
        w_arr = np.array(w_list)
        weights = w_arr / w_arr.sum() if w_arr.sum() > 0 else np.ones(n) / n
    else:  # Max Sharpe
        if not SCIPY_AVAILABLE:
            st.warning("scipy belum terinstall. Pakai equal weight.")
            weights = np.ones(n) / n
        else:
            opt = optimize_max_sharpe(prices)
            if opt:
                weights = np.array([opt.get(t, 0) for t in available])
                st.success("✅ Max-Sharpe optimizer berhasil.")
            else:
                weights = np.ones(n) / n
                st.warning("Optimizer gagal, pakai equal weight.")

    if weights is not None:
        st.write("**Bobot final:** " + ", ".join(
            f"{t} {w*100:.1f}%" for t, w in zip(available, weights)))

    rf = st.slider("Risk-free rate (annual %)", 0.0, 15.0, 4.0, 0.1, key="rf") / 100
    stats = compute_portfolio_stats(prices, weights, rf)
    if not stats:
        st.error("Gagal hitung statistik.")
        return

    st.subheader("📊 Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Annual Return", f"{stats['annual_return']*100:.2f}%")
    m2.metric("Annual Vol", f"{stats['annual_vol']*100:.2f}%")
    m3.metric("Sharpe", f"{stats['sharpe']:.2f}")
    m4.metric("Sortino", f"{stats['sortino']:.2f}" if pd.notna(stats['sortino']) else "—")
    m5.metric("Max Drawdown", f"{stats['max_drawdown']*100:.2f}%")

    m6, m7 = st.columns(2)
    m6.metric("Calmar", f"{stats['calmar']:.2f}" if pd.notna(stats['calmar']) else "—")
    m7.metric("Risk-Free", f"{rf*100:.1f}%")

    st.divider()

    st.subheader("📈 Equity Curve ($10,000 initial)")
    equity = stats['equity_curve'] * 10000
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                 name='Portfolio', line=dict(color='#1f77b4', width=2)))
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(equity)

    st.subheader("📉 Drawdown")
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stats['drawdown'].index,
                                 y=stats['drawdown'].values * 100,
                                 name='Drawdown', fill='tozeroy',
                                 line=dict(color='crimson')))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="Drawdown (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.area_chart(stats['drawdown'] * 100)

    st.subheader("📊 Rolling 30D Sharpe")
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stats['rolling_sharpe'].index,
                                 y=stats['rolling_sharpe'].values,
                                 name='Rolling Sharpe', line=dict(color='#2a9d8f')))
        fig.add_hline(y=1, line_dash="dash", line_color="green")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(stats['rolling_sharpe'])

    st.subheader("🔗 Correlation Matrix")
    corr = prices.pct_change().dropna().corr()
    if PLOTLY_AVAILABLE:
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(corr.round(2))


def render_ai_fund(openai_key, code):
    st.subheader("🤖 AI Fund — Paper Trading")
    st.caption("Simulasi portofolio yang dikelola bot dari Top-N screener.")

    raw = st.session_state.get('raw')
    if not raw or raw['df'].empty:
        st.info("⚠️ Jalankan **Stock Screener** (tab 1) dulu untuk punya kandidat.")
        return

    if 'fund' not in st.session_state:
        st.session_state['fund'] = {}
    fund = st.session_state['fund']

    if not fund.get('holdings'):
        st.info("🚀 Fund belum diinisialisasi. Setup dulu:")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            capital = st.number_input("Capital ($)", 1000.0, 100_000_000.0,
                                       100_000.0, 1000.0, key="fund_cap")
        with c2:
            top_n = st.number_input("Top N picks", 3, 50, 10, key="fund_topn")
        with c3:
            strategy_label = st.selectbox("Weighting:",
                                          ["Equal Weight", "Score Weighted", "Inverse Vol"],
                                          key="fund_wmode")
        with c4:
            benchmark = st.selectbox("Benchmark:",
                                     ["^GSPC", "^JKSE", "SPY", "QQQ"],
                                     key="fund_bench")
        if st.button("🚀 Launch AI Fund", type="primary", key="fund_init"):
            ok = init_fund_holdings(fund, raw['df'], int(top_n), strategy_label,
                                    float(capital), benchmark)
            if ok:
                st.success(f"✅ Fund launched: {top_n} saham, ${capital:,.0f}")
                st.rerun()
            else:
                st.error("Gagal inisialisasi fund.")
        return

    st.subheader(f"📊 AI Fund · {fund['strategy_label']} · Top {fund['top_n']}")
    with st.spinner("Menghitung equity curve..."):
        port, bench = compute_fund_equity_curve(fund)

    if port is not None and not port.empty:
        fund['current_value'] = float(port.iloc[-1])
        initial = fund['initial_capital']
        pnl = (fund['current_value'] / initial - 1) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Value", f"${fund['current_value']:,.2f}", f"{pnl:+.2f}%")
        m2.metric("Initial Capital", f"${initial:,.2f}")
        m3.metric("Benchmark", fund['benchmark'])

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=port.index, y=port.values, name='AI Fund',
                                     line=dict(color='#1f77b4', width=2)))
            if bench is not None and not bench.empty:
                fig.add_trace(go.Scatter(x=bench.index, y=bench.values,
                                         name='Benchmark',
                                         line=dict(color='orange', width=2, dash='dash')))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Value ($)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            compare = pd.DataFrame({'AI Fund': port})
            if bench is not None:
                compare['Benchmark'] = bench
            st.line_chart(compare)

    st.subheader("💼 Holdings")
    holdings_rows = []
    for t, h in fund['holdings'].items():
        try:
            current_price = float(yf.Ticker(t).history(period='1d')['Close'].iloc[-1])
        except Exception:
            current_price = h['entry_price']
        current_val = current_price * h['shares']
        pnl_pct = (current_price / h['entry_price'] - 1) * 100
        holdings_rows.append({
            'Ticker': t, 'Shares': f"{h['shares']:.2f}",
            'Entry': f"${h['entry_price']:.2f}",
            'Current': f"${current_price:.2f}",
            'Value': f"${current_val:,.0f}",
            'PnL %': f"{pnl_pct:+.2f}%",
        })
    st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🤖 AI Fund Review")
    macro_summary = st.session_state.get('macro_ai_result', '')[:1500]

    if st.button("🧠 Run AI Review", type="primary", key="fund_review"):
        if not openai_key:
            st.error("Butuh OpenAI API key.")
        else:
            with st.spinner("AI me-review fund..."):
                review = run_ai_fund_review(openai_key, fund, macro_summary)
            st.markdown(review)
            st.session_state['fund_ai_review'] = review
    elif 'fund_ai_review' in st.session_state:
        st.markdown(st.session_state['fund_ai_review'])

    if st.button("🗑️ Reset Fund", key="fund_reset"):
        st.session_state['fund'] = {}
        st.session_state.pop('fund_ai_review', None)
        st.rerun()


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("🔑 AI")
        openai_key = st.text_input("OpenAI API Key", type="password",
                                    key="openai_key",
                                    help="Untuk Macro AI & AI Fund review")
        if openai_key:
            st.success("✅ Key set")
        elif not OPENAI_AVAILABLE:
            st.warning("Install `openai` untuk aktifkan AI")

        st.divider()
        st.subheader("📈 Market")
        market = st.selectbox("Market:", list(MARKETS.keys()), key="market_select")
        cfg = MARKETS[market]
        code = cfg['code']
        is_idx = code == 'IDX'

        strategy = st.selectbox("Strategy:", list(STRATEGY_WEIGHTS.keys()),
                                index=len(STRATEGY_WEIGHTS) - 1 if is_idx else 0,
                                key="strategy_select")
        weights = STRATEGY_WEIGHTS[strategy]
        preset = get_preset(strategy, code)

        use_full = st.checkbox(f"Fetch SEMUA {cfg['universe_label']}",
                                value=cfg['default_full'], key=f"full_{code}")
        size = st.slider("Universe size:", 30, cfg['universe_max'],
                         cfg['default_size'], 10, disabled=use_full, key=f"size_{code}")
        top_n = st.slider("Show Top N:", 5, 50, 20, key="topn_slider")

        st.divider()
        st.subheader("🎛️ Filter (instan)")
        k = f"{code}|{strategy}"
        unit = cfg['mcap_unit']
        c1, c2 = st.columns(2)
        mcap_min = c1.number_input(f"Mcap min ({unit})", 0.0,
                                    float(preset['mcap_min']), 0.1, key=f"mmin|{k}")
        mcap_max = c2.number_input("Mcap max (0=∞)", 0.0,
                                    float(preset['mcap_max']), 0.5, key=f"mmax|{k}")
        min_price = st.number_input("Harga min", 0.0, float(cfg['min_price']),
                                    10.0 if is_idx else 1.0, key=f"mp|{k}")
        pos_value_u = st.number_input(cfg['pos_label'], 1.0,
                                       float(cfg['pos_default']), 10.0, key=f"pos|{code}")
        max_exit_days = st.slider("Maks. hari exit", 1, 10, 3, key=f"exit|{code}")
        max_zero_vol = st.slider("Maks. hari tanpa transaksi (%)", 0, 100, 25, key=f"zv|{code}")
        min_completeness = st.slider("Min. kelengkapan data (%)", 0, 100, 50, key=f"mc|{code}")
        max_pb = st.number_input("PBV maksimum", 0.5, float(cfg['max_pb']), 0.5, key=f"pb|{code}")
        max_vol = st.number_input("Volatilitas max (%)", 20.0,
                                    float(cfg['max_vol']), 10.0, key=f"vol|{code}")
        min_float = st.slider("Min. free float (%)", 0, 50,
                               10 if is_idx else 0, key=f"fl|{code}")
        max_runup = st.number_input("Maks. kenaikan 12M (%)", 0.0,
                                     float(preset['max_runup']), 10.0, key=f"ru|{k}")
        require_profit = st.checkbox("Wajib laba & OCF positif",
                                      value=preset['require_profit'], key=f"rp|{k}")

        st.divider()
        st.caption("**Bobot strategi:** " + " | ".join(
            f"{n.split()[0]} {w*100:.0f}%" for n, w in weights.items() if w > 0))

        if st.button("🗑️ Clear Cache", key="clear_cache"):
            st.cache_data.clear()
            st.session_state.pop('raw', None)
            st.success("Cache cleared.")
            st.rerun()

    return {
        'openai_key': openai_key,
        'market': market, 'cfg': cfg, 'code': code, 'is_idx': is_idx,
        'strategy': strategy, 'weights': weights, 'preset': preset,
        'use_full': use_full, 'size': size, 'top_n': top_n,
        'mcap_min': mcap_min, 'mcap_max': mcap_max, 'min_price': min_price,
        'pos_value_u': pos_value_u, 'max_exit_days': max_exit_days,
        'max_zero_vol': max_zero_vol, 'min_completeness': min_completeness,
        'max_pb': max_pb, 'max_vol': max_vol, 'min_float': min_float,
        'max_runup': max_runup, 'require_profit': require_profit,
    }


def main():
    st.set_page_config(page_title="Global Stock Screener Pro v5.0",
                       page_icon="📊", layout="wide")
    inject_custom_css()

    st.title("📊 Global Stock Screener Pro v5.0")
    st.caption(f"⚡ {MAX_WORKERS} threads | 💾 Cache {CACHE_TTL//3600}j | "
               f"🌏 US + IDX | 🪙 Crypto | 🤖 AI Macro")

    settings = render_sidebar()

    tabs = st.tabs(["📈 Stock Screener", "🪙 Crypto", "🌍 Macro AI",
                    "💼 Portfolio", "🤖 AI Fund"])
    with tabs[0]:
        render_stock_screener(settings)
    with tabs[1]:
        render_crypto_screener()
    with tabs[2]:
        render_macro_ai(settings['openai_key'])
    with tabs[3]:
        render_portfolio(settings['openai_key'], settings['code'])
    with tabs[4]:
        render_ai_fund(settings['openai_key'], settings['code'])


if __name__ == "__main__":
    main()