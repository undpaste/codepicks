"""
Global Stock Screener Pro v4.1
===============================
Perubahan v4.1 vs v4:
 [FIX-RATE] Fetch jadi 2-phase: batch download semua history (1 API call),
            lalu fetch .info HANYA untuk kandidat yang lolos pre-filter likuiditas.
            Dari ~900 request jadi ~50-80 request per screening → bebas rate limit.
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

# ============================================================
# KONFIGURASI
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
    },
    "🇮🇩 Indonesia (IDX)": {
        'code': 'IDX', 'suffix': '.JK', 'currency': 'Rp ',
        'mcap_mult': 1e12, 'mcap_unit': 'T',
        'pos_label': 'Target posisi per saham (Rp juta)', 'pos_mult': 1e6, 'pos_default': 50.0,
        'adv_mult': 1e9, 'adv_unit': 'Rp B', 'fetch_min_adv': 1e8,
        'universe_label': 'IDX', 'universe_max': 950, 'default_size': 300, 'default_full': True,
        'min_price': 100.0, 'max_pb': 10.0, 'max_vol': 120.0,
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
    raw = [
        'BBCA','BBRI','BMRI','BBNI','BRIS','BTPS','ARTO','BBTN','BJBR','BJTM',
        'BNGA','BNLI','PNBN','MEGA','NISP','BFIN','ADMF','PNLF','TUGU','ASBI',
        'LPGI','MCOR','BABP','AGRO','BNII','BBHI','BCIC','AMAR','MFIN','WOMF',
        'BNBA','BBKP','BKSW','BMAS','BSIM','BTPN','BVIC','INPC','MAYA',
        'NOBU','PNBS','BCAP','BHIT','BPFI','CFIN','HDFA','IMJS','JMAS',
        'KREN','PADI','PNIN','VRNA','APIC','BBYB','BDMN',
        'ANTM','INCO','TINS','SMGR','INTP','BRPT','TPIA','INKP','TKIM','MDKA',
        'NCKL','MBMA','NIKL','PSAB','IFSH','KRAS','ISSP','BAJA','JKSW','LION',
        'ALKA','ALMI','ANJT','APLI','ARNA','BMSR','BRMS','BTON','CTBN','DPNS',
        'EKAD','ESSA','GDST','GGRP','HKMU','IGAR','INAI','INDX','INTD','ITMA',
        'JSPT','KBLI','KDSI','KIAS','LMSH','LMPI','MARI','PICO',
        'POLY','PRAS','SMBR','SMKL','SPMA','SRIL','SSIA','SULI','TBMS','TIRT',
        'TRST','YPAS','ZBRA',
        'ADRO','PTBA','ITMG','MEDC','PGAS','HRUM','AKRA','ELSA','BUMI','DOID',
        'HRTA','TOBA','PTRO','KKGI','MYOH','DEWA','TGRA','AADI','APEX','ARTI',
        'BIPI','BSSR','BYAN','CNKO','DWGL','ENRG','FIRE','GEMS','GTBO','HITS',
        'INDY','JATI','MBAP','PKPK','RMKE','SGER','SHIP',
        'SMMT','SOCI','SUGI','TCPI','UNSP','WINS',
        'UNVR','ICBP','INDF','MYOR','SIDO','AMRT','CPIN','JPFA','MAIN','HMSP',
        'GGRM','WIIM','DMND','CAMP','ULTJ','STTP','TBLA','AISA','DLTA','MLBI',
        'INDR','KEJU','CEKA','GOOD','PSDN','SKBM','ADES','BTEK','CINT',
        'FOOD','HOKI','IIKP','IPPE','MGNA','MRAT','PANI','PCAR','ROTI',
        'SKLT','SMAR','TAST','TRGU','ULTR','WAPO','ALTO','BUDI',
        'ASII','AUTO','SMSM','MAPI','ACES','ERAA','LPPF','RALS','SCCO','MAPA',
        'CSAP','RANC','DIGI','FAST','RDTX','KIJA','AMFG','ARGO','BCIP',
        'BEBS','BLTA','BOGA','BRAM','CNTX','DIVA','DUCK','GDYR',
        'HOTL','HRME','IMAS','INDS','LPIN','MAMI',
        'MDIA','MINA','MPMX','MSKY','MYTX','NIPS','PBRX','PDES',
        'PMJS','PSKT','PTSN','RAJA','RICY','RIGS','SSTM','STAR',
        'TELE','TFCO','TRIS','UNIT','VOKS','YOII','ZONE',
        'TLKM','EXCL','ISAT','TOWR','MTEL','TBIG','CENT','MNCN','SCMA','FILM',
        'BMTR','IPTV','NETV','KBLV',
        'GOTO','BUKA','EMTK','DMMX','MTDL','WIFI','AWAN','MLPT','TECH','LMAS',
        'ATIC','CYBR','KETR','LUCK','NFCX','SIMS','TOSK','WGSH',
        'MIKA','SILO','HEAL','PRDA','KAEF','INAF','SAME','MTCN','DVLA','TSPC',
        'PYFA','PEHA','SRAJ','RSGK','BIMA','CARE','DGNS','IRRA','MERC',
        'PRIM','RSCH','SCPI','SOHO',
        'UNTR','HEXA','PTPP','WIKA','ADHI','WSKT','JSMR','IPCM','TMAS','BULL',
        'ASSA','SMDR','PSSI','LEAD','IATA','HATM',
        'APII','CANI','CITA','DPUM','GMFI',
        'ICON','INTA','JAST','JECC','JTPE','KARW','KOBX','KOPI',
        'MARK','MDRN','MFMI','MTLA','NELY','PJAA','PPRE',
        'PTIS','SCNP','TAMU','TIRA','TPMA','TRIM','WEHA',
        'BSDE','CTRA','PWON','SMRA','ASRI','LPKR','DILD','DART','APLN',
        'BEST','MKPI','AGRS','ARMY','BAPI',
        'BKDP','BKSL','CITY','COWL','CPRI','DMAS',
        'DUTI','ELTY','EMDE','FMII','GAMA','GMTD','GPRA','HOMI',
        'KOTA','LAND','LCGP','LPCK','MDLN','MTSM','NIRO',
        'NZIA','OMRE','PLIN','POLI','PUDP','RBMS','REAL',
        'ROCK','RODA','SATU','SCBD','SMDM','TARA','TOTL','TRAM','URBN',
        'CMPP','DEAL','GTRA','HUMI','IKAI','SAFE','TAXI',
        'AALI','LSIP','SGRO','TAPG','DSNG','SSMS','CSRA','BWPT',
        'GZCO','MAGP','PALM','SIMP','SLIS',
        'BNBR','BRNA','FASW','GJTL','POOL',
    ]
    return list(dict.fromkeys(raw))

# ============================================================
# UNIVERSE
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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        tickers, seen = [], set()
        for tbl in tables:
            for col_name in tbl.columns:
                if "Kode" in str(col_name) or "kode" in str(col_name):
                    for raw in tbl[col_name].astype(str):
                        code = raw.split(":")[-1].strip().upper()
                        code = code.replace("[", "").replace("]", "").strip()
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
# HELPERS
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
# [FIX-RATE] FETCH: 2-PHASE BATCH
# ============================================================
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_batch_histories(tickers_tuple, suffix):
    """Download SEMUA histories dalam 1 API call."""
    full = [f"{t}{suffix}" for t in tickers_tuple]
    try:
        return yf.download(full, period='2y', progress=False, auto_adjust=True,
                          threads=True, group_by='ticker', timeout=60)
    except Exception:
        return None

def _process_batch_history(batch, ticker_base, suffix, fetch_min_adv):
    """Ekstrak history dari batch. Return (status, payload)."""
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
    """Fetch .info dan bangun dict final. Dipanggil hanya untuk kandidat likuid."""
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
        if np.isfinite(ev_ebitda):
            ebitda_yield = (1.0 / ev_ebitda) if ev_ebitda > 0 else -1.0
        else:
            ebitda_yield = np.nan
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
    """2-phase: batch history (1 call) -> selective .info untuk kandidat likuid."""
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
# SCORING
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
# FILTER
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

# ============================================================
# UI
# ============================================================
def fmt_pct(x, mult=1.0, nd=1):
    return f"{x*mult:.{nd}f}%" if pd.notna(x) else "—"

def main():
    st.set_page_config(page_title="Global Stock Screener Pro", layout="wide")
    st.title("📊 Global Stock Screener Pro v4.1")
    st.caption(f"⚡ {MAX_WORKERS} threads | 💾 Cache {CACHE_TTL//3600} jam | 🌏 US + Indonesia | "
               f"Batch fetch (1 call untuk semua history) + selective info")

    with st.sidebar:
        st.header("⚙️ Settings")
        market = st.selectbox("Market:", list(MARKETS.keys()))
        cfg = MARKETS[market]
        code = cfg['code']
        is_idx = code == 'IDX'

        strategy = st.selectbox("Strategy:", list(STRATEGY_WEIGHTS.keys()),
                                index=len(STRATEGY_WEIGHTS) - 1 if is_idx else 0)
        weights = STRATEGY_WEIGHTS[strategy]
        preset = get_preset(strategy, code)

        use_full = st.checkbox(f"Fetch SEMUA {cfg['universe_label']}", value=cfg['default_full'], key=f"full_{code}")
        size = st.slider("Universe size (kalau tidak full):", 30, cfg['universe_max'], cfg['default_size'], 10,
                         disabled=use_full, key=f"size_{code}")
        top_n = st.slider("Show Top N:", 5, 50, 20)

        st.divider()
        st.subheader("🎛️ Filter (instan, tanpa fetch ulang)")
        k = f"{code}|{strategy}"
        unit = cfg['mcap_unit']
        c1, c2 = st.columns(2)
        mcap_min = c1.number_input(f"Mcap min ({cfg['currency'].strip()}{unit})", min_value=0.0,
                                   value=float(preset['mcap_min']), step=0.1, key=f"mmin|{k}")
        mcap_max = c2.number_input("Mcap max (0=∞)", min_value=0.0,
                                   value=float(preset['mcap_max']), step=0.5, key=f"mmax|{k}")
        min_price = st.number_input(f"Harga minimum ({cfg['currency'].strip()})", min_value=0.0,
                                    value=float(cfg['min_price']), step=10.0 if is_idx else 1.0, key=f"mp|{k}")
        pos_value_u = st.number_input(cfg['pos_label'], min_value=1.0,
                                      value=float(cfg['pos_default']), step=10.0, key=f"pos|{code}")
        max_exit_days = st.slider("Maks. hari untuk masuk/keluar posisi", 1, 10, 3, key=f"exit|{code}",
                                  help=f"Asumsi kamu maksimal {int(PARTICIPATION*100)}% dari volume harian.")
        max_zero_vol = st.slider("Maks. hari tanpa transaksi (60 hari, %)", 0, 100, 25, key=f"zv|{code}")
        min_completeness = st.slider("Min. kelengkapan data fundamental (%)", 0, 100, 50, key=f"mc|{code}")
        max_pb = st.number_input("PBV maksimum", min_value=0.5, value=float(cfg['max_pb']),
                                 step=0.5, key=f"pb|{code}")
        max_vol = st.number_input("Volatilitas tahunan maks (%)", min_value=20.0,
                                  value=float(cfg['max_vol']), step=10.0, key=f"vol|{code}")
        min_float = st.slider("Min. free float (%)", 0, 50, 10 if is_idx else 0, key=f"fl|{code}")
        max_runup = st.number_input("Maks. kenaikan 12 bulan (%) (0=off)", min_value=0.0,
                                    value=float(preset['max_runup']), step=10.0, key=f"ru|{k}")
        require_profit = st.checkbox("Wajib laba & arus kas operasi positif",
                                     value=preset['require_profit'], key=f"rp|{k}")

        st.divider()
        st.caption("**Bobot strategi:** " + " | ".join(f"{n[:3].title()} {w*100:.0f}%"
                                                       for n, w in weights.items() if w > 0))
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.session_state.pop('raw', None)
            st.success("Cache cleared.")
            st.rerun()

    if st.button("🚀 Fetch data & Run", type="primary"):
        all_tickers = get_sp500_tickers() if code == 'US' else get_idx_tickers()
        tickers = sample_universe(all_tickers, size, use_full)
        st.info(f"Fetch **{len(tickers)}** dari {len(all_tickers)} ticker {cfg['universe_label']} "
                f"(batch mode, hasil di-cache {CACHE_TTL//3600} jam).")
        bar = st.progress(0, text="Mulai...")
        def cb(c, t, tk):
            bar.progress(min(c / max(t, 1), 1.0), text=f"[{c}/{t}] {tk}")
        raw_df, counts, failed = fetch_universe(tickers, cfg['suffix'], cfg['fetch_min_adv'], cb)
        bar.empty()
        st.session_state['raw'] = {'code': code, 'df': raw_df, 'counts': counts, 'failed': failed,
                                   'n_requested': len(tickers), 'n_universe': len(all_tickers)}

    raw = st.session_state.get('raw')
    if not raw or raw['code'] != code:
        st.info("Klik **Fetch data & Run** dulu. Setelah data masuk, ubah slider/filter di sidebar → hasil langsung ter-update.")
        return
    raw_df = raw['df']
    if raw_df.empty:
        st.error(f"Tidak ada data. Status fetch: {raw['counts']}. Coba lagi beberapa menit (kemungkinan rate limit Yahoo).")
        return

    pos_value = pos_value_u * cfg['pos_mult']
    params = {
        'mcap_min': mcap_min * cfg['mcap_mult'], 'mcap_max': mcap_max * cfg['mcap_mult'],
        'min_price': min_price, 'pos_value': pos_value,
        'min_adv_required': pos_value / (PARTICIPATION * max_exit_days),
        'max_zero_vol': max_zero_vol / 100.0, 'min_completeness': float(min_completeness),
        'max_pb': max_pb, 'drop_negative_book': is_idx, 'max_vol': max_vol,
        'min_float_pct': float(min_float), 'max_runup': max_runup,
        'require_profit': require_profit, 'knife': preset['knife'],
    }
    df, funnel, eff_w = screen(raw_df, params, strategy, weights)

    st.success(f"✅ **{len(df)} saham** lolos semua filter (dari {len(raw_df)} yang datanya berhasil di-fetch).")
    with st.expander("ℹ️ Funnel: kenapa jumlahnya segini"):
        st.write(f"- Diminta: {raw['n_requested']} ticker | Data berhasil: {len(raw_df)}")
        st.write(f"- Status fetch: {raw['counts']}")
        for label, n in funnel:
            st.write(f"- {label}: **-{n}**")
        st.write(f"- **Lolos: {len(df)}**")
        if eff_w:
            st.write("- Bobot efektif: " +
                     ", ".join(f"{k} {v*100:.0f}%" for k, v in eff_w.items() if v > 0))
    if raw['failed']:
        with st.expander("⚠️ Ticker yang gagal / dibuang saat fetch"):
            for status, lst in raw['failed'].items():
                st.write(f"**{status}** ({len(lst)}): " + ", ".join(lst[:80]) + (" ..." if len(lst) > 80 else ""))

    if df.empty:
        st.warning("Tidak ada saham yang lolos. Longgarkan filter di sidebar.")
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
    st.caption("Skor 0-100 = persentil dalam sektor. Flag ⚠️/💧/🔸/🚀 = hal yang perlu dicek manual. "
               "Ini daftar KANDIDAT, bukan sinyal beli.")

    st.subheader("🏭 Sector Distribution")
    st.bar_chart(t['sector'].value_counts())
    st.subheader("📊 Factor Breakdown")
    st.bar_chart(t.set_index('ticker')[['value_score', 'quality_score', 'neglect_score',
                                        'momentum_score_final', 'low_vol_score_final']])
    st.download_button("📥 Download CSV (semua hasil)", df.to_csv(index=False).encode('utf-8'),
                       "screener_results.csv", "text/csv")

if __name__ == "__main__":
    main()