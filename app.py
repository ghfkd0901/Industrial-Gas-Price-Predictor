"""
도시가스 도매요금 예측 모델
실행: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import ccf, adfuller, grangercausalitytests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ══════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="도시가스 도매요금 예측",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-title  { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px; padding: 1rem;
        border-left: 4px solid #e74c3c;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 사이드바 설정
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    st.divider()

    st.markdown("### 🔑 API 키")
    st.caption("🔒 .streamlit/secrets.toml 에서 관리")
    eia_key  = st.secrets.get("EIA_API_KEY", "")
    ecos_key = st.secrets.get("ECOS_API_KEY", "")
    eia_ok = bool(eia_key)
    st.markdown(f"EIA: {'✅ 연결됨' if eia_ok else '❌ 키 없음'}")

    st.divider()
    st.markdown("### 📅 분석 기간")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("시작", "2018-01")
    with col2:
        end_date = st.text_input("종료", datetime.today().strftime("%Y-%m"))

    st.divider()
    st.markdown("### 🔧 모델 설정")
    test_months  = st.slider("테스트 기간(개월)", 6, 24, 12)
    max_lag      = st.slider("최대 Lag 탐색(개월)", 2, 12, 6)
    forecast_months = st.slider("예측 기간(개월)", 3, 24, 12)

    st.divider()
    run_btn = st.button("🚀 분석 실행", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════
# 데이터 수집 함수
# ══════════════════════════════════════════════════════════
CACHE_DIR = "data/raw_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# EIA v2 API - X-Params 헤더 방식 (파라미터 인코딩 문제 회피)
# JKM/TTF: natural-gas/pri/spt, Dubai/Brent: petroleum/pri/spt
EIA_ENDPOINTS = {
    "jkm":   {"url": "https://api.eia.gov/v2/natural-gas/pri/spt/data/",  "facet": "RNJPKM", "label": "JKM"},
    "ttf":   {"url": "https://api.eia.gov/v2/natural-gas/pri/spt/data/",  "facet": "RNTTTF", "label": "TTF"},
    "dubai": {"url": "https://api.eia.gov/v2/petroleum/pri/spt/data/",    "facet": "RBRTE",  "label": "Dubai"},
}

def fetch_eia(api_key: str, key: str, start: str, end: str, label: str) -> pd.Series:
    """EIA Open Data API v2 - X-Params 헤더 방식으로 파라미터 전달"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    today = datetime.today().strftime("%Y-%m-%d")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        if cached.get("date") == today:
            s = pd.Series(cached["data"], name=label)
            s.index = pd.to_datetime(s.index)
            return s.sort_index()

    ep = EIA_ENDPOINTS[key]

    # X-Params 헤더로 전달 (URL 인코딩 문제 완전 회피)
    x_params = {
        "frequency": "monthly",
        "data": ["value"],
        "facets": {"series": [ep["facet"]]},
        "start": start,
        "end": end,
        "sort": [{"column": "period", "direction": "asc"}],
        "offset": 0,
        "length": 500,
    }
    headers = {
        "X-Api-Key": api_key,
        "X-Params": json.dumps(x_params),
    }
    r = requests.get(ep["url"], headers=headers, timeout=20)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    if not rows:
        raise ValueError(f"데이터 없음: {key} ({ep['facet']})")
    s = pd.Series(
        {row["period"]: float(row["value"]) for row in rows if row["value"] is not None},
        name=label,
    )
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    with open(cache_file, "w") as f:
        json.dump({"date": today, "data": {str(k): v for k, v in s.items()}}, f)
    return s


def fetch_ecos_usdkrw(api_key: str, start: str, end: str) -> pd.Series:
    s_date = start.replace("-", "")[:6]
    e_date = end.replace("-", "")[:6]
    url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr"
           f"/1/1000/731Y001/MM/{s_date}/{e_date}/0000001")
    r = requests.get(url, timeout=15)
    rows = r.json()["StatisticSearch"]["row"]
    s = pd.Series(
        {pd.to_datetime(row["TIME"], format="%Y%m"): float(row["DATA_VALUE"]) for row in rows},
        name="USD_KRW",
    )
    return s.sort_index()


def sample_series(start, end, seed, base_range, shock_period=None, shock_range=(0, 30)):
    idx = pd.date_range(start, end, freq="MS")
    np.random.seed(seed)
    base = np.linspace(*base_range, len(idx))
    noise = np.random.normal(0, (base_range[1]-base_range[0])*0.05, len(idx))
    shock = np.zeros(len(idx))
    if shock_period:
        mask = (idx >= shock_period[0]) & (idx <= shock_period[1])
        n = mask.sum()
        if n > 0:
            shock[mask] = np.linspace(*shock_range, n)
    return pd.Series(base + shock + noise, index=idx)


COLLECTED_DIR = "data/collected"

def _read_csv(filename: str) -> pd.DataFrame | None:
    path = os.path.join(COLLECTED_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def _load_wholesale() -> pd.Series | None:
    """
    산업용도매원가 엑셀 로드 → ffill로 완전 월별 시계열 생성
    우선순위: data/collected/wholesale.csv → data/산업용도매원가.xlsx
    """
    # 1) collected CSV
    ws = _read_csv("wholesale.csv")
    if ws is not None and "도매요금" in ws.columns:
        s = ws["도매요금"].dropna()
        # NaT 인덱스 제거 후 날짜 정규화
        s = s[s.index.notna()]
        s.index = pd.to_datetime(s.index.astype(str).str[:7] + "-01", errors="coerce")
        s = s[s.index.notna()].sort_index()
        return s

    # 2) 엑셀 직접 읽기
    for path in ["data/산업용도매원가.xlsx", "data/산업용도매원가.xls"]:
        if os.path.exists(path):
            df_raw = pd.read_excel(path)
            # 날짜/요금 컬럼 자동 감지
            date_col  = next((c for c in df_raw.columns if any(k in str(c) for k in ["날짜","date","기간","연월","월"])), df_raw.columns[0])
            price_col = next((c for c in df_raw.columns if any(k in str(c) for k in ["원가","요금","price","단가","가격"])), df_raw.columns[1])
            df_raw[date_col] = pd.to_datetime(df_raw[date_col])
            s = df_raw.set_index(date_col)[price_col].dropna().astype(float)
            s = s[s.index.notna()]
            s.index = pd.to_datetime(s.index.astype(str).str[:7] + "-01", errors="coerce")
            s = s[s.index.notna()].sort_index()
            # ffill: 빈 달 = 직전 요금 (요금 변경 전까지 동일 요금 적용 규정)
            full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
            s = s.reindex(full_idx).ffill()
            s.name = "도매요금"
            return s
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(eia_key, ecos_key, start, end):
    errors = []

    def to_ms(s):
        """인덱스를 월 첫날(MS) 기준으로 정규화"""
        s = s.copy()
        s.index = pd.to_datetime(s.index.astype(str).str[:7] + "-01")
        return s.sort_index()

    def from_csv(filename, col=None):
        """CSV 읽어서 첫 번째(또는 지정) 컬럼 Series로 반환"""
        df = _read_csv(filename)
        if df is None:
            return None
        s = df[col] if (col and col in df.columns) else df.iloc[:, 0]
        return to_ms(s)

    # ── 도매요금: CSV 우선 → 엑셀 → 샘플 ───────────
    wholesale = _load_wholesale()
    if wholesale is not None:
        wholesale = to_ms(wholesale)
        errors.append(f"✅ 도매요금: {len(wholesale)}개월 ({wholesale.index[0].strftime('%Y-%m')} ~ {wholesale.index[-1].strftime('%Y-%m')})")
    else:
        wholesale = sample_series(start, end, 99, (13.5, 25.0), ("2022-02", "2023-06"), (0, 10))
        wholesale.name = "도매요금"
        errors.append("⚠️ 도매요금 없음 → 샘플")

    # ── EIA (HenryHub, Brent): CSV만 사용, 없으면 샘플 ──
    henry = from_csv("henry_hub_brent.csv", "HenryHub")
    brent = from_csv("henry_hub_brent.csv", "Brent")
    if henry is not None:
        errors.append(f"✅ HenryHub: {len(henry)}개월 ({henry.index[0].strftime('%Y-%m')} ~ {henry.index[-1].strftime('%Y-%m')})")
        errors.append(f"✅ Brent:    {len(brent)}개월 ({brent.index[0].strftime('%Y-%m')} ~ {brent.index[-1].strftime('%Y-%m')})")
    else:
        # CSV 없을 때만 API 호출
        if eia_key:
            try:
                henry = to_ms(fetch_eia(eia_key, "henry_hub", start, end, "HenryHub"))
                errors.append("✅ HenryHub: API 수집")
            except Exception as e:
                errors.append(f"⚠️ HenryHub API 실패: {e} → 샘플")
            try:
                brent = to_ms(fetch_eia(eia_key, "brent", start, end, "Brent"))
                errors.append("✅ Brent: API 수집")
            except Exception as e:
                errors.append(f"⚠️ Brent API 실패: {e} → 샘플")
        if henry is None:
            henry = sample_series(start, end, 1, (3, 6),   ("2022-02", "2023-03"), (0, 8));  henry.name = "HenryHub"
            errors.append("⚠️ HenryHub → 샘플 (데이터수집 페이지에서 수집하세요)")
        if brent is None:
            brent = sample_series(start, end, 3, (55, 80), ("2022-02", "2022-08"), (0, 40)); brent.name = "Brent"
            errors.append("⚠️ Brent → 샘플")

    # ── TTF: CSV만 사용, 없으면 샘플 ────────────────
    ttf = from_csv("ttf.csv")
    if ttf is not None:
        ttf.name = "TTF"
        errors.append(f"✅ TTF: {len(ttf)}개월 ({ttf.index[0].strftime('%Y-%m')} ~ {ttf.index[-1].strftime('%Y-%m')})")
    else:
        ttf = sample_series(start, end, 2, (10, 20), ("2022-02", "2023-06"), (0, 80)); ttf.name = "TTF"
        errors.append("⚠️ TTF → 샘플 (데이터수집 페이지에서 수집하세요)")

    # ── 환율: CSV만 사용, 없으면 샘플 ───────────────
    usdkrw = from_csv("usdkrw.csv")
    if usdkrw is not None:
        usdkrw.name = "USD_KRW"
        errors.append(f"✅ 환율: {len(usdkrw)}개월 ({usdkrw.index[0].strftime('%Y-%m')} ~ {usdkrw.index[-1].strftime('%Y-%m')})")
    else:
        usdkrw = sample_series(start, end, 7, (1150, 1350), ("2022-02", "2022-09"), (0, 100)); usdkrw.name = "USD_KRW"
        errors.append("⚠️ 환율 → 샘플 (데이터수집 페이지에서 수집하세요)")

    # ── 통합 병합 ─────────────────────────────────────
    df = pd.DataFrame({
        "도매요금": wholesale,
        "HenryHub": henry,
        "TTF":      ttf,
        "Brent":    brent,
        "USD_KRW":  usdkrw,
    })
    df = df.sort_index().loc[start:end]

    # 도매요금: ffill (변경일 사이 공백 메우기)
    df["도매요금"] = df["도매요금"].ffill()
    # 선행지표: 선형보간 (주말/공휴일 등 소수 결측)
    for col in FEAT_COLS:
        df[col] = df[col].interpolate(method="linear")
    df = df.dropna()

    errors.append(f"📊 최종 분석 데이터: {len(df)}개월 ({df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')})")
    return df, errors


# ══════════════════════════════════════════════════════════
# 분석 함수
# ══════════════════════════════════════════════════════════
FEAT_COLS   = ["HenryHub", "TTF", "Brent", "USD_KRW"]
FEAT_LABELS = {"HenryHub": "Henry Hub ($/MMBtu)", "TTF": "TTF ($/MMBtu)",
               "Brent": "Brent 유가 ($/bbl)", "USD_KRW": "USD/KRW (원)"}
FEAT_COLORS = {"HenryHub": "#e74c3c", "TTF": "#3498db", "Brent": "#2ecc71", "USD_KRW": "#9b59b6"}


def find_best_lags(df, max_lag):
    best = {}
    target = df["도매요금"].diff().dropna()
    for col in FEAT_COLS:
        if col not in df.columns:
            continue
        src = df[col].diff().dropna()
        aligned = pd.concat([target, src], axis=1).dropna()
        if len(aligned) < max_lag + 5:
            best[col] = 1
            continue
        corrs = ccf(aligned.iloc[:, 1], aligned.iloc[:, 0], nlags=max_lag+1, adjusted=False)
        best[col] = max(range(1, max_lag + 1), key=lambda i: abs(corrs[i]))
    return best


def run_sarimax(train, test, best_lags):
    exog_cols = [f"{c}_lag{l}" for c, l in best_lags.items()]

    def add_lags(df):
        out = df.copy()
        for c, l in best_lags.items():
            out[f"{c}_lag{l}"] = df[c].shift(l)
        return out

    full = add_lags(pd.concat([train, test])).dropna()
    tr   = full.iloc[:-len(test)]
    te   = full.iloc[-len(test):]

    model = SARIMAX(tr["도매요금"], exog=tr[exog_cols],
                    order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.forecast(steps=len(te), exog=te[exog_cols])

    mae  = mean_absolute_error(te["도매요금"], pred)
    rmse = np.sqrt(mean_squared_error(te["도매요금"], pred))
    mape = np.mean(np.abs((te["도매요금"].values - pred.values) / te["도매요금"].values)) * 100

    return pred, res, {"MAE": round(mae,3), "RMSE": round(rmse,3), "MAPE(%)": round(mape,2)}, te


def add_lag_features(df, best_lags):
    """Lag 피처 추가 헬퍼"""
    out = df.copy()
    for col, lag in best_lags.items():
        out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return out


def run_linear(train, test, best_lags):
    """선형회귀 + Lag 피처"""
    full = add_lag_features(pd.concat([train, test]), best_lags).dropna()
    tr = full.iloc[:-len(test)]
    te = full.iloc[-len(test):]

    feat_cols = [f"{c}_lag{l}" for c, l in best_lags.items()]
    X_train, y_train = tr[feat_cols], tr["도매요금"]
    X_test,  y_test  = te[feat_cols], te["도매요금"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=te.index)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test.values - pred.values) / y_test.values)) * 100

    coef_df = pd.DataFrame({
        "변수": feat_cols,
        "계수": model.coef_.round(4),
    })

    return pred, model, {"MAE": round(mae,3), "RMSE": round(rmse,3), "MAPE(%)": round(mape,2)}, coef_df, te



def run_var(train, test):
    """VAR - 변수 간 상호 인과 포착"""
    cols = ["도매요금"] + FEAT_COLS
    data = train[cols].diff().dropna()

    # 스케일링 없이 그대로 사용 (해석 편의)
    var_model = VAR(data)
    var_res = var_model.fit(maxlags=4, ic="aic")
    k = var_res.k_ar

    fc_input = data.values[-k:]
    fc = var_res.forecast(fc_input, steps=len(test))
    fc_df = pd.DataFrame(fc, index=test.index, columns=cols)

    # 역차분: 마지막 학습값 기준 누적합
    base = train["도매요금"].iloc[-1]
    pred = pd.Series(base + fc_df["도매요금"].cumsum().values, index=test.index)

    mae  = mean_absolute_error(test["도매요금"], pred)
    rmse = np.sqrt(mean_squared_error(test["도매요금"], pred))
    mape = np.mean(np.abs((test["도매요금"].values - pred.values) / test["도매요금"].values)) * 100

    return pred, var_res, {"MAE": round(mae,3), "RMSE": round(rmse,3), "MAPE(%)": round(mape,2)}


def run_granger(df, max_lag):
    records = []
    for col in FEAT_COLS:
        if col not in df.columns:
            continue
        try:
            data = df[["도매요금", col]].diff().dropna()
            test = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            for lag, res in test.items():
                p = res[0]["ssr_ftest"][1]
                records.append({"변수": FEAT_LABELS.get(col, col), "Lag": lag,
                                 "p-value": round(p, 4),
                                 "유의여부": "✅ 유의" if p < 0.05 else "❌"})
        except Exception:
            pass
    return pd.DataFrame(records)


def make_scenario_forecast(df, best_lags, forecast_months, scenario: str):
    """
    시나리오별 미래 예측
    - baseline: 최근 추세 유지
    - war_mild:   HenryHub+30%, TTF+50%, Brent+20%, 환율+8%
    - war_severe: HenryHub+80%, TTF+150%, Brent+50%, 환율+15%
    """
    SHOCKS = {
        "baseline":   {"HenryHub": 1.00, "TTF": 1.00, "Brent": 1.00, "USD_KRW": 1.00},
        "war_mild":   {"HenryHub": 1.30, "TTF": 1.50, "Brent": 1.20, "USD_KRW": 1.08},
        "war_severe": {"HenryHub": 1.80, "TTF": 2.50, "Brent": 1.50, "USD_KRW": 1.15},
    }
    shock = SHOCKS[scenario]

    last_vals = df[FEAT_COLS].iloc[-1]
    future_idx = pd.date_range(df.index[-1] + pd.DateOffset(months=1),
                               periods=forecast_months, freq="MS")

    future_df = pd.DataFrame(index=future_idx)
    for col in FEAT_COLS:
        future_df[col] = last_vals[col] * shock[col]

    # 단순 SARIMAX 재학습 후 예측
    exog_cols = [f"{c}_lag{l}" for c, l in best_lags.items()]

    def add_lags(d):
        out = d.copy()
        for c, l in best_lags.items():
            out[f"{c}_lag{l}"] = d[c].shift(l)
        return out

    hist_with_lags = add_lags(df).dropna()
    model = SARIMAX(hist_with_lags["도매요금"], exog=hist_with_lags[exog_cols],
                    order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # 미래 exog: lag는 과거 값 사용
    future_exog = pd.DataFrame(index=future_idx, columns=exog_cols)
    for col, lag in best_lags.items():
        key = f"{col}_lag{lag}"
        for i, dt in enumerate(future_idx):
            src_idx = i - lag
            if src_idx >= 0:
                future_exog.loc[dt, key] = future_df[col].iloc[src_idx]
            else:
                hist_i = len(df) + src_idx
                future_exog.loc[dt, key] = df[col].iloc[max(hist_i, 0)]

    future_exog = future_exog.astype(float).interpolate()
    pred = res.forecast(steps=forecast_months, exog=future_exog)
    return pd.Series(pred.values, index=future_idx)


# ══════════════════════════════════════════════════════════
# 메인 UI
# ══════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🔥 도시가스 도매요금 예측 모델</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EIA 에너지 지표 기반 선형회귀 · SARIMAX · VAR 비교 예측</div>', unsafe_allow_html=True)

if not run_btn:
    st.info("👈 사이드바에서 API 키와 도매요금 CSV를 설정한 후 **분석 실행** 버튼을 누르세요.")

    with st.expander("📋 CSV 형식 안내"):
        st.markdown("""
        **KESIS에서 다운받은 CSV는 아래 형식이면 자동 인식됩니다:**
        
        | date | price |
        |------|-------|
        | 2018-01 | 13.52 |
        | 2018-02 | 13.48 |
        
        - 날짜 컬럼명: `date`, `날짜`, `기간`, `년월`, `연월` 중 하나
        - 요금 컬럼명: `price`, `요금`, `단가`, `가격` 중 하나
        - 단위: **원/MJ** 권장
        """)

    with st.expander("🔑 EIA API 키 발급 방법"):
        st.markdown("""
        1. https://www.eia.gov/opendata/register.php 접속
        2. 이메일 입력 → 즉시 발급 (회원가입 불필요)
        3. 이메일에서 인증 링크 클릭
        4. 사이드바에 키 입력
        """)
    st.stop()


# ── 캐시 초기화 버튼 (사이드바) ────────────────────────────
with st.sidebar:
    if st.button("🔄 캐시 초기화", help="데이터 변경 후 캐시를 지웁니다"):
        st.cache_data.clear()
        st.success("캐시 초기화 완료!")
        st.rerun()

# ── 데이터 로드 ───────────────────────────────────────────
with st.spinner("📡 데이터 로드 중..."):
    df, errors = load_data(eia_key, ecos_key, start_date, end_date)

if errors:
    with st.expander("⚠️ 데이터 수집 알림", expanded=False):
        for e in errors:
            st.warning(e)

# ── Lag 분석 ──────────────────────────────────────────────
with st.spinner("🔍 Lag 분석 중..."):
    best_lags = find_best_lags(df, max_lag)

# ── 모델 학습 ──────────────────────────────────────────────
train = df.iloc[:-test_months]
test  = df.iloc[-test_months:]

with st.spinner("🤖 모델 학습 중..."):
    try:
        lr_pred, lr_model, lr_metrics, lr_coef, test_aligned = run_linear(train, test, best_lags)
        lr_ok = True
    except Exception as e:
        st.warning(f"선형회귀 오류: {e}")
        lr_ok = False

    try:
        sarimax_pred, sarimax_res, sarimax_metrics, _ = run_sarimax(train, test, best_lags)
        sarimax_ok = True
    except Exception as e:
        st.warning(f"SARIMAX 오류: {e}")
        sarimax_ok = False

    try:
        var_pred, var_res, var_metrics = run_var(train, test)
        var_ok = True
    except Exception as e:
        st.warning(f"VAR 오류: {e}")
        var_ok = False


# ══════════════════════════════════════════════════════════
# 탭 구성
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 현황", "🔍 Lag 분석", "🤖 모델 성능", "🎯 시나리오 예측"])


# ── TAB 1: 데이터 현황 ────────────────────────────────────
with tab1:
    # 주요 지표
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("분석 기간", f"{len(df)}개월")
    with c2:
        st.metric("최근 도매요금", f"{df['도매요금'].iloc[-1]:.2f} 원/MJ")
    with c3:
        delta = df['도매요금'].iloc[-1] - df['도매요금'].iloc[-13]
        st.metric("전년 대비", f"{delta:+.2f} 원/MJ")
    with c4:
        st.metric("최고 도매요금", f"{df['도매요금'].max():.2f} 원/MJ")

    st.divider()

    # 도매요금 추이
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["도매요금"], mode="lines+markers",
        name="도매요금(원/MJ)", line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=4),
    ))
    fig.add_vrect(x0="2022-02-01", x1="2023-06-01",
                  fillcolor="rgba(231,76,60,0.08)", line_width=0,
                  annotation_text="우크라이나 전쟁 충격", annotation_position="top left")
    fig.update_layout(title="도시가스 도매요금 추이", xaxis_title="", yaxis_title="원/MJ",
                      height=350, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 선행지표 추이
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=list(FEAT_LABELS.values()))
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (col, label), (r, c) in zip(FEAT_LABELS.items(), positions):
        fig2.add_trace(go.Scatter(
            x=df.index, y=df[col], name=label,
            line=dict(color=FEAT_COLORS[col], width=2), showlegend=False,
        ), row=r, col=c)
    fig2.update_layout(height=480, template="plotly_white", title_text="선행지표 추이")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**📋 원본 데이터**")
    st.dataframe(df.tail(24).style.format("{:.2f}"), use_container_width=True)


# ── TAB 2: Lag 분석 ───────────────────────────────────────
with tab2:
    st.markdown("### 최적 Lag (교차상관 기반)")

    lag_col = st.columns(len(FEAT_COLS))
    for i, col in enumerate(FEAT_COLS):
        with lag_col[i]:
            lag = best_lags.get(col, "-")
            st.metric(FEAT_LABELS.get(col, col), f"{lag}개월 후 반영")

    st.divider()

    # CCF 히트맵
    st.markdown("### 교차상관(CCF) 분석")
    target = df["도매요금"].diff().dropna()
    ccf_data = {}
    for col in FEAT_COLS:
        src = df[col].diff().dropna()
        aligned = pd.concat([target, src], axis=1).dropna()
        corrs = ccf(aligned.iloc[:, 1], aligned.iloc[:, 0], nlags=max_lag+1, adjusted=False)
        ccf_data[FEAT_LABELS[col]] = corrs[1:]

    ccf_df = pd.DataFrame(ccf_data, index=[f"Lag {i}" for i in range(1, max_lag+1)])
    fig_ccf = px.imshow(
        ccf_df.T, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="변수별 도매요금과의 교차상관계수 (lag별)",
    )
    fig_ccf.update_layout(height=280, template="plotly_white")
    st.plotly_chart(fig_ccf, use_container_width=True)

    # Granger 인과성
    st.markdown("### Granger 인과성 검정")
    with st.spinner("Granger 검정 중..."):
        granger_df = run_granger(df, min(max_lag, 4))

    if not granger_df.empty:
        # pivot으로 깔끔하게
        pivot = granger_df.pivot(index="변수", columns="Lag", values="p-value")
        st.dataframe(pivot.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.1)
                     .format("{:.4f}"), use_container_width=True)
        st.caption("🟢 p < 0.05: 해당 변수가 도매요금의 Granger 원인임 (통계적으로 유의)")


# ── TAB 3: 모델 성능 ──────────────────────────────────────
with tab3:
    st.markdown("### 📊 모델 성능 비교")

    metrics_rows = []
    if lr_ok:      metrics_rows.append({"모델": "선형회귀(+Lag)", **lr_metrics})
    if sarimax_ok: metrics_rows.append({"모델": "SARIMAX",       **sarimax_metrics})

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).set_index("모델")
        st.dataframe(metrics_df.style.highlight_min(axis=0, color="#d4edda"), use_container_width=True)
        st.caption("✅ 수치가 낮을수록 좋음  |  MAE·RMSE: 원/MJ 단위  |  MAPE: %")

    st.divider()

    # 예측 vs 실제 차트
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df.index, y=df["도매요금"], name="실제값",
        line=dict(color="#2c3e50", width=2.5),
    ))
    if lr_ok:
        fig3.add_trace(go.Scatter(
            x=lr_pred.index, y=lr_pred,
            name="선형회귀 예측", line=dict(color="#3498db", width=2, dash="dot"),
        ))
    if sarimax_ok:
        fig3.add_trace(go.Scatter(
            x=sarimax_pred.index, y=sarimax_pred,
            name="SARIMAX 예측", line=dict(color="#e74c3c", width=2, dash="dot"),
        ))
    fig3.add_vrect(
        x0=test.index[0], x1=test.index[-1],
        fillcolor="rgba(52,152,219,0.07)", line_width=0,
        annotation_text="테스트 구간", annotation_position="top left",
    )
    fig3.update_layout(title="예측 vs 실제 도매요금", xaxis_title="", yaxis_title="원/MJ",
                       height=400, template="plotly_white", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig3, use_container_width=True)

    # 선형회귀 계수 (해석용)
    if lr_ok:
        st.divider()
        st.markdown("### 📋 선형회귀 변수 계수")
        st.caption("계수가 클수록 도매요금에 미치는 영향이 큼")
        coef_display = lr_coef.copy()
        coef_display["영향"] = coef_display["계수"].apply(lambda x: "🔴 양" if x > 0 else "🔵 음")
        st.dataframe(coef_display, use_container_width=True, hide_index=True)


# ── TAB 4: 시나리오 예측 ──────────────────────────────────
with tab4:
    st.markdown("### 전쟁 시나리오별 도매요금 예측")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("""
        | 시나리오 | JKM | TTF | 두바이 | 환율 |
        |---------|-----|-----|--------|------|
        | 기준 (현상유지) | 동일 | 동일 | 동일 | 동일 |
        | 전쟁 (경미) | +30% | +50% | +20% | +8% |
        | 전쟁 (심각) | +80% | +150% | +50% | +15% |
        """)

    with st.spinner("시나리오 예측 중..."):
        sc_base   = make_scenario_forecast(df, best_lags, forecast_months, "baseline")
        sc_mild   = make_scenario_forecast(df, best_lags, forecast_months, "war_mild")
        sc_severe = make_scenario_forecast(df, best_lags, forecast_months, "war_severe")

    fig4 = go.Figure()

    # 과거 실적
    fig4.add_trace(go.Scatter(
        x=df.index, y=df["도매요금"], name="과거 실적",
        line=dict(color="#2c3e50", width=2),
    ))

    # 시나리오
    fig4.add_trace(go.Scatter(
        x=sc_base.index, y=sc_base, name="기준 시나리오",
        line=dict(color="#27ae60", width=2, dash="dot"),
    ))
    fig4.add_trace(go.Scatter(
        x=sc_mild.index, y=sc_mild, name="전쟁 경미",
        line=dict(color="#f39c12", width=2, dash="dot"),
    ))
    fig4.add_trace(go.Scatter(
        x=sc_severe.index, y=sc_severe, name="전쟁 심각",
        line=dict(color="#e74c3c", width=2.5, dash="dot"),
    ))

    # 불확실성 밴드 (심각 시나리오)
    upper = sc_severe * 1.10
    lower = sc_severe * 0.90
    fig4.add_trace(go.Scatter(
        x=pd.concat([sc_severe, sc_severe[::-1]]).index,
        y=pd.concat([upper, lower[::-1]]).values,
        fill="toself", fillcolor="rgba(231,76,60,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="심각 시나리오 신뢰구간(±10%)", showlegend=True,
    ))

    vline_x = df.index[-1].timestamp() * 1000
    fig4.add_shape(type="line", x0=vline_x, x1=vline_x, y0=0, y1=1,
                   xref="x", yref="paper", line=dict(color="gray", dash="dash"))
    fig4.add_annotation(x=vline_x, y=1, xref="x", yref="paper",
                        text="예측 시작", showarrow=False, yanchor="bottom")
    fig4.update_layout(
        title=f"도매요금 시나리오별 {forecast_months}개월 예측",
        xaxis_title="", yaxis_title="원/MJ",
        height=460, template="plotly_white",
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # 시나리오 수치 비교
    st.markdown("### 📋 예측값 비교표")
    sc_compare = pd.DataFrame({
        "기준": sc_base,
        "전쟁(경미)": sc_mild,
        "전쟁(심각)": sc_severe,
    })
    sc_compare.index = sc_compare.index.strftime("%Y-%m")
    st.dataframe(
        sc_compare.style.format("{:.2f}")
                  .background_gradient(cmap="YlOrRd", subset=["전쟁(심각)"]),
        use_container_width=True,
    )

    last_base   = sc_base.iloc[-1]
    last_mild   = sc_mild.iloc[-1]
    last_severe = sc_severe.iloc[-1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("기준", f"{last_base:.2f} 원/MJ")
    with c2:
        st.metric("전쟁 경미", f"{last_mild:.2f} 원/MJ",
                  delta=f"+{last_mild - last_base:.2f}", delta_color="inverse")
    with c3:
        st.metric("전쟁 심각", f"{last_severe:.2f} 원/MJ",
                  delta=f"+{last_severe - last_base:.2f}", delta_color="inverse")