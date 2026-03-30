import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

# ───────────────────────────────
# 페이지 설정
# ───────────────────────────────
st.set_page_config(
    page_title="도매요금 즉시 예측기",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────────
# 커스텀 CSS
# ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0, 120, 255, 0.15), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(255, 140, 0, 0.08), transparent);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0d1120; }

/* 메인 폰트 */
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
    color: #e8eaf0;
}

/* 타이틀 영역 */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.12em;
    color: #ffffff;
    line-height: 1.0;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4a7fd4;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* 입력 패널 */
.input-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.input-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #4a7fd4;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.input-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #556080;
    margin-top: 0.2rem;
}

/* 예측 카드 — st.columns 안에서 렌더링 */
.pred-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
    border-radius: 14px 14px 0 0;
}

.card-lr  { background: rgba(149,165,166,0.13); --accent: #b0bec5; border: 1px solid rgba(149,165,166,0.3); }
.card-rf  { background: rgba(39,174, 96,0.13);  --accent: #2ecc71; border: 1px solid rgba(39,174,96,0.3);  }
.card-gb  { background: rgba(243,156, 18,0.13); --accent: #f39c12; border: 1px solid rgba(243,156,18,0.3); }
.card-xgb { background: rgba(231, 76, 60,0.13); --accent: #e74c3c; border: 1px solid rgba(231,76,60,0.3);  }

.card-model-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.card-price {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: 0.04em;
    color: #ffffff;
    line-height: 1.0;
    text-shadow: 0 0 20px rgba(255,255,255,0.15);
}
.card-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #a0aabf;
    margin-top: 0.25rem;
    font-weight: 500;
}
.card-r2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #7888a8;
    margin-top: 1rem;
    padding-top: 0.6rem;
    border-top: 1px solid rgba(255,255,255,0.07);
}
.card-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 5px;
    margin-top: 0.6rem;
    letter-spacing: 0.1em;
}
.badge-max { background: rgba(255,215,0,0.18); color: #ffd700; border: 1px solid rgba(255,215,0,0.45); }
.badge-min { background: rgba(100,220,120,0.18); color: #6edc78; border: 1px solid rgba(100,220,120,0.45); }
.badge-mid { background: rgba(255,255,255,0.06); color: #7888a8; border: 1px solid rgba(255,255,255,0.12); }

/* 구분선 */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.5rem 0;
}

/* 데이터 기준 뱃지 */
.data-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4a7fd4;
    background: rgba(74,127,212,0.1);
    border: 1px solid rgba(74,127,212,0.2);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    margin-bottom: 1.5rem;
}

/* 학습 기간 슬라이더 레이블 */
.train-label {
    font-size: 0.68rem;
    color: #4a7fd4;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}

/* streamlit 기본 요소 덮어쓰기 */
div[data-testid="stNumberInput"] label { color: #8892a4 !important; font-size: 0.8rem !important; }
div[data-testid="stSelectSlider"] label { color: #8892a4 !important; font-size: 0.8rem !important; }
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #1a4fd8, #0d3aad) !important;
    border: 1px solid rgba(74,127,212,0.4) !important;
    border-radius: 10px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.15em !important;
    color: #fff !important;
    height: 3rem !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2560f0, #1a4fd8) !important;
    box-shadow: 0 0 20px rgba(74,127,212,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────
# 상수
# ───────────────────────────────
SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"
BRENT_LAG = 6
TRAIN_START_DEFAULT = pd.Timestamp("2015-01-01")

MODEL_META = {
    "Linear Regression": {"cls": "card-lr",  "short": "LR"},
    "Random Forest":     {"cls": "card-rf",  "short": "RF"},
    "Gradient Boosting": {"cls": "card-gb",  "short": "GB"},
    "XGBoost":           {"cls": "card-xgb", "short": "XGB"},
}

# ───────────────────────────────
# 데이터 로드
# ───────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_data(ttl=600)
def load_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    m_raw = get_as_dataframe(sh.worksheet("Master_Data")).dropna(how="all").dropna(axis=1, how="all")
    m_raw.iloc[:, 0] = pd.to_datetime(m_raw.iloc[:, 0])
    m_raw = m_raw.set_index(m_raw.columns[0]).sort_index().ffill()

    g_raw = get_as_dataframe(sh.worksheet("gas_price")).dropna(how="all").dropna(axis=1, how="all")
    g_raw.iloc[:, 0] = pd.to_datetime(g_raw.iloc[:, 0])
    g_raw = g_raw.set_index(g_raw.columns[0]).sort_index().ffill()
    g_raw.columns = ["Wholesale_Price"]

    merged = m_raw.join(g_raw, how="inner")
    merged[f"Brent_Lag{BRENT_LAG}"] = merged["Brent"].shift(BRENT_LAG)

    return merged


# ───────────────────────────────
# 모델 학습
# ───────────────────────────────
def train_models(merged: pd.DataFrame, train_start: pd.Timestamp, train_end: pd.Timestamp):
    feature_cols = [f"Brent_Lag{BRENT_LAG}", "USD_KRW"]
    required = feature_cols + ["Wholesale_Price"]

    df = merged.dropna(subset=required)
    df = df[(df.index >= train_start) & (df.index <= train_end)]

    X = df[feature_cols]
    y = df["Wholesale_Price"]

    models = {
        "Linear Regression": LinearRegression().fit(X, y),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, y),
        "XGBoost":           XGBRegressor(n_estimators=100, random_state=42, verbosity=0).fit(X, y),
    }

    r2_scores = {name: r2_score(y, m.predict(X)) for name, m in models.items()}
    n_train = len(X)

    return models, r2_scores, n_train, feature_cols


# ───────────────────────────────
# 예측 카드 렌더링 (st.columns 기반)
# ───────────────────────────────
def render_cards(predictions: dict, r2_scores: dict):
    prices  = list(predictions.values())
    max_val = max(prices)
    min_val = min(prices)

    items = list(predictions.items())
    # 2행 2열 레이아웃
    for row_items in [items[:2], items[2:]]:
        cols = st.columns(2, gap="small")
        for col, (name, price) in zip(cols, row_items):
            meta = MODEL_META[name]
            r2   = r2_scores.get(name, 0)

            if price == max_val and price != min_val:
                badge = '<span class="card-badge badge-max">▲ HIGHEST</span>'
            elif price == min_val and price != max_val:
                badge = '<span class="card-badge badge-min">▼ LOWEST</span>'
            else:
                badge = '<span class="card-badge badge-mid">— MID</span>'

            card_html = f"""
            <div class="pred-card {meta['cls']}">
                <div class="card-model-name">{name}</div>
                <div class="card-price">{price:.4f}</div>
                <div class="card-unit">원 / MJ</div>
                {badge}
                <div class="card-r2">R² = {r2:.4f}</div>
            </div>
            """
            col.markdown(card_html, unsafe_allow_html=True)


# ───────────────────────────────
# 메인
# ───────────────────────────────
def main():

    # ── 헤더 ──────────────────────────────────────────────
    st.markdown("""
    <div class="hero-title">QUICK PREDICT</div>
    <div class="hero-sub">⚡ 도매요금 즉시 예측기 · Daesung Energy</div>
    """, unsafe_allow_html=True)

    # ── 데이터 로드 ────────────────────────────────────────
    with st.spinner("데이터 로딩 중..."):
        try:
            merged = load_data()
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
            st.stop()

    last_date    = merged["Wholesale_Price"].dropna().index.max()
    latest_brent = float(merged["Brent"].dropna().iloc[-1])
    latest_krw   = float(merged["USD_KRW"].dropna().iloc[-1])

    st.markdown(
        f'<div class="data-badge">📅 데이터 기준: {last_date.strftime("%Y년 %m월")} &nbsp;|&nbsp; '
        f'최근 브렌트유: ${latest_brent:.1f} &nbsp;|&nbsp; 최근 환율: ₩{latest_krw:,.0f}</div>',
        unsafe_allow_html=True,
    )

    # ── 학습 기간 설정 (사이드바) ─────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 모델 설정")
        available_years = sorted(merged.index.year.unique())
        default_start   = max(TRAIN_START_DEFAULT.year, available_years[0])

        train_start_year, train_end_year = st.select_slider(
            "학습 기간 (연도)",
            options=available_years,
            value=(default_start, available_years[-1]),
        )
        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end   = pd.Timestamp(f"{train_end_year}-12-31")

        st.caption(f"📊 {train_start_year}년 ~ {train_end_year}년 데이터로 학습")
        st.divider()

        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── 모델 학습 ──────────────────────────────────────────
    with st.spinner("🤖 모델 학습 중..."):
        try:
            models, r2_scores, n_train, feature_cols = train_models(merged, train_start, train_end)
        except Exception as e:
            st.error(f"❌ 모델 학습 실패: {e}")
            st.stop()

    # ── 입력 패널 ──────────────────────────────────────────
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="input-label">입력값 설정</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        brent_input = st.number_input(
            f"브렌트유 가격 ($/배럴)  — {BRENT_LAG}개월 후 요금에 반영",
            min_value=10.0,
            max_value=300.0,
            value=latest_brent,
            step=0.5,
            format="%.2f",
        )
        st.markdown(
            f'<div class="input-hint">현재 최근값: ${latest_brent:.2f}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        krw_input = st.number_input(
            "환율 (원/$)",
            min_value=500.0,
            max_value=3000.0,
            value=latest_krw,
            step=1.0,
            format="%.0f",
        )
        st.markdown(
            f'<div class="input-hint">현재 최근값: ₩{latest_krw:,.0f}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 예측 실행 버튼 ─────────────────────────────────────
    run_btn = st.button("⚡  예측하기", type="primary", use_container_width=True)

    # ── 결과 표시 ──────────────────────────────────────────
    if run_btn:
        X_pred = pd.DataFrame(
            [[brent_input, krw_input]],
            columns=feature_cols,
        )

        predictions = {
            name: float(model.predict(X_pred)[0])
            for name, model in models.items()
        }

        avg_price = np.mean(list(predictions.values()))

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                        color:#4a7fd4; letter-spacing:0.2em; text-transform:uppercase;
                        margin-bottom:0.3rem;">
                브렌트유 ${brent_input:.2f} · 환율 ₩{krw_input:,.0f} →
                {BRENT_LAG}개월 후 예상 도매요금
            </div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem;
                        color:#8892a4; letter-spacing:0.15em;">
                4개 모델 평균 &nbsp;
                <span style="color:#ffffff; font-size:1.8rem;">{avg_price:.4f}</span>
                &nbsp; 원/MJ
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_cards(predictions, r2_scores)

        st.markdown(f"""
        <div style="text-align:center; margin-top:1.2rem;
                    font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#556080;">
            학습 기간: {train_start_year}년 ~ {train_end_year}년 ({n_train}개월) &nbsp;|&nbsp;
            피처: 브렌트유 Lag {BRENT_LAG}M · 환율
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()