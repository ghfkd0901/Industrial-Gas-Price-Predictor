import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

# ───────────────────────────────
# 🔑 1. 설정 및 데이터 로드 (캐싱 적용)
# ───────────────────────────────
st.set_page_config(page_title="에너지 도매요금 AI 예측 통합 분석", layout="wide")

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

# 변수별 최적 Lag (이미지 분석 결과 기반)
OPTIMAL_LAGS = {
    'WTI':      6,
    'Brent':    6,
    'HenryHub': 4,
    'JKM':      3,
    'TTF':      3,
    'USD_KRW':  0,
}

@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def load_all_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    # Master_Data 로드
    m_df = get_as_dataframe(sh.worksheet("Master_Data")).dropna(how='all').dropna(axis=1, how='all')
    m_df.iloc[:, 0] = pd.to_datetime(m_df.iloc[:, 0])
    m_df = m_df.set_index(m_df.columns[0])

    # gas_price 로드
    g_df = get_as_dataframe(sh.worksheet("gas_price")).dropna(how='all').dropna(axis=1, how='all')
    g_df.iloc[:, 0] = pd.to_datetime(g_df.iloc[:, 0])
    g_df = g_df.set_index(g_df.columns[0])
    g_df.columns = ['Wholesale_Price']

    df = m_df.join(g_df, how='outer')

    # ── 2015-01-01 이후 필터링
    df = df[df.index >= '2015-01-01']

    # ── 월별 인덱스로 리샘플 후 보간 (결측 구간 채우기)
    df = df.resample('MS').mean()          # 월 시작 기준 집계
    df = df.interpolate(method='time')     # 시간 기반 선형 보간
    df = df.ffill().bfill()                # 앞뒤 남은 NaN 제거

    return df.dropna(subset=['Wholesale_Price'])

# ───────────────────────────────
# 🛠️ 2. 사이드바 제어판
# ───────────────────────────────
data = load_all_data()

st.sidebar.header("⚙️ 모델 변수 설정")
all_features = [c for c in data.columns if c != 'Wholesale_Price']

default_features = [f for f in ['Brent', 'USD_KRW'] if f in all_features]

selected_features = st.sidebar.multiselect(
    "1. 분석에 포함할 지표 선택:",
    options=all_features,
    default=default_features
)

st.sidebar.divider()

# 변수별 Lag 표시 및 개별 조정
st.sidebar.subheader("2. 변수별 선행 시차(Lag) 설정")
st.sidebar.caption("각 변수의 최적 Lag가 자동 설정됩니다. 필요 시 조정하세요.")

custom_lags = {}
for feat in selected_features:
    default_lag = OPTIMAL_LAGS.get(feat, 3)
    custom_lags[feat] = st.sidebar.slider(
        f"{feat} Lag (개월)",
        min_value=0, max_value=12,
        value=default_lag,
        key=f"lag_{feat}"
    )

st.sidebar.divider()

# 학습 기간 설정
st.sidebar.subheader("3. 모델 학습 기간 설정")
st.sidebar.caption("선택한 기간의 데이터로만 모델을 학습합니다.")

available_years = sorted(data.index.year.unique())
min_year, max_year = available_years[0], available_years[-1]

train_start_year, train_end_year = st.sidebar.select_slider(
    "학습 기간 (연도)",
    options=available_years,
    value=(min_year, max_year)
)

train_start = pd.Timestamp(f"{train_start_year}-01-01")
train_end   = pd.Timestamp(f"{train_end_year}-12-31")

st.sidebar.caption(f"📅 학습 데이터: **{train_start_year}년** ~ **{train_end_year}년**")

st.sidebar.divider()
st.sidebar.info("변수 선택, Lag 조정, 학습 기간 변경 시 전 모델이 실시간으로 재학습됩니다.")

# ───────────────────────────────
# 🚀 3. 데이터 가공 및 4종 모델 학습
# ───────────────────────────────
if not selected_features:
    st.warning("👈 왼쪽 사이드바에서 지표를 선택해야 분석이 시작됩니다.")
else:
    # 변수별 개별 Lag 적용
    model_df = data[['Wholesale_Price']].copy()
    X_cols = []

    for col in selected_features:
        lag = custom_lags[col]
        col_name = f"{col}_Lag{lag}" if lag > 0 else col
        model_df[col_name] = data[col].shift(lag)
        X_cols.append(col_name)

    model_df = model_df.dropna()

    # 전체 데이터 (예측 시각화용)
    X_all = model_df[X_cols]
    y_all = model_df['Wholesale_Price']

    # 학습 기간 필터링
    train_mask = (model_df.index >= train_start) & (model_df.index <= train_end)
    X_train = model_df.loc[train_mask, X_cols]
    y_train = model_df.loc[train_mask, 'Wholesale_Price']

    if len(X_train) < 10:
        st.error("⚠️ 학습 데이터가 너무 적습니다. 학습 기간을 늘려주세요.")
        st.stop()

    # 모델 4종 학습 (학습 기간 데이터로만)
    models = {
        "Linear Regression": LinearRegression().fit(X_train, y_train),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train),
        "XGBoost":           XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    }

    # ───────────────────────────────
    # 📊 4. 성능 비교 (R² Score)
    # ───────────────────────────────
    st.title("⚖️ 에너지 도매요금 AI 모델 통합 분석")

    lag_summary = " | ".join([
        f"**{feat}** → Lag {custom_lags[feat]}개월"
        for feat in selected_features
    ])
    st.markdown(f"📅 **분석 기간:** 2015-01-01 ~ 현재 &nbsp;&nbsp;|&nbsp;&nbsp; 🎓 **학습 기간:** {train_start_year}년 ~ {train_end_year}년 ({len(X_train)}개월) &nbsp;&nbsp; {lag_summary}")

    # R² Score: 학습(train) / 전체(all) 두 가지 표시
    cols = st.columns(4)
    for i, (name, model) in enumerate(models.items()):
        r2_tr  = r2_score(y_train, model.predict(X_train))
        r2_all = r2_score(y_all,   model.predict(X_all))
        cols[i].metric(
            label=name,
            value=f"R² {r2_tr:.4f}",
            delta=f"전체기간 {r2_all:.4f}",
            delta_color="normal"
        )

    # ───────────────────────────────
    # 📉 5. 예측 추세 시각화
    # ───────────────────────────────
    st.divider()
    st.subheader("📉 실제 도매요금 vs 모델별 예측 결과")

    fig_pred = go.Figure()

    # 학습 기간 배경 표시
    fig_pred.add_vrect(
        x0=train_start, x1=train_end,
        fillcolor="lightyellow", opacity=0.3,
        layer="below", line_width=0,
        annotation_text=f"학습 기간 ({train_start_year}~{train_end_year})",
        annotation_position="top left"
    )

    fig_pred.add_trace(go.Scatter(
        x=model_df.index, y=y_all,
        name="실제 요금", line=dict(color='black', width=4)
    ))

    colors = ['gray', 'green', 'orange', 'red']
    for (name, model), color in zip(models.items(), colors):
        fig_pred.add_trace(go.Scatter(
            x=model_df.index, y=model.predict(X_all),
            name=name,
            line=dict(color=color, dash='dot' if name == "Linear Regression" else 'solid')
        ))

    fig_pred.update_layout(hovermode="x unified", template="plotly_white", height=600)
    st.plotly_chart(fig_pred, use_container_width=True)

    # ───────────────────────────────
    # 🧐 6. 변수 중요도 (Tree 3종)
    # ───────────────────────────────
    st.divider()
    st.subheader("🧐 AI 모델별 변수 중요도 (Feature Importance)")
    st.caption("선형 회귀를 제외한 트리 기반 모델들이 각 변수에 부여한 가중치입니다.")

    fi_cols = st.columns(3)
    tree_models = {
        "Random Forest":     ("viridis", models["Random Forest"]),
        "Gradient Boosting": ("plasma",  models["Gradient Boosting"]),
        "XGBoost":           ("rdylbu",  models["XGBoost"])
    }

    for i, (name, (cmap, model)) in enumerate(tree_models.items()):
        df_fi = (
            pd.DataFrame({'Feature': X_cols, 'Importance': model.feature_importances_})
            .sort_values(by='Importance', ascending=True)
        )
        fig_fi = px.bar(
            df_fi, x='Importance', y='Feature', orientation='h',
            title=f"{name} 중요도", color='Importance', color_continuous_scale=cmap
        )
        fig_fi.update_layout(coloraxis_showscale=False, height=350)
        fi_cols[i].plotly_chart(fig_fi, use_container_width=True)

    # ───────────────────────────────
    # 💡 7. 최종 인사이트
    # ───────────────────────────────
    st.divider()
    best_name = max(models, key=lambda k: r2_score(y_train, models[k].predict(X_train)))
    st.success(f"🎉 현재 조합에서 최적의 알고리즘은 **[{best_name}]** 입니다.")

    lag_detail = ", ".join([f"{feat}({custom_lags[feat]}개월)" for feat in selected_features])
    st.info(f"""
    **동적 분석 요약:**
    - **학습 기간:** {train_start_year}년 ~ {train_end_year}년 ({len(X_train)}개월 데이터로 학습)
    - **변수별 적용 Lag:** {lag_detail}
    - 학습 기간 기준 **{best_name}** 모델이 가장 높은 설명력을 보입니다.
    - 그래프의 **노란 영역**이 학습 기간이며, 그 외 구간은 학습하지 않은 데이터에 대한 예측입니다.
    - 학습 기간을 줄일수록 과적합 여부를 확인할 수 있습니다.
    """)