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

# 페이지 설정
st.set_page_config(page_title="머신러닝 모델 비교 및 중요도 분석", layout="wide")

# 한글 폰트 설정 (로컬 환경용, 배포 환경에서는 Plotly 권장)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

# ───────────────────────────────
# 🔑 1. 데이터 로드 및 전처리
# ───────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def load_compare_data():
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
    
    # 데이터 병합 및 결측치 처리
    df = m_df.join(g_df, how='inner')
    df['Wholesale_Price'] = df['Wholesale_Price'].ffill()
    
    # 분석용 파생 변수 생성 (시차 적용)
    df['Brent_Lag6'] = df['Brent'].shift(6)
    df['TTF_Lag3'] = df['TTF'].shift(3)
    df['War_Event'] = 0
    df.loc['2022-02-01':'2023-12-31', 'War_Event'] = 1
    
    return df

# ───────────────────────────────
# 🚀 2. 모델 학습 및 성능 평가
# ───────────────────────────────
try:
    data = load_compare_data()
    features = ['Brent_Lag6', 'TTF_Lag3', 'USD_KRW', 'War_Event']
    
    # 학습 데이터 셋 준비 (결측치 제거)
    model_df = data.dropna(subset=features + ['Wholesale_Price'])
    X = model_df[features]
    y = model_df['Wholesale_Price']

    # 4가지 모델 정의 및 학습
    lr_model = LinearRegression().fit(X, y)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42).fit(X, y)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42).fit(X, y)

    st.title("⚖️ 머신러닝 알고리즘별 성능 비교")
    st.markdown(f"**분석 데이터 범위:** {model_df.index.min().strftime('%Y-%m')} ~ {model_df.index.max().strftime('%Y-%m')}")

    # 상단 R2 점수 지표 (4개 컬럼)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("선형 회귀 R²", f"{r2_score(y, lr_model.predict(X)):.4f}")
    m2.metric("Random Forest R²", f"{r2_score(y, rf_model.predict(X)):.4f}")
    m3.metric("G-Boosting R²", f"{r2_score(y, gb_model.predict(X)):.4f}")
    m4.metric("XGBoost R²", f"{r2_score(y, xgb_model.predict(X)):.4f}")

    # 📉 예측 시각화 (오류 수정 반영)
    st.divider()
    st.subheader("📉 실제 요금 vs 모델별 예측 추세")
    
    fig_pred = go.Figure()
    
    # 실제값
    fig_pred.add_trace(go.Scatter(x=model_df.index, y=y, name="실제 도매요금", 
                                line=dict(color='black', width=3)))
    # 예측값들
    fig_pred.add_trace(go.Scatter(x=model_df.index, y=lr_model.predict(X), name="선형 회귀 (Base)", 
                                line=dict(color='gray', dash='dot', width=1)))
    fig_pred.add_trace(go.Scatter(x=model_df.index, y=rf_model.predict(X), name="Random Forest", 
                                line=dict(color='rgba(0, 128, 0, 0.6)', width=2)))
    fig_pred.add_trace(go.Scatter(x=model_df.index, y=gb_model.predict(X), name="Gradient Boosting", 
                                line=dict(color='orange', width=2)))
    fig_pred.add_trace(go.Scatter(x=model_df.index, y=xgb_model.predict(X), name="XGBoost (Strong)", 
                                line=dict(color='red', width=2)))

    fig_pred.update_layout(
        hovermode="x unified", 
        template="plotly_white", 
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ───────────────────────────────
    # 🔍 3. 트리 기반 모델 특성 중요도 (Feature Importance)
    # ───────────────────────────────
    st.divider()
    st.subheader("🧐 AI 모델별 변수 기여도 (Feature Importance) 비교")

    def create_fi_plot(model, model_name, colorscale):
        importances = model.feature_importances_
        df_fi = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        fig = px.bar(df_fi, x='Importance', y='Feature', orientation='h', 
                     text='Importance', color='Importance', 
                     color_continuous_scale=colorscale,
                     title=f"{model_name} 분석 가중치")
        
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, template="plotly_white", height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    # 3개 모델 중요도 비교 (컬럼 배치)
    fi_col1, fi_col2, fi_col3 = st.columns(3)

    with fi_col1:
        st.plotly_chart(create_fi_plot(rf_model, "Random Forest", "viridis"), use_container_width=True)
    with fi_col2:
        st.plotly_chart(create_fi_plot(gb_model, "Gradient Boosting", "plasma"), use_container_width=True)
    with fi_col3:
        st.plotly_chart(create_fi_plot(xgb_model, "XGBoost", "rdylbu"), use_container_width=True)

    # 💡 최종 분석 인사이트
    st.divider()
    scores = {
        "선형 회귀": r2_score(y, lr_model.predict(X)),
        "Random Forest": r2_score(y, rf_model.predict(X)),
        "Gradient Boosting": r2_score(y, gb_model.predict(X)),
        "XGBoost": r2_score(y, xgb_model.predict(X))
    }
    best_model = max(scores, key=scores.get)
    
    st.success(f"현재 데이터 기준, 가장 예측력이 높은 알고리즘은 **[{best_model}]** 입니다. (R² Score: {scores[best_model]:.4f})")
    st.info("""
    **💡 대리님을 위한 분석 포인트:**
    1. **R²(결정계수)**: 1.0에 가까울수록 실제 요금 변동을 완벽하게 설명한다는 뜻입니다.
    2. **변수 기여도**: 특정 모델에서 **Brent_Lag6**의 비중이 압도적이라면, 유가 변동이 6개월 뒤 요금 결정의 핵심 키(Key)임을 데이터가 증명하는 것입니다.
    3. **모델 선택**: 선형 회귀보다 트리 기반(XGBoost 등) 모델의 점수가 높다면, 요금 체계가 비선형적인 복합 요인에 의해 결정되고 있음을 의미합니다.
    """)

except Exception as e:
    st.error(f"비교 분석 중 오류가 발생했습니다: {e}")