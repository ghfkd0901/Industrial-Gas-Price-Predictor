import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

# 페이지 설정
st.set_page_config(page_title="머신러닝 모델 비교 및 중요도 분석", layout="wide")

# 한글 폰트 설정
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
    
    # 데이터 병합 로직
    m_df = get_as_dataframe(sh.worksheet("Master_Data")).dropna(how='all').dropna(axis=1, how='all')
    m_df.iloc[:, 0] = pd.to_datetime(m_df.iloc[:, 0])
    m_df = m_df.set_index(m_df.columns[0])
    
    g_df = get_as_dataframe(sh.worksheet("gas_price")).dropna(how='all').dropna(axis=1, how='all')
    g_df.iloc[:, 0] = pd.to_datetime(g_df.iloc[:, 0])
    g_df = g_df.set_index(g_df.columns[0])
    g_df.columns = ['Wholesale_Price']
    
    df = m_df.join(g_df, how='inner')
    df['Wholesale_Price'] = df['Wholesale_Price'].ffill()
    
    # 변수 생성
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
    
    model_df = data.dropna(subset=features + ['Wholesale_Price'])
    X = model_df[features]
    y = model_df['Wholesale_Price']

    # 모델 정의 및 학습
    lr_model = LinearRegression().fit(X, y)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42).fit(X, y)

    st.title("⚖️ 머신러닝 알고리즘별 성능 비교")
    st.markdown(f"**분석 범위:** {model_df.index.min().strftime('%Y-%m')} ~ {model_df.index.max().strftime('%Y-%m')}")

    # R2 점수 리포트
    c1, c2, c3 = st.columns(3)
    c1.metric("선형 회귀 R²", f"{r2_score(y, lr_model.predict(X)):.4f}")
    c2.metric("Random Forest R²", f"{r2_score(y, rf_model.predict(X)):.4f}")
    c3.metric("XGBoost R²", f"{r2_score(y, xgb_model.predict(X)):.4f}")

    # 📉 예측선 시각화
    st.divider()
    st.subheader("📉 실제 요금 vs 모델별 예측선 (Plotly)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_df.index, y=y, name="실제 요금", line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=model_df.index, y=lr_model.predict(X), name="선형 회귀 Prediction", line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=model_df.index, y=rf_model.predict(X), name="Random Forest Prediction", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=model_df.index, y=xgb_model.predict(X), name="XGBoost Prediction", line=dict(color='red')))
    fig.update_layout(hovermode="x unified", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ───────────────────────────────
    # 🔍 3. 트리 기반 모델 특성 중요도 (Feature Importance)
    # ───────────────────────────────
    st.divider()
    st.subheader("🧐 AI가 분석한 요금 결정 요인 (Feature Importance)")

    def create_fi_plot(model, model_name, colorscale):
        importances = model.feature_importances_
        df_fi = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        fig = px.bar(df_fi, x='Importance', y='Feature', orientation='h', 
                     text='Importance', color='Importance', 
                     color_continuous_scale=colorscale,
                     title=f"{model_name} 변수 기여도")
        
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, template="plotly_white")
        return fig

    col_xgb, col_rf = st.columns(2)

    with col_xgb:
        fig_xgb = create_fi_plot(xgb_model, "XGBoost", "rdylbu")
        st.plotly_chart(fig_xgb, use_container_width=True)

    with col_rf:
        fig_rf = create_fi_plot(rf_model, "Random Forest", "viridis")
        st.plotly_chart(fig_rf, use_container_width=True)

    # ───────────────────────────────
    # 📈 4. 선형 회귀 계수 분석 (Linear Regression Coefficients)
    # ───────────────────────────────
    st.divider()
    st.subheader("📊 선형 회귀(Linear Regression) 가중치 분석")
    
    # 선형 회귀는 계수(Coefficients)의 방향성(+, -)이 중요함
    lr_coefs = pd.DataFrame({
        'Feature': features, 
        'Weight': lr_model.coef_,
        'Abs_Weight': np.abs(lr_model.coef_)
    }).sort_values(by='Abs_Weight', ascending=True)

    fig_lr = px.bar(lr_coefs, x='Weight', y='Feature', orientation='h',
                    text='Weight', color='Weight',
                    color_continuous_scale="rdbu",
                    title="선형 회귀 변수별 영향도 (회귀 계수)")
    
    fig_lr.add_annotation(
        text="* 양수(+)는 요금 상승 요인, 음수(-)는 요금 하락 요인을 의미함",
        xref="paper", yref="paper", x=0, y=-0.25, showarrow=False, font=dict(size=12, color="gray")
    )

    fig_lr.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_lr.update_layout(coloraxis_showscale=True, template="plotly_white", height=400)
    
    st.plotly_chart(fig_lr, use_container_width=True)

    # 💡 AI 인사이트 요약
    st.divider()
    best_feature = features[np.argmax(xgb_model.feature_importances_)]
    st.success(f"분석 결과, 현재 요금 산정에 가장 지배적인 지표는 **[{best_feature}]** 입니다.")

except Exception as e:
    st.error(f"비교 분석 중 오류가 발생했습니다: {e}")