import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

st.set_page_config(page_title="도매요금 위기 분석 대시보드", layout="wide")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

# 🔑 1. 데이터 로드 및 모델 학습 (R2 0.9987 최적 모델)
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def load_and_train_final():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)
    
    # Master 데이터 전처리
    m_df = get_as_dataframe(sh.worksheet("Master_Data")).dropna(how='all').dropna(axis=1, how='all')
    m_df.iloc[:, 0] = pd.to_datetime(m_df.iloc[:, 0])
    m_df = m_df.set_index(m_df.columns[0]).fillna(method='ffill').fillna(method='bfill')
    
    # 요금 데이터 전처리
    g_df = get_as_dataframe(sh.worksheet("gas_price")).dropna(how='all').dropna(axis=1, how='all')
    g_df.iloc[:, 0] = pd.to_datetime(g_df.iloc[:, 0])
    g_df = g_df.set_index(g_df.columns[0]).fillna(method='ffill')
    g_df.columns = ['Wholesale_Price']
    
    df = m_df.join(g_df, how='inner')
    
    # 시차 변수 생성 (Brent 6M, TTF 3M)
    df['Brent_Lag6'] = df['Brent'].shift(6)
    df['TTF_Lag3'] = df['TTF'].shift(3)
    # 전쟁 변수 코딩 (학습용)
    df['War_Event'] = 0
    df.loc['2022-03-01':'2023-12-31', 'War_Event'] = 1
    
    features = ['Brent_Lag6', 'TTF_Lag3', 'USD_KRW', 'War_Event']
    model_df = df.dropna(subset=features + ['Wholesale_Price'])
    
    # XGBoost 학습
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(model_df[features], model_df['Wholesale_Price'])
    
    return model, df, features, model_df

# 🚀 2. 통합 대시보드 및 3중 축 시각화
try:
    xgb_model, df, features, model_df = load_and_train_final()
    
    st.title("🏛️ 대성에너지 도매요금 위기 분석 대시보드")
    
    # 사이드바 시나리오 설정
    st.sidebar.header("📊 미래 예측 시나리오")
    war_risk = st.sidebar.checkbox("전쟁/지정학적 리스크 반영", value=False)
    war_val = 1 if war_risk else 0
    
    # 미래 3개월 확정 예측 (환율 최신값 고정)
    future_dates = [df.index.max() + pd.DateOffset(months=i) for i in range(1, 4)]
    future_preds = []
    current_krw = df['USD_KRW'].iloc[-1]
    
    for i, date in enumerate(future_dates, 1):
        input_row = pd.DataFrame([[df['Brent'].iloc[-6 + (i-1)], df['TTF'].iloc[-3 + (i-1)], current_krw, war_val]], columns=features)
        future_preds.append(xgb_model.predict(input_row)[0])
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds}).set_index('Date')

    # 📈 3중 축 통합 시각화 (정제된 스타일)
    fig = go.Figure()

    # 1. 원자재 (투명도 조절로 지저분함 제거)
    fig.add_trace(go.Scatter(x=df.index, y=df['Brent'], name="Brent ($)", line=dict(color='rgba(255, 165, 0, 0.3)', width=1.5), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.index, y=df['TTF'], name="TTF (€)", line=dict(color='rgba(0, 128, 0, 0.3)', width=1.5), yaxis="y3"))

    # 2. 실제 요금 및 모델 추종
    fig.add_trace(go.Scatter(x=model_df.index, y=model_df['Wholesale_Price'], name="실제 요금", line=dict(color='#2C3E50', width=2.5)))
    
    # 3. 미래 3개월 확정 전망
    conn_x = [model_df.index[-1]] + list(future_df.index)
    conn_y = [model_df['Wholesale_Price'].iloc[-1]] + list(future_df['Predicted'])
    pred_color = '#E74C3C' if war_risk else '#3498DB'
    fig.add_trace(go.Scatter(x=conn_x, y=conn_y, name="3M 확정전망", line=dict(color=pred_color, width=4)))

    # 레이아웃 설정 (최신 Plotly 규격 반영)
    fig.update_layout(
        xaxis=dict(domain=[0, 0.88], showgrid=False),
        yaxis=dict(title="도매요금 (원/MJ)", side="left"),
        yaxis2=dict(title="Brent ($)", overlaying="y", side="right", showgrid=False, title_font=dict(color="#FFA500"), tickfont=dict(color="#FFA500")),
        yaxis3=dict(title="TTF (€)", overlaying="y", side="right", anchor="free", position=0.96, showgrid=False, title_font=dict(color="#008000"), tickfont=dict(color="#008000")),
        template="plotly_white", height=650, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 🔥 우크라이나 쇼크 기간 음영 (과거)
    fig.add_vrect(x0='2022-03-01', x1='2023-12-31', fillcolor="gray", opacity=0.1, line_width=0, layer="below", annotation_text="우크라이나 쇼크 기간", annotation_position="top left")

    # 🔥 미래 구간 음영 (예측)
    fig.add_vrect(x0=model_df.index[-1], x1=future_df.index[-1], fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
    
    st.plotly_chart(fig, use_container_width=True)

    # 보고서 요약
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 환율 기준", f"{current_krw:,.1f}원", "고정값 적용")
    c2.metric("3개월 뒤 전망", f"{future_preds[2]:.2f}원", f"{future_preds[2]-df['Wholesale_Price'].iloc[-1]:.2f}원")
    c3.info(f"💡 {'리스크 시나리오' if war_risk else '정상 시나리오'} 적용 중")

except Exception as e:
    st.error(f"오류 발생: {e}")