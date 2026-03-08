import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

# ───────────────────────────────
# 🔑 1. 설정 및 구글 인증
# ───────────────────────────────
st.set_page_config(page_title="데이터 탐색 및 상관관계", page_icon="🔍", layout="wide")

# 한글 폰트 설정 (Windows/Streamlit Cloud 범용)
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

# ───────────────────────────────
# 🛠️ 2. 분석 메인 함수
# ───────────────────────────────
def run_eda_analysis():
    try:
        gc = get_gcp_client()
        sh = gc.open_by_key(SHEET_ID)
    except Exception as e:
        st.error(f"구글 시트 연결 실패: {e}")
        return

    with st.spinner("시트 데이터를 분석용으로 변환 중..."):
        # 1) Master_Data 로드
        ws_master = sh.worksheet("Master_Data")
        master_df = get_as_dataframe(ws_master).dropna(how='all').dropna(axis=1, how='all')
        # 첫 번째 컬럼을 날짜로 강제 지정 (컬럼명이 Date든 date든 상관없음)
        master_df.iloc[:, 0] = pd.to_datetime(master_df.iloc[:, 0])
        master_df = master_df.set_index(master_df.columns[0])
        
        # 2) gas_price 로드
        ws_gas = sh.worksheet("gas_price")
        gas_df = get_as_dataframe(ws_gas).dropna(how='all').dropna(axis=1, how='all')
        gas_df.iloc[:, 0] = pd.to_datetime(gas_df.iloc[:, 0])
        gas_df = gas_df.set_index(gas_df.columns[0])
        # 가격 컬럼명 통일
        gas_df.columns = ['Wholesale_Price']

    # 3) 데이터 병합 및 ffill 보간
    # TTF 데이터가 있는 구간부터 분석하기 위해 inner join
    merged = master_df.join(gas_df, how='inner')
    # 도매요금 결측치 보간
    merged['Wholesale_Price'] = merged['Wholesale_Price'].fillna(method='ffill')
    # 나머지 수치형 데이터가 없는 행 제거
    merged = merged.dropna()

    st.markdown(f"### 🔍 분석 데이터 요약")
    st.info(f"**{merged.index.min().strftime('%Y-%m')}** ~ **{merged.index.max().strftime('%Y-%m')}** ({len(merged)}개월 데이터)")

    # ───────────────────────────────
    # 📈 3. 상관관계 히트맵 (사이즈 수정됨)
    # ───────────────────────────────
    st.subheader("1️⃣ 지표 간 상관관계 히트맵")
    
    # 🔥 이 부분을 수정했습니다!
    # figsize=(10, 7)에서 figsize=(7, 5)로 크기를 줄였습니다.
    fig_corr, ax_corr = plt.subplots(figsize=(7, 5)) 
    
    sns.heatmap(merged.corr(), annot=True, cmap='RdYlBu', fmt=".2f", ax=ax_corr)
    ax_corr.set_title("국제 지표와 도매요금의 상관계수")
    
    # Streamlit에서 차트를 표시할 때 너비를 자동으로 조절하지 않도록 설정
    # use_container_width=True를 제거했습니다.
    st.pyplot(fig_corr) 

    # ───────────────────────────────
    # ⏳ 4. 시차(Lag) 분석
    # ───────────────────────────────
    st.divider()
    st.subheader("2️⃣ 국제 지표 선행성(Lag) 분석")
    st.caption("국제 가격 변동이 몇 개월 뒤에 우리 요금에 반영될까요?")

    # 분석 대상 컬럼 (Wholesale_Price 제외)
    features = [c for c in merged.columns if c != 'Wholesale_Price']
    max_lag = 6
    
    lag_data = []
    for col in features:
        lags = {}
        for i in range(max_lag + 1):
            corr_val = merged['Wholesale_Price'].corr(merged[col].shift(i))
            lags[i] = corr_val
        lag_data.append(pd.Series(lags, name=col))

    df_lag = pd.concat(lag_data, axis=1)

    # 시차 그래프
    fig_lag, ax_lag = plt.subplots(figsize=(12, 5))
    df_lag.plot(kind='line', marker='o', ax=ax_lag)
    ax_lag.set_title("시차(Lag)별 상관계수 변화")
    ax_lag.set_xlabel("시차 (개월)")
    ax_lag.set_ylabel("상관계수")
    ax_lag.grid(True, alpha=0.3)
    ax_lag.legend(title="지표")
    st.pyplot(fig_lag)

    # 결과 Metric 출력
    st.write("#### 💡 최적 반영 시차 리포트")
    cols = st.columns(len(features))
    for i, col in enumerate(features):
        best_lag = df_lag[col].abs().idxmax() # 절대값 기준 가장 높은 상관관계
        best_corr = df_lag[col][best_lag]
        with cols[i]:
            st.metric(label=f"{col}", value=f"{best_lag}개월 후", delta=f"{best_corr:.2f}")

# ───────────────────────────────
# 🚀 5. 실행부
# ───────────────────────────────
st.markdown("## 📊 데이터 탐색: 상관관계 분석")
if st.button("분석 실행 (데이터 가져오기 및 상관분석)", type="primary", use_container_width=True):
    run_eda_analysis()