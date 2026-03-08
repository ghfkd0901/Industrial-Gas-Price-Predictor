import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe
import statsmodels.api as sm  # R2 계산을 위해 필요

# ───────────────────────────────
# 🔑 1. 설정 및 구글 인증
# ───────────────────────────────
st.set_page_config(page_title="에너지 쇼크 R2 비교 분석", page_icon="📊", layout="wide")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

# R2 계산 함수
def get_r2(df, x_col, y_col):
    if df.empty or len(df) < 2: return 0
    X = sm.add_constant(df[x_col])
    model = sm.OLS(df[y_col], X).fit()
    return model.rsquared

# ───────────────────────────────
# 🛠️ 2. 분석 메인 함수
# ───────────────────────────────
def run_scatter_analysis():
    try:
        gc = get_gcp_client()
        sh = gc.open_by_key(SHEET_ID)
    except Exception as e:
        st.error(f"구글 시트 연결 실패: {e}")
        return

    # 1. 데이터 로드 및 전처리
    ws_master = sh.worksheet("Master_Data")
    master_df = get_as_dataframe(ws_master).dropna(how='all').dropna(axis=1, how='all')
    master_df.iloc[:, 0] = pd.to_datetime(master_df.iloc[:, 0])
    master_df = master_df.set_index(master_df.columns[0])
    
    ws_gas = sh.worksheet("gas_price")
    gas_df = get_as_dataframe(ws_gas).dropna(how='all').dropna(axis=1, how='all')
    gas_df.iloc[:, 0] = pd.to_datetime(gas_df.iloc[:, 0])
    gas_df = gas_df.set_index(gas_df.columns[0])
    gas_df.columns = ['Wholesale_Price']

    merged = master_df.join(gas_df, how='inner')
    merged['Wholesale_Price'] = merged['Wholesale_Price'].ffill()
    merged['Brent_Lag6'] = merged['Brent'].shift(6)
    merged['TTF_Lag3'] = merged['TTF'].shift(3)
    
    all_df = merged.dropna(subset=['Brent_Lag6', 'TTF_Lag3', 'Wholesale_Price']).copy()
    
    # 쇼크 시기 정의
    shock_start = pd.to_datetime('2022-02-01')
    shock_end = pd.to_datetime('2023-03-31')
    all_df['Is_Shock'] = all_df.index.map(lambda x: shock_start <= x <= shock_end)
    all_df['Period'] = all_df['Is_Shock'].map({True: '쇼크기 (22.02~23.03)', False: '일반 기간'})
    all_df['Date_Str'] = all_df.index.strftime('%Y-%m-%d')

    # ───────────────────────────────
    # ⚙️ 상단 컨트롤러 (옵션)
    # ───────────────────────────────
    st.title("📊 에너지 쇼크 제외 시 모델 설명력($R^2$) 비교")
    
    exclude_shock = st.checkbox("🚫 분석에서 에너지 쇼크 시기(22.02 ~ 23.03) 데이터 제외하기", value=False)
    
    # 데이터 필터링 적용
    if exclude_shock:
        plot_df = all_df[~all_df['Is_Shock']].copy()
        st.warning("⚠️ 현재 쇼크 시기 데이터를 제외한 평시 데이터로만 분석 중입니다.")
    else:
        plot_df = all_df.copy()
        st.info("ℹ️ 현재 전체 기간 데이터를 포함하여 분석 중입니다.")

    # R2 계산
    r2_brent = get_r2(plot_df, 'Brent_Lag6', 'Wholesale_Price')
    r2_ttf = get_r2(plot_df, 'TTF_Lag3', 'Wholesale_Price')

    col1, col2 = st.columns(2)
    color_map = {'쇼크기 (22.02~23.03)': '#FF7F0E', '일반 기간': '#1f77b4'}

    # 1) Brent 분석
    with col1:
        st.subheader("⛽ 브렌트유 (6M 시차) vs 도매요금")
        fig1 = px.scatter(
            plot_df, x='Brent_Lag6', y='Wholesale_Price', color='Period',
            hover_data={'Date_Str': True, 'Brent_Lag6': ':.2f', 'Wholesale_Price': ':.2f', 'Period': False},
            trendline="ols",
            color_discrete_map=color_map,
            labels={'Brent_Lag6': 'Brent ($/bbl)', 'Wholesale_Price': '도매요금 (원/MJ)'},
            template="plotly_white", height=600
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.metric(f"결정계수 ($R^2$)", f"{r2_brent:.4f}")

    # 2) TTF 분석
    with col2:
        st.subheader("🇪🇺 TTF 가스 (3M 시차) vs 도매요금")
        fig2 = px.scatter(
            plot_df, x='TTF_Lag3', y='Wholesale_Price', color='Period',
            hover_data={'Date_Str': True, 'TTF_Lag3': ':.2f', 'Wholesale_Price': ':.2f', 'Period': False},
            trendline="ols",
            color_discrete_map=color_map,
            labels={'TTF_Lag3': 'TTF (EUR/MWh)', 'Wholesale_Price': '도매요금 (원/MJ)'},
            template="plotly_white", height=600
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.metric(f"결정계수 ($R^2$)", f"{r2_ttf:.4f}")

    st.divider()
    st.write("💡 **Tip:** 쇼크 시기를 제외했을 때 $R^2$ 값이 크게 올라간다면, 해당 시기가 기존 요금 결정 모델을 벗어난 이상치(Outlier)였다는 과학적 근거가 됩니다.")

# 실행
run_scatter_analysis()