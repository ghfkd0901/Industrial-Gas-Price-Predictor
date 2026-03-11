import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import requests, io, os
from datetime import datetime
import yfinance as yf

# 구글 API 관련
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# 페이지 설정
st.set_page_config(page_title="에너지 마스터 데이터 수집", page_icon="📊", layout="wide")

# ───────────────────────────────
# 🔑 1. 구글 서비스 계정 인증
# ───────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets"
    ]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

gc = get_gcp_client()
SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

# ───────────────────────────────
# 📡 2. yfinance 통합 데이터 수집 함수
# ───────────────────────────────

def fetch_yfinance_master(start, end):
    """
    yfinance를 통해 모든 에너지 지표 수집
    - WTI, Brent, HenryHub, JKM, TTF, 환율
    """
    tickers = {
        "CL=F": "WTI",
        "BZ=F": "Brent",       # 브렌트유 선물
        "NG=F": "HenryHub",    # 헨리허브 천연가스 선물
        "JKM=F": "JKM",        # 동북아 LNG (JKM)
        "TTF=F": "TTF",        # 유럽 천연가스 (TTF)
        "KRW=X": "USD_KRW"     # 원/달러 환율
    }
    
    start_dt = pd.to_datetime(start).strftime("%Y-%m-%d")
    # 종료일은 이번 달 데이터를 포함하기 위해 한 달 뒤로 설정
    end_dt = (pd.to_datetime(end) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    
    combined_monthly = pd.DataFrame()

    for ticker, name in tickers.items():
        with st.spinner(f"Yahoo Finance {name} 데이터 로드 중..."):
            try:
                df_daily = yf.download(ticker, start=start_dt, end=end_dt, interval="1d", progress=False)
                
                if not df_daily.empty:
                    # 데이터 구조에 따른 클렌징
                    if "Close" in df_daily.columns:
                        close_series = df_daily["Close"]
                        # 멀티인덱스 방지 (yfinance 버전에 따라 다를 수 있음)
                        if isinstance(close_series, pd.DataFrame):
                            close_series = close_series[ticker] if ticker in close_series.columns else close_series.iloc[:, 0]
                    
                    # 월평균(MS: Month Start) 계산
                    s_monthly = close_series.resample('MS').mean()
                    s_monthly.name = name
                    
                    if combined_monthly.empty:
                        combined_monthly = s_monthly.to_frame()
                    else:
                        combined_monthly = combined_monthly.join(s_monthly, how='outer')
            except Exception as e:
                st.warning(f"{name} 수집 실패: {e}")

    return combined_monthly

# ───────────────────────────────
# 🖥️ 3. 스트림릿 UI 및 메인 로직
# ───────────────────────────────

st.markdown("## 🗃️ 에너지 지표 통합 마스터 (All-in-One)")
st.info(f"연결된 스프레드시트: [Master_Data 탭 열기](https://docs.google.com/spreadsheets/d/{SHEET_ID})")

c1, c2 = st.columns(2)
with c1: start_input = st.text_input("수집 시작 (YYYY-MM)", "2015-01")
with c2: end_input = st.text_input("수집 종료 (YYYY-MM)", datetime.today().strftime("%Y-%m"))

if st.button("🚀 전체 데이터 수집 및 구글 시트 업데이트", type="primary", use_container_width=True):
    try:
        with st.spinner("Yahoo Finance에서 모든 지표를 가져오는 중입니다..."):
            # 1. 신규 데이터 수집
            new_combined = fetch_yfinance_master(start_input, end_input)
            new_combined.index.name = "Date"

            # 2. 구글 시트에서 기존 데이터 로드
            sh = gc.open_by_key(SHEET_ID)
            try:
                ws = sh.worksheet("Master_Data")
                existing_df = get_as_dataframe(ws).dropna(how='all').dropna(axis=1, how='all')
                if not existing_df.empty:
                    existing_df["Date"] = pd.to_datetime(existing_df["Date"])
                    existing_df = existing_df.set_index("Date")
            except:
                ws = sh.add_worksheet(title="Master_Data", rows="1000", cols="20")
                existing_df = pd.DataFrame()

            # 3. 데이터 통합 (중복 제거 및 정렬)
            if not existing_df.empty:
                # 신규 데이터로 기존 데이터 덮어쓰기(업데이트) 및 병합
                final_df = new_combined.combine_first(existing_df).sort_index()
            else:
                final_df = new_combined.sort_index()

            # 4. 저장 준비 (NaN 상태 그대로 유지)
            ws.clear()
            df_save = final_df.reset_index()
            df_save["Date"] = df_save["Date"].dt.strftime('%Y-%m-%d')
            
            set_with_dataframe(ws, df_save)
            
            st.success("✅ Yahoo Finance 통합 업데이트 완료!")
            st.dataframe(final_df.tail(15), use_container_width=True)

    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.exception(e)

st.divider()

# KESIS 도매요금 통합 부분 (수동 업로드용 유지)
st.markdown("### 📥 KESIS 도매요금 파일 통합")
uploaded = st.file_uploader("KESIS 엑셀/CSV 파일 업로드", type=["csv","xlsx"])

if uploaded:
    df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    col1, col2 = st.columns(2)
    d_col = col1.selectbox("날짜 컬럼 선택", df_raw.columns)
    p_col = col2.selectbox("요금 컬럼 선택", df_raw.columns)
    
    if st.button("💾 도매요금 병합 실행"):
        try:
            df_raw[d_col] = pd.to_datetime(df_raw[d_col])
            new_price = df_raw.set_index(d_col)[[p_col]].rename(columns={p_col: "Wholesale_Price"})
            
            sh = gc.open_by_key(SHEET_ID)
            ws = sh.worksheet("Master_Data")
            master_df = get_as_dataframe(ws).dropna(how='all').dropna(axis=1, how='all')
            master_df["Date"] = pd.to_datetime(master_df["Date"])
            master_df = master_df.set_index("Date")
            
            updated_master = master_df.combine_first(new_price).sort_index()
            
            ws.clear()
            df_to_save = updated_master.reset_index()
            df_to_save["Date"] = df_to_save["Date"].dt.strftime('%Y-%m-%d')
            set_with_dataframe(ws, df_to_save)
            
            st.success("✅ 도매요금 병합 완료!")
            st.dataframe(updated_master.tail(5))
        except Exception as e:
            st.error(f"병합 실패: {e}")