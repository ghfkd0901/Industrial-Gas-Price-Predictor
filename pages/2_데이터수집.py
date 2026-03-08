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

st.set_page_config(page_title="에너지 마스터 데이터 수집", page_icon="📊", layout="wide")

# ───────────────────────────────
# 🔑 1. 구글 서비스 계정 인증
# ───────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    # 읽기/쓰기 권한을 위해 전체 Scope 설정
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

gc = get_gcp_client()
SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

# ───────────────────────────────
# 📡 2. 정밀 데이터 수집 함수 (일 단위 -> 월평균 변환)
# ───────────────────────────────

def fetch_eia_master(api_key, start, end):
    """EIA: Henry Hub & Brent 수집 및 통합"""
    series_map = {
        "RNGWHHD": "HenryHub",
        "RBRTE": "Brent"
    }
    dfs = []
    # EIA는 API 자체에서 Monthly 데이터를 안정적으로 제공하므로 기존 방식 유지하되 결측치 보정
    for facet, name in series_map.items():
        base_url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/" if facet == "RNGWHHD" else "https://api.eia.gov/v2/petroleum/pri/spt/data/"
        url = (f"{base_url}?api_key={api_key}&frequency=monthly&data[0]=value"
               f"&facets[series][]={facet}&start={start}&end={end}&length=500")
        try:
            r = requests.get(url, timeout=20).json()
            rows = r["response"]["data"]
            df = pd.DataFrame([{"Date": row["period"], name: float(row["value"])} for row in rows if row["value"] is not None])
            df["Date"] = pd.to_datetime(df["Date"])
            dfs.append(df.set_index("Date"))
        except Exception as e:
            st.warning(f"EIA {name} 수집 중 경고: {e}")
    
    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

def fetch_yfinance_master(start, end):
    """yfinance: TTF & 환율을 '일 단위'로 받아 '월평균'으로 정밀 계산"""
    tickers = {"TTF=F": "TTF", "KRW=X": "USD_KRW"}
    
    # 넉넉하게 시작일을 하루 앞당김
    start_dt = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_dt = (pd.to_datetime(end) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    
    combined_monthly = pd.DataFrame()

    for ticker, name in tickers.items():
        # '1d' 단위로 수집하여 누락 방지
        df_daily = yf.download(ticker, start=start_dt, end=end_dt, interval="1d", progress=False)
        
        if not df_daily.empty:
            # yfinance 최신 버전 MultiIndex 대응
            if isinstance(df_daily.columns, pd.MultiIndex):
                close_series = df_daily["Close"][ticker]
            else:
                close_series = df_daily["Close"]
            
            # 일 단위 데이터를 '월 단위 평균'으로 변환 (Resampling)
            # 'MS'는 Month Start(그 달의 1일)를 의미함
            s_monthly = close_series.resample('MS').mean()
            s_monthly.name = name
            
            if combined_monthly.empty:
                combined_monthly = s_monthly.to_frame()
            else:
                combined_monthly = combined_monthly.join(s_monthly, how='outer')

    return combined_monthly

# ───────────────────────────────
# 🖥️ 3. 스트림릿 UI 및 메인 로직
# ───────────────────────────────

st.markdown("## 🗃️ 에너지 지표 통합 마스터 수집기")
st.info(f"연결된 스프레드시트: [Master_Data 탭 열기](https://docs.google.com/spreadsheets/d/{SHEET_ID})")

EIA_KEY = st.secrets.get("EIA_API_KEY", "")

c1, c2 = st.columns(2)
with c1: start_input = st.text_input("수집 시작 (YYYY-MM)", "2015-01")
with c2: end_input = st.text_input("수집 종료 (YYYY-MM)", datetime.today().strftime("%Y-%m"))

if st.button("🚀 전체 데이터 통합 및 구글 시트 업데이트", type="primary", use_container_width=True):
    try:
        with st.spinner("EIA & Yahoo Finance 데이터 정밀 수집 중..."):
            # 1. 구글 시트에서 기존 데이터 읽기
            sh = gc.open_by_key(SHEET_ID)
            try:
                ws = sh.worksheet("Master_Data")
                existing_df = get_as_dataframe(ws).dropna(how='all').dropna(axis=1, how='all')
                if not existing_df.empty:
                    existing_df.iloc[:, 0] = pd.to_datetime(existing_df.iloc[:, 0])
                    existing_df = existing_df.set_index(existing_df.columns[0])
            except gspread.WorksheetNotFound:
                existing_df = pd.DataFrame()

            # 2. 신규 데이터 정밀 수집
            df_eia = fetch_eia_master(EIA_KEY, start_input, end_input)
            df_yf = fetch_yfinance_master(start_input, end_input)
            
            # 3. 데이터 가로 병합 (Index: Date 기준)
            new_combined = pd.concat([df_eia, df_yf], axis=1)
            
            # 4. 기존 데이터와 병합 및 중복 제거 (Upsert)
            if not existing_df.empty:
                # 신규 데이터가 기존 데이터의 빈칸을 채우거나 최신화함
                final_df = pd.concat([existing_df, new_combined])
                final_df = final_df[~final_df.index.duplicated(keep="last")].sort_index()
            else:
                final_df = new_combined.sort_index()

            # 5. 구글 시트 저장 (전체 덮어쓰기 방식으로 동기화)
            if 'ws' not in locals():
                try: ws = sh.worksheet("Master_Data")
                except: ws = sh.add_worksheet(title="Master_Data", rows="1000", cols="10")
            
            ws.clear()
            df_save = final_df.reset_index()
            df_save.rename(columns={df_save.columns[0]: "Date"}, inplace=True)
            # 시트 가독성을 위해 날짜 형식 고정
            df_save["Date"] = df_save["Date"].dt.strftime('%Y-%m-%d')
            
            set_with_dataframe(ws, df_save)
            
            st.success("✅ 모든 데이터가 'Master_Data' 시트에 통합되었습니다!")
            st.dataframe(final_df.tail(15).style.format("{:.2f}"), use_container_width=True)

    except Exception as e:
        st.error(f"데이터 수집 중 오류 발생: {e}")

st.divider()

# KESIS 도매요금 업로드 (Master_Data에 합치기)
st.markdown("### 📥 도매요금(KESIS) 파일 통합")
uploaded = st.file_uploader("KESIS 엑셀/CSV 파일 업로드", type=["csv","xlsx"])

if uploaded:
    df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    col1, col2 = st.columns(2)
    d_col = col1.selectbox("날짜 컬럼 선택", df_raw.columns)
    p_col = col2.selectbox("요금 컬럼 선택", df_raw.columns)
    
    if st.button("💾 도매요금 마스터 시트에 병합"):
        try:
            # 업로드 데이터 정리
            df_raw[d_col] = pd.to_datetime(df_raw[d_col])
            new_ws_data = df_raw.set_index(d_col)[[p_col]].rename(columns={p_col: "Wholesale_Price"})
            
            # 기존 마스터 로드 및 병합
            sh = gc.open_by_key(SHEET_ID)
            ws = sh.worksheet("Master_Data")
            master_df = get_as_dataframe(ws).dropna(how='all').dropna(axis=1, how='all')
            master_df.iloc[:, 0] = pd.to_datetime(master_df.iloc[:, 0])
            master_df = master_df.set_index(master_df.columns[0])
            
            # Left Join으로 마스터 데이터에 도매요금 컬럼 추가/업데이트
            updated_master = master_df.combine_first(new_ws_data) # 혹은 pd.concat([master_df, new_ws_data], axis=1)
            # 중복 컬럼 방지 및 정리
            updated_master = updated_master[~updated_master.index.duplicated(keep='last')].sort_index()
            
            # 다시 저장
            ws.clear()
            df_to_save = updated_master.reset_index()
            df_to_save.rename(columns={df_to_save.columns[0]: "Date"}, inplace=True)
            df_to_save["Date"] = df_to_save["Date"].dt.strftime('%Y-%m-%d')
            set_with_dataframe(ws, df_to_save)
            
            st.success("✅ 도매요금이 Master_Data에 성공적으로 병합되었습니다!")
            st.dataframe(updated_master.tail(5))
        except Exception as e:
            st.error(f"병합 실패: {e}")