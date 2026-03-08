import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

st.set_page_config(page_title="시차(Lag) 최적화 분석 시스템", layout="wide")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def load_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)
    m_df = get_as_dataframe(sh.worksheet("Master_Data")).dropna(how='all').dropna(axis=1, how='all')
    m_df.iloc[:, 0] = pd.to_datetime(m_df.iloc[:, 0])
    m_df = m_df.set_index(m_df.columns[0]).ffill()

    g_df = get_as_dataframe(sh.worksheet("gas_price")).dropna(how='all').dropna(axis=1, how='all')
    g_df.iloc[:, 0] = pd.to_datetime(g_df.iloc[:, 0])
    g_df = g_df.set_index(g_df.columns[0]).ffill()
    g_df.columns = ['Wholesale_Price']

    return m_df.join(g_df, how='inner')

try:
    df = load_data()
    st.title("🧪 원자재 시차(Lag) 최적화 검증 시스템")

    # --- 사이드바 컨트롤 설정 ---
    st.sidebar.header("📊 분석 설정")
    variables = ['Brent', 'TTF', 'USD_KRW']
    selected_vars = st.sidebar.multiselect("비교 지표 선택", variables, default=['Brent', 'TTF'])

    apply_lag = st.sidebar.toggle("최적 시차(Lag) 적용하기", value=False)

    if apply_lag:
        st.sidebar.success("✅ 현재 최적 시차가 적용된 상태입니다.")
    else:
        st.sidebar.warning("⚠️ 현재 시차가 적용되지 않은(Raw) 상태입니다.")

    target = 'Wholesale_Price'
    max_lag = 12

    # 1. 상관계수 및 최적 시차 계산
    lag_results = []
    best_lags = {}

    for var in variables:
        best_corr = -1
        best_lag_val = 0
        for lag in range(max_lag + 1):
            correlation = df[target].corr(df[var].shift(lag))
            lag_results.append({'Variable': var, 'Lag': lag, 'Correlation': correlation})
            if correlation > best_corr:
                best_corr = correlation
                best_lag_val = lag
        best_lags[var] = best_lag_val

    lag_df = pd.DataFrame(lag_results)

    # 2. 메인 화면: 추세 그래프 (3중 축)
    st.subheader("📈 도매요금 vs 원자재 가격 추세 비교")

    fig_line = go.Figure()

    colors      = {'Brent': '#e74c3c', 'TTF': '#2ecc71', 'USD_KRW': '#3498db'}
    yaxis_map   = {'Brent': 'y2',      'TTF': 'y3',      'USD_KRW': 'y4'}
    yaxis_title = {'Brent': 'Brent ($/bbl)', 'TTF': 'TTF (€/MWh)', 'USD_KRW': 'USD/KRW (원)'}

    # 도매요금 (y1, 왼쪽)
    fig_line.add_trace(go.Scatter(
        x=df.index, y=df[target],
        name="도매요금 (Wholesale)",
        line=dict(color='#2c3e50', width=3),
        yaxis="y1"
    ))

    # 원자재 변수 (각각 독립 y축)
    for var in selected_vars:
        current_lag = best_lags[var] if apply_lag else 0
        display_name = f"{var} (Lag {current_lag}M)" if apply_lag else var
        fig_line.add_trace(go.Scatter(
            x=df.index, y=df[var].shift(current_lag),
            name=display_name,
            line=dict(dash='dot', color=colors[var], width=2),
            yaxis=yaxis_map[var]
        ))

    # position은 반드시 0.0 ~ 1.0 사이 고정값
    # xaxis domain [0, 0.70] → 오른쪽 30%에 3개 축 배치
    # y2: 0.72 / y3: 0.84 / y4: 0.96
    XDOMAIN_END = 0.70
    POS_Y2      = 0.72
    POS_Y3      = 0.84
    POS_Y4      = 0.96

    fig_line.update_layout(
        height=580,
        title=dict(
            text="<b>[시차 적용]</b> 실시간 추세 정렬 상태" if apply_lag else "<b>[시차 미적용]</b> 실시간 추세 정렬 상태"
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),

        xaxis=dict(domain=[0, XDOMAIN_END]),

        # y1: 도매요금 (왼쪽)
        yaxis=dict(
            title=dict(text="도매요금 (원)", font=dict(color='#2c3e50')),
            tickfont=dict(color='#2c3e50'),
        ),
        # y2: Brent
        yaxis2=dict(
            title=dict(text=yaxis_title['Brent'], font=dict(color=colors['Brent'])),
            tickfont=dict(color=colors['Brent']),
            anchor="free",
            overlaying="y",
            side="right",
            position=POS_Y2,
            showgrid=False,
        ),
        # y3: TTF
        yaxis3=dict(
            title=dict(text=yaxis_title['TTF'], font=dict(color=colors['TTF'])),
            tickfont=dict(color=colors['TTF']),
            anchor="free",
            overlaying="y",
            side="right",
            position=POS_Y3,
            showgrid=False,
        ),
        # y4: USD/KRW
        yaxis4=dict(
            title=dict(text=yaxis_title['USD_KRW'], font=dict(color=colors['USD_KRW'])),
            tickfont=dict(color=colors['USD_KRW']),
            anchor="free",
            overlaying="y",
            side="right",
            position=POS_Y4,
            showgrid=False,
        ),
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # 3. 데이터 분석 요약 및 히트맵 (2단 구성)
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("💡 최적 시차 리포트")
        for var in selected_vars:
            res = lag_df[lag_df['Variable'] == var].sort_values(by='Correlation', ascending=False).iloc[0]
            st.metric(f"{var}의 최적 시차", f"{int(res['Lag'])}개월", f"상관도: {res['Correlation']:.3f}")

    with col2:
        st.subheader("📊 시차별 상관계수 (Heatmap)")
        pivot_df = lag_df.pivot(index='Variable', columns='Lag', values='Correlation')
        fig_heat = px.imshow(pivot_df, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
        fig_heat.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.info("""
    **💡 보고용 시나리오 가이드:**
    1. 왼쪽 사이드바의 **'시차 적용하기'**를 끕니다.
    2. "보시는 것처럼 유가(Brent)나 가스(TTF)가 급등해도 도매요금은 즉각 반응하지 않습니다."라고 설명합니다.
    3. 토글을 **On**으로 바꿉니다.
    4. "하지만 통계적으로 산출된 **최적 시차**를 적용하면, 원자재의 고점과 저점이 도매요금과 일치하는 것을 볼 수 있습니다."라고 시각적 근거를 제시합니다.
    """)

except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")