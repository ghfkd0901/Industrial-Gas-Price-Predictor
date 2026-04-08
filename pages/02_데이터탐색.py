import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gspread
from google.oauth2 import service_account

# ───────────────────────────────
# 🔑 1. 설정 및 데이터 로드 (캐싱)
# ───────────────────────────────
st.set_page_config(page_title="도매요금 예측 지표 분석", page_icon="📈", layout="wide")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def load_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    # ── Master_Data ──
    ws_master = sh.worksheet("Master_Data")
    raw_master = ws_master.get_all_values()
    master_df = pd.DataFrame(raw_master[1:], columns=raw_master[0])
    master_df = master_df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')

    date_col = master_df.columns[0]
    master_df[date_col] = pd.to_datetime(master_df[date_col], errors='coerce')
    master_df = master_df.dropna(subset=[date_col])
    master_df = master_df.set_index(date_col)

    for col in master_df.columns:
        master_df[col] = pd.to_numeric(
            master_df[col].astype(str).str.replace(',', ''), errors='coerce'
        )

    # ── gas_price ──
    ws_gas = sh.worksheet("gas_price")
    raw_gas = ws_gas.get_all_values()
    gas_df = pd.DataFrame(raw_gas[1:], columns=raw_gas[0])
    gas_df = gas_df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')

    date_col_g = gas_df.columns[0]
    gas_df[date_col_g] = pd.to_datetime(gas_df[date_col_g], errors='coerce')
    gas_df = gas_df.dropna(subset=[date_col_g])
    gas_df = gas_df.set_index(date_col_g)
    gas_df.columns = ['Wholesale_Price']
    gas_df['Wholesale_Price'] = pd.to_numeric(
        gas_df['Wholesale_Price'].astype(str).str.replace(',', ''), errors='coerce'
    )

    # ── 월별 인덱스 정렬 및 병합 ──
    monthly_idx = pd.date_range(
        start=max(master_df.index.min(), gas_df.index.min()),
        end=min(master_df.index.max(), gas_df.index.max()),
        freq='MS'
    )

    master_df = (master_df
                 .reindex(master_df.index.union(monthly_idx))
                 .interpolate(method='time')
                 .reindex(monthly_idx))

    gas_df = (gas_df
              .reindex(gas_df.index.union(monthly_idx))
              .interpolate(method='time')
              .reindex(monthly_idx))

    return master_df.join(gas_df, how='inner').bfill().ffill().dropna()

# ───────────────────────────────
# 🛠️ 2. 최적 래그(Lag) 계산 로직
# ───────────────────────────────
merged = load_data()
features = [c for c in merged.columns if c != 'Wholesale_Price']
max_lag = 6

best_lags = {}
lagged_df = pd.DataFrame(index=merged.index)
lagged_df['Wholesale_Price'] = merged['Wholesale_Price']

for col in features:
    best_corr = -1
    best_lag_val = 0
    for i in range(max_lag + 1):
        current_corr = merged['Wholesale_Price'].corr(merged[col].shift(i))
        if abs(current_corr) > best_corr:
            best_corr = abs(current_corr)
            best_lag_val = i

    best_lags[col] = {
        'Lag': best_lag_val,
        'Corr': merged['Wholesale_Price'].corr(merged[col].shift(best_lag_val))
    }
    lagged_df[f"{col}(Lag:{best_lag_val})"] = merged[col].shift(best_lag_val)

lagged_df = lagged_df.dropna()

# ───────────────────────────────
# 🎨 Plotly 색상 팔레트
# ───────────────────────────────
BLUE = "#1f77b4"
RED  = "#d62728"

axis_colors = [
    "#1f77b4",
    "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf"
]

# ───────────────────────────────
# 🖥️ 3. UI 출력
# ───────────────────────────────
st.markdown("## 🎯 도매요금 예측을 위한 최적 선행 지표 분석")
st.caption(
    f"국제 에너지 지표가 실제 도매요금에 반영되는 최적의 시차(Lag)를 분석합니다. "
    f"| 데이터 기간: {merged.index.min().strftime('%Y-%m')} ~ {merged.index.max().strftime('%Y-%m')} "
    f"({len(merged)}개월)"
)

# ─────────────────────────────────────────
# 1️⃣ 최적 래그 상관계수 표
# ─────────────────────────────────────────
st.subheader("1️⃣ 지표별 최적 래그(Lag) 및 상관계수")

df_lag_res = pd.DataFrame(best_lags).T.sort_values(by='Corr', ascending=False)
df_lag_res.columns = ['최적 시차 (개월)', '상관계수 (R)']

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=df_lag_res.index,
    y=df_lag_res['상관계수 (R)'].abs(),
    text=[
        f"Lag: {int(row['최적 시차 (개월)'])}개월<br>R: {row['상관계수 (R)']:.3f}"
        for _, row in df_lag_res.iterrows()
    ],
    textposition='outside',
    marker=dict(
        color=df_lag_res['상관계수 (R)'].abs(),
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="|R|"),
        line=dict(color='white', width=1.5)
    ),
    hovertemplate="<b>%{x}</b><br>|상관계수|: %{y:.3f}<extra></extra>"
))
fig_bar.update_layout(
    title="지표별 절대 상관계수 (최적 래그 적용)",
    xaxis_title="지표명",
    yaxis_title="|상관계수|",
    yaxis=dict(range=[0, 1.15]),
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=13)
)
fig_bar.update_xaxes(showgrid=False)
fig_bar.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

col_table, col_bar = st.columns([1, 2])
with col_table:
    df_display = df_lag_res.copy()
    df_display['최적 시차 (개월)'] = df_display['최적 시차 (개월)'].astype(int)
    df_display['상관계수 (R)'] = df_display['상관계수 (R)'].round(4)
    st.dataframe(
        df_display.style.background_gradient(cmap='Blues', subset=['상관계수 (R)']),
        use_container_width=True
    )
with col_bar:
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# 2️⃣ 최적 래그 산점도 (Plotly)
# ─────────────────────────────────────────
st.subheader("2️⃣ 최적 래그 지표 vs 도매요금 산점도")
st.info("각 지표를 위에서 찾은 '최적 시차'만큼 뒤로 밀어서 현재 도매요금과 비교한 그래프입니다.")

lagged_cols = [c for c in lagged_df.columns if c != 'Wholesale_Price']

for i in range(0, len(lagged_cols), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(lagged_cols):
            target_col = lagged_cols[i + j]
            with cols[j]:
                x_vals = lagged_df[target_col]
                y_vals = lagged_df['Wholesale_Price']

                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = np.polyval(coeffs, x_line)

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(color=BLUE, opacity=0.5, size=6,
                                line=dict(width=0.5, color='white')),
                    hovertemplate=f"{target_col}: %{{x:.2f}}<br>도매요금: %{{y:.2f}}<extra></extra>"
                ))
                fig_sc.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color=RED, width=2.5),
                    name='회귀선',
                    hoverinfo='skip'
                ))
                fig_sc.update_layout(
                    title=dict(text=f"{target_col}와 도매요금의 관계", font=dict(size=13)),
                    xaxis_title=f"{target_col} (USD/Unit)",
                    yaxis_title="도매요금 (원/MJ)",
                    height=380,
                    showlegend=False,
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=50, r=20, t=50, b=50)
                )
                fig_sc.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
                fig_sc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
                st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# 3️⃣ 통합 시계열 비교 차트 (정규화 단일 Y축)
# ─────────────────────────────────────────
st.subheader("3️⃣ 전 지표 통합 시계열 비교 (최적 시차 적용)")
st.caption(
    "모든 지표를 0~1 정규화하여 하나의 Y축에서 비교합니다. "
    "실제 수치는 아래 테이블에서 확인하세요. 범례 클릭으로 지표를 표시/숨길 수 있습니다."
)

def minmax_norm(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx != mn else series * 0

fig_ts = go.Figure()

fig_ts.add_trace(go.Scatter(
    x=merged.index,
    y=minmax_norm(merged['Wholesale_Price']),
    name='도매요금',
    line=dict(color=axis_colors[0], width=3),
    customdata=merged['Wholesale_Price'].round(2),
    hovertemplate="날짜: %{x|%Y-%m}<br>도매요금: %{customdata} 원/MJ<extra></extra>"
))

for idx, col_name in enumerate(features):
    lag_val = int(best_lags[col_name]['Lag'])
    color   = axis_colors[(idx + 1) % len(axis_colors)]
    shifted = merged[col_name].shift(lag_val)

    fig_ts.add_trace(go.Scatter(
        x=merged.index,
        y=minmax_norm(shifted),
        name=f"{col_name} (Lag {lag_val}M)",
        line=dict(color=color, width=1.8, dash='dash'),
        customdata=shifted.round(2),
        hovertemplate=f"날짜: %{{x|%Y-%m}}<br>{col_name}: %{{customdata}} USD<extra></extra>",
        visible='legendonly' if idx >= 3 else True
    ))

fig_ts.update_layout(
    title=dict(
        text="📊 도매요금 vs 국제 지표 통합 비교 (정규화, 각 지표별 최적 래그 적용)",
        font=dict(size=15)
    ),
    xaxis=dict(
        title="날짜",
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        rangeslider=dict(visible=True),
        type='date'
    ),
    yaxis=dict(
        title="정규화 값 (0 ~ 1)",
        showgrid=True,
        gridcolor='rgba(128,128,128,0.15)',
        range=[-0.05, 1.15]
    ),
    height=550,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                xanchor='left', x=0, font=dict(size=11)),
    plot_bgcolor='rgba(248,249,250,1)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=60, r=40, t=80, b=80)
)

st.plotly_chart(fig_ts, use_container_width=True)

# ── 하단 실제 수치 테이블 ──
with st.expander("📋 실제 수치 데이터 테이블 (클릭하여 펼치기)", expanded=False):
    table_df = pd.DataFrame(index=merged.index)
    table_df['도매요금 (원/MJ)'] = merged['Wholesale_Price'].round(2)
    for col_name in features:
        lag_val = int(best_lags[col_name]['Lag'])
        table_df[f"{col_name} (Lag {lag_val}M, USD)"] = merged[col_name].shift(lag_val).round(2)
    table_df.index = table_df.index.strftime('%Y-%m')
    table_df = table_df.dropna().sort_index(ascending=False)
    st.dataframe(table_df, use_container_width=True, height=350)

# ── 지표별 요약 메트릭 ──
st.markdown("#### 📋 지표별 분석 요약")
summary_cols = st.columns(min(len(features), 4))
for idx, col_name in enumerate(features):
    lag_val  = int(best_lags[col_name]['Lag'])
    corr_val = best_lags[col_name]['Corr']
    with summary_cols[idx % 4]:
        direction = "▲ 양의" if corr_val > 0 else "▼ 음의"
        st.metric(
            label=col_name,
            value=f"Lag {lag_val}개월",
            delta=f"{direction} 상관 | R={corr_val:.3f}"
        )