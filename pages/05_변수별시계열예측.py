import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from dateutil.relativedelta import relativedelta
import gspread
from google.oauth2 import service_account

# ───────────────────────────────
# 상수
# ───────────────────────────────
SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

VAR_CONFIG = {
    "브렌트유 (Brent)": {
        "col":   "Brent",
        "unit":  "$/배럴",
        "color": "#e74c3c",
        "icon":  "🛢️",
    },
    "JKM": {
        "col":   "JKM",
        "unit":  "$/MMBtu",
        "color": "#2980b9",
        "icon":  "🔵",
    },
    "환율 (USD/KRW)": {
        "col":   "USD_KRW",
        "unit":  "원/$",
        "color": "#27ae60",
        "icon":  "💱",
    },
}

# ───────────────────────────────
# GCP / 데이터 로드
# ───────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_data(ttl=600)
def load_master() -> pd.DataFrame:
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    ws = sh.worksheet("Master_Data")
    raw = ws.get_all_values()
    df = pd.DataFrame(raw[1:], columns=raw[0])
    df = df.replace('', np.nan).dropna(how='all').dropna(axis=1, how='all')

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', ''), errors='coerce'
        )

    return df.ffill()


# ───────────────────────────────
# Holt 예측
# ───────────────────────────────
def holt_forecast(series: pd.Series, n: int) -> tuple[np.ndarray, object]:
    s = series.astype(float).dropna()
    try:
        fit = Holt(s, exponential=False).fit(optimized=True)
        vals = fit.forecast(n).values
        vals = np.clip(vals, float(s.min()) * 0.5, float(s.max()) * 2.0)
        return vals, fit
    except Exception:
        fit = SimpleExpSmoothing(s).fit(optimized=True)
        return fit.forecast(n).values, fit


# ───────────────────────────────
# 차트 생성
# ───────────────────────────────
def make_chart(
    series: pd.Series,
    forecast_vals: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    var_name: str,
    cfg: dict,
    hist_start: pd.Timestamp,
) -> go.Figure:
    color      = cfg["color"]
    unit       = cfg["unit"]
    last_date  = series.index[-1]
    last_val   = float(series.iloc[-1])

    hist = series[series.index >= hist_start]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="실제값",
        line=dict(color=color, width=2.5),
        hovertemplate=f"%{{x|%Y-%m}}<br>실제: %{{y:,.2f}} {unit}<extra></extra>",
    ))

    connect_x = [last_date] + list(forecast_dates)
    connect_y = [last_val]  + list(forecast_vals)
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        name="Holt 지수평활 예측",
        line=dict(color=color, width=2.5, dash="dot"),
        hovertemplate=f"%{{x|%Y-%m}}<br>예측: %{{y:,.2f}} {unit}<extra></extra>",
    ))

    upper = np.array(forecast_vals) * 1.10
    lower = np.array(forecast_vals) * 0.90
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
        line=dict(width=0),
        name="±10% 범위",
        hoverinfo="skip",
    ))

    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_dash="dash", line_color="gray", line_width=1.2,
        annotation_text="현재",
        annotation_position="top right",
        annotation_font_color="gray",
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        title=dict(
            text=f"{cfg['icon']} {var_name} — Holt 이중지수평활 예측",
            font=dict(size=16),
        ),
        yaxis_title=unit,
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=40),
    )
    return fig


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════
def main():
    st.title("📈 변수 시계열 예측")
    st.caption("브렌트유 · JKM · 환율의 Holt 이중지수평활법(Double Exponential Smoothing) 예측 결과를 시각화합니다.")

    with st.spinner("📡 데이터 불러오는 중..."):
        try:
            master = load_master()
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
            st.stop()

    last_date = master.index.max()

    with st.sidebar:
        st.header("⚙️ 예측 설정")

        forecast_months = st.slider(
            "예측 개월 수",
            min_value=3, max_value=36, value=12, step=1,
            help="현재 기준 몇 개월 앞까지 예측할지 설정합니다."
        )

        st.divider()

        available_years = sorted(master.index.year.unique())
        hist_start_year = st.select_slider(
            "과거 데이터 표시 시작 연도",
            options=available_years,
            value=max(2015, available_years[0]),
        )
        hist_start = pd.Timestamp(f"{hist_start_year}-01-01")

        st.divider()

        st.subheader("📊 표시 변수 선택")
        show_vars = {
            name: st.checkbox(f"{cfg['icon']} {name}", value=True)
            for name, cfg in VAR_CONFIG.items()
        }

        st.divider()
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"📅 데이터 기준: **{last_date.strftime('%Y-%m')}**")

    forecast_dates = pd.date_range(
        start=last_date + relativedelta(months=1),
        periods=forecast_months,
        freq="MS",
    )

    selected = [name for name, show in show_vars.items() if show]
    if not selected:
        st.warning("⚠️ 사이드바에서 변수를 1개 이상 선택하세요.")
        st.stop()

    summary_rows = []

    for var_name in selected:
        cfg = VAR_CONFIG[var_name]
        col = cfg["col"]

        if col not in master.columns:
            st.warning(f"⚠️ '{col}' 컬럼이 데이터에 없습니다.")
            continue

        series = master[col].dropna()
        forecast_vals, fit = holt_forecast(series, forecast_months)

        fig = make_chart(series, forecast_vals, forecast_dates, var_name, cfg, hist_start)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"🔧 {var_name} — 모델 파라미터 & 예측 수치", expanded=False):
            p_col1, p_col2, p_col3 = st.columns(3)
            try:
                alpha = getattr(fit, "params", {}).get("smoothing_level", None)
                beta  = getattr(fit, "params", {}).get("smoothing_trend", None)
                sse   = getattr(fit, "sse", None)
                p_col1.metric("α (수준 평활)", f"{alpha:.4f}" if alpha is not None else "-")
                p_col2.metric("β (추세 평활)", f"{beta:.4f}"  if beta  is not None else "-")
                p_col3.metric("SSE", f"{sse:,.2f}" if sse is not None else "-")
            except Exception:
                st.caption("파라미터 정보를 불러올 수 없습니다.")

            pred_df = pd.DataFrame({
                "날짜":  forecast_dates.strftime("%Y-%m"),
                f"예측값 ({cfg['unit']})": np.round(forecast_vals, 2),
                f"상단 (+10%)": np.round(forecast_vals * 1.10, 2),
                f"하단 (-10%)": np.round(forecast_vals * 0.90, 2),
            })
            st.dataframe(pred_df, hide_index=True, use_container_width=True)

        summary_rows.append({
            "변수": f"{cfg['icon']} {var_name}",
            "단위": cfg["unit"],
            "현재값": round(float(series.iloc[-1]), 2),
            "1개월 후": round(float(forecast_vals[0]), 2) if len(forecast_vals) > 0 else "-",
            "3개월 후": round(float(forecast_vals[2]), 2) if len(forecast_vals) > 2 else "-",
            "6개월 후": round(float(forecast_vals[5]), 2) if len(forecast_vals) > 5 else "-",
            "12개월 후": round(float(forecast_vals[11]), 2) if len(forecast_vals) > 11 else "-",
        })

        st.divider()

    if summary_rows:
        st.subheader("📋 예측 요약")
        summary_df = pd.DataFrame(summary_rows).set_index("변수")

        show_cols = ["단위", "현재값", "1개월 후"]
        if forecast_months >= 3:  show_cols.append("3개월 후")
        if forecast_months >= 6:  show_cols.append("6개월 후")
        if forecast_months >= 12: show_cols.append("12개월 후")

        st.dataframe(summary_df[show_cols], use_container_width=True)
        st.caption("※ Holt 이중지수평활법 예측값 기준 | ±10% 범위는 단순 참고용입니다.")


if __name__ == "__main__":
    main()