import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2 import service_account
from dateutil.relativedelta import relativedelta

st.set_page_config(
    page_title="LNG 가격 예측 시뮬레이터",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"
FORECAST_MONTHS = 4

JCC_A, JCC_B = -1.3852, 1.0667
LNG_A, LNG_B =  1.3673, 0.1317

@st.cache_resource
def get_gc():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)

def ws_to_df(ws):
    vals = ws.get_all_values()
    if not vals: return pd.DataFrame()
    h = vals[0]; rows = vals[1:]
    padded = [r + [""] * (len(h) - len(r)) for r in rows]
    df = pd.DataFrame(padded, columns=h).replace("", np.nan).dropna(how="all")
    return df

@st.cache_data(ttl=600)
def load_master():
    sh  = get_gc().open_by_key(SHEET_ID)
    raw = ws_to_df(sh.worksheet("Master_Data"))
    raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    raw = raw.dropna(subset=[raw.columns[0]]).set_index(raw.columns[0]).sort_index()
    raw.index.name = "date"
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw

def get_val(master, col, date):
    if master.empty or col not in master.columns: return np.nan
    if date in master.index:
        v = master.loc[date, col]
        return float(v) if pd.notna(v) else np.nan
    return np.nan

def build_result(master, oil_col, edited_df):
    rows = []
    for _, r in edited_df.iterrows():
        oil    = float(r["oil"])    if pd.notna(r.get("oil"))    else np.nan
        jcc_fx = float(r["jcc_fix"])if pd.notna(r.get("jcc_fix"))else np.nan
        fx     = float(r["fx"])     if pd.notna(r.get("fx"))     else np.nan

        # JCC: 확정값 우선, 없으면 oil[t-1]로 예측
        jcc_pred = JCC_A + JCC_B * oil if not np.isnan(oil) else np.nan
        jcc_use  = jcc_fx if not np.isnan(jcc_fx) else jcc_pred
        jcc_tag  = "확정" if not np.isnan(jcc_fx) else ("예측" if not np.isnan(jcc_pred) else "-")

        lng_usd = LNG_A + LNG_B * jcc_use if not np.isnan(jcc_use) else np.nan
        lng_krw = lng_usd * fx             if not (np.isnan(lng_usd) or np.isnan(fx)) else np.nan
        lng_mj  = lng_krw / 1055.06        if not np.isnan(lng_krw) else np.nan

        rows.append({
            "예측 월":           r["date"],
            f"{oil_col}[t-1]기준월": r.get("oil_date", ""),
            f"{oil_col}($/배럴)": round(oil, 2)      if not np.isnan(oil)      else None,
            "JCC[t-3]기준월":    r.get("jcc_date", ""),
            "JCC($/배럴)":       round(jcc_use, 4)   if not np.isnan(jcc_use)  else None,
            "JCC구분":           jcc_tag,
            "LNG($/mmbtu)":      round(lng_usd, 5)   if not np.isnan(lng_usd)  else None,
            "LNG(원/mmbtu)":     round(lng_krw, 2)   if not np.isnan(lng_krw)  else None,
            "LNG(원/MJ)":        round(lng_mj, 5)    if not np.isnan(lng_mj)   else None,
            "환율(원/$)":         round(fx, 2)        if not np.isnan(fx)       else None,
        })
    return pd.DataFrame(rows)


def main():
    st.title("🔥 LNG 도입가 예측 시뮬레이터")
    st.caption(
        "가스공사 관계식 기반  ·  "
        "**JCC** = −1.3852 + 1.0667 × oil[−1]  ·  "
        "**LNG** = 1.3673 + 0.1317 × JCC[−3]"
    )

    with st.sidebar:
        st.header("⚙️ 설정")
        oil_col = st.radio("JCC 예측 기준 유가", ["Dubai", "Brent", "WTI"], index=0)
        st.divider()
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            if "result" in st.session_state: del st.session_state["result"]
            st.rerun()

    # ── 데이터 로드 ──────────────────────────────────────
    with st.spinner("📡 데이터 로드 중..."):
        try:
            master = load_master()
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
            master = pd.DataFrame()

    # 마지막 데이터월 감지
    if not master.empty and oil_col in master.columns:
        last_date = master[oil_col].dropna().index.max()
    else:
        last_date = pd.Timestamp("2026-03-01")

    forecast_dates = [last_date + relativedelta(months=i+1) for i in range(FORECAST_MONTHS)]

    # ── 현황 카드 ─────────────────────────────────────────
    st.markdown("#### 📊 최근 데이터 현황")
    disp_pairs = [("WTI","WTI"), ("Dubai","두바이유"), ("Brent","브렌트유"),
                  ("JCC","JCC"), ("USD_KRW","환율(원/$)")]
    c_cards = st.columns(5)
    for i, (col, lbl) in enumerate(disp_pairs):
        if not master.empty and col in master.columns:
            s = master[col].dropna()
            if not s.empty:
                cur = s.iloc[-1]; prev = s.iloc[-2] if len(s) > 1 else cur
                c_cards[i].metric(lbl, f"{cur:,.2f}", f"{cur-prev:+.2f}")
            else:
                c_cards[i].metric(lbl, "-")
        else:
            c_cards[i].metric(lbl, "-")

    st.caption(
        f"📅 **{oil_col}** 기준 최신 데이터: **{last_date.strftime('%Y년 %m월')}**  →  "
        f"예측 대상: **{forecast_dates[0].strftime('%Y.%m')} ~ {forecast_dates[-1].strftime('%Y.%m')}**"
    )

    st.divider()

    # ── 입력 테이블 ───────────────────────────────────────
    st.markdown("#### 📋 시나리오 입력")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.info(f"**{oil_col}[t-1]**: 실제 확정값 자동 입력")
    col_info2.info("**JCC[t-3]**: 실제 확정값 자동 입력 (없으면 유가로 자동 계산)")
    col_info3.info("**환율**: 직접 수정 가능")

    default_rows = []
    for dt in forecast_dates:
        oil_date = dt - relativedelta(months=1)  # oil[t-1]
        jcc_date = dt - relativedelta(months=3)  # JCC[t-3]

        oil_val = get_val(master, oil_col, oil_date)
        jcc_val = get_val(master, "JCC", jcc_date)

        # JCC[t-3]이 없으면 → oil[t-1]로 JCC 예측
        if np.isnan(jcc_val) and not np.isnan(oil_val):
            # JCC 예측에 필요한 oil은 JCC 기준 t-1 → 즉 LNG 기준 t-4
            oil_for_jcc_date = dt - relativedelta(months=4)  # JCC(t-3) 예측용 oil[t-4]
            oil_for_jcc = get_val(master, oil_col, oil_for_jcc_date)
            if not np.isnan(oil_for_jcc):
                jcc_val = JCC_A + JCC_B * oil_for_jcc  # JCC 자동 예측

        fx_val = get_val(master, "USD_KRW", dt)
        if np.isnan(fx_val) and not master.empty and "USD_KRW" in master.columns:
            s = master["USD_KRW"].dropna()
            fx_val = float(s.iloc[-1]) if not s.empty else np.nan

        # oil[t-1] 소스 표시
        oil_source = "확정" if not np.isnan(get_val(master, oil_col, oil_date)) else "-"
        jcc_source = "확정" if not np.isnan(get_val(master, "JCC", jcc_date)) else "예측"

        default_rows.append({
            "date":         dt.strftime("%Y-%m"),
            "oil_date":     oil_date.strftime("%Y-%m"),
            "oil":          round(oil_val, 2) if not np.isnan(oil_val) else None,
            "jcc_date":     jcc_date.strftime("%Y-%m"),
            "jcc_fix":      round(jcc_val, 4) if not np.isnan(jcc_val) else None,
            "jcc_source":   jcc_source,
            "fx":           round(fx_val,  2) if not np.isnan(fx_val)  else None,
        })

    edited = st.data_editor(
        pd.DataFrame(default_rows),
        column_config={
            "date":       st.column_config.TextColumn("📅 예측 월", disabled=True, width="small"),
            "oil_date":   st.column_config.TextColumn(f"{oil_col} 기준월 [t-1]", disabled=True, width="small"),
            "oil":        st.column_config.NumberColumn(f"{oil_col} ($/배럴)", format="%.2f", min_value=0.0,
                                                         help="예측월 1개월 전 유가 (자동 입력, 수정 가능)"),
            "jcc_date":   st.column_config.TextColumn("JCC 기준월 [t-3]", disabled=True, width="small"),
            "jcc_fix":    st.column_config.NumberColumn("JCC ($/배럴)", format="%.4f", min_value=0.0,
                                                         help="확정값 우선 사용, 없으면 유가로 자동 계산"),
            "jcc_source": st.column_config.TextColumn("JCC 구분", disabled=True, width="small"),
            "fx":         st.column_config.NumberColumn("환율 (원/$)", format="%.2f", min_value=0.0),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        height=215,
    )

    st.divider()

    # ── 실행 버튼 ─────────────────────────────────────────
    col_b1, col_b2 = st.columns([1, 5])
    with col_b1:
        run = st.button("🚀 예측 실행", type="primary", use_container_width=True)
    with col_b2:
        if "result" in st.session_state:
            st.caption("✅ 최근 예측 결과가 표시됩니다. 값 수정 후 재실행하세요.")
        else:
            st.caption("👆 입력값 확인 후 예측 실행을 누르세요.")

    if run:
        st.session_state["result"] = build_result(master, oil_col, edited)

    if "result" not in st.session_state:
        st.stop()

    result = st.session_state["result"]

    # ── 결과 메트릭 ───────────────────────────────────────
    st.markdown("#### 📈 예측 결과")
    valid = result.dropna(subset=["LNG(원/MJ)"])
    if not valid.empty:
        m_cols = st.columns(len(valid))
        for i, (_, row) in enumerate(valid.iterrows()):
            m_cols[i].metric(
                label=row["예측 월"],
                value=f"{row['LNG(원/MJ)']:.5f} 원/MJ",
                delta=f"${row['LNG($/mmbtu)']:.5f}/mmbtu",
            )

    st.divider()

    # ── 결과 테이블 ───────────────────────────────────────
    st.markdown("#### 📋 상세 결과")
    lng_cols = ["LNG($/mmbtu)", "LNG(원/mmbtu)", "LNG(원/MJ)"]

    def hl(row):
        return ["background-color:#E6F1FB;font-weight:bold"
                if c in lng_cols else "" for c in row.index]

    fmt = {
        f"{oil_col}($/배럴)": "{:.2f}",
        "JCC($/배럴)":        "{:.4f}",
        "LNG($/mmbtu)":       "{:.5f}",
        "LNG(원/mmbtu)":      "{:,.2f}",
        "LNG(원/MJ)":         "{:.5f}",
        "환율(원/$)":           "{:,.2f}",
    }
    st.dataframe(
        result.style.apply(hl, axis=1).format(fmt, na_rep="-"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "🔵 파란 열: LNG 예측 결과  ·  "
        "JCC 확정값 입력 시 해당 값으로 LNG 계산  ·  "
        "제세 및 기타 부담금 제외된 순수 도입가 기준"
    )

    st.divider()

    # ── 차트 ──────────────────────────────────────────────
    st.markdown("#### 📊 예측 차트")
    tab1, tab2 = st.tabs(["LNG 도입가 추이", "유가 → JCC → LNG 단계별"])

    with tab1:
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Bar(
            x=result["예측 월"], y=result["LNG(원/MJ)"],
            name="LNG (원/MJ)", marker_color="#378ADD",
            text=result["LNG(원/MJ)"].apply(lambda x: f"{x:.5f}" if pd.notna(x) else ""),
            textposition="outside",
        ), secondary_y=False)
        fig1.add_trace(go.Scatter(
            x=result["예측 월"], y=result["LNG($/mmbtu)"],
            name="LNG ($/mmbtu)",
            mode="lines+markers+text",
            line=dict(color="#E24B4A", width=2.5), marker=dict(size=10),
            text=result["LNG($/mmbtu)"].apply(lambda x: f"  ${x:.4f}" if pd.notna(x) else ""),
            textposition="top right",
        ), secondary_y=True)
        fig1.update_layout(
            title=f"LNG 도입가 예측 ({forecast_dates[0].strftime('%Y.%m')} ~ {forecast_dates[-1].strftime('%Y.%m')})",
            template="plotly_white", height=430, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        fig1.update_yaxes(title_text="원/MJ", secondary_y=False)
        fig1.update_yaxes(title_text="$/mmbtu", secondary_y=True)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=(
                f"① {oil_col} 유가 ($/배럴)  [예측월 t-1 기준]",
                "② JCC 예측 ($/배럴)  [예측월 t-3 기준]",
                "③ LNG 도입가 ($/mmbtu)",
            ),
        )
        fig2.add_trace(go.Bar(
            x=result["예측 월"], y=result[f"{oil_col}($/배럴)"],
            name=f"{oil_col}", marker_color="#BA7517",
            text=result[f"{oil_col}($/배럴)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
            textposition="outside",
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=result["예측 월"], y=result["JCC 예측($/배럴)"],
            name="JCC 예측", mode="lines+markers",
            line=dict(color="#1D9E75", width=2.5), marker=dict(size=9),
        ), row=2, col=1)
        if result["JCC 확정값"].notna().any():
            fig2.add_trace(go.Scatter(
                x=result["예측 월"], y=result["JCC 확정값"],
                name="JCC 확정", mode="markers",
                marker=dict(size=13, color="#E24B4A", symbol="star"),
            ), row=2, col=1)
        fig2.add_trace(go.Bar(
            x=result["예측 월"], y=result["LNG($/mmbtu)"],
            name="LNG ($/mmbtu)", marker_color="#378ADD",
            text=result["LNG($/mmbtu)"].apply(lambda x: f"${x:.4f}" if pd.notna(x) else ""),
            textposition="outside",
        ), row=3, col=1)
        fig2.update_layout(
            template="plotly_white", height=680, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── 관계식 ────────────────────────────────────────────
    with st.expander("📐 예측 관계식 및 주의사항"):
        st.markdown(f"""
| 단계 | 관계식 | 적용 시차 |
|------|--------|----------|
| ① JCC 예측 | `JCC(t) = {JCC_A} + {JCC_B} × {oil_col}(t-1)` | 유가 **1개월 전** |
| ② LNG 예측 | `LNG(t) = {LNG_A} + {LNG_B} × JCC(t-3)` | JCC **3개월 전** |
| ③ 원화 환산 | `LNG(원/mmbtu) = LNG($/mmbtu) × 환율(원/$)` | — |
| ④ MJ 환산 | `LNG(원/MJ) = LNG(원/mmbtu) ÷ 1,055.06` | — |

> ※ 본 예측가격은 과거 자료 기반 시계열 분석으로, 시차 반영 시기 및 국제 환경 변화 등으로 오차 발생 가능  
> ※ 제세 및 기타 부담금이 제외된 순수 LNG 도입가 기준  
> ※ 자료 출처: 일본석유연맹, 무역협회, 페트로넷, 한국은행 등
        """)


if __name__ == "__main__":
    main()