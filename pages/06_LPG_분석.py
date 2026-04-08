import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

st.set_page_config(page_title="LPG 가격 상관관계 분석기", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

OIL_VARS = {
    "브렌트유 (Brent)": "Brent",
    "두바이유 (Dubai)": "Dubai",
    "WTI":             "WTI",
    "JKM (LNG 현물)":  "JKM",
}

LPG_VARS = {
    "CP 프로판 ($/톤)":    "CP_propane_usd",
    "CP 부탄 ($/톤)":      "CP_butane_usd",
    "MB 프로판 ($/톤)":    "MB_price_usd_propane",
    "MB 부탄 ($/톤)":      "MB_price_usd_butane",
    "SK 수입가격 (원/kg)": "수입가격_SK가",
    "E1 수입가격 (원/kg)": "수입가격_E1_원",
}

@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive","https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def ws_to_df(ws):
    """get_all_values()로 워크시트를 안전하게 DataFrame으로 변환.
    - 마지막 열 누락 버그(get_as_dataframe) 완전 우회
    - 중간 빈 열(TTF 등)도 보존
    - 행마다 길이 다를 때 헤더 길이 기준으로 패딩
    """
    all_vals = ws.get_all_values()
    if not all_vals:
        return pd.DataFrame()
    headers = all_vals[0]
    rows    = all_vals[1:]
    padded  = [r + [""] * (len(headers) - len(r)) for r in rows]
    df = pd.DataFrame(padded, columns=headers)
    df = df.replace("", np.nan).dropna(how="all")
    return df

@st.cache_data(ttl=600)
def load_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    # LPG 시트
    lpg_raw = ws_to_df(sh.worksheet("LPG"))
    lpg_raw.iloc[:, 0] = pd.to_datetime(lpg_raw.iloc[:, 0], errors="coerce")
    lpg_raw = lpg_raw.dropna(subset=[lpg_raw.columns[0]])
    lpg_raw = lpg_raw.set_index(lpg_raw.columns[0]).sort_index()
    lpg_raw.index.name = "date"
    for col in lpg_raw.columns:
        lpg_raw[col] = pd.to_numeric(lpg_raw[col], errors="coerce")

    # Master_Data 시트
    master_raw = ws_to_df(sh.worksheet("Master_Data"))
    master_raw.iloc[:, 0] = pd.to_datetime(master_raw.iloc[:, 0], errors="coerce")
    master_raw = master_raw.dropna(subset=[master_raw.columns[0]])
    master_raw = master_raw.set_index(master_raw.columns[0]).sort_index()
    master_raw.index.name = "date"
    for col in master_raw.columns:
        master_raw[col] = pd.to_numeric(master_raw[col], errors="coerce")

    merged = master_raw.join(lpg_raw, how="inner", rsuffix="_lpg")
    merged = merged.ffill()
    return merged, lpg_raw, master_raw

def calc_lag_corr(df, oil_col, lpg_col, max_lag):
    rows = []
    for lag in range(0, max_lag + 1):
        oil_lagged = df[oil_col].shift(lag)
        valid = df[[lpg_col]].join(oil_lagged.rename("oil_lagged")).dropna()
        corr = valid[lpg_col].corr(valid["oil_lagged"]) if len(valid) >= 10 else np.nan
        rows.append({"Lag (개월)": lag, "상관계수": round(corr, 4) if not np.isnan(corr) else np.nan, "샘플수": len(valid)})
    return pd.DataFrame(rows)

def find_best_lag(corr_df):
    best_row = corr_df.loc[corr_df["상관계수"].abs().idxmax()]
    return {"lag": int(best_row["Lag (개월)"]), "corr": float(best_row["상관계수"]), "samples": int(best_row["샘플수"])}

def plot_lag_bar(corr_df, oil_name, lpg_name, best_lag):
    colors = ["#E24B4A" if r["Lag (개월)"] == best_lag else ("#378ADD" if r["상관계수"] >= 0 else "#BA7517") for _, r in corr_df.iterrows()]
    fig = go.Figure(go.Bar(x=corr_df["Lag (개월)"], y=corr_df["상관계수"], marker_color=colors,
        text=corr_df["상관계수"].apply(lambda x: f"{x:.3f}"), textposition="outside",
        hovertemplate="Lag %{x}개월<br>상관계수: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=f"{oil_name} → {lpg_name} | Lag별 상관계수",
        xaxis_title="Lag (개월)", yaxis_title="Pearson 상관계수",
        yaxis=dict(range=[-1.05, 1.05]), xaxis=dict(tickmode="linear", dtick=1),
        template="plotly_white", height=380, margin=dict(t=50, b=40))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    best_val = corr_df.loc[corr_df["Lag (개월)"] == best_lag, "상관계수"].values
    if len(best_val):
        fig.add_annotation(x=best_lag, y=best_val[0], text=f"  최적 Lag: {best_lag}개월",
            showarrow=True, arrowhead=2, arrowcolor="#E24B4A", font=dict(color="#E24B4A", size=12), yshift=15)
    return fig

def plot_timeseries(df, oil_col, lpg_col, best_lag, oil_name, lpg_name, show_lag1=True):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # LPG 가격 (좌축)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[lpg_col], name=lpg_name,
        line=dict(color="#378ADD", width=2.5),
        hovertemplate="%{x|%Y-%m}<br>" + lpg_name + ": %{y:.2f}<extra></extra>",
    ), secondary_y=False)

    # 최적 Lag (우축, 빨간 점선)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[oil_col].shift(best_lag),
        name=f"{oil_name} — 최적 Lag {best_lag}M",
        line=dict(color="#E24B4A", width=2, dash="dot"),
        hovertemplate="%{x|%Y-%m}<br>" + f"Lag {best_lag}M: %{{y:.2f}}<extra></extra>",
    ), secondary_y=True)

    # Lag 1 비교선 (우축, 초록 점선) — 최적 Lag가 1이 아닐 때만 추가
    if show_lag1 and best_lag != 1:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[oil_col].shift(1),
            name=f"{oil_name} — 이론 Lag 1M",
            line=dict(color="#1D9E75", width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m}<br>Lag 1M: %{y:.2f}<extra></extra>",
        ), secondary_y=True)

    title_suffix = f"최적 Lag {best_lag}M" + (" vs 이론 Lag 1M" if show_lag1 and best_lag != 1 else "")
    fig.update_layout(
        title=f"{lpg_name} vs {oil_name} ({title_suffix})",
        template="plotly_white", height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=55, b=40),
    )
    fig.update_yaxes(title_text=lpg_name, secondary_y=False)
    fig.update_yaxes(title_text=f"{oil_name} ($/배럴)", secondary_y=True)
    return fig

def plot_heatmap(df, oil_cols, lpg_cols, oil_labels, lpg_labels, lag):
    matrix = []
    for o_col in oil_cols:
        row = []
        for l_col in lpg_cols:
            if o_col not in df.columns or l_col not in df.columns:
                row.append(np.nan); continue
            valid = df[[l_col]].join(df[o_col].shift(lag).rename("oil")).dropna()
            row.append(round(valid[l_col].corr(valid["oil"]), 3) if len(valid) >= 10 else np.nan)
        matrix.append(row)
    fig = go.Figure(go.Heatmap(z=matrix, x=lpg_labels, y=oil_labels, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.3f}" if not np.isnan(v) else "N/A" for v in r] for r in matrix],
        texttemplate="%{text}", textfont=dict(size=13),
        hovertemplate="유가: %{y}<br>LPG: %{x}<br>상관계수: %{z:.3f}<extra></extra>",
        colorbar=dict(title="상관계수")))
    fig.update_layout(title=f"유가 × LPG 상관관계 히트맵 (Lag {lag}개월)",
        template="plotly_white", height=320, margin=dict(t=50, b=40))
    return fig

def build_summary_table(df, oil_cols, lpg_cols, oil_labels, lpg_labels, max_lag):
    rows = []
    for o_col, o_lbl in zip(oil_cols, oil_labels):
        for l_col, l_lbl in zip(lpg_cols, lpg_labels):
            if o_col not in df.columns or l_col not in df.columns: continue
            corr_df = calc_lag_corr(df, o_col, l_col, max_lag)
            best    = find_best_lag(corr_df)

            # Lag 0, 1 상관계수 개별 추출
            lag0 = corr_df.loc[corr_df["Lag (개월)"] == 0, "상관계수"].values
            lag1 = corr_df.loc[corr_df["Lag (개월)"] == 1, "상관계수"].values
            lag0_val = float(lag0[0]) if len(lag0) else np.nan
            lag1_val = float(lag1[0]) if len(lag1) else np.nan
            diff     = round(abs(lag1_val) - abs(lag0_val), 4) if not np.isnan(lag0_val) and not np.isnan(lag1_val) else np.nan
            # 이론 Lag(1개월) 채택 여부: |Lag1 - Lag0| < 0.01 이면 "차이 없음 → Lag1 채택"
            verdict = "✅ Lag1 채택" if (not np.isnan(diff) and abs(diff) < 0.01) else ("Lag1 우위" if (not np.isnan(diff) and diff > 0) else "Lag0 우위")

            rows.append({
                "유가 변수":       o_lbl,
                "LPG 변수":       l_lbl,
                "최적 Lag (개월)": best["lag"],
                "최대 상관계수":   best["corr"],
                "Lag 0 상관계수":  round(lag0_val, 4),
                "Lag 1 상관계수":  round(lag1_val, 4),
                "Lag1-Lag0 차이":  diff,
                "이론 Lag 판정":   verdict,
                "절대값":          abs(best["corr"]),
                "샘플수":          best["samples"],
            })
    return pd.DataFrame(rows).sort_values("절대값", ascending=False).drop(columns=["절대값"])

def main():
    st.title("🔍 LPG 가격 상관관계 분석기")
    st.caption("브렌트유 / 두바이유 / WTI / JKM → CP · MB · 수입가격 | 최적 Lag 자동 탐색")

    with st.spinner("📡 Google Sheets 데이터 불러오는 중..."):
        try:
            merged, lpg_raw, master_raw = load_data()
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
            st.stop()

    # ── 디버그 섹션 ──────────────────────────────────────
    with st.expander("🔧 데이터 로드 디버그 (컬럼 확인)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Master_Data 컬럼 목록**")
            st.write(list(master_raw.columns))
            st.caption(f"shape: {master_raw.shape}")
            if "Dubai" in master_raw.columns:
                st.success(f"✅ Dubai 컬럼 정상 로드 | 유효값: {master_raw['Dubai'].notna().sum()}개월 | 최근값: {master_raw['Dubai'].dropna().iloc[-1]:.2f}")
            else:
                st.error("❌ Dubai 컬럼 없음 — 구글 시트 H열 컬럼명이 정확히 'Dubai'인지 확인하세요")
        with c2:
            st.markdown("**Master_Data 최근 5행**")
            st.dataframe(master_raw.tail(5), use_container_width=True)

    available_oil = [c for c in OIL_VARS.values() if c in merged.columns]
    available_lpg = [c for c in LPG_VARS.values() if c in merged.columns]

    if not available_oil:
        st.error("❌ 분석 가능한 유가 컬럼이 없습니다."); st.stop()
    if not available_lpg:
        st.error("❌ 분석 가능한 LPG 컬럼이 없습니다."); st.stop()

    oil_label_map = {v: k for k, v in OIL_VARS.items() if v in available_oil}
    lpg_label_map = {v: k for k, v in LPG_VARS.items() if v in available_lpg}

    with st.sidebar:
        st.header("⚙️ 분석 설정")
        sel_oil_labels = st.multiselect("🛢️ 유가 변수", [oil_label_map[c] for c in available_oil], default=[oil_label_map[c] for c in available_oil])
        sel_oil_cols   = [OIL_VARS[l] for l in sel_oil_labels]
        sel_lpg_labels = st.multiselect("⛽ LPG 변수", [lpg_label_map[c] for c in available_lpg], default=[lpg_label_map[c] for c in available_lpg])
        sel_lpg_cols   = [LPG_VARS[l] for l in sel_lpg_labels]
        max_lag = st.slider("최대 Lag (개월)", 3, 24, 12)
        years = sorted(merged.index.year.unique())
        start_y, end_y = st.select_slider("분석 기간 (연도)", options=years, value=(years[0], years[-1]))
        st.divider()
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        st.caption(f"📅 {merged.index.min().strftime('%Y-%m')} ~ {merged.index.max().strftime('%Y-%m')}")

    if not sel_oil_cols or not sel_lpg_cols:
        st.warning("사이드바에서 변수를 선택하세요."); st.stop()

    df = merged[(merged.index >= pd.Timestamp(f"{start_y}-01-01")) & (merged.index <= pd.Timestamp(f"{end_y}-12-31"))].copy()

    tab1, tab2, tab3 = st.tabs(["📊 전체 요약 (최적 Lag)", "🔬 개별 상세 분석", "🗺️ 히트맵 (Lag별)"])

    with tab1:
        st.subheader("📊 유가 × LPG 최적 Lag 요약")
        with st.spinner("상관관계 계산 중..."):
            summary_df = build_summary_table(df, sel_oil_cols, sel_lpg_cols, sel_oil_labels, sel_lpg_labels, max_lag)

        def color_corr(val):
            if isinstance(val, float):
                if val >= 0.9: return "background-color:#1a7a1a;color:white"
                if val >= 0.7: return "background-color:#5cb85c;color:white"
                if val >= 0.5: return "background-color:#f0ad4e"
                if val <= -0.7: return "background-color:#d9534f;color:white"
            return ""

        def color_diff(val):
            if isinstance(val, float):
                if abs(val) < 0.01: return "background-color:#1a7a1a;color:white"  # 거의 동일 → Lag1 채택
                if val > 0:         return "background-color:#5cb85c;color:white"   # Lag1 우위
                return "background-color:#f0ad4e"                                   # Lag0 우위
            return ""

        def color_verdict(val):
            if "✅" in str(val): return "background-color:#1a7a1a;color:white;font-weight:bold"
            if "Lag1 우위" in str(val): return "background-color:#5cb85c;color:white"
            return ""

        styled = (summary_df.style
            .applymap(color_corr,   subset=["최대 상관계수", "Lag 0 상관계수", "Lag 1 상관계수"])
            .applymap(color_diff,   subset=["Lag1-Lag0 차이"])
            .applymap(color_verdict, subset=["이론 Lag 판정"])
            .format({"최대 상관계수": "{:.4f}", "Lag 0 상관계수": "{:.4f}",
                     "Lag 1 상관계수": "{:.4f}", "Lag1-Lag0 차이": "{:+.4f}"})
            .background_gradient(subset=["최적 Lag (개월)"], cmap="YlOrRd"))
        st.dataframe(styled, use_container_width=True, height=400)
        st.caption("🟢 상관계수 ≥0.9  🟡 0.7~0.9  🟠 0.5~0.7  |  이론 Lag 판정: |Lag1-Lag0| < 0.01 이면 ✅ Lag1 채택")

        if not summary_df.empty:
            b = summary_df.iloc[0]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🏆 최고 상관 유가", b["유가 변수"])
            c2.metric("🏆 최고 상관 LPG", b["LPG 변수"])
            c3.metric("⏱️ 최적 Lag", f"{int(b['최적 Lag (개월)'])}개월")
            c4.metric("📈 최대 상관계수", f"{b['최대 상관계수']:.4f}")

    with tab2:
        st.subheader("🔬 개별 조합 상세 분석")
        ca, cb = st.columns(2)
        detail_oil_lbl = ca.selectbox("유가 변수", sel_oil_labels, key="d_oil")
        detail_lpg_lbl = cb.selectbox("LPG 변수", sel_lpg_labels, key="d_lpg")
        o_col = OIL_VARS[detail_oil_lbl]
        l_col = LPG_VARS[detail_lpg_lbl]

        if o_col not in df.columns:
            st.error(f"'{o_col}' 컬럼이 없습니다. 디버그 섹션에서 컬럼명을 확인하세요.")
        elif l_col not in df.columns:
            st.error(f"'{l_col}' 컬럼이 없습니다. LPG 시트 컬럼명을 확인하세요.")
        else:
            corr_df = calc_lag_corr(df, o_col, l_col, max_lag)
            best    = find_best_lag(corr_df)

            lag0_val = corr_df.loc[corr_df["Lag (개월)"] == 0, "상관계수"].values
            lag1_val = corr_df.loc[corr_df["Lag (개월)"] == 1, "상관계수"].values
            lag0_str = f"{lag0_val[0]:.4f}" if len(lag0_val) else "N/A"
            lag1_str = f"{lag1_val[0]:.4f}" if len(lag1_val) else "N/A"
            diff_str = f"{abs(lag1_val[0]) - abs(lag0_val[0]):+.4f}" if len(lag0_val) and len(lag1_val) else "N/A"

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("최적 Lag",       f"{best['lag']}개월")
            m2.metric("최대 상관계수",   f"{best['corr']:.4f}")
            m3.metric("Lag 0 상관계수",  lag0_str)
            m4.metric("Lag 1 상관계수",  lag1_str, delta=diff_str, delta_color="normal")
            m5.metric("유효 샘플수",     f"{best['samples']}개월")

            # 바 차트 — Lag1도 별도 색으로 강조
            colors = []
            for _, r in corr_df.iterrows():
                lag = r["Lag (개월)"]
                if lag == best["lag"]:      colors.append("#E24B4A")   # 최적 Lag: 빨강
                elif lag == 1:              colors.append("#1D9E75")   # 이론 Lag1: 초록
                elif r["상관계수"] >= 0:   colors.append("#378ADD")
                else:                       colors.append("#BA7517")

            fig_bar = go.Figure(go.Bar(
                x=corr_df["Lag (개월)"], y=corr_df["상관계수"],
                marker_color=colors,
                text=corr_df["상관계수"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
                hovertemplate="Lag %{x}개월<br>상관계수: %{y:.4f}<extra></extra>",
            ))
            fig_bar.update_layout(
                title=f"{detail_oil_lbl} → {detail_lpg_lbl} | Lag별 상관계수",
                xaxis_title="Lag (개월)", yaxis_title="Pearson 상관계수",
                yaxis=dict(range=[-1.05, 1.05]), xaxis=dict(tickmode="linear", dtick=1),
                template="plotly_white", height=380, margin=dict(t=50, b=40),
            )
            fig_bar.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            # 최적 Lag 어노테이션
            best_y = corr_df.loc[corr_df["Lag (개월)"] == best["lag"], "상관계수"].values
            if len(best_y):
                fig_bar.add_annotation(x=best["lag"], y=best_y[0],
                    text=f"  최적 Lag: {best['lag']}개월",
                    showarrow=True, arrowhead=2, arrowcolor="#E24B4A",
                    font=dict(color="#E24B4A", size=11), yshift=15)
            # 이론 Lag1 어노테이션 (최적과 다를 때만)
            if best["lag"] != 1 and len(lag1_val):
                fig_bar.add_annotation(x=1, y=lag1_val[0],
                    text="  이론 Lag 1M",
                    showarrow=True, arrowhead=2, arrowcolor="#1D9E75",
                    font=dict(color="#1D9E75", size=11), yshift=-25)

            st.plotly_chart(fig_bar, use_container_width=True)
            st.plotly_chart(
                plot_timeseries(df, o_col, l_col, best["lag"], detail_oil_lbl, detail_lpg_lbl, show_lag1=True),
                use_container_width=True,
            )

            with st.expander("📋 Lag별 상관계수 전체 테이블"):
                def highlight_lag(row):
                    if row["Lag (개월)"] == best["lag"]: return ["font-weight:bold;border:2px solid #E24B4A"] * len(row)
                    if row["Lag (개월)"] == 1:           return ["font-weight:bold;border:2px solid #1D9E75"] * len(row)
                    return [""] * len(row)
                st.dataframe(corr_df.style
                    .background_gradient(subset=["상관계수"], cmap="RdBu", vmin=-1, vmax=1)
                    .format({"상관계수": "{:.4f}"})
                    .apply(highlight_lag, axis=1),
                    use_container_width=True, hide_index=True)
            st.caption("🔴 최적 Lag  🟢 이론 Lag 1개월")

    with tab3:
        st.subheader("🗺️ Lag별 상관관계 히트맵")
        heatmap_lag = st.slider("Lag 선택 (개월)", 0, max_lag, 0, key="hm_lag")
        st.plotly_chart(plot_heatmap(df, sel_oil_cols, sel_lpg_cols, sel_oil_labels, sel_lpg_labels, heatmap_lag), use_container_width=True)

        st.divider()
        st.subheader("📈 Lag 변화에 따른 상관계수 추이 (전체 조합)")
        fig_line = go.Figure()
        for o_col, o_lbl in zip(sel_oil_cols, sel_oil_labels):
            for l_col, l_lbl in zip(sel_lpg_cols, sel_lpg_labels):
                if o_col not in df.columns or l_col not in df.columns: continue
                corr_df = calc_lag_corr(df, o_col, l_col, max_lag)
                fig_line.add_trace(go.Scatter(x=corr_df["Lag (개월)"], y=corr_df["상관계수"],
                    name=f"{o_lbl} → {l_lbl}", mode="lines+markers", marker=dict(size=5),
                    hovertemplate=f"{o_lbl} → {l_lbl}<br>Lag %{{x}}개월: %{{y:.4f}}<extra></extra>"))
        fig_line.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_line.update_layout(xaxis_title="Lag (개월)", yaxis_title="Pearson 상관계수",
            yaxis=dict(range=[-1.05, 1.05]), xaxis=dict(tickmode="linear", dtick=1),
            template="plotly_white", height=480, hovermode="x unified",
            legend=dict(orientation="v", x=1.01, y=1), margin=dict(t=30, b=40, r=220))
        st.plotly_chart(fig_line, use_container_width=True)

if __name__ == "__main__":
    main()