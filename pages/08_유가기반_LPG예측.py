import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gspread
from google.oauth2 import service_account
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title="LPG 가격 예측 시뮬레이터", page_icon="🔮", layout="wide")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

OIL_VARS = {"브렌트유": "Brent", "두바이유": "Dubai", "WTI": "WTI", "JKM (LNG)": "JKM"}
LPG_VARS = {
    "CP 프로판 ($/톤)":    "CP_propane_usd",
    "CP 부탄 ($/톤)":      "CP_butane_usd",
    "SK 수입가격 (원/kg)": "수입가격_SK가스_원kg",
    "E1 수입가격 (원/kg)": "수입가격_E1_원kg",
}

# 모델 정의 ─ 키: 표시명 / 값: (인스턴스, 색상)
MODELS = {
    "RandomForest":     (RandomForestRegressor(n_estimators=200, max_depth=5,
                                               min_samples_leaf=2, random_state=42),
                         "#2ECC71"),
    "GradientBoosting": (GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                    learning_rate=0.05, subsample=0.8,
                                                    random_state=42),
                         "#E74C3C"),
    "LinearRegression": (LinearRegression(), "#95A5A6"),   # 베이스라인 비교용
}

# ── GCP / 데이터 로드 ─────────────────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


def ws_to_df(ws):
    all_vals = ws.get_all_values()
    if not all_vals:
        return pd.DataFrame()
    headers = all_vals[0]
    padded = [r + [""] * (len(headers) - len(r)) for r in all_vals[1:]]
    df = pd.DataFrame(padded, columns=headers)
    return df.replace("", np.nan).dropna(how="all")


@st.cache_data(ttl=600)
def load_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    def clean_sheet(name):
        raw = ws_to_df(sh.worksheet(name))
        raw.columns = raw.columns.str.strip()          # 컬럼명 앞뒤 공백 제거
        raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
        raw = raw.dropna(subset=[raw.columns[0]]).set_index(raw.columns[0]).sort_index()
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        return raw

    lpg    = clean_sheet("LPG")
    master = clean_sheet("Master_Data")
    merged = master.join(lpg, how="inner", rsuffix="_lpg").ffill()
    return merged


def resolve_col(df, candidates):
    """후보 컬럼명 리스트 중 실제 df에 존재하는 첫 번째 컬럼명 반환"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── 피처 생성 ─────────────────────────────────────────────────
def build_features(df: pd.DataFrame, oil_col: str, lpg_col: str, usd_col):
    """
    Features:
      oil_lag1  : 1개월 전 유가 (핵심)
      oil_lag2  : 2개월 전 유가
      oil_chg1  : 전월 대비 유가 변화율
      usd_krw   : 환율 (컬럼 존재 시)
    """
    data = df[[oil_col, lpg_col]].copy()
    if usd_col:
        data["usd_krw"] = df[usd_col]

    data["oil_lag1"] = data[oil_col].shift(1)
    data["oil_lag2"] = data[oil_col].shift(2)
    data["oil_chg1"] = data[oil_col].pct_change(1)

    feat_cols = ["oil_lag1", "oil_lag2", "oil_chg1"]
    if usd_col:
        feat_cols.append("usd_krw")

    train_df = data.dropna(subset=feat_cols + [lpg_col]).copy()
    return train_df, feat_cols


# ── 전체 모델 학습 ────────────────────────────────────────────
@st.cache_data(ttl=600)
def train_all_models(_df, oil_col: str, lpg_col: str):
    df = _df

    # 컬럼 존재 여부 사전 검증
    if oil_col not in df.columns:
        avail = [c for c in df.columns if any(k in c.upper() for k in ["BRENT","DUBAI","WTI","JKM"])]
        raise ValueError(f"유가 컬럼 '{oil_col}' 없음. 실제 컬럼: {avail}")
    if lpg_col not in df.columns:
        avail = [c for c in df.columns if any(k in c.upper() for k in ["CP","수입","LPG","PRICE"])]
        raise ValueError(f"LPG 컬럼 '{lpg_col}' 없음. 실제 컬럼: {avail}")

    usd_col = next(
        (c for c in df.columns if any(k in c.upper() for k in ["USD", "KRW", "환율", "EXCHANGE"])),
        None,
    )

    train_df, feat_cols = build_features(df, oil_col, lpg_col, usd_col)
    X = train_df[feat_cols].values
    y = train_df[lpg_col].values

    results = {}
    for name, (model, color) in MODELS.items():
        model.fit(X, y)
        preds = model.predict(X)
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

        results[name] = {
            "model":     model,
            "color":     color,
            "preds":     preds,
            "r2":        r2_score(y, preds),
            "mae":       mean_absolute_error(y, preds),
            "cv_r2":     cv_r2,
            "train_df":  train_df,
            "feat_cols": feat_cols,
        }

    return results, usd_col


# ── 단일 입력 예측 ────────────────────────────────────────────
def predict_now(result: dict, input_oil: float, prev2_oil: float, usd_val: float):
    """
    input_oil  : 사용자 입력 전월 유가 → oil_lag1
    prev2_oil  : 데이터 자동 추출 2개월 전 유가 → oil_lag2
    """
    feat_cols = result["feat_cols"]
    oil_chg   = (input_oil - prev2_oil) / prev2_oil if prev2_oil != 0 else 0.0

    row = {
        "oil_lag1": input_oil,
        "oil_lag2": prev2_oil,
        "oil_chg1": oil_chg,
        "usd_krw":  usd_val,
    }
    X_new = np.array([[row[f] for f in feat_cols]])
    return float(result["model"].predict(X_new)[0])


# ── Feature Importance 차트 ───────────────────────────────────
def plot_importance(results: dict, feat_cols: list):
    fig = go.Figure()
    for name in ["RandomForest", "GradientBoosting"]:
        imp = results[name]["model"].feature_importances_
        fig.add_trace(go.Bar(
            name=name, x=feat_cols, y=imp,
            marker_color=results[name]["color"],
            text=[f"{v:.3f}" for v in imp], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", template="plotly_white", height=320,
        title="Feature Importance 비교 (RF vs GB)",
        yaxis_title="Importance", xaxis_title="Feature",
        legend=dict(orientation="h", y=1.15),
        margin=dict(t=60, b=20),
    )
    return fig


# ── 메인 ──────────────────────────────────────────────────────
def main():
    st.title("🔮 LPG 가격 예측 시뮬레이터")
    st.caption("1개월 전 유가 + 환율 기반 | 🌲 RandomForest · 🚀 GradientBoosting · 📏 LinearRegression 비교")

    try:
        df_all = load_data()
    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}"); st.stop()

    # ── 사이드바 ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 설정")
        sel_oil = st.selectbox("🛢️ 기준 유가", list(OIL_VARS.keys()))
        sel_lpg = st.selectbox("⛽ 예측 대상 LPG", list(LPG_VARS.keys()))
        o_col   = OIL_VARS[sel_oil]
        l_col   = LPG_VARS[sel_lpg]

        # 실제 컬럼명 불일치 조기 경고
        if o_col not in df_all.columns:
            st.warning(f"⚠️ 유가 컬럼 **{o_col}** 없음\n실제 컬럼: {list(df_all.columns)}")
        if l_col not in df_all.columns:
            st.warning(f"⚠️ LPG 컬럼 **{l_col}** 없음\n실제 컬럼: {list(df_all.columns)}")

        st.divider()
        st.markdown("#### 📥 예측 입력값")
        st.caption("**전월** 유가와 환율을 입력하면 당월 LPG 가격을 즉시 예측합니다.")

        oil_series = df_all[o_col].dropna() if o_col in df_all.columns else pd.Series([80.0, 79.0])
        latest_oil = float(oil_series.iloc[-1])
        prev2_oil  = float(oil_series.iloc[-2]) if len(oil_series) >= 2 else latest_oil

        input_oil = st.number_input(
            f"전월 {sel_oil} (달러)",
            min_value=0.0, max_value=300.0,
            value=round(latest_oil, 2), step=0.5,
        )
        input_usd = st.number_input(
            "전월 USD/KRW 환율 (원)",
            min_value=800.0, max_value=2500.0,
            value=1380.0, step=1.0,
        )

        st.divider()
        sel_model = st.radio(
            "📊 Backtest 차트 기준 모델",
            list(MODELS.keys()), index=0,
        )

        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear(); st.rerun()

    # ── 모델 학습 ──────────────────────────────────────────────
    try:
        results, usd_col = train_all_models(df_all, o_col, l_col)
    except Exception as e:
        st.error(f"모델 학습 오류: {e}"); st.stop()

    train_df    = results["RandomForest"]["train_df"]
    feat_cols   = results["RandomForest"]["feat_cols"]
    last_actual = float(train_df[l_col].iloc[-1])
    unit        = "원/kg" if "원" in sel_lpg else "$/톤"

    # ── 모델별 당월 예측 ──────────────────────────────────────
    preds_now = {
        name: predict_now(res, input_oil, prev2_oil,
                          usd_val=input_usd if "usd_krw" in feat_cols else 1350.0)
        for name, res in results.items()
    }
    ensemble_pred = np.mean([preds_now["RandomForest"], preds_now["GradientBoosting"]])

    # ── 모델 성능 카드 ────────────────────────────────────────
    st.subheader("📊 모델 성능 비교")
    icons = {"RandomForest": "🌲", "GradientBoosting": "🚀", "LinearRegression": "📏"}
    cols  = st.columns(3)
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx]:
            delta = preds_now[name] - last_actual
            st.markdown(f"##### {icons[name]} {name}")
            st.metric("R² (학습)",   f"{res['r2']:.1%}")
            st.metric("MAE",         f"{res['mae']:,.2f}")
            st.metric("CV R² (5f)",  f"{res['cv_r2']:.1%}")
            st.metric(
                "당월 예측가",
                f"{preds_now[name]:,.1f} {unit}",
                delta=f"{delta:+,.1f}",
                delta_color="inverse",
            )

    # ── 앙상블 결과 요약 ──────────────────────────────────────
    delta_ens = ensemble_pred - last_actual
    st.success(
        f"**입력** | {sel_oil}: **${input_oil:,.2f}**  ·  환율: **₩{input_usd:,.0f}** "
        f"→  🌲 RF: **{preds_now['RandomForest']:,.1f}**  /  "
        f"🚀 GB: **{preds_now['GradientBoosting']:,.1f}**  /  "
        f"⚖️ 앙상블 평균: **{ensemble_pred:,.1f} {unit}**  ({delta_ens:+,.1f})"
    )

    # ── Backtest 그래프 ───────────────────────────────────────
    st.subheader(f"📈 {sel_lpg} — 실제 vs 모델 예측 (Backtest)")

    dash_map = {"RandomForest": "dot", "GradientBoosting": "dash", "LinearRegression": "dashdot"}
    fig = go.Figure()

    # 실제값 라인
    fig.add_trace(go.Scatter(
        x=train_df.index, y=train_df[l_col],
        name="실제 가격", line=dict(color="#378ADD", width=3),
    ))

    # 모델별 예측 라인 (선택 모델만 기본 표시)
    for name, res in results.items():
        visible = True if name == sel_model else "legendonly"
        fig.add_trace(go.Scatter(
            x=train_df.index, y=res["preds"],
            name=f"{icons[name]} {name} (R²={res['r2']:.1%})",
            line=dict(color=res["color"], width=2, dash=dash_map[name]),
            visible=visible,
        ))

    # 당월 예측 포인트
    next_date = train_df.index[-1] + pd.DateOffset(months=1)
    for name in ["RandomForest", "GradientBoosting"]:
        fig.add_trace(go.Scatter(
            x=[next_date], y=[preds_now[name]],
            name=f"{icons[name]} {name} 당월",
            mode="markers+text",
            marker=dict(color=results[name]["color"], size=14, symbol="star"),
            text=[f"  {preds_now[name]:,.1f}"],
            textposition="middle right",
            textfont=dict(size=12, color=results[name]["color"]),
        ))

    fig.update_layout(
        template="plotly_white", height=530, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="연월", yaxis_title=f"{sel_lpg} ({unit})",
        margin=dict(r=110),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────
    st.subheader("🔍 Feature Importance")
    st.plotly_chart(plot_importance(results, feat_cols), use_container_width=True)

    # ── 탭: 데이터 / 파라미터 ─────────────────────────────────
    t1, t2 = st.tabs(["📋 Backtest 데이터", "🧮 모델 파라미터"])

    with t1:
        disp = train_df[[l_col]].copy()
        for name, res in results.items():
            disp[f"{name}"] = res["preds"]
        disp.rename(columns={l_col: "실제가격"}, inplace=True)
        for name in MODELS:
            disp[f"{name}_오차율(%)"] = ((disp["실제가격"] - disp[name]) / disp["실제가격"] * 100).round(2)
        st.dataframe(disp.sort_index(ascending=False), use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🌲 RandomForest**")
            rf = results["RandomForest"]["model"]
            st.json({"n_estimators": rf.n_estimators, "max_depth": rf.max_depth,
                     "min_samples_leaf": rf.min_samples_leaf})
        with c2:
            st.markdown("**🚀 GradientBoosting**")
            gb = results["GradientBoosting"]["model"]
            st.json({"n_estimators": gb.n_estimators, "max_depth": gb.max_depth,
                     "learning_rate": gb.learning_rate, "subsample": gb.subsample})

        st.markdown("**피처 목록**")
        st.write(feat_cols)
        if usd_col is None:
            st.info("💡 환율 컬럼 미감지 → 유가 기반 3-feature 모델로 학습했습니다.")


if __name__ == "__main__":
    main()