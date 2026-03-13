import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from dateutil.relativedelta import relativedelta
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

# ───────────────────────────────
# 상수 정의
# ───────────────────────────────
st.set_page_config(
    page_title="도매요금 AI 시뮬레이터",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

MODEL_COLORS = {
    "Linear Regression":  '#95a5a6',
    "Random Forest":      '#27ae60',
    "Gradient Boosting":  '#f39c12',
    "XGBoost":            '#e74c3c',
}

FEATURE_PRESETS = {
    "브렌트유 + 환율":       (["Brent"],        ["USD_KRW"]),
    "JKM + 환율":           (["JKM"],          ["USD_KRW"]),
    "브렌트유 + JKM + 환율": (["Brent", "JKM"], ["USD_KRW"]),
}
COL_LAG = {"Brent": 6, "JKM": 3}

COL_LABELS = {
    "Brent":   "브렌트유($)",
    "JKM":     "JKM($)",
    "USD_KRW": "환율(원)",
}
TABLE_LABELS = {
    "브렌트유($)": "브렌트유($/배럴)",
    "JKM($)":     "JKM($/MMBtu)",
    "환율(원)":    "환율(원/$)",
}

TRAIN_START_DEFAULT = pd.Timestamp("2015-01-01")


# ───────────────────────────────
# Google Sheets 연결
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
def load_all_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    m_raw = get_as_dataframe(sh.worksheet("Master_Data")).dropna(how="all").dropna(axis=1, how="all")
    m_raw.iloc[:, 0] = pd.to_datetime(m_raw.iloc[:, 0])
    m_raw = m_raw.set_index(m_raw.columns[0]).sort_index().ffill()

    g_raw = get_as_dataframe(sh.worksheet("gas_price")).dropna(how="all").dropna(axis=1, how="all")
    g_raw.iloc[:, 0] = pd.to_datetime(g_raw.iloc[:, 0])
    g_raw = g_raw.set_index(g_raw.columns[0]).sort_index().ffill()
    g_raw.columns = ["Wholesale_Price"]

    merged = m_raw.join(g_raw, how="inner")
    for col, lag in COL_LAG.items():
        if col in merged.columns:
            merged[f"{col}_Lag{lag}"] = merged[col].shift(lag)

    return merged, m_raw


# ───────────────────────────────
# 피처 컬럼명 헬퍼
# ───────────────────────────────
def get_feature_cols(preset_key: str) -> list:
    lag_cols, nolag_cols = FEATURE_PRESETS[preset_key]
    return [f"{c}_Lag{COL_LAG[c]}" for c in lag_cols] + nolag_cols


def get_train_xy(merged: pd.DataFrame, feature_cols: list, train_start: pd.Timestamp, train_end: pd.Timestamp):
    required = feature_cols + ["Wholesale_Price"]
    train_df = merged.dropna(subset=required)
    train_df = train_df[(train_df.index >= train_start) & (train_df.index <= train_end)]
    return train_df[feature_cols], train_df["Wholesale_Price"]


# ───────────────────────────────
# 모델 학습
# ───────────────────────────────
def build_models(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "Linear Regression":  LinearRegression().fit(X, y),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, y),
        "XGBoost":            XGBRegressor(n_estimators=100, random_state=42, verbosity=0).fit(X, y),
    }


# ───────────────────────────────
# Holt 이중지수평활 예측
# ───────────────────────────────
def holt_forecast(series: pd.Series, n: int) -> np.ndarray:
    s = series.astype(float).dropna()
    try:
        fit = Holt(s, exponential=False).fit(optimized=True)
        vals = fit.forecast(n).values
        return np.clip(vals, float(s.min()) * 0.5, float(s.max()) * 2.0)
    except Exception:
        fit = SimpleExpSmoothing(s).fit(optimized=True)
        return fit.forecast(n).values


# ───────────────────────────────
# 시나리오 DataFrame 생성
# ───────────────────────────────
def build_scenario_df(
    forecast_months: int,
    last_date: pd.Timestamp,
    master_raw: pd.DataFrame,
    input_mode: str,
    start_offset: int,
    preset_key: str,
) -> pd.DataFrame:
    use_ses = "지수평활법" in input_mode
    lag_cols, nolag_cols = FEATURE_PRESETS[preset_key]

    total_needed = abs(start_offset) + forecast_months + max(COL_LAG.values())
    ses_cache: dict = {}
    if use_ses:
        for col in lag_cols + nolag_cols:
            if col in master_raw.columns:
                ses_cache[col] = holt_forecast(master_raw[col], total_needed)

    rows = []
    for step in range(forecast_months):
        i = start_offset + step
        target_date = last_date + relativedelta(months=i)
        row: dict = {"날짜": target_date.strftime("%Y-%m")}
        status = "수동(시나리오)"

        for col in lag_cols:
            label = COL_LABELS[col]
            lag = COL_LAG[col]
            if col not in master_raw.columns:
                row[label] = np.nan
                continue

            history = master_raw[col].values
            history_index = master_raw.index
            needed_date = target_date - relativedelta(months=lag)

            if needed_date <= master_raw.index[-1]:
                if needed_date in history_index:
                    val = float(master_raw.loc[needed_date, col])
                else:
                    pos = history_index.searchsorted(needed_date, side="right") - 1
                    pos = max(0, min(pos, len(history) - 1))
                    val = float(history[pos])
                status = "자동(실제값)"
            else:
                months_from_last = (needed_date.year - master_raw.index[-1].year) * 12 \
                                   + (needed_date.month - master_raw.index[-1].month)
                if use_ses and col in ses_cache:
                    idx = min(max(months_from_last - 1, 0), len(ses_cache[col]) - 1)
                    val = float(ses_cache[col][idx])
                else:
                    val = float(history[-1])

            row[label] = round(val, 2)

        for col in nolag_cols:
            label = COL_LABELS[col]
            if col not in master_raw.columns:
                row[label] = np.nan
                continue
            if target_date <= master_raw.index[-1] and target_date in master_raw.index:
                row[label] = round(float(master_raw.loc[target_date, col]), 1)
                status = "자동(실제값)"
            elif use_ses and col in ses_cache:
                months_from_last = (target_date.year - master_raw.index[-1].year) * 12 \
                                   + (target_date.month - master_raw.index[-1].month)
                idx = min(max(months_from_last - 1, 0), len(ses_cache[col]) - 1)
                row[label] = round(float(ses_cache[col][idx]), 1)
            else:
                row[label] = round(float(master_raw[col].iloc[-1]), 1)

        row["구분"] = status
        rows.append(row)

    return pd.DataFrame(rows)


# ───────────────────────────────
# 예측 실행
# ───────────────────────────────
def run_forecast(
    edited_df: pd.DataFrame,
    models: dict,
    feature_cols: list,
    preset_key: str,
) -> pd.DataFrame:
    lag_cols, nolag_cols = FEATURE_PRESETS[preset_key]
    ordered_labels = [COL_LABELS[c] for c in lag_cols] + [COL_LABELS[c] for c in nolag_cols]

    records = []
    for _, row in edited_df.iterrows():
        x_vals = [row[lbl] for lbl in ordered_labels]
        X_pred = pd.DataFrame([x_vals], columns=feature_cols)
        for name, model in models.items():
            records.append({
                "Date":  pd.to_datetime(row["날짜"]),
                "Price": float(model.predict(X_pred)[0]),
                "Model": name,
            })
    return pd.DataFrame(records)


# ───────────────────────────────
# Plotly 차트
# ───────────────────────────────
def plot_forecast(
    train_y: pd.Series,
    train_index: pd.Index,
    future_df: pd.DataFrame,
    last_date: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    selected_models: list,
) -> go.Figure:
    fig = go.Figure()

    fig.add_vrect(
        x0=train_start, x1=train_end,
        fillcolor="lightyellow", opacity=0.25,
        layer="below", line_width=0,
        annotation_text=f"학습 기간 ({train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')})",
        annotation_position="top left",
        annotation_font_color="#b8860b",
        annotation_font_size=11,
    )

    fig.add_trace(go.Scatter(
        x=train_index, y=train_y,
        name="과거 실제 요금",
        line=dict(color="black", width=3),
        hovertemplate="%{x|%Y-%m}<br>실제: %{y:.4f}원/MJ<extra></extra>",
    ))

    for name in selected_models:
        color = MODEL_COLORS[name]
        mdf = future_df[future_df["Model"] == name].sort_values("Date")
        px_vals = [train_index[-1]] + list(mdf["Date"])
        py_vals = [float(train_y.iloc[-1])] + list(mdf["Price"])
        fig.add_trace(go.Scatter(
            x=px_vals, y=py_vals,
            name=name,
            line=dict(color=color, width=2.5, dash="dot"),
            hovertemplate=f"%{{x|%Y-%m}}<br>{name}: %{{y:.4f}}원/MJ<extra></extra>",
        ))

    confirm_limit = last_date + relativedelta(months=COL_LAG["Brent"])
    fig.add_vline(
        x=confirm_limit.timestamp() * 1000,
        line_dash="dash", line_color="royalblue", line_width=1.5,
        annotation_text=f"확정유가 한계 (Brent {COL_LAG['Brent']}M / JKM {COL_LAG['JKM']}M)",
        annotation_position="top left",
        annotation_font_color="royalblue",
    )

    fig.update_layout(
        template="plotly_white", height=600, hovermode="x unified",
        yaxis_title="도매요금 (원/MJ)", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════
def main():
    st.title("🏛️ 대성에너지 도매요금 AI 시뮬레이션 시스템")

    # ── session_state 초기화 ──────────────────────────────
    if "forecast_result" not in st.session_state:
        st.session_state.forecast_result = None   # 예측 결과 DataFrame
    if "forecast_snapshot" not in st.session_state:
        st.session_state.forecast_snapshot = None # 예측 당시 edited_df 스냅샷

    with st.spinner("📡 Google Sheets 데이터 불러오는 중..."):
        try:
            merged, master_raw = load_all_data()
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
            st.stop()

    last_date = merged["Wholesale_Price"].dropna().index.max()

    # ── 사이드바 ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 시뮬레이션 설정")

        # 1) 피처 조합 선택
        st.subheader("🔢 입력 변수 선택")
        preset_key = st.radio(
            "예측에 사용할 변수 조합을 선택하세요",
            list(FEATURE_PRESETS.keys()),
            help=(
                "• 브렌트유 + 환율: 기존 모델 (원유 기준)\n"
                "• JKM + 환율: LNG 현물가 직접 반영\n"
                "• 브렌트유 + JKM + 환율: 두 지수 동시 활용"
            ),
        )
        feature_cols = get_feature_cols(preset_key)
        st.caption(f"학습 피처: `{'`, `'.join(feature_cols)}`")

        st.divider()

        # 2) 모델 학습 기간 설정
        st.subheader("🎓 모델 학습 기간")
        st.caption("선택한 기간의 데이터로만 모델을 학습합니다.")

        available_years = sorted(merged.index.year.unique())
        default_train_start_year = max(TRAIN_START_DEFAULT.year, available_years[0])

        train_start_year, train_end_year = st.select_slider(
            "학습 기간 (연도)",
            options=available_years,
            value=(default_train_start_year, available_years[-1]),
        )
        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end   = pd.Timestamp(f"{train_end_year}-12-31")

        _X_check, _y_check = get_train_xy(merged, feature_cols, train_start, train_end)
        n_train = len(_X_check)
        if n_train < 10:
            st.error(f"⚠️ 학습 데이터 {n_train}개 — 기간을 늘려주세요.")
        else:
            st.caption(f"📊 학습 데이터: **{train_start_year}년 ~ {train_end_year}년** ({n_train}개월)")

        st.divider()

        # 3) 예측 기간
        st.subheader("📅 예측 기간 설정")
        default_start = last_date + relativedelta(months=1)
        default_end   = last_date + relativedelta(months=12)

        col_s, col_e = st.columns(2)
        with col_s:
            start_year  = st.number_input("시작 연도", min_value=2000,
                                          max_value=last_date.year + 10, value=default_start.year)
            start_month = st.number_input("시작 월", min_value=1, max_value=12, value=default_start.month)
        with col_e:
            end_year  = st.number_input("종료 연도", min_value=2000,
                                        max_value=last_date.year + 10, value=default_end.year)
            end_month = st.number_input("종료 월", min_value=1, max_value=12, value=default_end.month)

        forecast_start = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
        forecast_end   = pd.Timestamp(year=int(end_year),   month=int(end_month),   day=1)

        if forecast_end < forecast_start:
            st.warning("⚠️ 종료월이 시작월보다 앞설 수 없습니다.")
            forecast_end = forecast_start + relativedelta(months=11)

        forecast_months = (
            (forecast_end.year - forecast_start.year) * 12
            + (forecast_end.month - forecast_start.month) + 1
        )
        start_offset = (
            (forecast_start.year - last_date.year) * 12
            + (forecast_start.month - last_date.month)
        )
        st.caption(
            f"📊 총 **{forecast_months}개월** "
            f"({forecast_start.strftime('%Y-%m')} ~ {forecast_end.strftime('%Y-%m')})"
        )

        st.divider()

        # 4) 지수평활법 선택
        input_mode = st.radio(
            "미래 지표 생성 방식",
            ["직접 입력 (사용자 지정)", "지수평활법 (자동 트렌드 예측)"],
            index=1,
            help="지수평활법: Holt 이중지수평활법으로 트렌드 반영\n직접 입력: 표에서 직접 수정",
        )

        st.divider()

        # 5) 표시 모델 선택
        st.subheader("📊 표시 모델 선택")
        selected_models = [n for n in MODEL_COLORS if st.checkbox(n, value=True)]

        st.divider()
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.session_state.forecast_result = None
            st.session_state.forecast_snapshot = None
            st.rerun()
        st.caption(f"📅 데이터 기준: **{last_date.strftime('%Y-%m')}**")

    # ── 모델 학습 ─────────────────────────────────────────
    X, y = get_train_xy(merged, feature_cols, train_start, train_end)
    if X.empty or len(X) < 10:
        st.error(
            f"❌ 학습 데이터가 부족합니다 ({len(X)}개월). "
            "사이드바에서 학습 기간을 늘려주세요."
        )
        st.stop()

    y_all = merged["Wholesale_Price"].dropna()
    X_all_r2, y_all_r2 = get_train_xy(merged, feature_cols,
                                       merged.index.min(), merged.index.max())

    with st.spinner("🤖 AI 모델 학습 중..."):
        models = build_models(X, y)

    # ── R² 스코어 표시 ─────────────────────────────────────
    r2_cols = st.columns(4)
    for i, (name, model) in enumerate(models.items()):
        r2_tr  = r2_score(y,        model.predict(X))
        r2_all = r2_score(y_all_r2, model.predict(X_all_r2))
        r2_cols[i].metric(
            label=name,
            value=f"R² {r2_tr:.4f}",
            delta=f"전체기간 {r2_all:.4f}",
            delta_color="normal",
            help=f"학습기간({train_start_year}~{train_end_year}) R² vs 전체기간 R²",
        )

    # ── Step 1: 시나리오 에디터 ───────────────────────────
    st.markdown("### 📅 Step 1. 미래 지표 시나리오 설정")

    scenario_df = build_scenario_df(
        forecast_months, last_date, master_raw, input_mode, start_offset, preset_key
    )

    st.info(
        f"💡 **{input_mode}** 모드 | 피처: **{preset_key}** | "
        f"🎓 학습: **{train_start_year}년 ~ {train_end_year}년** ({n_train}개월) | "
        + ("**Holt 이중지수평활법** 적용. " if "지수평활법" in input_mode else "")
        + "확정유가 구간은 과거 실제값이 자동 반영됩니다. 숫자를 더블클릭하여 수정 가능합니다."
    )

    lag_cols_sel, nolag_cols_sel = FEATURE_PRESETS[preset_key]
    col_cfg: dict = {
        "날짜": st.column_config.TextColumn("날짜", disabled=True),
        "구분": st.column_config.TextColumn("구분", disabled=True),
        "환율(원)": st.column_config.NumberColumn("환율 (₩)", format="%.1f", min_value=0.0),
    }
    if "Brent" in lag_cols_sel:
        col_cfg["브렌트유($)"] = st.column_config.NumberColumn("브렌트유 ($)", format="%.2f", min_value=0.0)
    if "JKM" in lag_cols_sel:
        col_cfg["JKM($)"] = st.column_config.NumberColumn("JKM ($)", format="%.2f", min_value=0.0)

    edited_df = st.data_editor(
        scenario_df,
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
        height=min(35 * forecast_months + 60, 500),
    )

    # ── 예측하기 버튼 ─────────────────────────────────────
    st.divider()

    btn_col, hint_col = st.columns([1, 4])
    with btn_col:
        run_btn = st.button(
            "🚀 예측하기",
            type="primary",
            use_container_width=True,
            disabled=(not selected_models),
        )
    with hint_col:
        if not selected_models:
            st.warning("⚠️ 사이드바에서 표시할 모델을 1개 이상 선택하세요.")
        elif st.session_state.forecast_result is not None:
            st.caption("✅ 아래에 가장 최근 예측 결과가 표시됩니다. 값을 수정한 뒤 **예측하기**를 다시 누르세요.")
        else:
            st.caption("👆 시나리오 값을 설정한 뒤 **예측하기** 버튼을 눌러주세요.")

    # 버튼 클릭 시 → session_state에 결과 저장
    if run_btn and selected_models:
        active_models = {k: v for k, v in models.items() if k in selected_models}
        with st.spinner("🔮 예측 계산 중..."):
            future_df = run_forecast(edited_df, active_models, feature_cols, preset_key)
        st.session_state.forecast_result = future_df
        st.session_state.forecast_snapshot = {
            "edited_df":       edited_df.copy(),
            "selected_models": selected_models[:],
            "train_start":     train_start,
            "train_end":       train_end,
            "preset_key":      preset_key,
            "feature_cols":    feature_cols[:],
        }

    # ── Step 2: 저장된 예측 결과 표시 ────────────────────
    if st.session_state.forecast_result is not None:
        snap        = st.session_state.forecast_snapshot
        future_df   = st.session_state.forecast_result
        s_edited_df     = snap["edited_df"]
        s_selected      = snap["selected_models"]
        s_train_start   = snap["train_start"]
        s_train_end     = snap["train_end"]
        s_preset_key    = snap["preset_key"]
        s_feature_cols  = snap["feature_cols"]

        st.markdown("### 📈 Step 2. AI 모델별 도매요금 예측 결과")

        fig = plot_forecast(
            y_all, y_all.index, future_df, last_date,
            s_train_start, s_train_end, s_selected
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Step 3: 가로형 수치 테이블 ───────────────────
        st.subheader("📋 산업용 가스요금 전망 (원/MJ)")

        date_cols = [d.strftime("%Y-%m") for d in sorted(future_df["Date"].unique())]
        idx_df = s_edited_df.set_index("날짜")

        pivot_raw = future_df.pivot(index="Date", columns="Model", values="Price")
        pivot_raw.index = pivot_raw.index.strftime("%Y-%m")

        table_rows: dict = {}
        for lbl, tbl_lbl in TABLE_LABELS.items():
            if lbl in idx_df.columns:
                table_rows[tbl_lbl] = {
                    d: idx_df.loc[d, lbl] if d in idx_df.index else "" for d in date_cols
                }

        for model_name in s_selected:
            if model_name in pivot_raw.columns:
                table_rows[f"예상요금 · {model_name}"] = {
                    d: pivot_raw.loc[d, model_name] if d in pivot_raw.index else ""
                    for d in date_cols
                }

        display_df = pd.DataFrame(table_rows, index=date_cols).T
        display_df.index.name = "구분"

        def highlight_table(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            indicator_rows = [r for r in df.index if "예상요금" not in r]
            for r in indicator_rows:
                styles.loc[r, :] = "background-color: #f5f5f5; font-weight: bold"
            model_rows = [r for r in df.index if "예상요금" in r]
            if len(model_rows) > 1:
                model_vals = df.loc[model_rows].apply(pd.to_numeric, errors="coerce")
                for col in model_vals.columns:
                    col_vals = model_vals[col].dropna()
                    if len(col_vals) == 0:
                        continue
                    styles.loc[col_vals.idxmax(), col] = "background-color: #FFD700; font-weight: bold"
                    styles.loc[col_vals.idxmin(), col] = "background-color: #90EE90; font-weight: bold"
            return styles

        styled = display_df.style.apply(highlight_table, axis=None)
        for row_label in display_df.index:
            if "환율" in row_label:
                fmt_str = "{:.0f}"
            elif "예상요금" in row_label:
                fmt_str = "{:.4f}"
            else:
                fmt_str = "{:.2f}"
            styled = styled.format(fmt_str, subset=pd.IndexSlice[row_label, :], na_rep="-")

        st.dataframe(styled, use_container_width=True)
        st.caption("🟡 노란색: 해당 월 최고 예상요금 모델  |  🟢 초록색: 최저 예상요금 모델")


if __name__ == "__main__":
    main()