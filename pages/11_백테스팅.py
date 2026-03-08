import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import gspread
from google.oauth2 import service_account
from gspread_dataframe import get_as_dataframe

st.set_page_config(page_title="도매요금 위기 분석 대시보드", layout="wide")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

# ─── 모델 정의 ────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "XGBoost": {
        "model": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        "color": "#E74C3C",
        "dash": "dash",
    },
    "Random Forest": {
        "model": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "color": "#2980B9",
        "dash": "dot",
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        "color": "#27AE60",
        "dash": "dashdot",
    },
}

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)


@st.cache_data
def load_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    # Master 데이터
    m_df = (
        get_as_dataframe(sh.worksheet("Master_Data"))
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    m_df.iloc[:, 0] = pd.to_datetime(m_df.iloc[:, 0])
    m_df = (
        m_df.set_index(m_df.columns[0])
        .sort_index()
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # 요금 데이터
    g_df = (
        get_as_dataframe(sh.worksheet("gas_price"))
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )
    g_df.iloc[:, 0] = pd.to_datetime(g_df.iloc[:, 0])
    g_df = g_df.set_index(g_df.columns[0]).sort_index().fillna(method="ffill")
    g_df.columns = ["Wholesale_Price"]

    df = m_df.join(g_df, how="inner")

    # 피처 생성
    df["Brent_Lag6"] = df["Brent"].shift(6)
    df["TTF_Lag3"] = df["TTF"].shift(3)
    df["War_Event"] = 0
    df.loc["2022-03-01":"2023-12-31", "War_Event"] = 1

    features = ["Brent_Lag6", "TTF_Lag3", "USD_KRW", "War_Event"]
    model_df = df.dropna(subset=features + ["Wholesale_Price"])
    return model_df, features, df


# ─── 성능 지표 계산 ────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ─── 메인 ─────────────────────────────────────────────────────────────────────
try:
    model_df, features, full_df = load_data()

    st.title("🏛️ 대성에너지 도매요금 위기 분석 및 모델 비교")

    # ── 사이드바 ────────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ 분석 설정")
    mode = st.sidebar.radio("모드 선택", ["미래 예측 (Latest)", "백테스팅 (Validation)"])

    # 모델 선택 (다중)
    selected_models = st.sidebar.multiselect(
        "비교할 모델 선택",
        options=list(MODEL_CONFIG.keys()),
        default=list(MODEL_CONFIG.keys()),
    )
    if not selected_models:
        st.warning("⚠️ 모델을 하나 이상 선택해 주세요.")
        st.stop()

    # ── 미래 예측 모드 ──────────────────────────────────────────────────────────
    if mode == "미래 예측 (Latest)":
        st.subheader("📊 전체 데이터 학습 및 미래 3개월 전망")

        war_risk   = st.sidebar.checkbox("미래 리스크 시나리오 반영", value=False)
        war_val    = 1 if war_risk else 0
        current_krw = model_df["USD_KRW"].iloc[-1]

        future_dates = [
            model_df.index.max() + pd.DateOffset(months=i) for i in range(1, 4)
        ]

        fig = go.Figure()
        # 실제 요금 (공통)
        fig.add_trace(
            go.Scatter(
                x=model_df.index,
                y=model_df["Wholesale_Price"],
                name="실제 요금",
                line=dict(color="#2C3E50", width=2),
            )
        )

        # 각 모델별 미래 예측
        future_results = {}
        for model_name in selected_models:
            cfg = MODEL_CONFIG[model_name]
            m = cfg["model"]
            m.fit(model_df[features], model_df["Wholesale_Price"])

            preds = []
            for i in range(1, 4):
                row = pd.DataFrame(
                    [[
                        full_df["Brent"].iloc[-6 + (i - 1)],
                        full_df["TTF"].iloc[-3 + (i - 1)],
                        current_krw,
                        war_val,
                    ]],
                    columns=features,
                )
                preds.append(m.predict(row)[0])

            future_results[model_name] = preds

            fig.add_trace(
                go.Scatter(
                    x=[model_df.index[-1]] + future_dates,
                    y=[model_df["Wholesale_Price"].iloc[-1]] + preds,
                    name=f"{model_name} 예측",
                    line=dict(color=cfg["color"], width=3, dash=cfg["dash"]),
                )
            )

        fig.update_layout(
            template="plotly_white",
            height=600,
            xaxis_title="날짜",
            yaxis_title="도매요금 (원/MJ)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 모델별 3개월 후 예측값 카드
        current_price = model_df["Wholesale_Price"].iloc[-1]
        cols = st.columns(len(selected_models) + 1)
        cols[0].metric("현재 요금", f"{current_price:.2f}원")
        for i, model_name in enumerate(selected_models):
            pred_3m = future_results[model_name][2]
            delta   = pred_3m - current_price
            cols[i + 1].metric(
                f"{model_name} (3개월 후)",
                f"{pred_3m:.2f}원",
                f"{delta:+.2f}원",
                delta_color="inverse",
            )

        st.info(f"💡 {'전쟁 리스크' if war_risk else '정상'} 시나리오 반영됨")

        # 미래 예측 테이블
        future_table = pd.DataFrame(
            {name: future_results[name] for name in selected_models},
            index=[d.strftime("%Y-%m") for d in future_dates],
        )
        future_table.index.name = "예측 월"
        with st.expander("📋 미래 예측 상세 수치"):
            st.dataframe(future_table.style.format("{:.2f}원"))

    # ── 백테스팅 모드 ──────────────────────────────────────────────────────────
    else:
        st.subheader("🧪 모델 백테스팅 검증")

        available_months = (
            model_df.index.to_period("M").unique().astype(str).tolist()
        )
        default_idx = max(0, len(available_months) - 4)

        selected_month_str = st.sidebar.selectbox(
            "학습 종료(Cut-off) 월 선택",
            options=available_months,
            index=default_idx,
            help="선택한 달까지 학습 → 이후 데이터로 성능 검증",
        )

        split_date = (
            pd.to_datetime(selected_month_str).to_period("M").to_timestamp(how="end")
        )
        train_df = model_df[model_df.index <= split_date]
        test_df  = model_df[model_df.index > split_date].copy()

        if test_df.empty:
            st.warning("⚠️ 선택한 달 이후의 실제 데이터가 없습니다. 더 과거의 달을 선택해 주세요.")
        else:
            # ── 예측 & 지표 계산 ──────────────────────────────────────────────
            all_metrics = {}
            fig = go.Figure()

            # 학습 구간
            fig.add_trace(
                go.Scatter(
                    x=train_df.index,
                    y=train_df["Wholesale_Price"],
                    name="학습 데이터 (Actual)",
                    line=dict(color="gray", width=1.5, dash="dot"),
                )
            )
            # 테스트 실제값
            fig.add_trace(
                go.Scatter(
                    x=test_df.index,
                    y=test_df["Wholesale_Price"],
                    name="테스트 실제 (Actual)",
                    line=dict(color="#2C3E50", width=3),
                )
            )

            for model_name in selected_models:
                cfg = MODEL_CONFIG[model_name]
                m   = cfg["model"]
                m.fit(train_df[features], train_df["Wholesale_Price"])
                preds = m.predict(test_df[features])
                test_df[f"Pred_{model_name}"] = preds

                all_metrics[model_name] = calc_metrics(
                    test_df["Wholesale_Price"], preds
                )

                fig.add_trace(
                    go.Scatter(
                        x=test_df.index,
                        y=preds,
                        name=f"{model_name} 예측",
                        line=dict(color=cfg["color"], width=2.5, dash=cfg["dash"]),
                    )
                )

            fig.add_vrect(
                x0=test_df.index[0],
                x1=test_df.index[-1],
                fillcolor="rgba(231,76,60,0.07)",
                layer="below",
                line_width=0,
                annotation_text="검증 구간",
            )
            fig.update_layout(
                title=f"백테스팅 결과 ({selected_month_str} 이후 검증)",
                xaxis_title="날짜",
                yaxis_title="도매요금 (원/MJ)",
                template="plotly_white",
                height=600,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── 성능 지표 비교 테이블 ─────────────────────────────────────────
            st.markdown("### 📊 모델 성능 비교")
            metrics_df = pd.DataFrame(all_metrics).T
            metrics_df.index.name = "모델"

            # 최고 성능 강조를 위해 스타일 적용
            def highlight_best(s):
                is_r2   = s.name == "R²"
                is_best = s == s.max() if is_r2 else s == s.min()
                return ["background-color: #d4efdf; font-weight:bold" if v else "" for v in is_best]

            styled = (
                metrics_df.style
                .apply(highlight_best, axis=0)
                .format({"R²": "{:.4f}", "MAE": "{:.2f}원", "RMSE": "{:.2f}원", "MAPE": "{:.2f}%"})
            )
            st.dataframe(styled, use_container_width=True)

            # ── 개별 지표 카드 ─────────────────────────────────────────────────
            metric_tabs = st.tabs(["R² (결정계수)", "MAE", "RMSE", "MAPE"])
            metric_keys = ["R²", "MAE", "RMSE", "MAPE"]
            for tab, key in zip(metric_tabs, metric_keys):
                with tab:
                    cols = st.columns(len(selected_models))
                    best_val = (
                        max(all_metrics[n][key] for n in selected_models)
                        if key == "R²"
                        else min(all_metrics[n][key] for n in selected_models)
                    )
                    for i, model_name in enumerate(selected_models):
                        val = all_metrics[model_name][key]
                        is_best = val == best_val
                        fmt = f"{val:.4f}" if key == "R²" else f"{val:.2f}"
                        if key != "R²":
                            fmt += "원" if key != "MAPE" else "%"
                        label = f"{'🏆 ' if is_best else ''}{model_name}"
                        cols[i].metric(label, fmt)

            # ── 레이더 차트 (정규화 점수) ──────────────────────────────────────
            if len(selected_models) >= 2:
                st.markdown("### 🕸️ 종합 성능 레이더 차트")

                # 각 지표를 0~1로 정규화 (R²는 높을수록, 나머지는 낮을수록 좋음)
                norm_metrics = {}
                for key in metric_keys:
                    vals = {n: all_metrics[n][key] for n in selected_models}
                    mn, mx = min(vals.values()), max(vals.values())
                    rng = mx - mn if mx != mn else 1
                    for n in selected_models:
                        if n not in norm_metrics:
                            norm_metrics[n] = {}
                        score = (vals[n] - mn) / rng
                        norm_metrics[n][key] = score if key == "R²" else 1 - score

                radar_fig = go.Figure()
                categories = metric_keys + [metric_keys[0]]  # 닫힌 다각형

                for model_name in selected_models:
                    scores = [norm_metrics[model_name][k] for k in metric_keys]
                    scores += [scores[0]]
                    radar_fig.add_trace(
                        go.Scatterpolar(
                            r=scores,
                            theta=categories,
                            fill="toself",
                            name=model_name,
                            line_color=MODEL_CONFIG[model_name]["color"],
                            opacity=0.6,
                        )
                    )

                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template="plotly_white",
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(radar_fig, use_container_width=True)
                st.caption("※ 레이더 차트: 각 지표를 0~1로 정규화. 넓을수록 종합 성능 우수")

            # ── 상세 비교 테이블 ───────────────────────────────────────────────
            with st.expander("📝 검증 데이터 상세 비교"):
                display_cols = ["Wholesale_Price"] + [
                    f"Pred_{n}" for n in selected_models
                ]
                rename_map = {"Wholesale_Price": "실제값"}
                rename_map.update({f"Pred_{n}": f"{n} 예측" for n in selected_models})
                st.dataframe(
                    test_df[display_cols]
                    .rename(columns=rename_map)
                    .style.format("{:.2f}원"),
                    use_container_width=True,
                )

except Exception as e:
    st.error(f"오류 발생: {e}")
    st.exception(e)