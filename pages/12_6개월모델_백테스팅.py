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

st.set_page_config(page_title="도매요금 예측 대시보드", layout="wide")

SHEET_ID = "1Y5hxWDA_SRXyXhZF7bZLmI8MFQg9r78HLiHoSFVeneE"

# ─── 피처셋 정의 ──────────────────────────────────────────────────────────────
FEATURE_SETS = {
    "6개월 선행 v2 (Brent+TTF_Lag6+환율+전쟁)": {
        "features": ["Brent_Lag6", "TTF_Lag6", "USD_KRW", "War_Event"],
        "horizon": 6,
        "color": "#8E44AD",
        "description": "Brent_Lag6 + TTF_Lag6 — 완전한 선행 변수만 사용. TTF 충격 신호 포함.",
    },
    "6개월 선행 v1 (Brent+환율+전쟁)": {
        "features": ["Brent_Lag6", "USD_KRW", "War_Event"],
        "horizon": 6,
        "color": "#E74C3C",
        "description": "TTF 없는 순수 6개월 선행 모델. 안정적이나 단기 충격 미반영.",
    },
    "3개월 선행 (Brent+TTF_Lag3+환율+전쟁)": {
        "features": ["Brent_Lag6", "TTF_Lag3", "USD_KRW", "War_Event"],
        "horizon": 3,
        "color": "#2980B9",
        "description": "기존 모델. TTF_Lag3으로 단기 충격 포착. 예측 지평 3개월.",
    },
}

# ─── 모델 정의 ────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "XGBoost": {
        "fn": lambda: XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        "color": "#E74C3C", "dash": "dash",
    },
    "Random Forest": {
        "fn": lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "color": "#2980B9", "dash": "dot",
    },
    "Gradient Boosting": {
        "fn": lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        "color": "#27AE60", "dash": "dashdot",
    },
}

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_gcp_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data
def load_data():
    gc = get_gcp_client()
    sh = gc.open_by_key(SHEET_ID)

    m_df = get_as_dataframe(sh.worksheet("Master_Data")).dropna(how="all").dropna(axis=1, how="all")
    m_df.iloc[:, 0] = pd.to_datetime(m_df.iloc[:, 0])
    m_df = m_df.set_index(m_df.columns[0]).sort_index().fillna(method="ffill").fillna(method="bfill")

    g_df = get_as_dataframe(sh.worksheet("gas_price")).dropna(how="all").dropna(axis=1, how="all")
    g_df.iloc[:, 0] = pd.to_datetime(g_df.iloc[:, 0])
    g_df = g_df.set_index(g_df.columns[0]).sort_index().fillna(method="ffill")
    g_df.columns = ["Wholesale_Price"]

    df = m_df.join(g_df, how="inner")
    df["Brent_Lag6"] = df["Brent"].shift(6)
    df["TTF_Lag3"]   = df["TTF"].shift(3)
    df["TTF_Lag6"]   = df["TTF"].shift(6)   # ★ 추가
    df["War_Event"]  = 0
    df.loc["2022-03-01":"2023-12-31", "War_Event"] = 1
    return df

# ─── 유틸 ─────────────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def fmt_metric(key, val):
    if key == "R²":   return f"{val:.4f}"
    if key == "MAPE": return f"{val:.2f}%"
    return f"{val:.2f}원"

def render_metrics_table(all_metrics):
    df = pd.DataFrame(all_metrics).T
    df.index.name = "모델"
    def highlight_best(s):
        is_best = s == s.max() if s.name == "R²" else s == s.min()
        return ["background-color:#d4efdf;font-weight:bold" if v else "" for v in is_best]
    st.dataframe(
        df.style.apply(highlight_best, axis=0)
          .format({"R²": "{:.4f}", "MAE": "{:.2f}원", "RMSE": "{:.2f}원", "MAPE": "{:.2f}%"}),
        use_container_width=True,
    )

def render_radar(all_metrics, selected_models):
    if len(selected_models) < 2:
        return
    metric_keys = ["R²", "MAE", "RMSE", "MAPE"]
    norm = {}
    for key in metric_keys:
        vals = {n: all_metrics[n][key] for n in selected_models}
        mn, mx = min(vals.values()), max(vals.values())
        rng = mx - mn if mx != mn else 1
        for n in selected_models:
            if n not in norm:
                norm[n] = {}
            score = (vals[n] - mn) / rng
            norm[n][key] = score if key == "R²" else 1 - score
    cats = metric_keys + [metric_keys[0]]
    fig  = go.Figure()
    for name in selected_models:
        scores = [norm[name][k] for k in metric_keys] + [norm[name][metric_keys[0]]]
        fig.add_trace(go.Scatterpolar(
            r=scores, theta=cats, fill="toself", name=name,
            line_color=MODEL_CONFIG[name]["color"], opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("※ 각 지표를 0~1 정규화. 넓을수록 종합 성능 우수")

def build_future_row(fset_name, fset_cfg, full_df, i, current_krw, war_val):
    """미래 i번째 달의 예측 입력 행 생성"""
    features = fset_cfg["features"]
    if "TTF_Lag3" in features:
        vals = [
            full_df["Brent"].iloc[-6 + (i - 1)],
            full_df["TTF"].iloc[-3 + (i - 1)],
            current_krw, war_val,
        ]
    elif "TTF_Lag6" in features:
        vals = [
            full_df["Brent"].iloc[max(-7 + (i - 1), -len(full_df))],
            full_df["TTF"].iloc[max(-7 + (i - 1), -len(full_df))],
            current_krw, war_val,
        ]
    else:
        vals = [
            full_df["Brent"].iloc[max(-7 + (i - 1), -len(full_df))],
            current_krw, war_val,
        ]
    return pd.DataFrame([vals], columns=features)

# ─── 메인 ─────────────────────────────────────────────────────────────────────
try:
    full_df = load_data()

    st.title("🏛️ 대성에너지 도매요금 예측 대시보드")

    # ── 사이드바 ────────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ 분석 설정")

    fset_name = st.sidebar.radio("📐 피처셋 선택", options=list(FEATURE_SETS.keys()))
    fset      = FEATURE_SETS[fset_name]
    features  = fset["features"]
    horizon   = fset["horizon"]
    st.sidebar.caption(f"ℹ️ {fset['description']}")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio("🔍 모드 선택", ["미래 예측 (Latest)", "백테스팅 (Validation)", "피처셋 전체 비교"])

    selected_models = st.sidebar.multiselect(
        "🤖 모델 선택",
        options=list(MODEL_CONFIG.keys()),
        default=list(MODEL_CONFIG.keys()),
    )
    if not selected_models:
        st.warning("⚠️ 모델을 하나 이상 선택해 주세요.")
        st.stop()

    model_df = full_df.dropna(subset=features + ["Wholesale_Price"])

    # 배너
    c1, c2, c3, c4 = st.columns(4)
    c1.info(f"**피처셋** {fset_name}")
    c2.info(f"**예측 지평** {horizon}개월")
    c3.info(f"**학습 샘플** {len(model_df)}개월")
    c4.info(f"**피처수** {len(features)}개")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════════
    # 미래 예측
    # ════════════════════════════════════════════════════════════════════════════
    if mode == "미래 예측 (Latest)":
        st.subheader(f"📊 전체 학습 → 미래 {horizon}개월 전망")

        war_risk    = st.sidebar.checkbox("전쟁/지정학 리스크 반영", value=False)
        war_val     = 1 if war_risk else 0
        current_krw = model_df["USD_KRW"].iloc[-1]
        future_dates = [model_df.index.max() + pd.DateOffset(months=i) for i in range(1, horizon + 1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=model_df.index, y=model_df["Wholesale_Price"],
            name="실제 요금", line=dict(color="#2C3E50", width=2),
        ))

        future_results = {}
        for mname in selected_models:
            m = MODEL_CONFIG[mname]["fn"]()
            m.fit(model_df[features], model_df["Wholesale_Price"])
            preds = [
                m.predict(build_future_row(fset_name, fset, full_df, i, current_krw, war_val))[0]
                for i in range(1, horizon + 1)
            ]
            future_results[mname] = preds
            fig.add_trace(go.Scatter(
                x=[model_df.index[-1]] + future_dates,
                y=[model_df["Wholesale_Price"].iloc[-1]] + preds,
                name=f"{mname} 예측",
                line=dict(color=MODEL_CONFIG[mname]["color"], width=3, dash=MODEL_CONFIG[mname]["dash"]),
            ))

        fig.update_layout(
            template="plotly_white", height=600,
            xaxis_title="날짜", yaxis_title="도매요금 (원/MJ)", hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        current_price = model_df["Wholesale_Price"].iloc[-1]
        cols = st.columns(len(selected_models) + 1)
        cols[0].metric("현재 요금", f"{current_price:.2f}원")
        for i, mname in enumerate(selected_models):
            pred_end = future_results[mname][-1]
            cols[i + 1].metric(
                f"{mname} ({horizon}개월 후)",
                f"{pred_end:.2f}원",
                f"{pred_end - current_price:+.2f}원",
                delta_color="inverse",
            )
        st.info(f"💡 {'전쟁 리스크' if war_risk else '정상'} 시나리오 | 환율 고정: {current_krw:,.0f}원")

        with st.expander("📋 월별 예측 상세"):
            tbl = pd.DataFrame(
                {n: future_results[n] for n in selected_models},
                index=[d.strftime("%Y-%m") for d in future_dates],
            )
            tbl.index.name = "예측 월"
            st.dataframe(tbl.style.format("{:.2f}원"), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # 백테스팅
    # ════════════════════════════════════════════════════════════════════════════
    elif mode == "백테스팅 (Validation)":
        st.subheader("🧪 모델 백테스팅 검증")

        available_months = model_df.index.to_period("M").unique().astype(str).tolist()
        default_idx      = max(0, len(available_months) - (horizon + 1))
        selected_month_str = st.sidebar.selectbox(
            "학습 종료(Cut-off) 월", options=available_months, index=default_idx,
            help=f"선택 월까지 학습 → 이후 {horizon}개월 검증",
        )

        split_date = pd.to_datetime(selected_month_str).to_period("M").to_timestamp(how="end")
        train_df   = model_df[model_df.index <= split_date]
        test_df    = model_df[model_df.index > split_date].copy()

        if test_df.empty:
            st.warning("⚠️ 선택한 달 이후 데이터가 없습니다.")
        else:
            all_metrics = {}
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_df.index, y=train_df["Wholesale_Price"],
                name="학습 데이터", line=dict(color="gray", width=1.5, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=test_df.index, y=test_df["Wholesale_Price"],
                name="테스트 실제", line=dict(color="#2C3E50", width=3),
            ))

            for mname in selected_models:
                m = MODEL_CONFIG[mname]["fn"]()
                m.fit(train_df[features], train_df["Wholesale_Price"])
                preds = m.predict(test_df[features])
                test_df[f"Pred_{mname}"] = preds
                all_metrics[mname] = calc_metrics(test_df["Wholesale_Price"], preds)
                fig.add_trace(go.Scatter(
                    x=test_df.index, y=preds, name=f"{mname} 예측",
                    line=dict(color=MODEL_CONFIG[mname]["color"], width=2.5, dash=MODEL_CONFIG[mname]["dash"]),
                ))

            fig.add_vrect(
                x0=test_df.index[0], x1=test_df.index[-1],
                fillcolor="rgba(231,76,60,0.07)", layer="below", line_width=0,
                annotation_text="검증 구간",
            )
            fig.update_layout(
                title=f"백테스팅: {selected_month_str} 이후 ({fset_name})",
                xaxis_title="날짜", yaxis_title="도매요금 (원/MJ)",
                template="plotly_white", height=600, hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📊 모델 성능 비교")
            render_metrics_table(all_metrics)

            metric_tabs = st.tabs(["R²", "MAE", "RMSE", "MAPE"])
            for tab, key in zip(metric_tabs, ["R²", "MAE", "RMSE", "MAPE"]):
                with tab:
                    cols   = st.columns(len(selected_models))
                    best_v = max(all_metrics[n][key] for n in selected_models) if key == "R²" \
                             else min(all_metrics[n][key] for n in selected_models)
                    for i, mname in enumerate(selected_models):
                        val = all_metrics[mname][key]
                        cols[i].metric(f"{'🏆 ' if val == best_v else ''}{mname}", fmt_metric(key, val))

            st.markdown("### 🕸️ 종합 성능 레이더 차트")
            render_radar(all_metrics, selected_models)

            with st.expander("📝 검증 데이터 상세"):
                disp = ["Wholesale_Price"] + [f"Pred_{n}" for n in selected_models]
                rmap = {"Wholesale_Price": "실제값"}
                rmap.update({f"Pred_{n}": n for n in selected_models})
                st.dataframe(
                    test_df[disp].rename(columns=rmap).style.format("{:.2f}원"),
                    use_container_width=True,
                )

    # ════════════════════════════════════════════════════════════════════════════
    # 피처셋 전체 비교 (3종 나란히)
    # ════════════════════════════════════════════════════════════════════════════
    else:
        st.subheader("⚖️ 피처셋 전체 비교: 6개월 v2 vs 6개월 v1 vs 3개월")
        st.caption("동일 Cut-off · 동일 모델 기준으로 3가지 피처셋을 나란히 비교합니다.")

        # 공통 유효 기간 기준 (가장 제약 많은 TTF_Lag6 기준)
        base_months = (
            full_df.dropna(subset=["Brent_Lag6", "TTF_Lag6", "TTF_Lag3", "USD_KRW", "War_Event", "Wholesale_Price"])
            .index.to_period("M").unique().astype(str).tolist()
        )
        default_idx = max(0, len(base_months) - 7)
        selected_month_str = st.sidebar.selectbox(
            "학습 종료(Cut-off) 월", options=base_months, index=default_idx,
        )
        compare_model = st.sidebar.selectbox(
            "비교 기준 모델", options=list(MODEL_CONFIG.keys()), index=0,
        )

        split_date = pd.to_datetime(selected_month_str).to_period("M").to_timestamp(how="end")

        results_by_fset = {}
        for fname, fset_cfg in FEATURE_SETS.items():
            fdf = full_df.dropna(subset=fset_cfg["features"] + ["Wholesale_Price"])
            tr  = fdf[fdf.index <= split_date]
            te  = fdf[fdf.index > split_date].copy()
            if te.empty:
                continue
            m = MODEL_CONFIG[compare_model]["fn"]()
            m.fit(tr[fset_cfg["features"]], tr["Wholesale_Price"])
            preds = m.predict(te[fset_cfg["features"]])
            te["Predicted"] = preds
            results_by_fset[fname] = {
                "train": tr, "test": te,
                "metrics": calc_metrics(te["Wholesale_Price"], preds),
                "horizon": fset_cfg["horizon"],
                "color": fset_cfg["color"],
            }

        if len(results_by_fset) < 2:
            st.warning("⚠️ 비교 데이터 부족. 더 과거 월을 선택해 주세요.")
        else:
            # ── 3열 차트 ──────────────────────────────────────────────────────
            chart_cols = st.columns(len(results_by_fset))
            for col, (fname, res) in zip(chart_cols, results_by_fset.items()):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=res["train"].index, y=res["train"]["Wholesale_Price"],
                    name="학습", line=dict(color="gray", width=1.5, dash="dot"),
                ))
                fig.add_trace(go.Scatter(
                    x=res["test"].index, y=res["test"]["Wholesale_Price"],
                    name="실제", line=dict(color="#2C3E50", width=3),
                ))
                fig.add_trace(go.Scatter(
                    x=res["test"].index, y=res["test"]["Predicted"],
                    name="예측", line=dict(color=res["color"], width=2.5, dash="dash"),
                ))
                fig.add_vrect(
                    x0=res["test"].index[0], x1=res["test"].index[-1],
                    fillcolor="rgba(231,76,60,0.06)", layer="below", line_width=0,
                )
                r2_val = res["metrics"]["R²"]
                fig.update_layout(
                    title=f"{fname}<br><sup>{compare_model} | R²={r2_val:.4f}</sup>",
                    template="plotly_white", height=380,
                    hovermode="x unified", showlegend=False,
                    margin=dict(t=70),
                )
                col.plotly_chart(fig, use_container_width=True)

            # ── 성능 비교 테이블 ──────────────────────────────────────────────
            st.markdown("### 📊 피처셋별 성능 비교")
            comp_df = pd.DataFrame({fname: res["metrics"] for fname, res in results_by_fset.items()}).T
            comp_df.index.name = "피처셋"

            def highlight_best_col(s):
                is_best = s == s.max() if s.name == "R²" else s == s.min()
                return ["background-color:#d4efdf;font-weight:bold" if v else "" for v in is_best]

            st.dataframe(
                comp_df.style.apply(highlight_best_col, axis=0)
                    .format({"R²": "{:.4f}", "MAE": "{:.2f}원", "RMSE": "{:.2f}원", "MAPE": "{:.2f}%"}),
                use_container_width=True,
            )

            # ── 레이더 (피처셋 간 비교) ───────────────────────────────────────
            if len(results_by_fset) >= 2:
                st.markdown("### 🕸️ 피처셋 종합 성능 레이더")
                metric_keys = ["R²", "MAE", "RMSE", "MAPE"]
                norm = {}
                for key in metric_keys:
                    vals = {n: results_by_fset[n]["metrics"][key] for n in results_by_fset}
                    mn, mx = min(vals.values()), max(vals.values())
                    rng = mx - mn if mx != mn else 1
                    for n in results_by_fset:
                        if n not in norm:
                            norm[n] = {}
                        score = (vals[n] - mn) / rng
                        norm[n][key] = score if key == "R²" else 1 - score

                cats     = metric_keys + [metric_keys[0]]
                radar_fig = go.Figure()
                for fname, res in results_by_fset.items():
                    scores = [norm[fname][k] for k in metric_keys] + [norm[fname][metric_keys[0]]]
                    radar_fig.add_trace(go.Scatterpolar(
                        r=scores, theta=cats, fill="toself",
                        name=fname, line_color=res["color"], opacity=0.65,
                    ))
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template="plotly_white", height=430,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=11)),
                )
                st.plotly_chart(radar_fig, use_container_width=True)
                st.caption("※ 각 지표를 0~1 정규화. 넓을수록 종합 성능 우수")

            # ── 자동 해석 ─────────────────────────────────────────────────────
            st.markdown("### 💡 분석 결과 해석")
            fnames = list(results_by_fset.keys())
            if len(fnames) >= 3:
                mv2  = results_by_fset[fnames[0]]["metrics"]  # 6개월 v2 (TTF_Lag6 포함)
                mv1  = results_by_fset[fnames[1]]["metrics"]  # 6개월 v1
                m3   = results_by_fset[fnames[2]]["metrics"]  # 3개월

                d_v2_vs_v1_r2   = mv2["R²"]   - mv1["R²"]
                d_v2_vs_v1_mape = mv2["MAPE"] - mv1["MAPE"]
                d_v2_vs_3m_r2   = mv2["R²"]   - m3["R²"]
                d_v2_vs_3m_mape = mv2["MAPE"] - m3["MAPE"]

                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown("**v2 vs v1 (TTF_Lag6 추가 효과)**")
                    if d_v2_vs_v1_r2 > 0:
                        st.success(f"✅ TTF_Lag6 추가로 R² {d_v2_vs_v1_r2:+.4f}, MAPE {d_v2_vs_v1_mape:+.2f}%p → 성능 개선")
                    else:
                        st.warning(f"⚠️ TTF_Lag6 추가 후 R² {d_v2_vs_v1_r2:+.4f}, MAPE {d_v2_vs_v1_mape:+.2f}%p → 개선 없음")

                with ic2:
                    st.markdown("**v2 vs 3개월 모델 (예측 지평 연장 비용)**")
                    if abs(d_v2_vs_3m_r2) < 0.05 and d_v2_vs_3m_mape < 3:
                        st.success(f"✅ 3개월 모델 대비 R² 차이 {d_v2_vs_3m_r2:+.4f}, MAPE {d_v2_vs_3m_mape:+.2f}%p → 6개월 지평 실용적")
                    else:
                        st.warning(f"⚠️ 3개월 모델 대비 R² 차이 {d_v2_vs_3m_r2:+.4f}, MAPE {d_v2_vs_3m_mape:+.2f}%p → 정확도 손실 있음")

except Exception as e:
    st.error(f"오류 발생: {e}")
    st.exception(e)