import streamlit as st
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="도매요금 예측 모델 v2.0", layout="wide")

# 메인 타이틀 및 개요
st.title("📊 산업용 천연가스 도매요금 예측 시스템")
st.markdown("""
이 시스템은 국제 에너지 지표와 국내 요금 간의 **시차(Lag) 영향**을 분석하여, 미래 도매요금을 예측하고 리스크를 관리합니다.
**XGBoost, Random Forest, Gradient Boosting** 등 최신 머신러닝 앙상블 모델을 활용하며,
**변수별 개별 Lag 최적화**와 **사용자 정의 학습 기간 설정**을 통해 높은 예측 정밀도를 유지합니다.
""")

st.divider()

# 1. 핵심 지표 요약
st.subheader("💡 모델 핵심 변수 (Key Drivers)")
v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)

with v_col1:
    st.markdown("#### 🛢️ Brent Oil")
    st.caption("글로벌 유가 기준점")
    st.metric(label="최적 Lag", value="6개월", delta="R=0.852", delta_color="normal")

with v_col2:
    st.markdown("#### ⛽ WTI Oil")
    st.caption("미국 원유 기준 지수")
    st.metric(label="최적 Lag", value="6개월", delta="R=0.874", delta_color="normal")

with v_col3:
    st.markdown("#### 🌏 JKM Gas")
    st.caption("아시아 LNG 현물가")
    st.metric(label="최적 Lag", value="3개월", delta="R=0.825", delta_color="normal")

with v_col4:
    st.markdown("#### 🇪🇺 TTF Gas")
    st.caption("유럽 천연가스 지수")
    st.metric(label="최적 Lag", value="3개월", delta="R=0.850", delta_color="normal")

with v_col5:
    st.markdown("#### 💵 USD/KRW")
    st.caption("원/달러 환율")
    st.metric(label="최적 Lag", value="0개월", delta="R=0.491", delta_color="normal")

st.divider()

# 2. 분석 프로세스 설명
st.subheader("🛠️ 주요 분석 및 검증 프로세스")
cols = st.columns(2)

with cols[0]:
    st.info("""
### 1단계: 데이터 분석 및 최적화
**• 데이터 수집:** GCP Google Sheets 연동을 통한 실시간 지표 확보  
**• 분석 기간:** 2015년 1월 ~ 현재 (결측 구간 시간 보간 처리)  
**• 상관관계/산점도:** 변수 간 선행·후행 관계 도출  
**• 최적 Lag 탐색:** 변수별 개별 시차 최적화 (Brent 6M, JKM 3M 등)
""")

with cols[1]:
    st.success("""
### 2단계: 모델링 및 성과 검증
**• 앙상블 모델 비교:** XGBoost, RF, GBR, Linear Regression 4종 비교  
**• 학습 기간 설정:** 사용자 정의 학습 구간으로 과적합 여부 실시간 확인  
**• 통합 예측:** 미래 리스크 시나리오 기반 월별 요금 전망  
**• 백테스팅:** R², MAE 지표를 통한 학습기간/전체기간 이중 검증
""")

st.divider()

# 3. 최근 업데이트 내역
st.subheader("🆕 최근 업데이트 내역")
upd_col1, upd_col2, upd_col3 = st.columns(3)

with upd_col1:
    st.markdown("""
    <div style="background:#f0f7ff; border-left:4px solid #2196F3; padding:16px; border-radius:6px;">
    <b>🎓 모델 학습 기간 설정</b><br><br>
    사용자가 직접 학습 구간을 지정할 수 있습니다.
    연도 슬라이더로 시작~종료 연도를 선택하면 모든 모델이 해당 기간 데이터만으로 즉시 재학습됩니다.<br><br>
    <span style="color:#888; font-size:0.85em;">→ 학습기간 R² + 전체기간 R² 동시 표시로 과적합 여부 즉시 확인 가능</span>
    </div>
    """, unsafe_allow_html=True)

with upd_col2:
    st.markdown("""
    <div style="background:#f0fff4; border-left:4px solid #4CAF50; padding:16px; border-radius:6px;">
    <b>📐 변수별 개별 Lag 최적화</b><br><br>
    기존 단일 Lag 방식에서 <b>변수별 개별 Lag</b> 적용 방식으로 개선되었습니다.
    상관분석 결과를 반영한 최적값이 자동 설정되며, 사이드바에서 개별 조정(0~12개월)도 가능합니다.<br><br>
    <span style="color:#888; font-size:0.85em;">→ Brent 6M / JKM 3M / TTF 3M / USD_KRW 0M</span>
    </div>
    """, unsafe_allow_html=True)

with upd_col3:
    st.markdown("""
    <div style="background:#fff8f0; border-left:4px solid #FF9800; padding:16px; border-radius:6px;">
    <b>📅 데이터 기간 확장 및 보간</b><br><br>
    분석 기준 시작일이 <b>2015년 1월</b>로 통일되었습니다.
    월별 리샘플링 후 시간 기반 선형 보간을 적용하여 중간 결측 구간도 자연스럽게 처리됩니다.<br><br>
    <span style="color:#888; font-size:0.85em;">→ resample('MS') → interpolate(method='time') → ffill/bfill</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# 4. 시스템 구성 및 기대효과
with st.expander("📌 시스템 운영 개요 및 기대효과", expanded=True):
    e_col1, e_col2, e_col3 = st.columns(3)

    with e_col1:
        st.markdown("**[예측 알고리즘]**")
        st.write("- 통계적 시계열 분석과 머신러닝 기법의 결합")
        st.write("- 과거 데이터 학습을 통한 변동성 자동 학습")
        st.write("- Holt 이중지수평활법으로 미래 지표 자동 생성")

    with e_col2:
        st.markdown("**[시스템 구성]**")
        st.write("- 통합 분석 탭: 4종 모델 성능 비교 및 변수 중요도")
        st.write("- 시뮬레이터 탭: 시나리오 편집 및 월별 요금 전망")
        st.write("- 데이터 소스: Google Sheets (Master_Data, gas_price)")

    with e_col3:
        st.markdown("**[기대 효과]**")
        st.write("- 데이터 기반의 객관적인 마케팅 전략 수립")
        st.write("- 국제 정세 변화에 따른 신속한 수요 이탈 대응")
        st.write("- 학습 구간 조정으로 시장 국면별 맞춤 예측 가능")

# 하단 푸터
st.divider()
st.caption(f"운영 부서: 대성에너지 마케팅팀 기획 부서 | 개발자: 배경호 대리 | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d')}")