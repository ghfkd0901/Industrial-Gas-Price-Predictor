import streamlit as st
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="도매요금 예측 모델 v2.0", layout="wide")

# 메인 타이틀 및 개요
st.title("📊 산업용 천연가스 도매요금 예측 시스템")
st.markdown("""
이 시스템은 국제 에너지 지표와 국내 요금 간의 **시차(Lag) 영향**을 분석하여, 미래 도매요금을 예측하고 리스크를 관리합니다.
현재 **XGBoost, Random Forest** 등 최신 머신러닝 모델을 활용하여 높은 예측 정밀도를 유지하고 있습니다.
""")

st.divider()

# 1. 핵심 지표 요약 (Key Variables)
st.subheader("💡 모델 핵심 변수 (Key Drivers)")
v_col1, v_col2, v_col3, v_col4 = st.columns(4)

with v_col1:
    st.markdown("#### 🛢️ Brent Oil")
    st.caption("글로벌 유가 기준점 (6개월 시차 반영)")
with v_col2:
    st.markdown("#### 🇪🇺 TTF Gas")
    st.caption("유럽 천연가스 지수 (3개월 시차 반영)")
with v_col3:
    st.markdown("#### 💵 USD/KRW")
    st.caption("원/달러 환율 (수입 단가 결정)")
with v_col4:
    st.markdown("#### ⚠️ War Risk")
    st.caption("지정학적 리스크 가중치 시나리오")

st.divider()

# 2. 분석 프로세스 설명
st.subheader("🛠️ 주요 분석 및 검증 프로세스")
cols = st.columns(2)

with cols[0]:
    st.info("### 1단계: 데이터 분석 및 최적화")
    st.write("**• 데이터 수집:** GCP 연동을 통한 실시간 지표 확보")
    st.write("**• 상관관계/산점도:** 변수 간 선행·후행 관계 도출")
    st.write("**• 최적의 Lag 탐색:** 요금에 가장 영향력이 큰 시차 지점 탐색")

with cols[1]:
    st.success("### 2단계: 모델링 및 성과 검증")
    st.write("**• 앙상블 모델 비교:** XGBoost, RF, GBR 모델 성능 비교")
    st.write("**• 통합 예측:** 미래 리스크 시나리오 기반 3개월 전망")
    st.write("**• 백테스팅:** $R^2$, MAE 등 지표를 통한 모델 신뢰도 검증")

st.divider()

# 3. 모델 아키텍처 및 기대효과
with st.expander("📌 시스템 운영 개요 및 기대효과", expanded=True):
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown("**[예측 알고리즘]**")
        st.write("- 통계적 시계열 분석과 머신러닝 기법의 결합")
        st.write("- 과거 데이터 학습을 통한 변동성 자동 학습")
    with e_col2:
        st.markdown("**[기대 효과]**")
        st.write("- 데이터 기반의 객관적인 마케팅 전략 수립")
        st.write("- 국제 정세 변화에 따른 신속한 수요 이탈 대응")

# 하단 푸터
st.divider()
st.caption(f"운영 부서: 대성에너지 마케팅팀 기획 부서 | 개발자: 배경호 대리 | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d')}")