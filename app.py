import streamlit as st

# 페이지 설정
st.set_page_config(page_title="도매요금 예측 모델", layout="wide")

# 메인 타이틀
st.title("📊 산업용 천연가스 도매요금 예측 시스템")
st.markdown("---")

# 각 서브페이지 요약 설명
st.subheader("🛠️ 주요 분석 프로세스")

cols = st.columns(2)

with cols[0]:
    st.info("### 1단계: 데이터 및 탐색")
    st.write("**• 데이터수집:** 분석에 필요한 기초 지표 확보")
    st.write("**• 데이터탐색(상관/산점도):** 변수 간 영향력 분석")
    st.write("**• 최적의 Lag 찾기:** 선행 지표의 시차 결정")

with cols[1]:
    st.success("### 2단계: 모델링 및 검증")
    st.write("**• 모델비교:** 다양한 알고리즘별 성능 평가")
    st.write("**• 통합예측대시보드:** 최종 예측 결과 시각화")
    st.write("**• 백테스팅(일반/6개월):** 과거 데이터를 통한 모델 신뢰도 검증")

st.markdown("---")
st.caption("대성에너지 마케팅팀 기획 부서 운영")