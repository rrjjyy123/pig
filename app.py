# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import os

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="아기 돼지 삼형제 AI", layout="wide", page_icon="🐷")


# CSS 스타일 적용 (블록 코딩 스타일)
st.markdown("""
<style>
    /* 전체 폰트 및 배경 */
    .block-container {
        padding-top: 2rem;
    }
    
    /* 블록 공통 스타일 */
    .code-block {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        color: white;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    /* 데이터 블록 (파랑) */
    .block-data {
        background-color: #4C97FF; /* Scratch Motion Blue */
        border: 2px solid #3373CC;
    }
    
    /* 인공지능/연산 블록 (초록) */
    .block-ai {
        background-color: #59C059; /* Scratch Operator Green */
        border: 2px solid #389438;
    }
    
    /* 이벤트/실행 블록 (노랑) */
    .block-run {
        background-color: #FFBF00; /* Scratch Events Yellow */
        border: 2px solid #CC9900;
        color: #333 !important;
    }
    
    /* 제어/결과 블록 (오렌지) */
    .block-control {
        background-color: #FFAB19; /* Scratch Control Orange */
        border: 2px solid #CF8B17;
    }

    /* 제목 스타일 */
    .main-header {
        font-family: 'Ownglyph_ci', sans-serif; /* 귀여운 폰트 가정 */
        font-size: 3rem;
        color: #FF5733;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333333;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 인트로 ---
st.markdown('<div class="main-header">🐷  슈퍼 태풍 ‘울프(Wolf)를 이겨라!</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">아기 돼지 삼형제와 함께하는 <b>블록 코딩 AI</b> 만들기</div>', unsafe_allow_html=True)

st.info("📢 목표: 블록을 조립해서 슈퍼 태풍 ‘울프(Wolf)를 견딜 수 있는 인공지능 집을 만들어보세요!")

# 사이드바
st.sidebar.header("🧩 AI 블록 조립소")

# --- Step 1: Ready! ---
st.header("Step 1. Ready! 데이터 블록 준비하기")
st.markdown("""
<div class="code-block block-data">
    🧱 <b>[데이터 불러오기]</b> 블록<br>
    <small>건축 자재 데이터(csv)를 가져와서 '공부할 준비'를 합니다.</small>
</div>
""", unsafe_allow_html=True)

# 1. 데이터 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
# 단일 페이지이므로 같은 폴더에 있음
data_path = os.path.join(current_dir, 'construction_materials_data.csv')

try:
    df = pd.read_csv(data_path)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 건축 자재 데이터")
        st.write(df.head(10))
        st.caption("데이터의 앞부분 10개만 보여줍니다.")
    
    with col2:
        st.subheader("💡 데이터 설명")
        st.markdown("""
        - **density (밀도)**: 재료가 얼마나 단단하게 뭉쳐있는지 (kg/m³)
        - **thickness (두께)**: 재료의 두께 (cm)
        - **wind_resistance (버팀 강도)**: 바람을 얼마나 잘 견디는지 (m/s)
        """)
        
    # 소스 코드 보기
    with st.expander("📜 [소스 코드] 데이터 불러오기"):
        st.code("""
import pandas as pd

# 건축 자재 데이터 준비
df = pd.read_csv('construction_materials_data.csv')
print(df.head(10))
        """, language='python')

    # 2. 데이터 시각화
    st.markdown("""
    <div class="code-block block-data">
        👀 <b>[데이터 확인하기]</b> 블록<br>
        <small>데이터가 어떻게 생겼는지 3차원 그래프로 확인합니다.</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("마우스로 그래프를 돌려보세요! 밀도와 두께가 커지면 버팀 강도는 어떻게 변하나요?")
    
    fig = px.scatter_3d(df, x='density', y='thickness', z='wind_resistance',
                        color='wind_resistance',
                        labels={'density':'밀도 (density)', 'thickness':'두께 (thickness)', 'wind_resistance':'버팀 강도'},
                        color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔍 블록 속 내용 보기 (Python 코드)"):
        st.code("""
import matplotlib.pyplot as plt

fig = plt.figure(dpi = 300)
ax = fig.gca(projection = '3d')

# 3차원 그래프 그리기
ax.scatter(df['density'], df['thickness'], df['wind_resistance'])
ax.set_xlabel('density')
ax.set_ylabel('thickness')
ax.set_zlabel('wind resistance')
plt.show()
        """, language='python')

    # 3. 데이터 전처리 (Train/Test Split)
    st.markdown("""
    <div class="code-block block-data">
        ✂️ <b>[데이터 나누기]</b> 블록<br>
        <small>전체 데이터를 <b>공부용(Train)</b>과 <b>시험용(Test)</b>으로 나눕니다.</small>
    </div>
    """, unsafe_allow_html=True)
    
    material_full = df[['density', 'thickness']].to_numpy()
    material_strength = df['wind_resistance'].to_numpy()
    
    train_input, test_input, train_target, test_target = train_test_split(
        material_full, material_strength, random_state=42
    )
    
    st.write(f"- **훈련 데이터(공부용)**: {train_input.shape[0]}개")
    st.write(f"- **테스트 데이터(시험용)**: {test_input.shape[0]}개")
    
    with st.expander("🔍 블록 속 내용 보기 (Python 코드)"):
        st.code("""
from sklearn.model_selection import train_test_split

# 데이터를 넘파이 배열로 변환
material_full = df[['density', 'thickness']].to_numpy()
material_strength = df['wind_resistance'].to_numpy()

# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(
    material_full, material_strength, random_state=42
)
        """, language='python')

except FileNotFoundError:
    st.error("데이터 파일(construction_materials_data.csv)을 찾을 수 없습니다. setup_pig_project.py를 먼저 실행해주세요.")
    st.stop()


st.divider()

# --- Step 2: Make! ---
st.header("Step 2. Make! 인공지능 블록 조립하기")
st.markdown("AI가 공부하는 방식을 선택해서 **학습 블록**을 실행해봅시다.")

if 'model_poly' not in st.session_state:
    st.session_state['model_poly'] = None
    st.session_state['poly_features'] = None

col_m1, col_m2 = st.columns(2)

# 모델 1: 단순 선형 회귀
with col_m1:
    st.markdown("""
    <div class="code-block block-ai">
        🤖 <b>[기본 AI]</b> 만들기<br>
        <small>밀도와 두께만 가지고 단순하게 생각합니다.</small>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("▶️ 기본 학습 블록 실행"):
        lr = LinearRegression()
        lr.fit(train_input, train_target)
        score = lr.score(test_input, test_target)
        st.write(f"**점수: {score*100:.5f}점**")
        if score < 0.8:
            st.warning("음... 점수가 좀 낮네요. 더 똑똑한 블록이 필요해요!")
            
    with st.expander("🔍 블록 속 내용 보기 (Python 코드)"):
        st.code("""
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(test_input, test_target))
        """, language='python')

# 모델 2: 선형 회귀 (특성 공학)
with col_m2:
    st.markdown("""
    <div class="code-block block-ai">
        🧠 <b>[슈퍼 AI]</b> 만들기 (특성 공학)<br>
        <small>데이터를 응용(제곱, 서로 곱하기)해서 더 깊게 생각합니다!</small>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("▶️ 슈퍼 학습 블록 실행", type="primary"):
        # 특성 공학
        poly = PolynomialFeatures(degree=3, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        
        # 모델 학습
        lr_poly = LinearRegression()
        lr_poly.fit(train_poly, train_target)
        score_poly = lr_poly.score(test_poly, test_target)
        
        st.session_state['model_poly'] = lr_poly
        st.session_state['poly_features'] = poly
        
        st.success(f"**점수: {score_poly*100:.5f}점!** 🎉")
        st.balloons()
        st.markdown("대단해요! 이제 **예측 블록**을 사용할 준비가 되었어요.")
        
    with st.expander("🔍 블록 속 내용 보기 (Python 코드)"):
        st.code("""
from sklearn.preprocessing import PolynomialFeatures

# 특성 공학 (3차원)
poly = PolynomialFeatures(degree=3, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

# 모델 학습
lr.fit(train_poly, train_target)
print(lr.score(test_poly, test_target))
        """, language='python')

st.divider()

# --- Step 3: Predict! ---
st.header("Step 3. Predict! 예측 블록 실행하기")

st.markdown("""
<div class="code-block block-run">
    🌪️ <b>태풍 '울프' 시뮬레이션</b> 블록<br>
    <small>완성된 AI 모델을 사용하여 80m/s 태풍을 견딜 수 있는지 확인합니다.</small>
</div>
""", unsafe_allow_html=True)

if st.session_state['model_poly'] is None:
    st.warning("☝️ 먼저 Step 2에서 **[슈퍼 학습 블록 실행]** 버튼을 눌러주세요!")
else:
    # 돼지 집 데이터 로드
    pig_data_path = os.path.join(current_dir, 'pig_houses.csv')
    try:
        df_pig = pd.read_csv(pig_data_path)
        
        st.subheader("🏘️ 아기 돼지 삼형제의 집")
        st.dataframe(df_pig)
        
        if st.button("🚩 예측 블록 실행 (Run)"):
            
            # 예측 준비
            pig_input = df_pig[['density', 'thickness']].to_numpy()
            pig_poly = st.session_state['poly_features'].transform(pig_input)
            
            # 예측 실행
            predictions = st.session_state['model_poly'].predict(pig_poly)
            
            # 결과 표시
            st.markdown("### 🌪️ 시뮬레이션 결과")
            
            cols = st.columns(3)
            for i, (name, pred) in enumerate(zip(df_pig['name'], predictions)):
                with cols[i % 3]:
                    result_card = st.container()
                    with result_card:
                        st.markdown(f"#### {name}")
                        st.metric("예측된 버팀 강도", f"{pred:.5f} m/s")
                        
                        if pred >= 80:
                            st.success("✅ **안전함!**")
                            st.markdown("늑대 바람(80m/s)보다 튼튼합니다.\n\n🎉")
                        else:
                            st.error("😱 **위험함!**")
                            st.markdown("늑대 바람(80m/s)에 날아가버렸어요.\n\n💨🏚️")
            
            # 소스 코드
            with st.expander("🔍 블록 속 내용 보기 (Python 코드)"):
                st.code("""
# 돼지들의 집 데이터 준비
df = pd.read_csv('pig_houses.csv')
pig_houses_full = df[['density', 'thickness']].to_numpy()

# 예측하기 (다항 특성 변환 후 예측)
# x0, x1, x0^2, x0x1, ... 등의 특성을 자동으로 만들어줘요.
prediction = lr.predict(poly.transform(pig_houses_full))

# 결과 확인 (80m/s 이상인지)
for i in range(len(prediction)):
    if prediction[i] >= 80:
        print(f"{df['name'][i]}: 안전!")
    else:
        print(f"{df['name'][i]}: 위험!")
                """, language='python')

    except FileNotFoundError:
        st.error("pig_houses.csv 파일을 찾을 수 없습니다.")

# (Optional) 직접 집 지어보기
st.divider()
with st.expander("🛠️ 보너스: 나만의 튼튼한 집 설계하기"):
    st.markdown("여러분이 직접 자재를 골라보세요. 과연 늑대를 이길 수 있을까요?")
    c1, c2 = st.columns(2)
    with c1:
        my_density = st.slider("재료 밀도 (kg/m³)", 0, 3000, 1000)
    with c2:
        my_thickness = st.slider("벽 두께 (cm)", 0, 100, 20)
        
    if st.button("내 집 테스트하기"):
        if st.session_state['model_poly']:
            my_input = np.array([[my_density, my_thickness]])
            my_poly = st.session_state['poly_features'].transform(my_input)
            my_pred = st.session_state['model_poly'].predict(my_poly)[0]
            
            st.metric("내 집의 버팀 강도", f"{my_pred:.1f} m/s")
            
            if my_pred >= 80:
                st.balloons()
                st.success("대단해요! 늑대도 울고 갈 튼튼한 집이네요! 🏆")
            else:
                st.error("아이고... 바람에 날아가버렸어요. 더 튼튼한 재료를 써보세요! 🍃")
        else:
            st.warning("먼저 '모델 훈련 (특성 공학 적용)'을 해주세요.")
