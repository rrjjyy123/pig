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

# CSS 스타일 적용 (예쁜 디자인)
st.markdown("""
<style>
    .main-header {
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
st.markdown('<div class="sub-header">아기 돼지 삼형제의 후손들과 함께 튼튼한 집을 설계하는 AI를 만들어보아요!</div>', unsafe_allow_html=True)

st.info("📢 목표: 슈퍼 태풍 ‘울프(Wolf)를 견딜 수 있는 튼튼한 집을 찾아라!")

# 사이드바
st.sidebar.header("🚀 프로젝트 단계")

# --- Step 1: Ready! ---
st.header("Step 1. Ready! 데이터 준비하고 전처리 하기")
st.markdown("인공지능을 가르치려면 먼저 **공부할 재료(데이터)**가 필요해요. 건축 자재들의 정보를 살펴봅시다.")

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
    st.markdown("### 👀 눈으로 확인하기 (3D 그래프)")
    st.markdown("마우스로 그래프를 돌려보세요! 밀도와 두께가 커지면 버팀 강도는 어떻게 변하나요?")
    
    fig = px.scatter_3d(df, x='density', y='thickness', z='wind_resistance',
                        color='wind_resistance',
                        labels={'density':'밀도 (density)', 'thickness':'두께 (thickness)', 'wind_resistance':'버팀 강도'},
                        color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📜 [소스 코드] 데이터 시각화하기"):
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
    st.markdown("### ✂️ 데이터 나누기 (훈련용 vs 시험용)")
    st.markdown("AI에게 모든 데이터를 다 보여주면 안 돼요. 나중에 잘 배웠는지 시험 보기 위해 일부는 남겨둡니다.")
    
    material_full = df[['density', 'thickness']].to_numpy()
    material_strength = df['wind_resistance'].to_numpy()
    
    train_input, test_input, train_target, test_target = train_test_split(
        material_full, material_strength, random_state=42
    )
    
    st.write(f"- **훈련 데이터(공부용)**: {train_input.shape[0]}개")
    st.write(f"- **테스트 데이터(시험용)**: {test_input.shape[0]}개")
    
    with st.expander("📜 [소스 코드] 데이터 전처리하기"):
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
st.header("Step 2. Make! 인공지능 모델 만들기")
st.markdown("이제 AI에게 '밀도와 두께를 알면 버팀 강도를 맞추는 법'을 가르쳐봅시다.")

if 'model_poly' not in st.session_state:
    st.session_state['model_poly'] = None
    st.session_state['poly_features'] = None

col_m1, col_m2 = st.columns(2)

# 모델 1: 단순 선형 회귀
with col_m1:
    st.subheader("선형 회귀 모델 훈련하기1 - 2개의 특성으로 모델 훈련")
    st.markdown("밀도와 두께, 2가지 특성을 그대로 사용하여 학습합니다.")
    if st.button("모델 훈련 (기본)"):
        lr = LinearRegression()
        lr.fit(train_input, train_target)
        score = lr.score(test_input, test_target)
        st.write(f"**점수: {score*100:.5f}점**")
        if score < 0.8:
            st.warning("점수가 높지 않아요... 좀 더 똑똑한 방법이 필요해요! 🤔")
            
    with st.expander("📜 [소스 코드] 선형 회귀"):
        st.code("""
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(test_input, test_target))
        """, language='python')

# 모델 2: 선형 회귀 (특성 공학)
with col_m2:
    st.subheader("선형 회귀 모델 훈련하기2 - 특성 공학을 사용하여 모델 훈련")
    st.markdown("특성 공학(PolynomialFeatures)을 사용하여 데이터의 특징을 확장해 학습합니다.")
    
    if st.button("모델 훈련 (특성 공학 적용)", type="primary"):
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
        st.markdown("와우! 점수가 훨씬 높아졌어요. 이제 돼지들의 집을 더 정확하게 감정할 수 있겠어요.")
        
    with st.expander("📜 [소스 코드] 특성 공학을 사용한 선형 회귀"):
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
st.header("Step 3. Predict! 슈퍼 태풍 ‘울프(Wolf)를 이겨라")

if st.session_state['model_poly'] is None:
    st.warning("☝️ 먼저 위에서 '모델 훈련 (특성 공학 적용)' 버튼을 눌러주세요!")
else:
    st.markdown("드디어 태풍이 다가왔어요! 슈퍼 태풍 ‘울프(Wolf)’의 태풍 바람은 **80m/s**입니다. 돼지들의 집이 과연 무사할까요?")
    
    # 돼지 집 데이터 로드
    pig_data_path = os.path.join(current_dir, 'pig_houses.csv')
    try:
        df_pig = pd.read_csv(pig_data_path)
        
        st.subheader("🏘️ 아기 돼지 삼형제의 집")
        st.dataframe(df_pig)
        
        if st.button("🏠 운명의 순간! 예측 결과 확인하기"):
            
            # 예측 준비
            pig_input = df_pig[['density', 'thickness']].to_numpy()
            pig_poly = st.session_state['poly_features'].transform(pig_input)
            
            # 예측 실행
            predictions = st.session_state['model_poly'].predict(pig_poly)
            
            # 결과 표시
            st.markdown("### 🌪️ 결과 발표")
            
            cols = st.columns(3)
            for i, (name, pred) in enumerate(zip(df_pig['name'], predictions)):
                with cols[i % 3]:
                    result_card = st.container()
                    with result_card:
                        st.markdown(f"#### {name}")
                        st.metric("예측된 버팀 강도", f"{pred:.1f} m/s")
                        
                        if pred >= 80:
                            st.success("✅ **안전함!**")
                            st.markdown("늑대 바람(80m/s)보다 튼튼합니다.\n\n🎉")
                        else:
                            st.error("😱 **위험함!**")
                            st.markdown("늑대 바람(80m/s)에 날아가버렸어요.\n\n💨🏚️")
            
            # 소스 코드
            with st.expander("📜 [소스 코드] 예측하기"):
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
