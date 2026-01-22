# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Step 2. Make!", page_icon="🧠")

st.header("Step 2. Make! 인공지능 모델 만들기")
st.markdown("이제 AI에게 '밀도와 두께를 알면 버팀 강도를 맞추는 법'을 가르쳐봅시다.")

# 데이터 확인
if 'train_input' not in st.session_state:
    st.warning("⚠️ 데이터가 준비되지 않았습니다. **Step 1. Ready!** 페이지를 먼저 실행해주세요.")
    st.stop()

train_input = st.session_state['train_input']
test_input = st.session_state['test_input']
train_target = st.session_state['train_target']
test_target = st.session_state['test_target']

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
        st.write(f"**점수: {score*100:.1f}점**")
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
        
        st.success(f"**점수: {score_poly*100:.1f}점!** 🎉")
        st.balloons()
        st.markdown("와우! 점수가 훨씬 높아졌어요. 이제 돼지들의 집을 더 정확하게 감정할 수 있겠어요.")
        st.info("이제 **Step 3**로 이동해서 예측을 해봅시다!")
        
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
