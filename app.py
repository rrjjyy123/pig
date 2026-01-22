# -*- coding: utf-8 -*-
import streamlit as st

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

st.markdown("""
### 👋 환영합니다!
이 프로그램은 **머신러닝(Machine Learning)**을 통해 튼튼한 집을 짓는 방법을 인공지능에게 가르치는 과정을 담고 있습니다.

### 📚 학습 순서
왼쪽 사이드바(Sidebar)에서 단계를 순서대로 선택해서 진행해주세요.

1. **Step 1. Ready!** 🏗️
   - 건축 자재 데이터를 살펴보고, AI가 공부할 수 있도록 준비합니다.
2. **Step 2. Make!** 🧠
   - AI에게 데이터를 학습시켜 똑똑한 모델을 만듭니다.
3. **Step 3. Predict!** 🌪️
   - 완성된 AI로 태풍을 이기는 튼튼한 집을 찾아냅니다.

---
**👈 왼쪽 사이드바에서 [Step 1. Ready!]를 클릭하여 시작하세요!**
""")
