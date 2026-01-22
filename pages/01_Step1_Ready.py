# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Step 1. Ready!", page_icon="🏗️")

st.header("Step 1. Ready! 데이터 준비하고 전처리 하기")
st.markdown("인공지능을 가르치려면 먼저 **공부할 재료(데이터)**가 필요해요. 건축 자재들의 정보를 살펴봅시다.")

# 1. 데이터 불러오기
# app.py가 있는 폴더(pig)의 절대 경로를 구한 뒤 csv 파일을 찾습니다.
# 현재 파일: pig/pages/01_Step1_Ready.py
# 부모 폴더: pig
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # pig 폴더
data_path = os.path.join(parent_dir, 'construction_materials_data.csv')

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
    
    # 세션 상태에 데이터 저장 (다음 페이지에서 쓰기 위해)
    st.session_state['train_input'] = train_input
    st.session_state['test_input'] = test_input
    st.session_state['train_target'] = train_target
    st.session_state['test_target'] = test_target
    
    st.write(f"- **훈련 데이터(공부용)**: {train_input.shape[0]}개")
    st.write(f"- **테스트 데이터(시험용)**: {test_input.shape[0]}개")
    
    st.success("데이터 준비 완료! 왼쪽 사이드바에서 **Step 2**로 이동하세요.")
    
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
    st.error("데이터 파일(construction_materials_data.csv)을 찾을 수 없습니다. setup_pig_project.py를먼저 실행해주세요.")
