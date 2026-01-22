# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Step 3. Predict!", page_icon="🌪️")

st.header("Step 3. Predict! 슈퍼 태풍 ‘울프(Wolf)를 이겨라")

if 'model_poly' not in st.session_state or st.session_state['model_poly'] is None:
    st.warning("☝️ 모델이 학습되지 않았습니다. **Step 2. Make!** 페이지에서 '똑똑한 모델 학습시키기'를 먼저 완료해주세요!")
else:
    st.markdown("드디어 태풍이 다가왔어요! 슈퍼 태풍 ‘울프(Wolf)’의 태풍 바람은 **80m/s**입니다. 돼지들의 집이 과연 무사할까요?")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # pig 폴더
    pig_data_path = os.path.join(parent_dir, 'pig_houses.csv')
    
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
        if st.session_state.get('model_poly'):
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
            st.warning("먼저 '똑똑한 모델 학습시키기'를 해주세요.")
