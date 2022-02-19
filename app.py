import numpy as np
import pandas as pd
import streamlit as st

import pickle

pipe_GNB = pickle.load(open('E:\\ML_models\\Nlp\\GNB_NLP_classifier.pkl','rb'))

def main():
    menu = ['Home','Monitor']
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home")

        with st.form(key="spam-clf"):
            text_input = st.text_area("Input here")
            submit_buttton = st.form_submit_button(label="Predict")
        if submit_buttton:
            col1, col2 = st.columns(2)

            with col1:
                st.success("Input:")
                st.write(text_input)

                st.success(prediction(text_input))

            with col2:
                st.success(pred_prob(text_input))    
    else:
        st.subheader("Monitor")    

def prediction(input):
    pred = pipe_GNB.predict([input])
    return pred[0]

def pred_prob(input):
    prob = pipe_GNB.predict_proba([input])
    return prob

main()