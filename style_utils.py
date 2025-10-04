# style_utils.py
import streamlit as st

def add_custom_css():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(to right, #0A192F, #1E3A8A);
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] {
            background-color: #0A192F;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
