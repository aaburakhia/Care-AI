# style_utils.py
import streamlit as st

def add_custom_css():
    st.markdown(
        """
        <style>
        /* --- 1. BACKGROUND STYLING (Existing) --- */
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(to right, #0A192F, #1E3A8A);
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] {
            background-color: #0A192F;
        }

        /* --- 2. HIDE STREAMLIT BRANDING (New & Important) --- */
        /* This hides the "Made with Streamlit" footer */
        footer {visibility: hidden;}

        /* This hides the top-right hamburger menu */
        #MainMenu {visibility: hidden;}

        /* This hides the colored line at the very top of the page */
        header[data-testid="stHeader"] {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
