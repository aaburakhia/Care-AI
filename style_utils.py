# style_utils.py
import streamlit as st

def add_custom_css():
    st.markdown(
        """
        <style>
        /* --- 1. BACKGROUND STYLING --- */
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(to right, #0A192F, #1E3A8A);
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] {
            background-color: #0A192F;
        }

        /* --- 2. HIDE STREAMLIT BRANDING (AGGRESSIVE) --- */
        /* This hides the Streamlit Toolbar, which includes the Share, Star, and GitHub icons */
        [data-testid="stToolbar"] {
            display: none !important;
        }
        
        /* This hides the "Made with Streamlit" footer */
        footer {
            display: none !important;
        }
        
        /* This hides the top-right hamburger menu */
        #MainMenu {
            display: none !important;
        }

        /* This hides the colored line at the very top of the screen */
        header[data-testid="stHeader"] {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
