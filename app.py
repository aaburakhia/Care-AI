# 0_Home.py

import streamlit as st
from supabase_client import get_supabase_client
from style_utils import add_custom_css 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Care-AI Analytics | Home", 
    page_icon="🩺",
    layout="centered"
)

# --- CUSTOM STYLING ---
add_custom_css()

# --- INITIALIZE SUPABASE CLIENT ---
supabase = get_supabase_client()

# --- USER AUTHENTICATION ---
st.title("Care-AI Analytics: The Integrated Health Co-Pilot")

# Initialize session state to keep track of user's login status
if 'user' not in st.session_state:
    st.session_state.user = None

# --- Main App Logic ---

# If user is not logged in, show the login/signup form
if st.session_state.user is None:
    st.write("Welcome! Please sign in or create an account to continue.")
    
    signup_tab, login_tab = st.tabs(["Sign Up", "Login"])

    with signup_tab:
        with st.form("signup_form", clear_on_submit=True):
            st.write("### Create a New Account")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            
            if st.form_submit_button("Sign Up"):
                try:
                    user = supabase.auth.sign_up({
                        "email": new_email,
                        "password": new_password,
                    })
                    st.success("Account created successfully! Please check your email to verify.")
                except Exception as e:
                    st.error(f"Error during sign-up: {e}")

    with login_tab:
        with st.form("login_form", clear_on_submit=True):
            st.write("### Sign In to Your Account")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.form_submit_button("Login"):
                try:
                    user = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password
                    })
                    st.session_state.user = user
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during login: {e}")

# If the user IS logged in, show the main application content
else:
    st.success(f"Welcome back, {st.session_state.user.user.email}!")
    st.markdown("---")
    st.header("Your Integrated Health Co-Pilot")
    st.write("Please select a feature from the sidebar on the left to get started.")
    st.info("Start with the **Symptom Checker** to get a preliminary analysis of your symptoms.", icon="🩺")
    
    if st.button("Logout"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()
