import streamlit as st
from supabase import create_client, Client

# This function creates and returns a Supabase client instance.
# It uses Streamlit's secrets management to get the credentials.
# The @st.cache_resource annotation ensures we only create one client object
# per user session, which is more efficient.

@st.cache_resource
def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client, caching the resource for efficiency.
    """
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    
    # The create_client function initializes the connection to your Supabase project.
    return create_client(supabase_url, supabase_key)
