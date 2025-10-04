import streamlit as st
from supabase import create_client, Client
import json # Import the json library for handling the analysis result

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

# --- NEW FUNCTION ADDED BELOW ---

def save_symptom_analysis(supabase_client, symptoms, analysis):
    """
    Saves the user's symptom analysis to the database.
    Note: The user_id is handled automatically by the database default value.
    """
    try:
        # Supabase's python client expects a JSON string, not a dictionary, for jsonb columns.
        analysis_json_string = json.dumps(analysis)
        
        data_to_insert = {
            "symptoms_input": symptoms,
            "analysis_result": analysis_json_string
        }
        
        # Insert the data into the 'symptom_history' table
        response = supabase_client.table('symptom_history').insert(data_to_insert).execute()
        
        # Check if the insert was successful
        if len(response.data) > 0:
            return True
        return False
        
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False
