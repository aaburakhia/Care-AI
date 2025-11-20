import streamlit as st
from supabase import create_client, Client
import json

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

def get_symptom_history(supabase_client):
    """
    Retrieves the symptom analysis history for the currently logged-in user.
    """
    try:
        # The .select() query automatically uses the RLS policies you created.
        # It will only ever return rows where the user_id matches the logged-in user.
        # We order by 'created_at' descending to show the newest entries first.
        response = supabase_client.table('symptom_history').select('*').order('created_at', desc=True).execute()
        
        if response.data:
            return response.data
        return [] # Return an empty list if there's no history
        
    except Exception as e:
        print(f"Error fetching history from database: {e}")
        return []

# --- NEW CHAT FUNCTIONS ADDED BELOW ---

def create_chat_conversation(supabase_client):
    """Creates a new chat conversation for the current user and returns its ID."""
    try:
        # We only need to insert a row; user_id is handled by the default value.
        response = supabase_client.table('chat_conversations').insert({}).execute()
        
        if response.data:
            # The database returns the new row's data, including the generated ID.
            return response.data[0]['id']
        return None
    except Exception as e:
        print(f"Error creating conversation: {e}")
        return None

def save_chat_message(supabase_client, conversation_id, role, content):
    """Saves a single chat message to the database."""
    try:
        supabase_client.table('chat_messages').insert({
            "conversation_id": conversation_id,
            "role": role,
            "content": content
        }).execute()
        return True
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return False

def get_chat_history(supabase_client):
    """Retrieves all conversations and their messages for the current user."""
    # This is a placeholder for a future feature to view past chats.
    pass

# --- MEDICATION MANAGEMENT FUNCTIONS ---

def save_medication(supabase_client, medication_data):
    """
    Saves a medication record to the database.
    Args:
        supabase_client: Supabase client instance
        medication_data: Dictionary containing medication information
    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        # Convert medication data to JSON string for storage
        medication_json = json.dumps(medication_data)
        
        data_to_insert = {
            "medication_data": medication_json
        }
        
        response = supabase_client.table('medications').insert(data_to_insert).execute()
        
        if response.data and len(response.data) > 0:
            return (True, response.data[0]['id'])
        return (False, "No data returned from database")
        
    except Exception as e:
        error_msg = str(e)
        # Check for common errors
        if "relation" in error_msg and "does not exist" in error_msg:
            return (False, "medications_table_missing")
        elif "violates row-level security policy" in error_msg or "RLS" in error_msg:
            return (False, "rls_policy_error")
        else:
            print(f"Error saving medication: {e}")
            return (False, f"Database error: {error_msg}")

def get_medications(supabase_client):
    """
    Retrieves all medications for the currently logged-in user.
    Returns:
        List of medication records
    """
    try:
        response = supabase_client.table('medications').select('*').order('created_at', desc=False).execute()
        
        if response.data:
            # Parse the JSON data for each medication
            medications = []
            for record in response.data:
                med_data = json.loads(record['medication_data'])
                med_data['id'] = record['id']
                med_data['db_created_at'] = record['created_at']
                medications.append(med_data)
            return medications
        return []
        
    except Exception as e:
        print(f"Error fetching medications: {e}")
        return []

def update_medication(supabase_client, medication_id, medication_data):
    """
    Updates an existing medication record.
    Args:
        supabase_client: Supabase client instance
        medication_id: ID of the medication to update
        medication_data: Updated medication data
    Returns:
        True if successful, False otherwise
    """
    try:
        medication_json = json.dumps(medication_data)
        
        response = supabase_client.table('medications').update({
            "medication_data": medication_json
        }).eq('id', medication_id).execute()
        
        return len(response.data) > 0
        
    except Exception as e:
        print(f"Error updating medication: {e}")
        return False

def delete_medication(supabase_client, medication_id):
    """
    Deletes a medication record.
    Args:
        supabase_client: Supabase client instance
        medication_id: ID of the medication to delete
    Returns:
        True if successful, False otherwise
    """
    try:
        response = supabase_client.table('medications').delete().eq('id', medication_id).execute()
        return True
        
    except Exception as e:
        print(f"Error deleting medication: {e}")
        return False
