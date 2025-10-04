# pages/4_Analysis_History.py

import streamlit as st
import json
from datetime import datetime
from supabase_client import get_supabase_client, get_symptom_history
from style_utils import add_custom_css

# --- Page Configuration & Styling ---
st.set_page_config(page_title="Analysis History", page_icon="📜")
add_custom_css()
st.title("Your Symptom Analysis History")

# --- Authentication & Supabase Client ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access your history.")
    st.stop()

supabase = get_supabase_client()

# --- Fetch and Display History ---
history_data = get_symptom_history(supabase)

if not history_data:
    st.info("You have no saved analyses yet. Use the Symptom Checker to get started.")
else:
    st.write("Here are your past analyses, with the most recent first.")
    
    # Loop through each saved record and display it in an expandable container
    for record in history_data:
        # The date from Supabase is a detailed string; let's format it nicely.
        record_date = datetime.strptime(record['created_at'], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%B %d, %Y at %I:%M %p")
        
        with st.expander(f"**Analysis from {record_date}**"):
            
            # --- Display the user's original input ---
            st.markdown(f"**Your Symptoms:**")
            st.info(record['symptoms_input'])
            
            # --- Display the AI's analysis ---
            st.markdown(f"**AI Analysis Results:**")
            
            # The result is stored as a JSON string, so we must parse it back into a Python dictionary
            analysis_result = json.loads(record['analysis_result'])

            # Now we can display the details from the dictionary
            if "disclaimer" in analysis_result:
                st.warning(f"**Disclaimer:** {analysis_result['disclaimer']}")
            if "overall_assessment" in analysis_result:
                st.write(f"**Overall Assessment:** {analysis_result['overall_assessment']}")
            
            st.write("---")
            for condition in analysis_result.get("analysis", []):
                st.subheader(f"Potential Condition: {condition.get('condition', 'N/A')}")
                st.write(f"**Explanation:** {condition.get('explanation', 'No explanation provided.')}")
                st.write("---")
