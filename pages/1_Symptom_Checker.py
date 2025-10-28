# pages/Symptom_Checker.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
from datetime import datetime
from gemini_client import get_symptom_analysis
from supabase_client import get_supabase_client, save_symptom_analysis, get_symptom_history
from style_utils import add_custom_css

# --- Page Configuration & Styling ---
st.set_page_config(page_title="Symptom Analysis", page_icon="🩺")
add_custom_css()
st.title("AI-Powered Symptom Analysis")

# --- Authentication & Supabase Client ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()
    
supabase = get_supabase_client()

# --- Create Tabs for Checker and History ---
tab1, tab2 = st.tabs(["Check Symptoms", "View History"])


# --- Tab 1: The Symptom Checker Tool ---
with tab1:
    st.header("Check Your Symptoms")
    st.write("Describe your symptoms, and our AI co-pilot will provide a preliminary analysis and suggest next steps.")
    st.info("This tool does **not** provide a medical diagnosis. Always consult with a healthcare professional for health concerns.", icon="⚠️")

    # Input validation function (specific to this tab)
    def is_valid_input(symptoms: str) -> bool:
        if len(symptoms.split()) < 3:
            st.error("Please provide a more detailed description (at least 3 words).")
            return False
        symptom_keywords = ['pain', 'ache', 'fever', 'headache', 'cough', 'sore', 'throat', 'nausea', 'dizzy', 'fatigue', 'tired', 'rash', 'itchy', 'swelling', 'breathing', 'stomach', 'cramp', 'chills', 'vomit', 'diarrhea', 'runny', 'nose']
        if not any(keyword in symptoms.lower() for keyword in symptom_keywords):
            st.error("Your description does not seem to contain common medical symptoms.")
            return False
        return True

    with st.form("symptom_form"):
        symptoms_input = st.text_area("Please describe your symptoms in detail:", height=150, placeholder="e.g., 'I have a high fever, a persistent dry cough, and I'm feeling very tired.'")
        submitted = st.form_submit_button("Analyze Symptoms")

    if submitted and symptoms_input:
        if is_valid_input(symptoms_input):
            with st.spinner("Analyzing your symptoms..."):
                analysis_result = get_symptom_analysis(symptoms_input)

            st.divider()
            
            if "error" in analysis_result:
                st.error(analysis_result["error"])
            else:
                save_success = save_symptom_analysis(supabase, symptoms_input, analysis_result)
                if save_success:
                    st.toast("Analysis saved to your history!", icon="✅")
                else:
                    st.toast("Could not save to history.", icon="❌")
                
                # Display logic
                st.subheader("Analysis Results")
                # ... (display logic remains the same)
                if "disclaimer" in analysis_result:
                    st.warning(f"**Disclaimer:** {analysis_result['disclaimer']}")
                if "overall_assessment" in analysis_result:
                    st.write(f"**Overall Assessment:** {analysis_result['overall_assessment']}")
                st.write("---")
                for condition in analysis_result.get("analysis", []):
                    st.subheader(f"Potential Condition: {condition.get('condition', 'N/A')}")
                    prob_score = condition.get('probability_score', 0)
                    st.write(f"**Probability Score:** {prob_score}%")
                    st.progress(prob_score)
                    st.write(f"**Explanation:** {condition.get('explanation', 'No explanation provided.')}")
                    next_steps = condition.get('suggested_next_steps', '')
                    if next_steps == 'Immediate Care':
                        st.error(f"**Suggested Next Steps:** {next_steps}", icon="🚨")
                    elif next_steps == 'Consult a Doctor':
                        st.warning(f"**Suggested Next Steps:** {next_steps}", icon="👨‍⚕️")
                    else:
                        st.info(f"**Suggested Next Steps:** {next_steps}", icon="🏠")
                    st.write("---")

# --- Tab 2: The User's Analysis History ---
with tab2:
    st.header("Your Analysis History")
    history_data = get_symptom_history(supabase)

    if not history_data:
        st.info("You have no saved analyses yet. Use the 'Check Symptoms' tab to get started.")
    else:
        st.write("Here are your past analyses, with the most recent first.")
        
        for record in history_data:
            record_date = datetime.strptime(record['created_at'], "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%B %d, %Y at %I:%M %p")
            
            with st.expander(f"**Analysis from {record_date}**"):
                st.markdown(f"**Your Symptoms:**")
                st.info(record['symptoms_input'])
                
                st.markdown(f"**AI Analysis Results:**")
                analysis_result = json.loads(record['analysis_result'])

                if "disclaimer" in analysis_result:
                    st.warning(f"**Disclaimer:** {analysis_result['disclaimer']}")
                if "overall_assessment" in analysis_result:
                    st.write(f"**Overall Assessment:** {analysis_result['overall_assessment']}")
                
                st.write("---")
                for condition in analysis_result.get("analysis", []):
                    st.subheader(f"Potential Condition: {condition.get('condition', 'N/A')}")
                    st.write(f"**Explanation:** {condition.get('explanation', 'No explanation provided.')}")
                    st.write("---")
