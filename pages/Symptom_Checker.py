import streamlit as st
import pandas as pd
from gemini_client import get_symptom_analysis

# --- Page Configuration ---
st.set_page_config(page_title="Symptom Checker", page_icon="🩺")
st.title("AI-Powered Symptom Checker")

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()

# --- Main Page Content ---
st.write("Describe your symptoms, and our AI co-pilot will provide a preliminary analysis and suggest next steps.")
st.info("This tool does **not** provide a medical diagnosis. Always consult with a healthcare professional for health concerns.", icon="⚠️")

with st.form("symptom_form"):
    symptoms_input = st.text_area("Please describe your symptoms in detail:", height=150, placeholder="e.g., 'I have a high fever, a persistent dry cough, and I'm feeling very tired.'")
    submitted = st.form_submit_button("Analyze Symptoms")

if submitted and symptoms_input:
    with st.spinner("Analyzing your symptoms... This may take a moment."):
        analysis_result = get_symptom_analysis(symptoms_input)

    st.divider()
    
    if "error" in analysis_result:
        st.error(analysis_result["error"])
    else:
        st.subheader("Analysis Results")
        
        # Display Disclaimer
        if "disclaimer" in analysis_result:
            st.warning(f"**Disclaimer:** {analysis_result['disclaimer']}")

        # Display Overall Assessment
        if "overall_assessment" in analysis_result:
            st.write(f"**Overall Assessment:** {analysis_result['overall_assessment']}")
        
        st.write("---")

        # Display each potential condition
        for condition in analysis_result.get("analysis", []):
            st.subheader(f"Potential Condition: {condition.get('condition', 'N/A')}")
            
            # Probability Score as a progress bar
            prob_score = condition.get('probability_score', 0)
            st.write(f"**Probability Score:** {prob_score}%")
            st.progress(prob_score)

            # Explanation
            st.write(f"**Explanation:** {condition.get('explanation', 'No explanation provided.')}")
            
            # Next Steps with color coding
            next_steps = condition.get('suggested_next_steps', '')
            if next_steps == 'Immediate Care':
                st.error(f"**Suggested Next Steps:** {next_steps}", icon="🚨")
            elif next_steps == 'Consult a Doctor':
                st.warning(f"**Suggested Next Steps:** {next_steps}", icon="👨‍⚕️")
            else: # Self-Care or other
                st.info(f"**Suggested Next Steps:** {next_steps}", icon="🏠")

            st.write("---")

elif submitted:
    st.error("Please enter your symptoms before analyzing.")
