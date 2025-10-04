import streamlit as st
from gemini_client import get_symptom_analysis
from supabase_client import get_supabase_client, save_symptom_analysis
from style_utils import add_custom_css 

# --- Page Configuration ---
st.set_page_config(page_title="Symptom Checker", page_icon="🩺")
add_custom_css() # <--- ADD THIS LINE
st.title("AI-Powered Symptom Checker")

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access this page.")
    st.stop()
    
# --- Get the Supabase client object ---
supabase = get_supabase_client()

# --- Input Pre-Validation Function ---
def is_valid_input(symptoms: str) -> bool:
    """
    A simple check to guard against clearly invalid input before calling the API.
    """
    if len(symptoms.split()) < 3:
        st.error("Please provide a more detailed description of your symptoms (at least 3 words).")
        return False
    
    symptom_keywords = [
        'pain', 'ache', 'fever', 'headache', 'cough', 'sore', 'throat', 'nausea', 
        'dizzy', 'fatigue', 'tired', 'rash', 'itchy', 'swelling', 'breathing', 
        'stomach', 'cramp', 'chills', 'vomit', 'diarrhea', 'runny', 'nose'
    ]
    if not any(keyword in symptoms.lower() for keyword in symptom_keywords):
        st.error("Your description does not seem to contain common medical symptoms. Please describe how you are feeling.")
        return False
        
    return True

# --- Main Page Content ---
st.write("Describe your symptoms, and our AI co-pilot will provide a preliminary analysis and suggest next steps.")
st.info("This tool does **not** provide a medical diagnosis. Always consult with a healthcare professional for health concerns.", icon="⚠️")

with st.form("symptom_form"):
    symptoms_input = st.text_area("Please describe your symptoms in detail:", height=150, placeholder="e.g., 'I have a high fever, a persistent dry cough, and I'm feeling very tired.'")
    submitted = st.form_submit_button("Analyze Symptoms")

if submitted and symptoms_input:
    # --- Execute Pre-Validation FIRST ---
    if is_valid_input(symptoms_input):
        with st.spinner("Analyzing your symptoms... This may take a moment."):
            analysis_result = get_symptom_analysis(symptoms_input)

        st.divider()
        
        if "error" in analysis_result:
            st.error(analysis_result["error"])
        else:
            # --- NEW: Save the result to the database ---
            save_success = save_symptom_analysis(supabase, symptoms_input, analysis_result)
            if save_success:
                st.toast("Analysis saved to your history!", icon="✅")
            else:
                st.toast("Could not save analysis to your history.", icon="❌")
            # ---------------------------------------------
            
            # The rest of the display logic is the same as before
            st.subheader("Analysis Results")
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

elif submitted:
    st.error("Please enter your symptoms before analyzing.")
