import streamlit as st
import google.generativeai as genai
import json

# --- Helper Function for Schema Validation ---
def validate_json_schema(data: dict) -> bool:
    """
    Validates if the received JSON matches our expected schema.
    Returns True if valid, False otherwise.
    """
    if "disclaimer" not in data or "analysis" not in data or "overall_assessment" not in data:
        return False
    
    if not isinstance(data["analysis"], list):
        return False
        
    for item in data["analysis"]:
        if "condition" not in item or "probability_score" not in item or "explanation" not in item or "suggested_next_steps" not in item:
            return False
            
    return True

# --- Main Analysis Function ---
def get_symptom_analysis(symptoms_description: str) -> dict:
    """
    Analyzes user symptoms using the Gemini API with a robust, structured prompt.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error("Error configuring the Gemini API. Please check your API key in the secrets.")
        st.stop()

    system_prompt = """
    You are an AI-powered clinical decision support model. Your role is to assist users by providing a preliminary analysis of their symptoms.

    **IMPORTANT RULES:**
    1.  **DO NOT PROVIDE A DIAGNOSIS.** You are not a doctor.
    2.  **ALWAYS INCLUDE A STRONG DISCLAIMER.**
    3.  **OUTPUT MUST BE A VALID JSON OBJECT.** No other text or explanation should be outside of the JSON structure.
    4.  **IF THE USER INPUT DOES NOT DESCRIBE MEDICAL SYMPTOMS**, you must respond with a specific JSON error object: {"error": "The input provided does not seem to describe medical symptoms. Please describe your symptoms to get an analysis."} Do not try to answer the question.
    5.  **SUGGESTED NEXT STEPS MUST BE SAFE AND RESPONSIBLE.**

    **JSON OUTPUT SCHEMA:**
    {
      "disclaimer": "This is an AI-generated analysis...",
      "analysis": [
        {
          "condition": "Name of the potential condition",
          "probability_score": <integer>,
          "explanation": "...",
          "suggested_next_steps": "Immediate Care / Consult a Doctor / Self-Care"
        }
      ],
      "overall_assessment": "..."
    }

    ---
    **EXAMPLE:**

    **User Input:** "I have a high fever, a pounding headache, and my body aches all over. I also feel very tired."

    **Expected JSON Output:**
    {
      "disclaimer": "This is an AI-generated analysis and not a medical diagnosis. Please consult a qualified healthcare professional for any health concerns.",
      "analysis": [
        {
          "condition": "Influenza (Flu)",
          "probability_score": 85,
          "explanation": "The combination of high fever, headache, body aches, and fatigue are classic and strong indicators of influenza.",
          "suggested_next_steps": "Consult a Doctor"
        },
        {
          "condition": "Common Cold",
          "probability_score": 30,
          "explanation": "While possible, the high fever and severe body aches make a common cold less likely than the flu.",
          "suggested_next_steps": "Self-Care"
        }
      ],
      "overall_assessment": "The symptoms strongly suggest a viral infection like the flu. Medical consultation is recommended for proper diagnosis and management."
    }
    ---

    Now, analyze the following user-provided symptoms and generate a response strictly following the rules and JSON schema.
    """

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-001",
        system_instruction=system_prompt
    )
    
    try:
        response = model.generate_content(symptoms_description)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON
        data = json.loads(cleaned_response)

        # Handle the specific non-symptom error case from our prompt
        if "error" in data:
            return data

        # Validate the schema of the successful response
        if not validate_json_schema(data):
             raise ValueError("JSON output does not match the required schema.")

        return data
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON validation/parsing error: {e}")
        return {"error": "The model returned an invalid response format. Please try rephrasing your symptoms."}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "An unexpected error occurred while analyzing your symptoms. Please try again later."}
