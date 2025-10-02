import streamlit as st
import google.generativeai as genai
import json

def get_symptom_analysis(symptoms_description: str) -> dict:
    """
    Analyzes user symptoms using the Gemini API with a robust, structured prompt.

    Args:
        symptoms_description: A string of symptoms provided by the user.

    Returns:
        A dictionary containing the structured analysis from the model.
    """
    # Configure the Gemini API client
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error("Error configuring the Gemini API. Please check your API key in the secrets.")
        st.stop()

    # This is the advanced system prompt
    system_prompt = """
    You are an AI-powered clinical decision support model. Your role is to assist users by providing a preliminary analysis of their symptoms.

    **IMPORTANT RULES:**
    1.  **DO NOT PROVIDE A DIAGNOSIS.** You are not a doctor. Your primary function is to suggest possibilities and recommend next steps.
    2.  **ALWAYS INCLUDE A STRONG DISCLAIMER.** The first part of your output must be a clear warning that this is not a substitute for professional medical advice.
    3.  **OUTPUT MUST BE A VALID JSON OBJECT.** No other text or explanation should be outside of the JSON structure.
    4.  **PROBABILITIES MUST BE EDUCATED GUESSES.** They should be represented as integers out of 100 and reflect the likelihood based on the provided symptoms. The total may not add up to 100.
    5.  **SUGGESTED NEXT STEPS MUST BE SAFE AND RESPONSIBLE.** Categorize them as 'Immediate Care', 'Consult a Doctor', or 'Self-Care'.

    **JSON OUTPUT SCHEMA:**
    {
      "disclaimer": "This is an AI-generated analysis and not a medical diagnosis. Please consult a qualified healthcare professional for any health concerns.",
      "analysis": [
        {
          "condition": "Name of the potential condition",
          "probability_score": <integer between 0 and 100>,
          "explanation": "A brief explanation of why the symptoms match this condition.",
          "suggested_next_steps": "Immediate Care / Consult a Doctor / Self-Care"
        }
      ],
      "overall_assessment": "A brief summary of the situation based on the symptoms."
    }

    Analyze the following user-provided symptoms and generate a response strictly following the JSON schema.
    """

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-001",
        system_instruction=system_prompt
    )
    
    try:
        response = model.generate_content(symptoms_description)
        # Clean the response to ensure it's valid JSON
        # Sometimes the model might wrap the JSON in ```json ... ```
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_response)
    except Exception as e:
        # Handle potential errors like API failures or JSON parsing issues
        print(f"An error occurred: {e}") # Log error for debugging
        return {
            "error": "Failed to get a valid analysis. The model may be overloaded or the input is unclear. Please try again."
        }
