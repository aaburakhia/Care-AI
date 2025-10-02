import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(page_title="MindWell Chatbot", page_icon="🧠")
st.title("MindWell: Your Mental Health Support Chatbot")

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access the MindWell Chatbot.")
    st.stop()

# --- Chatbot Introduction ---
st.info("This is a safe space to express your thoughts and feelings. I am an AI chatbot trained on therapeutic dialogues to offer support and guidance. I am not a replacement for a human therapist. If you are in crisis, please seek immediate help.", icon="❤️")

# --- Initialize Chat History ---
# We use st.session_state to keep the chat history persistent across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Existing Chat Messages ---
# Loop through the history and display each message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
# The st.chat_input widget is used to get user input at the bottom of the screen.
if prompt := st.chat_input("How are you feeling today?"):
    # 1. Add user's message to the chat history and display it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. (Placeholder for AI response) - We will add the Gemini call here next.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Simulate a "typing" effect for the placeholder response.
        assistant_response = "Thank you for sharing. I'm processing your thoughts... (AI integration coming soon!)"
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # 3. Add the AI's response to the chat history.
    st.session_state.messages.append({"role": "assistant", "content": full_response})
