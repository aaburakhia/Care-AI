import streamlit as st
from style_utils import add_custom_css
from gemini_client import get_mindwell_response_stream # Import our new function

# --- Page Configuration & Styling ---
st.set_page_config(page_title="MindWell Chatbot", page_icon="🧠")
add_custom_css()
st.title("MindWell: Your Mental Health Support Chatbot")

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access the MindWell Chatbot.")
    st.stop()

# --- Chatbot Introduction ---
st.info("This is a safe space to express your thoughts and feelings. I am an AI chatbot trained on therapeutic dialogues to offer support and guidance. I am not a replacement for a human therapist. If you are in crisis, please seek immediate help.", icon="❤️")

# --- Initialize Chat History ---
# We use a unique key for this page's messages
if "mindwell_messages" not in st.session_state:
    st.session_state.mindwell_messages = [
        {"role": "assistant", "content": "Hello! I'm MindWell, your AI support companion. How are you feeling today?"}
    ]

# --- Display Existing Chat Messages ---
for message in st.session_state.mindwell_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message to session state and display it
    st.session_state.mindwell_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display the AI's streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call the new function from gemini_client
        stream = get_mindwell_response_stream(st.session_state.mindwell_messages)
        
        # Iterate through the stream and update the placeholder
        for chunk in stream:
            # Check for empty chunks and handle them
            if chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
        
        # Display the final response
        message_placeholder.markdown(full_response)
    
    # Add the final AI response to the session state
    st.session_state.mindwell_messages.append({"role": "assistant", "content": full_response})
