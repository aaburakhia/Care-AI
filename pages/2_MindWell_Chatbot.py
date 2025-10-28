import streamlit as st
from style_utils import add_custom_css
from gemini_client import get_mindwell_response_stream
from supabase_client import get_supabase_client, create_chat_conversation, save_chat_message

# --- Page Configuration & Styling ---
st.set_page_config(page_title="MindWell Chatbot", page_icon="🧠")
add_custom_css()
st.title("MindWell: Your Mental Health Support Chatbot")

# --- Authentication & Supabase Client ---
if 'user' not in st.session_state or st.session_state.user is None:
    st.warning("Please log in to access the MindWell Chatbot.")
    st.stop()
    
supabase = get_supabase_client()

# --- Chatbot Introduction ---
st.info("This is a safe space to express your thoughts and feelings. All conversations are automatically saved privately to your account for you to review later.", icon="❤️")

# --- Initialize Session State for Chat ---
# Use unique keys for this chat's state to avoid conflicts with other pages
if "mindwell_messages" not in st.session_state:
    st.session_state.mindwell_messages = [
        {"role": "assistant", "content": "Hello! I'm MindWell, your AI support companion. How are you feeling today?"}
    ]
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# --- Display Existing Chat Messages from the current session ---
for message in st.session_state.mindwell_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Processing ---
if prompt := st.chat_input("How are you feeling today?"):
    # 1. Create a new conversation record in the DB if this is the first message of the session.
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = create_chat_conversation(supabase)

    # 2. Add user message to the session state and display it on the screen.
    st.session_state.mindwell_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 3. Save the user's message to the database in the background.
    if st.session_state.conversation_id:
        save_chat_message(supabase, st.session_state.conversation_id, "user", prompt)

    # 4. Get and display the AI's streaming response.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call the Gemini API to get a response
        stream = get_mindwell_response_stream(st.session_state.mindwell_messages)
        
        # Stream the response to the screen
        for chunk in stream:
            if hasattr(chunk, 'text') and chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # 5. Add the final, complete AI response to the session state.
    st.session_state.mindwell_messages.append({"role": "assistant", "content": full_response})

    # 6. Save the AI's message to the database in the background.
    if st.session_state.conversation_id:
        save_chat_message(supabase, st.session_state.conversation_id, "assistant", full_response)
