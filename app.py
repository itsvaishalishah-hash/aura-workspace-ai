"""
Language: Python 3.14.5
Key Libraries: streamlit, google-genai, python-dotenv, streamlit-mic-recorder, duckduckgo-search, sqlite3
Purpose: Execute final master application logic with permanent database, audio, and live web search capabilities.
Book: Build a GenAI Desktop Assistant With Streamlit
"""

import streamlit as st
import os
import sqlite3
from dotenv import load_dotenv
from google import genai
from google.genai import types
from streamlit_mic_recorder import speech_to_text
from duckduckgo_search import DDGS

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="Aura Workspace Pro", page_icon="⚡", layout="wide")
load_dotenv()

# 2. Database Architect Functions
def initialize_database():
    """Creates the permanent table schema if it does not exist."""
    conn = sqlite3.connect('aura_memory.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_message_to_db(role_string, content_string):
    """Executes live database insert commands."""
    conn = sqlite3.connect('aura_memory.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_logs (role, content) VALUES (?, ?)", (role_string, content_string))
    conn.commit()
    conn.close()

def retrieve_historical_logs():
    """Pulls all past conversations from the permanent node."""
    conn = sqlite3.connect('aura_memory.db')
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_logs ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def clear_database():
    """Wipes the database cleanly."""
    conn = sqlite3.connect('aura_memory.db')
    conn.cursor().execute("DELETE FROM chat_logs")
    conn.commit()
    conn.close()
    st.session_state.messages = []

# 3. Initialize Memory Vault
initialize_database()

if 'messages' not in st.session_state:
    # Load from permanent database instead of starting empty
    st.session_state.messages = retrieve_historical_logs()
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get('GEMINI_API_KEY') or ""

PERSONA_MAP = {
    "General Assistant": "You are Aura, a helpful and concise AI desktop assistant.",
    "Python Expert": "You are an expert Python developer. Provide highly optimized, PEP-8 compliant code.",
    "Creative Writer": "You are a creative writing assistant. Focus on engaging narrative.",
    "Data Analyst": "You are a senior data analyst. Explain trends and statistics clearly.",
    "Code Reviewer": "You are a strict code reviewer. Analyze context for flaws."
}

# 4. Header Layout
col1, col2 = st.columns([1, 8])
with col1: st.title("🌌")
with col2: st.title("Aura Workspace AI Pro")

# 5. Sidebar Control Panel
with st.sidebar:
    st.title("Workspace Controls")
    st.caption("Configure your AI environment.")
    st.divider()
    
    api_key_input = st.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
    if st.checkbox("Remember API Key", value=bool(st.session_state.api_key)):
        st.session_state.api_key = api_key_input
        
    persona_selection = st.selectbox("Select Persona", list(PERSONA_MAP.keys()))
    grounding_toggle = st.checkbox("Enable Live Web Search", value=False)
    
    st.divider()
    
    if st.button("Clear Conversation History", type="primary"):
        clear_database()
        st.rerun()
        
    export_data = "\n\n".join([f"[{msg['role'].upper()}]\n{msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        label="Export Chat History",
        data=export_data if export_data else "No active session history.",
        file_name="aura_chat_log.txt",
        mime="text/plain"
    )

# 6. Core stream generation function logic
def get_live_internet_context(query_string):
    try:
        results = DDGS().text(query_string, max_results=3)
        context_block = "Live Web Context Data:\n"
        for item in results:
            context_block += f"- {item['body']}\n"
        return context_block
    except Exception:
        return ""

def get_gemini_stream(prompt_text, active_persona, uploaded_file, use_web):
    if not st.session_state.api_key:
        yield "Aura cannot connect: Please provide a valid Gemini API key in the sidebar."
        return

    try:
        client = genai.Client(api_key=st.session_state.api_key)
        system_prompt = PERSONA_MAP.get(active_persona, PERSONA_MAP["General Assistant"])
        
        formatted_history = []
        for msg in st.session_state.messages:
            api_role = 'model' if msg['role'] == 'assistant' else 'user'
            formatted_history.append({'role': api_role, 'parts': [{'text': msg['content']}]})
            
        current_parts = []
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            current_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=uploaded_file.type))
            
        final_prompt = prompt_text
        if use_web:
            web_data = get_live_internet_context(prompt_text)
            final_prompt = f"Use this live search data to answer accurately:\n{web_data}\n\nUser Request: {prompt_text}"
            
        current_parts.append({'text': final_prompt})
        formatted_history.append({'role': 'user', 'parts': current_parts})
        
        response_stream = client.models.generate_content_stream(
            model='gemini-3.1-flash-lite',
            contents=formatted_history,
            config=types.GenerateContentConfig(system_instruction=system_prompt)
        )
        
        for chunk in response_stream:
            yield chunk.text

    except Exception as e:
        yield f"\n\nAura encountered a network error: {str(e)}"

# 7. Main Chat Interface setup
st.caption("Your private, multimodal desktop assistant.")
active_document = st.file_uploader("Attach a reference document", type=["txt", "pdf"])

chat_container = st.container(height=450)
with chat_container:
    if len(st.session_state.messages) == 0:
        st.write("Welcome to Aura Workspace! Configure your settings in the sidebar to begin.")
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

# 8. Chat Input and Execution Flow
colA, colB = st.columns([1, 15])
with colA:
    vocal_prompt = speech_to_text(
        start_prompt="🎙️",
        stop_prompt="🛑",
        language='en', 
        use_container_width=True, 
        just_once=True, 
        key='mic'
    )
with colB:
    text_prompt = st.chat_input("Ask Aura anything...")

active_prompt = vocal_prompt if vocal_prompt else text_prompt

if active_prompt:
    st.session_state.messages.append({'role': 'user', 'content': active_prompt})
    save_message_to_db('user', active_prompt)
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(active_prompt)
            
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_machine_response = ""
            
            for partial_text in get_gemini_stream(active_prompt, persona_selection, active_document, grounding_toggle):
                full_machine_response += partial_text
                response_placeholder.markdown(full_machine_response + "▌")
                
            response_placeholder.markdown(full_machine_response)
            
    st.session_state.messages.append({'role': 'assistant', 'content': full_machine_response})
    save_message_to_db('assistant', full_machine_response)

# 9. Footer
st.markdown(
    "<br><br><div style='text-align: center; color: gray; font-size: small;'>"
    "Build a GenAI Desktop Assistant With Streamlit<br>"
    "© 2026 Sharanam and Vaishali Shah</div>", 
    unsafe_allow_html=True
)