"""
Language: Python 3.14.5
Key Libraries: streamlit, google-genai, python-dotenv
Purpose: Execute final application logic with cross-environment secret fallback capabilities.
Book: Build a GenAI Desktop Assistant With Streamlit
"""

import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="Aura Workspace", page_icon="⚡", layout="wide")

# 2. Load environment variables for local testing
load_dotenv()

# 3. Initialize memory vault components and handle cross-environment secrets
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    try:
        # Prioritize secure Streamlit Cloud secrets during production
        cloud_key = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        # Fall back to local .env execution during development
        cloud_key = None

    st.session_state.api_key = cloud_key or os.environ.get("GEMINI_API_KEY") or ""

# 4. Define Persona Dictionary Map
PERSONA_MAP = {
    "General Assistant": "You are Aura, a helpful and concise AI desktop assistant.",
    "Python Expert": "You are an expert Python developer. Provide highly optimized, PEP-8 compliant code.",
    "Creative Writer": "You are a creative writing assistant. Focus on engaging narrative.",
    "Data Analyst": "You are a senior data analyst. Explain trends and data structures systematically.",
    "Code Reviewer": "You are a strict code reviewer. Analyze the provided context for flaws.",
}

# 5. Header Layout
col1, col2 = st.columns([1, 8])
with col1:
    st.title("🌌")
with col2:
    st.title("Aura Workspace AI")

# 6. Sidebar Control Panel
with st.sidebar:
    st.title("Workspace Controls")
    st.caption("Configure your AI environment.")
    st.divider()

    api_key_input = st.text_input(
        "Gemini API Key", type="password", value=st.session_state.api_key
    )
    remember_key_toggle = st.checkbox(
        "Remember API Key", value=bool(st.session_state.api_key)
    )

    if remember_key_toggle:
        st.session_state.api_key = api_key_input

    persona_selection = st.selectbox(
        "Select Persona",
        [
            "General Assistant",
            "Python Expert",
            "Creative Writer",
            "Data Analyst",
            "Code Reviewer",
        ],
    )
    st.divider()

    export_data = "\n\n".join(
        [
            f"[{msg['role'].upper()}]\n{msg['content']}"
            for msg in st.session_state.messages
        ]
    )
    st.download_button(
        label="Export Chat History",
        data=export_data if export_data else "No active session history.",
        file_name="aura_chat_log.txt",
        mime="text/plain",
    )


# 7. Core stream generation function logic
def get_gemini_stream(prompt_text, active_persona, uploaded_file):
    """Sends the context and file data to the API, returning a stream generator."""
    if not st.session_state.api_key:
        yield "Aura cannot connect: Please provide a valid Gemini API key in the sidebar."
        return

    try:
        client = genai.Client(api_key=st.session_state.api_key)
        system_prompt = PERSONA_MAP.get(
            active_persona, PERSONA_MAP["General Assistant"]
        )

        formatted_history = []
        for msg in st.session_state.messages:
            api_role = "model" if msg["role"] == "assistant" else "user"
            formatted_history.append(
                {"role": api_role, "parts": [{"text": msg["content"]}]}
            )

        current_parts = []
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            current_parts.append(
                types.Part.from_bytes(data=file_bytes, mime_type=uploaded_file.type)
            )

        current_parts.append({"text": prompt_text})
        formatted_history.append({"role": "user", "parts": current_parts})

        response_stream = client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=formatted_history,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )

        for chunk in response_stream:
            yield chunk.text

    except Exception as e:
        yield f"\n\nAura encountered a network error: {str(e)}"


# 8. Main Chat Interface setup
st.caption("Your private, multimodal desktop assistant.")
active_document = st.file_uploader("Attach a reference document", type=["txt", "pdf"])

# 9. Chat Container and History Render Loop
chat_container = st.container(height=450)
with chat_container:
    if len(st.session_state.messages) == 0:
        st.write(
            "Welcome to Aura Workspace! Configure your settings in the sidebar to begin."
        )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 10. Chat Input and Execution Flow
if prompt := st.chat_input("Ask Aura anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_machine_response = ""

            for partial_text in get_gemini_stream(
                prompt, persona_selection, active_document
            ):
                full_machine_response += partial_text
                response_placeholder.markdown(full_machine_response + "▌")

            response_placeholder.markdown(full_machine_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_machine_response}
    )

# 11. Footer
st.markdown(
    "<br><hr><center><small>Build a GenAI Desktop Assistant With Streamlit<br>"
    "© 2026 Sharanam and Vaishali Shah</small></center>",
    unsafe_allow_html=True,
)
