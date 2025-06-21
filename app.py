
import os
import streamlit as st
from langchain_community.llms import OpenAI
from utils import load_and_split_pdf, create_vector_db, query_pdf

st.set_page_config(page_title="Chat with PDF", layout="centered")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #F1E7E7;
        color: #443850;
        font-family: 'Segoe UI', sans-serif;
    }
    .chat-bubble {
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user {
        background-color: #E3EEB2;
        align-self: flex-end;
    }
    .bot {
        background-color: #443850;
        color: white;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Chat with PDF (Modern AI Assistant)")
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📎 Upload a PDF", type="pdf")

# Reset button
if st.button("🔁 Reset"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Process the PDF
if uploaded_file and openai_api_key:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("📄 PDF uploaded and processing...")
    docs = load_and_split_pdf("temp.pdf")
    db = create_vector_db(docs, openai_api_key)
    query = st.text_input("💬 Ask something about the PDF:")

    if query:
        with st.spinner("🤔 Thinking..."):
            answer, matched_docs = query_pdf(db, openai_api_key, query)
            st.session_state.chat_history.append({"q": query, "a": answer})

# Display chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for entry in reversed(st.session_state.chat_history):
    st.markdown(f'<div class="chat-bubble user">🟣 You: {entry["q"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble bot">🟢 AI: {entry["a"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
