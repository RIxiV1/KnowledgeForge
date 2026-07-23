import os
import shutil
import hashlib
import time
import streamlit as st
from document_loader import load_file
from vector_store import get_vectorstore
from rag_engine import add_documents, ask_question
from database import create_tables, save_chat, get_chat_history, clear_history

create_tables()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config( page_title="Personal RAG Assistant", layout="wide", initial_sidebar_state="expanded" )

# Basic designing

st.markdown("""
<style>

/*BG*/
.stApp {
    background: linear-gradient( 135deg, #020617 0%, #0f172a 25%, #1e293b 75%, #334155 100% ); }

h1, h2, h3, p, div, span, label { color: #f8fafc; }

/*Menubar*/
[data-testid="stSidebar"] { background: #111827; border-right: 1px solid rgba(255,255,255,0.08); }

/*Chat msg*/
.stChatMessage { background-color: rgba(30, 41, 59, 0.5); border-radius: 8px; padding: 12px; margin-bottom: 8px; }

/* Input box */
.stChatInputContainer { background-color: rgba(15, 23, 42, 0.8); }

.stInfo { background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; }

/* Hide Streamlit Branding */
#MainMenu { visibility: hidden; }

footer { visibility: hidden; }

header { visibility: hidden; }

</style>
            
""", unsafe_allow_html=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "uploaded_hashes" not in st.session_state:
    st.session_state.uploaded_hashes = set()

if "show_history" not in st.session_state:
    st.session_state.show_history = False

if "show_delete_warning" not in st.session_state:
    st.session_state.show_delete_warning = False

with st.sidebar:
    st.title("Arsath Mohamed")
    st.caption("AI & Data Engineering")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("GitHub", "https://github.com/ArsathMohamed351", use_container_width=True)
    with col2:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/arsath-mohamed-710067323/", use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Knowledge Base")
    
    # Chat history info
    history_count = len(get_chat_history())
    st.metric("Chat History", history_count)
    
    if st.button("View Chat History", use_container_width=True):
        st.session_state.show_history = not st.session_state.get("show_history", False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear History", use_container_width=True):
            clear_history()
            st.session_state.conversation_history = []
            st.success("Chat history cleared")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("Delete Uploads", use_container_width=True):
            st.session_state.show_delete_warning = True
    
    # Show warning if delete uploads button was clicked
    if st.session_state.get("show_delete_warning", False):
        st.warning("This will delete ALL uploaded files!")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Yes, Delete", use_container_width=True, key="confirm_delete"):
                try:
                    if os.path.exists("uploads"):
                        shutil.rmtree("uploads")
                        os.makedirs("uploads", exist_ok=True)    
                    st.session_state.uploaded_hashes = set()
                    st.session_state.show_delete_warning = False
                    st.success("All uploads deleted!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("Cancel", use_container_width=True, key="cancel_delete"):
                st.session_state.show_delete_warning = False
                st.info("Cancelled")
                time.sleep(1)
                st.rerun()

st.title("Ask me anything, I remember stuff")

uploaded_files = st.file_uploader(
    "Throw your docs in here, I got you",
    accept_multiple_files=True,
    type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "pptx", "json"]
)

if uploaded_files:
    progress_placeholder = st.empty()
    
    for uploaded_file in uploaded_files:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Skip if already uploaded
        if file_hash in st.session_state.uploaded_hashes:
            st.info(f"{uploaded_file.name} already indexed")
            continue
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # file processing
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                docs = load_file(file_path)
                add_documents(st.session_state.vectorstore, docs)
                st.session_state.uploaded_hashes.add(file_hash)
                st.success(f"{uploaded_file.name} indexed ({len(docs)} chunks)")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
if st.session_state.get("show_history", False):
    with st.expander("what we talked about", expanded=True):
        history = get_chat_history()
        
        if history:
            for idx, (question, answer, created_at) in enumerate(history, 1):
                with st.container():
                    st.caption(f"**{idx}. {created_at}**")
                    st.write(f"**Q:** {question}")
                    # Handle None answer
                    if answer is None:
                        st.write("**A:** (No answer saved)")
                    # long answer truncation with expander
                    elif len(answer) > 300:
                        with st.expander("View full answer"):
                            st.write(answer)
                    else:
                        st.write(f"**A:** {answer}")
                    
                    st.divider()
        else:
            st.info("bro's memory is on airplane mode")

# MAIN INTERFACE

st.subheader("Bro,Ask Something...")

# conversation history
for exchange in st.session_state.conversation_history:
    with st.chat_message("user"):
        st.write(exchange["question"])    
    with st.chat_message("assistant"):
        st.write(exchange["answer"])
        if exchange.get("sources"):
            with st.expander("Sources"):
                for source in exchange["sources"]:
                    if isinstance(source, dict):
                        source_text = f"{source.get('filename', 'Unknown')}"
                        if source.get('file_type'):
                            source_text += f" ({source['file_type'].upper()})"
                        if source.get('rows'):
                            source_text += f" - Rows: {source['rows']}"
                        st.info(source_text)
                    else:
                        st.info(f"{source}")

question = st.chat_input("Ask me anything about your docs… I got you")

if question:
    # input validation
    if len(question.strip()) < 3:
        st.warning("Please ask a longer question")
    else:
        # user message
        with st.chat_message("user"):
            st.write(question)
        
        # question Process
        start_time = time.time()
        # Defaults so the metrics row below never hits an undefined name if the
        # call raises before these are assigned.
        answer = "No response generated"
        sources = []
        is_analytics = False
        with st.spinner("Searching and analyzing..."):
            try:
                result = ask_question( st.session_state.vectorstore, question, conversation_history=st.session_state.conversation_history )
                answer = result.get("answer", "No response generated")
                sources = result.get("sources", [])
                is_analytics = result.get("is_analytics", False)

                save_chat(question, answer)

                st.session_state.conversation_history.append({ "question": question, "answer": answer, "sources": sources, "is_analytics": is_analytics })

                if len(st.session_state.conversation_history) > 10:
                    st.session_state.conversation_history = st.session_state.conversation_history[-10:]
            except Exception as e:
                answer = f"Error: {str(e)}"
                sources = []
        latency = round(time.time() - start_time, 2)
        with st.chat_message("assistant"):
            st.write(answer)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{latency}s")
            with col2:
                if is_analytics:
                    st.caption("Analytics")
                else:
                    st.caption("Document Q&A")
            with col3:
                st.caption(f"{len(sources)} source(s)")
            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources, 1):
                        if isinstance(source, dict):
                            source_text = f"**{i}. {source.get('filename', 'Unknown')}**"
                            if source.get('file_type'):
                                source_text += f" ({source['file_type'].upper()})"
                            st.info(source_text)
                            if source.get('rows'):
                                st.caption(f"Rows: {source['rows']}")
                        else:
                            st.info(f"**{i}. {source}**")
