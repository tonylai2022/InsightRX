import streamlit as st
from pdf_utils import extract_text_from_pdf
from rag_pipeline import build_vectorstore, create_qa_chain
from summarize import get_summary
import os
from dotenv import load_dotenv

load_dotenv()

# Verify token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

st.set_page_config(page_title="InsightRX", layout="wide")
st.title("🧠 InsightRX – Clinical Research Summarizer")

# Initialize session state
if "summary" not in st.session_state:
    st.session_state.summary = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        try:
            text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            st.stop()

    # Generate summary
    if st.button("📝 Generate Summary") or st.session_state.summary:
        if not st.session_state.summary:
            with st.spinner("Generating summary..."):
                try:
                    st.session_state.summary = get_summary(text)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.stop()

        st.subheader("Summary")
        st.markdown(st.session_state.summary)

    # RAG QA System
    try:
        vectordb = build_vectorstore(text)
        qa_chain = create_qa_chain(vectordb)
    except Exception as e:
        st.error(f"Error building vector store or QA chain: {str(e)}")
        st.stop()

    # Q&A section
    st.subheader("💬 Ask questions about the paper")
    question = st.text_input("Ask a question")
    if question:
        try:
            response = qa_chain({"question": question, "chat_history": st.session_state.chat_history})
            st.markdown(f"**Answer:** {response['answer']}")
            st.session_state.chat_history.append((question, response['answer']))
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    # Display previous Q&A
    if st.session_state.chat_history:
        st.markdown("### 🔁 Chat History")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
