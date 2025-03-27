import streamlit as st
from pdf_utils import extract_text_from_pdf
from rag_pipeline import build_vectorstore, create_qa_chain
from summarize import get_summary
import os
from dotenv import load_dotenv

load_dotenv()

# Verify token is loaded
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

st.set_page_config(page_title="InsightRX", layout="wide")
st.title("🧠 InsightRX – Clinical Research Summarizer")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        try:
            text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            st.stop()

    if st.button("📝 Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                summary = get_summary(text)
                st.subheader("Summary")
                st.markdown(summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

    try:
        vectordb = build_vectorstore(text)
        qa_chain = create_qa_chain(vectordb)
    except Exception as e:
        st.error(f"Error building vector store or QA chain: {str(e)}")
        st.stop()

    st.subheader("💬 Ask questions about the paper")
    chat_history = []
    question = st.text_input("Ask a question")
    if question:
        try:
            response = qa_chain({"question": question, "chat_history": chat_history})
            st.markdown(f"**Answer:** {response['answer']}")
            chat_history.append((question, response['answer']))
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")