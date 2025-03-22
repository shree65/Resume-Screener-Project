import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
    return text.strip()

# Function to calculate similarity score
def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]  # Cosine similarity score

# Streamlit UI
st.title("📄 AI Resume Screener")

st.subheader("Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload multiple PDFs", type=["pdf"], accept_multiple_files=True)

job_desc = st.text_area("Enter Job Description", "Looking for a Data Scientist with Python, ML, and SQL skills.")

if st.button("Process Resumes"):
    if uploaded_files and job_desc:
        resume_scores = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            resume_text = extract_text_from_pdf(temp_file_path)
            score = calculate_similarity(resume_text, job_desc)

            resume_scores.append({"Filename": uploaded_file.name, "Score": round(score * 100, 2)})

            os.remove(temp_file_path)  # Clean up temp files

        # Show results
        st.subheader("📊 Resume Ranking")
        sorted_resumes = sorted(resume_scores, key=lambda x: x["Score"], reverse=True)
        st.table(sorted_resumes)
    else:
        st.warning("⚠️ Please upload resumes and enter a job description.")
