import pandas as pd

# File paths
job_desc_path = "job_title_des.csv"
resume_path = "UpdatedResumeDataSet.csv"

# Load datasets
job_desc_df = pd.read_csv(job_desc_path)
resume_df = pd.read_csv(resume_path)

import pandas as pd
import re

# Ensure DataFrames exist before proceeding
if 'job_desc_df' not in locals() or 'resume_df' not in locals():
    raise ValueError("DataFrames job_desc_df or resume_df are not defined!")

# Drop unnecessary index column if it exists
if "Unnamed: 0" in job_desc_df.columns:
    job_desc_df = job_desc_df.drop(columns=["Unnamed: 0"])

# Function to clean text (remove special characters, extra spaces, and normalize case)
def clean_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9., ]', '', text)  # Remove unwanted characters
    return text

# Apply cleaning function to Job Description dataset
job_desc_df["Job Title"] = job_desc_df["Job Title"].astype(str).apply(clean_text)
job_desc_df["Job Description"] = job_desc_df["Job Description"].astype(str).apply(clean_text)

# Remove duplicate rows
job_desc_df = job_desc_df.drop_duplicates()

# Apply cleaning function to Resume dataset
resume_df["Category"] = resume_df["Category"].astype(str).apply(clean_text)
resume_df["Resume"] = resume_df["Resume"].astype(str).apply(clean_text)

# Remove duplicate rows
resume_df = resume_df.drop_duplicates()

# Save cleaned datasets
job_desc_df.to_csv("cleaned_job_descriptions.csv", index=False)
resume_df.to_csv("cleaned_resumes.csv", index=False)

from sentence_transformers import SentenceTransformer

# Load a pre-trained BERT model
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

print("BERT model loaded successfully ✅")

from sentence_transformers import SentenceTransformer
import pandas as pd

# Load pre-trained BERT model
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Load cleaned datasets
resume_df = pd.read_csv("cleaned_resumes.csv")
job_desc_df = pd.read_csv("cleaned_job_descriptions.csv")

# Convert Resume text into embeddings
resume_df["Resume_Embeddings"] = resume_df["Resume"].apply(lambda x: bert_model.encode(x).tolist())

# Convert Job Description text into embeddings
job_desc_df["Job_Embeddings"] = job_desc_df["Job Description"].apply(lambda x: bert_model.encode(x).tolist())

# Save embeddings to CSV
resume_df.to_csv("resume_embeddings.csv", index=False)
job_desc_df.to_csv("job_desc_embeddings.csv", index=False)

print("BERT embeddings generated & saved successfully ✅")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the resume & job description embeddings
resume_df = pd.read_csv("resume_embeddings.csv")
job_desc_df = pd.read_csv("job_desc_embeddings.csv")

# Convert embeddings from list to array
resume_embeddings = resume_df["Resume_Embeddings"].apply(eval).apply(pd.Series).values
job_desc_embeddings = job_desc_df["Job_Embeddings"].apply(eval).apply(pd.Series).values

# Calculate cosine similarity for each resume and job description
similarity_matrix = cosine_similarity(resume_embeddings, job_desc_embeddings)

# Create a function to get the top N most similar resumes for each job description
def get_top_matches(job_index, top_n=5):
    # Get the similarity scores for the job description (row of the matrix)
    job_similarities = similarity_matrix[:, job_index]

    # Get the indices of the top N most similar resumes
    top_indices = job_similarities.argsort()[-top_n:][::-1]

    # Get the top N resumes
    top_resumes = resume_df.iloc[top_indices]

    return top_resumes

# Example: Get the top 5 resumes for the first job description
top_resumes_for_first_job = get_top_matches(0, top_n=5)

# Display the top 5 resumes for the first job description
print(top_resumes_for_first_job[["Resume", "Category"]])

import re

def extract_experience(resume_text):
    exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
    matches = re.findall(exp_pattern, resume_text)
    if matches:
        return max(map(int, matches))  # Return highest experience found
    return 0  # If no experience mentioned, return 0

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_top_skills_tfidf(resume_texts, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(resume_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the most important words
    top_skills = []
    for i in range(len(resume_texts)):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        top_skills.append(set(top_words))
    
    return top_skills

# Example Usage
resumes = [
    "Experienced in Python, TensorFlow, and data visualization using Matplotlib and Seaborn.",
    "Strong knowledge of SQL, Excel, and business analytics. Proficient in Power BI."
]
print(extract_top_skills_tfidf(resumes))

import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Sample skills database (Replace this with a real dataset)
skills_db = {"python", "sql", "machine learning", "deep learning", "tensorflow", "pandas", "numpy"}

def extract_skills(resume_text):
    """Extracts skills from resume text using keyword matching."""
    words = set(re.findall(r'\b\w+\b', resume_text.lower()))  # Tokenize words
    matched_skills = skills_db.intersection(words)  # Match against predefined skill set
    return matched_skills

def calculate_similarity(text1, text2):
    """Calculates similarity using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
    return similarity

def compute_resume_score(resume_text, job_desc_text):
    """Computes a final score based on skills match, experience, and relevance."""
    experience = extract_experience(resume_text)  # Assuming this function exists
    matched_skills = extract_skills(resume_text)
    relevance_score = calculate_similarity(resume_text, job_desc_text)

    skills_match_percent = (len(matched_skills) / len(skills_db)) * 100 if skills_db else 0
    experience_bonus = min(10, experience * 2)  # Max 10 points for experience
    final_score = (relevance_score * 5) + (skills_match_percent * 3) + (experience_bonus * 2)

    return final_score

resumes = [
    {"name": "Resume A", "text": "Experienced in Python, SQL, and ML with 3+ years of experience."},
    {"name": "Resume B", "text": "Deep Learning, TensorFlow, 5 years of AI experience."},
]

job_description = "Looking for a Machine Learning Engineer with Python, SQL, and Deep Learning experience."

# Rank resumes
ranked_resumes = sorted(resumes, key=lambda x: compute_resume_score(x["text"], job_description), reverse=True)

# Print results
for i, res in enumerate(ranked_resumes):
    print(f"Rank {i+1}: {res['name']} - Score: {compute_resume_score(res['text'], job_description)}")

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
    return text.strip()

'''
import os

# Use an absolute path to ensure the folder is found
resumes_folder = os.path.abspath("resumes")  

resume_texts = {}

if os.path.exists(resumes_folder) and os.path.isdir(resumes_folder):
    for filename in os.listdir(resumes_folder):
        if filename.lower().endswith(".pdf"):  # Case-insensitive check
            resume_path = os.path.join(resumes_folder, filename)
            resume_texts[filename] = extract_text_from_pdf(resume_path)

    # Print extracted text from first resume (if any)
    if resume_texts:
        print(resume_texts[list(resume_texts.keys())[0]])
    else:
        print("No PDF files found in the 'resumes' folder.")
else:
    print("Error: 'resumes' folder not found.")
'''