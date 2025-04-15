import os
import re
import faiss
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery
from dotenv import load_dotenv

nltk.download('punkt')
load_dotenv()

# === CONFIG ===
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = 'vector_store.index'
DOCUMENTS_PATH = 'documents.npy'

# === INIT ===
model = SentenceTransformer(EMBEDDING_MODEL)
bq_client = bigquery.Client()

# === UTILS ===

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def embed_text(texts):
    return model.encode(texts, convert_to_numpy=True)

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    docs = np.load(DOCUMENTS_PATH, allow_pickle=True)
    return index, docs

def similarity_search(query_embedding, index, top_k=3):
    D, I = index.search(np.array([query_embedding]), top_k)
    return I[0], D[0]

def generate_sql_from_query(query):
    # Very basic, replace with LangChain or your logic
    if "revenue" in query:
        return "SELECT year, revenue FROM `your_project.dataset.table` ORDER BY year DESC LIMIT 5"
    else:
        return None

def query_bigquery(sql):
    try:
        df = bq_client.query(sql).to_dataframe()
        return df.to_string(index=False)
    except Exception as e:
        return f"Error querying BigQuery: {e}"

# === MAIN ===

def main():
    question = input("Enter your question: ")
    pdf_path = input("Enter path to optional PDF file (or press Enter to skip): ").strip()

    if pdf_path:
        pdf_text = extract_text_from_pdf(pdf_path)
        text_corpus = [pdf_text]
        embeddings = embed_text([preprocess(pdf_text)])

        # Save for future runs
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(DOCUMENTS_PATH, np.array(text_corpus))

        print("PDF processed and indexed.")
    else:
        try:
            index, docs = load_faiss_index()
        except Exception as e:
            print("No FAISS index found. Please process a PDF first.")
            return

    processed_q = preprocess(question)
    q_embed = embed_text([processed_q])[0]

    index, docs = load_faiss_index()
    ids, scores = similarity_search(q_embed, index)

    print("\nTop matching documents:")
    for i, score in zip(ids, scores):
        print(f"[Score: {score:.4f}]\n{docs[i][:300]}...\n")

    sql_query = generate_sql_from_query(question)
    if sql_query:
        print("\nRunning SQL query on BigQuery...")
        sql_result = query_bigquery(sql_query)
        print("\nBigQuery Result:\n", sql_result)

    print("\n🎯 Final Response (merged):")
    print("-" * 40)
    for i in ids:
        print(docs[i][:300], "\n...\n")
    if sql_query:
        print("Structured Data:\n", sql_result)

if __name__ == "__main__":
    main()
