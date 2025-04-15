import os
import fitz
import faiss
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

load_dotenv()

# ========== Config ==========
FAISS_INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"
EMBEDDINGS_PATH = "embeddings.pkl"
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

model = SentenceTransformer("all-MiniLM-L6-v2")


# ========== Utilities ==========

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def clean_text(text):
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ========== FAISS and Embeddings ==========

def create_or_load_faiss(text_chunks):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH) and os.path.exists(CHUNKS_PATH):
        print("🔄 Loading existing FAISS index and embeddings...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        embeddings = load_pickle(EMBEDDINGS_PATH)
        chunks = load_pickle(CHUNKS_PATH)
    else:
        print("⚙️ Creating new FAISS index...")
        embeddings = model.encode(text_chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        faiss.write_index(index, FAISS_INDEX_PATH)
        save_pickle(embeddings, EMBEDDINGS_PATH)
        save_pickle(text_chunks, CHUNKS_PATH)
        chunks = text_chunks
    return index, embeddings, chunks

def search_chunks(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    _, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

# ========== BigQuery ==========

def query_bigquery(sql):
    client = bigquery.Client()
    query_job = client.query(sql)
    return query_job.result().to_dataframe()

# ========== SQL Generation via HuggingFace ==========

def generate_sql(nl_query):
    db = SQLDatabase.from_uri("bigquery://bcdap-scrr-dev")
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.1, "max_length": 512})
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
    return agent.run(nl_query)

# ========== Main ==========

def run(query, pdf_path=None):
    query_cleaned = clean_text(query)
    context = ""

    index = None
    chunks = []

    # Process PDF if available
    if pdf_path and os.path.exists(pdf_path):
        print(f"📄 Processing PDF: {pdf_path}")
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(clean_text(raw_text))
        index, _, chunks = create_or_load_faiss(chunks)

    elif os.path.exists(FAISS_INDEX_PATH):
        print("🔁 Loading cached FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        chunks = load_pickle(CHUNKS_PATH)

    # FAISS search
    if index and chunks:
        context = "\n".join(search_chunks(query_cleaned, index, chunks))

    # SQL generation
    sql_query = None
    try:
        sql_query = generate_sql(query)
        print(f"\n🧾 SQL Generated: {sql_query}\n")
    except Exception as e:
        print("❌ Failed to generate SQL:", e)

    df = None
    if sql_query:
        try:
            df = query_bigquery(sql_query)
        except Exception as e:
            print("❌ SQL execution failed:", e)

    # Final Response
    print("\n🤖 Final Contextual Response:\n")
    if context:
        print("🧠 From FAISS:\n", context)
    if df is not None and not df.empty:
        print("\n📊 From BigQuery:\n", df.to_string(index=False))
    elif df is not None:
        print("\n📊 No structured results found.")

# ========== Run ==========
if __name__ == "__main__":
    user_query = input("🗣️ Enter your query: ")
    pdf_input = input("📎 PDF file path (optional): ").strip()
    pdf_path = pdf_input if pdf_input else None

    run(user_query, pdf_path)
