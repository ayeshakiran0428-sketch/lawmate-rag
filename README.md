# LawMate — AI Legal Analyzer

An AI-powered legal assistant that analyzes case descriptions and identifies relevant sections of the **Pakistan Penal Code (PPC)** using Retrieval-Augmented Generation (RAG). Built as a final year project and extended into a fully deployed, cloud-hosted application.

**🔗 Live demo:** _add your Streamlit Cloud URL here, e.g. https://lawmate-rag.streamlit.app/
**🔗 Backend API (Swagger docs):** https://lawmate-rag.onrender.com/

---

## About This Project

LawMate lets a user describe a legal situation in plain language — in **English, Urdu, or Roman Urdu** — and returns:
1. The likely **case type / offence**
2. The **relevant section(s)** of the Pakistan Penal Code for an FIR
3. A brief explanation grounded strictly in the actual law text (not invented by the model)

It uses a Retrieval-Augmented Generation (RAG) pipeline: the user's query is translated/cleaned, relevant law sections are retrieved from a vector database, and an LLM generates a grounded answer using only that retrieved context — reducing hallucination compared to asking an LLM cold.

## Architecture

```
User (Browser)
      │
      ▼
Streamlit Cloud (Frontend UI)
      │
      ▼
   RAG Pipeline
      │
      ├──▶ Query cleaning + translation (deep-translator)
      │
      ├──▶ HuggingFace Embeddings (all-MiniLM-L6-v2)
      │
      ├──▶ Pinecone Vector DB  ── retrieves relevant PPC sections
      │
      └──▶ Groq LLM (openai/gpt-oss-120b) ── generates grounded answer
```

A separate **FastAPI backend**, deployed on Render, exposes the same RAG logic as a REST API for programmatic access.

## Tech Stack

**AI / RAG**
- LangChain — orchestration of the retrieval + generation pipeline
- Pinecone — managed vector database for semantic search over PPC text
- HuggingFace (`sentence-transformers`) — text embeddings
- Groq — fast LLM inference (`openai/gpt-oss-120b`)
- `deep-translator` — Urdu / Roman Urdu → English translation for multilingual support

**Backend**
- FastAPI — REST API
- Uvicorn — ASGI server
- Deployed on **Render** (PaaS)

**Frontend**
- Streamlit — interactive web UI
- Deployed on **Streamlit Community Cloud**

**DevOps**
- Git / GitHub — version control
- Environment-based secrets management (Render & Streamlit Cloud secrets, no keys in source control)

## Project Structure

```
lawmate-rag/
├── app.py                 # Original terminal-based chatbot
├── api.py                 # FastAPI backend (REST API)
├── streamlit_app.py       # Streamlit web UI
├── ingest.py               # Loads and embeds the PPC document into Pinecone
├── utils.py                # Query cleaning + translation helpers
├── ppc.pdf                 # Source legal document (Pakistan Penal Code)
├── requirements.txt
├── runtime.txt              # Pins Python version for deployment
├── Procfile                 # Render start command
└── .env.example              # Template for required environment variables
```

## Project Status

- [x] Core RAG pipeline built and tested locally
- [x] FastAPI backend deployed live on Render
- [x] Streamlit frontend built with custom UI matching the LawMate app theme
- [x] Streamlit frontend deployed live on Streamlit Community Cloud
- [x] Multilingual support (English / Urdu / Roman Urdu)
- [x] Secrets properly managed (no API keys in source control)
- [ ] Automated tests / CI pipeline
- [ ] Custom domain

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/ayeshakiran0428-sketch/lawmate-rag.git
cd lawmate-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# then fill in your own PINECONE_API_KEY, PINECONE_INDEX_NAME, GROQ_API_KEY

# 4. Ingest the legal document into Pinecone (first-time setup only)
python ingest.py

# 5a. Run the terminal chatbot
python app.py

# 5b. OR run the API server
uvicorn api:app --reload

# 5c. OR run the Streamlit UI
streamlit run streamlit_app.py
```

## What I'm Learning

This project was built to gain hands-on experience with:
- Designing and implementing a Retrieval-Augmented Generation (RAG) pipeline end-to-end
- Working with vector databases and semantic search (Pinecone)
- Integrating LLM inference APIs (Groq) into a real application
- Deploying Python backends and frontends to cloud PaaS platforms (Render, Streamlit Cloud)
- Managing secrets and environment configuration securely across multiple deployment environments
- Building multilingual NLP features for a real-world use case

## Disclaimer

LawMate is an educational/portfolio project demonstrating AI-powered legal research assistance. The legal reasoning and section identification were validated in consultation with practicing advocates during development to ensure accuracy against the Pakistan Penal Code. However, this tool is not a substitute for professional legal advice — outputs should be verified by a qualified lawyer before being used in any real legal proceeding.

## Author

**Ayesha Kiran**
[GitHub](https://github.com/ayeshakiran0428-sketch)
