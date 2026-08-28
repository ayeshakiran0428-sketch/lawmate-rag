import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from utils import clean_query, translate_to_english

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="LawMate — AI Legal Analyzer",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- FONTS + THEME CSS (matches lawmate_redesign.html) ----------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#14231c;
  --forest:#0e3b2c;
  --forest-deep:#0a2b20;
  --forest-light:#1e5c45;
  --parchment:#faf8f3;
  --sage:#eef2ec;
  --gold:#b8863a;
  --gold-light:#f4e9d6;
  --line:#e4e0d5;
  --text-muted:#6b7568;
  --white:#ffffff;
}
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: var(--parchment); font-family: 'Inter', sans-serif; }
.block-container { max-width: 560px; padding-top: 20px; padding-bottom: 40px; }

.lm-header{
  display:flex; align-items:center; justify-content:space-between;
  padding:16px 20px; border-bottom:1px solid var(--line);
  background:var(--white); border-radius:14px; margin-bottom:20px;
}
.lm-logo{ font-family:'Fraunces',serif; font-size:21px; font-weight:600; display:flex; align-items:center; gap:8px; }
.lm-logo .mark{
  display:inline-flex; align-items:center; justify-content:center;
  width:28px; height:28px; border-radius:7px; background:var(--forest); color:white; font-size:14px;
}
.lm-logo .law{ color:var(--ink); }
.lm-logo .mate{ color:var(--forest); }
.lm-bell{
  width:38px; height:38px; border-radius:10px; background:var(--sage);
  display:flex; align-items:center; justify-content:center; color:var(--forest); font-size:16px; position:relative;
}

.lm-hero{
  position:relative; background:linear-gradient(155deg, var(--forest) 0%, var(--forest-deep) 100%);
  border-radius:20px; padding:28px 24px 26px; color:var(--white); margin-bottom:20px; overflow:hidden;
}
.lm-hero-badge{
  width:44px; height:44px; border-radius:12px; background:rgba(255,255,255,0.12);
  display:flex; align-items:center; justify-content:center; margin-bottom:14px; font-size:20px;
}
.lm-hero h1{ font-family:'Fraunces',serif; font-size:24px; font-weight:600; margin:0 0 6px; }
.lm-hero p{ margin:0; font-size:14px; line-height:1.5; color:rgba(255,255,255,0.72); max-width:320px; }
.lm-hero-stat{ margin-top:18px; display:flex; gap:22px; }
.lm-hero-stat div{ font-size:11px; color:rgba(255,255,255,0.55); text-transform:uppercase; letter-spacing:0.04em; }
.lm-hero-stat strong{ display:block; font-family:'Fraunces',serif; font-size:17px; font-weight:600; color:var(--white); margin-bottom:2px; }

.lm-card{ background:var(--white); border:1px solid var(--line); border-radius:18px; padding:20px 20px 6px 20px; margin-bottom:20px; }
.lm-card-label{ font-size:13px; font-weight:600; color:var(--ink); margin-bottom:10px; }
.lm-confidential{ display:flex; align-items:center; gap:6px; margin:10px 0 6px 0; font-size:12px; color:var(--text-muted); }

.stTextArea textarea{
  border:1.5px solid var(--line) !important; border-radius:12px !important;
  padding:14px !important; font-family:'Inter',sans-serif !important; font-size:14.5px !important;
  line-height:1.55 !important; color:var(--ink) !important; background:var(--parchment) !important;
}
.stTextArea textarea:focus{
  border-color:var(--forest) !important; box-shadow:0 0 0 3px rgba(14,59,44,0.10) !important; background:var(--white) !important;
}

.lm-tips{ background:var(--sage); border-radius:18px; padding:20px; margin-bottom:20px; }
.lm-tips-head{ display:flex; align-items:center; gap:8px; margin-bottom:14px; }
.lm-tips-head .icon{
  width:26px; height:26px; border-radius:8px; background:var(--gold-light);
  display:flex; align-items:center; justify-content:center; color:var(--gold); font-size:13px;
}
.lm-tips-head h2{ font-family:'Fraunces',serif; font-size:15px; font-weight:600; margin:0; color:var(--ink); }
.lm-tips-grid{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.lm-tip-item{ background:var(--white); border-radius:12px; padding:12px 12px 13px; border:1px solid var(--line); }
.lm-tip-item .n{ font-family:'Fraunces',serif; font-size:11px; color:var(--gold); font-weight:600; letter-spacing:0.03em; margin-bottom:5px; display:block; }
.lm-tip-item p{ margin:0; font-size:12.5px; line-height:1.45; color:var(--ink); }

div.stButton > button{
  width:100%; border:none; background:var(--forest); color:var(--white) !important;
  font-family:'Inter',sans-serif; font-weight:600; font-size:15px; padding:16px; border-radius:14px;
  box-shadow:0 10px 24px -8px rgba(14,59,44,0.45); transition:0.15s ease;
}
div.stButton > button:hover{ background:var(--forest-light); transform:translateY(-1px); box-shadow:0 14px 28px -8px rgba(14,59,44,0.5); }

.lm-results-head{ font-family:'Fraunces',serif; font-size:16px; font-weight:600; color:var(--ink); margin:24px 0 10px 0; display:flex; align-items:center; gap:8px; }
.lm-results-box{
  background:var(--white); border:1px solid var(--line); border-radius:16px; padding:20px;
  font-size:14px; color:var(--ink); white-space:pre-wrap; line-height:1.65;
}
.lm-placeholder{ color:var(--text-muted); font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ---------- SECRETS ----------
load_dotenv()

def get_secret(key, default=None):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "ppc-rag-index")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    st.error("Missing API keys. Please set PINECONE_API_KEY and GROQ_API_KEY in Streamlit secrets.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource(show_spinner="Connecting to legal database...")
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = load_retriever()

def generate_answer(question, context):
    prompt = f"""
You are a highly knowledgeable legal AI assistant specialized in the Pakistan Penal Code (PPC).
Rules:
- Understand Urdu, Roman Urdu, and English
- Reply in the SAME language the user used
- Your goal is to identify the legal "Case Type" (e.g., Theft, Robbery, Assault) and the exact "Article/Section" for the FIR based on the user's situation.
- Provide the exact and correct answer strictly based on the provided context.
- If you don't know the answer based on the context, say so clearly. Do not invent sections or laws.

User's Case Description:
{question}

Context from Law Document (PPC):
{context}

Please provide:
1. Case Type / Offence
2. Relevant Section(s) for the FIR
3. Brief explanation based on context

Answer:
"""
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error connecting to Groq API: {e}"

# ---------- HEADER ----------
st.markdown("""
<div class="lm-header">
    <div class="lm-logo"><span class="mark">⚖️</span><span class="law">Law</span><span class="mate">Mate</span></div>
    <div class="lm-bell">🔔</div>
</div>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="lm-hero">
    <div class="lm-hero-badge">🧠</div>
    <h1>AI Legal Analyzer</h1>
    <p>Describe what happened in plain language. We'll identify the legal issues and outline your options.</p>
    <div class="lm-hero-stat">
        <div><strong>2 min</strong>Avg. analysis</div>
        <div><strong>50+</strong>Case types</div>
        <div><strong>256-bit</strong>Encryption</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- INPUT CARD ----------
st.markdown('<div class="lm-card">', unsafe_allow_html=True)
st.markdown('<div class="lm-card-label">Describe your situation</div>', unsafe_allow_html=True)

query = st.text_area(
    label="Case description",
    label_visibility="collapsed",
    placeholder="e.g. On August 12th, another driver rear-ended my car at a red light on Main Street. There was visible damage to my bumper and the other driver admitted fault at the scene...",
    height=130
)

st.markdown(
    '<div class="lm-confidential">🔒 Your information is kept confidential and secure.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- TIPS ----------
st.markdown("""
<div class="lm-tips">
    <div class="lm-tips-head"><span class="icon">💡</span><h2>Tips for a better analysis</h2></div>
    <div class="lm-tips-grid">
        <div class="lm-tip-item"><span class="n">01</span><p>Be specific about dates, locations, and parties involved</p></div>
        <div class="lm-tip-item"><span class="n">02</span><p>Mention any existing contracts, agreements, or documents</p></div>
        <div class="lm-tip-item"><span class="n">03</span><p>Describe the outcome you're hoping to achieve</p></div>
        <div class="lm-tip-item"><span class="n">04</span><p>Include any previous legal actions taken</p></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- ANALYZE BUTTON ----------
analyze = st.button("Analyze My Case →")

# ---------- RESULTS ----------
st.markdown('<div class="lm-results-head">📄 Applicable Pakistani Law Sections</div>', unsafe_allow_html=True)

if analyze and query.strip():
    with st.spinner("Searching relevant law sections..."):
        cleaned = clean_query(query)
        translated = translate_to_english(cleaned)
        retrieved_docs = retriever.invoke(translated)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    if not context.strip():
        result_text = "No relevant context found in the document for this query."
    else:
        with st.spinner("Generating analysis..."):
            result_text = generate_answer(query, context)

    st.markdown(f'<div class="lm-results-box">{result_text}</div>', unsafe_allow_html=True)

elif analyze and not query.strip():
    st.warning("Please describe your case before analyzing.")
else:
    st.markdown(
        '<div class="lm-results-box lm-placeholder">Result will appear here...</div>',
        unsafe_allow_html=True
    )
