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

# ---------- THEME ----------
st.markdown("""
<style>
:root{
  --green:#0F6B3C;
  --green-dark:#0A4D2B;
  --green-light:#DCEFE3;
  --green-text:#BFE3CC;
  --white:#ffffff;
  --text-muted:#8A9A91;
  --border:#E8ECE9;
}
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: var(--white); font-family: -apple-system, "Segoe UI", Roboto, Inter, sans-serif; }
.block-container { max-width: 520px; padding-top: 20px; padding-bottom: 50px; }

/* Header */
.lm-header{
  display:flex; align-items:center; justify-content:space-between;
  padding:4px 2px 20px 2px; border-bottom:1px solid var(--border); margin-bottom:24px;
}
.lm-logo{ font-size:1.3rem; font-weight:800; display:flex; align-items:center; gap:8px; letter-spacing:0.02em; }
.lm-logo .badge{
  width:28px; height:28px; border-radius:50%; background:var(--green);
  display:flex; align-items:center; justify-content:center; color:white; font-size:14px;
}
.lm-logo .law{ color:#1a1a1a; }
.lm-logo .mate{ color:var(--green); }
.lm-header-icons{ display:flex; align-items:center; gap:16px; }
.lm-bell{ font-size:1.2rem; color:#444; }
.lm-avatar{
  width:32px; height:32px; border-radius:50%; background:var(--green);
  color:white; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px;
}

/* Hero card */
.lm-hero{
  background:var(--green-dark); border-radius:18px; padding:26px 24px;
  color:var(--white); margin-bottom:20px;
}
.lm-hero-top{ display:flex; align-items:center; justify-content:space-between; }
.lm-hero-left{ display:flex; align-items:center; gap:14px; }
.lm-hero-icon{
  width:48px; height:48px; border-radius:50%; border:1.5px solid rgba(255,255,255,0.45);
  display:flex; align-items:center; justify-content:center; font-size:1.4rem; flex-shrink:0;
  background:rgba(255,255,255,0.06);
}
.lm-hero-title{ font-size:1.35rem; font-weight:700; margin:0; color:var(--white); line-height:1.25; }
.lm-hero-sub{ font-size:0.87rem; color:var(--green-text); margin:3px 0 0 0; }
.lm-hero-wrench{ font-size:1.15rem; opacity:0.8; }

/* Input card */
.lm-card{
  background:var(--white); border:1px solid var(--border); border-radius:16px;
  padding:22px 22px 14px 22px; box-shadow:0 4px 16px rgba(15,107,60,0.06); margin-bottom:20px;
}
.lm-card-label{ font-size:0.9rem; font-weight:700; color:#1a1a1a; margin-bottom:12px; }
.lm-note{
  font-size:0.8rem; color:var(--text-muted); display:flex; align-items:center; gap:6px; margin:10px 0 6px 0;
}

.stTextArea textarea{
  border:1px solid var(--border) !important; border-radius:10px !important; font-size:0.93rem !important;
  color:#333 !important; background:#FAFBFA !important; padding:12px !important;
}
.stTextArea textarea:focus{
  border-color:var(--green) !important; box-shadow:0 0 0 3px rgba(15,107,60,0.10) !important; background:var(--white) !important;
}
.stTextArea textarea::placeholder{ color:#a3aca6 !important; }

/* Tips card */
.lm-tips{ background:var(--green-dark); color:var(--white); border-radius:16px; padding:22px 24px; margin-bottom:24px; }
.lm-tips h4{ margin:0 0 14px 0; font-size:1rem; font-weight:700; display:flex; align-items:center; gap:8px; }
.lm-tips ul{ margin:0; padding-left:20px; font-size:0.87rem; line-height:2; color:var(--green-text); }
.lm-tips li::marker{ color:var(--green-text); }

/* Analyze button */
.stButton{ margin-bottom: 28px; }
.stButton > button{
  background:var(--white) !important; color:var(--green-dark) !important;
  border:1.5px solid var(--green-dark) !important; border-radius:999px !important;
  padding:14px 0 !important; font-weight:700 !important; font-size:1rem !important;
  width:100% !important; box-shadow:0 2px 8px rgba(0,0,0,0.04);
}
.stButton > button:hover{
  background:var(--green-dark) !important; color:var(--white) !important;
}
.stButton > button:focus:not(:active){
  color: var(--green-dark) !important;
}

/* Results */
.lm-results-head{
  display:flex; align-items:center; gap:8px; font-weight:700; color:var(--green-dark);
  margin:0 0 12px 0; font-size:1.05rem;
}
.lm-results-box{
  background:var(--green-light); border:1px solid #CDE7D8; border-radius:14px; padding:20px 22px;
  font-size:0.93rem; color:#1f2b24; white-space:pre-wrap; line-height:1.7;
}
.lm-placeholder{ color:#8A9A91; font-style:italic; }
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
    <div class="lm-logo"><span class="badge">⚖</span><span class="law">LAW</span><span class="mate">MATE</span></div>
    <div class="lm-header-icons">
        <span class="lm-bell">🔔</span>
        <div class="lm-avatar">A</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="lm-hero">
    <div class="lm-hero-top">
        <div class="lm-hero-left">
            <div class="lm-hero-icon">🧠</div>
            <div>
                <p class="lm-hero-title">AI Legal Analyzer</p>
                <p class="lm-hero-sub">Powered by Pakistani Law</p>
            </div>
        </div>
        <div class="lm-hero-wrench">🔧</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- INPUT CARD ----------
st.markdown('<div class="lm-card">', unsafe_allow_html=True)
st.markdown('<div class="lm-card-label">Describe your situation</div>', unsafe_allow_html=True)

query = st.text_area(
    label="Case description",
    label_visibility="collapsed",
    placeholder="Describe your legal situation — what happened, who was involved, and any documents you have. The more detail you provide, the more accurate the analysis will be...",
    height=140
)

st.markdown(
    '<div class="lm-note">ⓘ Your information is kept confidential and secure.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- TIPS ----------
st.markdown("""
<div class="lm-tips">
    <h4>💡 Tips for Better Analysis</h4>
    <ul>
        <li>Be specific about dates, locations, and parties involved</li>
        <li>Mention any existing contracts, agreements, or documents</li>
        <li>Describe the outcome you're hoping to achieve</li>
        <li>Include any previous legal actions taken</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------- ANALYZE BUTTON ----------
analyze = st.button("Analyze My Case", use_container_width=True)

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
