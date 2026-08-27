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

# ---------- THEME (Pakistan flag: deep green + white) ----------
DARK_GREEN = "#01411C"
GREEN = "#046A38"
LIGHT_GREEN = "#B7D9C4"
WHITE = "#FFFFFF"

st.markdown(f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{
        background-color: {WHITE};
    }}
    .block-container {{
        padding-top: 1.5rem;
        max-width: 640px;
    }}

    .lm-navbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 2px 18px 2px;
        border-bottom: 1px solid #eee;
        margin-bottom: 20px;
    }}
    .lm-logo {{
        font-size: 1.3rem;
        font-weight: 800;
        color: #1a1a1a;
    }}
    .lm-logo span {{
        color: {GREEN};
    }}

    .lm-hero {{
        background: linear-gradient(180deg, {DARK_GREEN} 0%, {GREEN} 100%);
        border-radius: 18px 18px 0 0;
        padding: 28px 24px 20px 24px;
        color: {WHITE};
    }}
    .lm-hero-top {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}
    .lm-hero-icon {{
        width: 46px;
        height: 46px;
        border-radius: 50%;
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
    }}
    .lm-hero-title {{
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }}
    .lm-hero-sub {{
        font-size: 0.85rem;
        color: {LIGHT_GREEN};
        margin: 2px 0 0 0;
    }}

    .lm-card {{
        background: {WHITE};
        border: 1px solid #eee;
        border-top: none;
        border-radius: 0 0 16px 16px;
        padding: 20px 20px 10px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }}
    .lm-note {{
        font-size: 0.78rem;
        color: #888;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
    }}

    .lm-tips {{
        background: {DARK_GREEN};
        color: {WHITE};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }}
    .lm-tips h4 {{
        margin: 0 0 10px 0;
        font-size: 0.95rem;
    }}
    .lm-tips ul {{
        margin: 0;
        padding-left: 18px;
        font-size: 0.85rem;
        line-height: 1.7;
        color: {LIGHT_GREEN};
    }}

    .lm-results-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        color: {DARK_GREEN};
        margin: 4px 0 10px 0;
        font-size: 1rem;
    }}
    .lm-results-box {{
        background: #F5FAF7;
        border: 1px solid {LIGHT_GREEN};
        border-radius: 14px;
        padding: 18px 20px;
        font-size: 0.92rem;
        color: #222;
        white-space: pre-wrap;
        line-height: 1.6;
    }}
    .lm-placeholder {{
        color: #999;
        font-style: italic;
    }}

    .stTextArea textarea {{
        border-radius: 10px !important;
        border: 1px solid #ddd !important;
        font-size: 0.92rem !important;
    }}

    div.stButton > button {{
        background-color: {WHITE};
        color: {DARK_GREEN};
        border: 2px solid {DARK_GREEN};
        border-radius: 12px;
        padding: 10px 0;
        font-weight: 700;
        width: 100%;
        transition: 0.2s;
    }}
    div.stButton > button:hover {{
        background-color: {DARK_GREEN};
        color: {WHITE};
        border-color: {DARK_GREEN};
    }}
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

# ---------- NAVBAR ----------
st.markdown("""
<div class="lm-navbar">
    <div class="lm-logo">LAW<span>MATE</span></div>
    <div style="font-size:1.2rem;">🔔</div>
</div>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="lm-hero">
    <div class="lm-hero-top">
        <div class="lm-hero-icon">🧠</div>
        <div>
            <p class="lm-hero-title">AI Legal Analyzer</p>
            <p class="lm-hero-sub">Powered by Pakistani Law</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- CARD: INPUT (attached directly under hero) ----------
st.markdown('<div class="lm-card">', unsafe_allow_html=True)

query = st.text_area(
    label="Case description",
    label_visibility="collapsed",
    placeholder="Describe your legal situation — what happened, who was involved, and any documents you have. The more detail you provide, the more accurate the analysis will be...",
    height=140
)

st.markdown(
    '<div class="lm-note">🔒 Your information is kept confidential and secure.</div>',
    unsafe_allow_html=True
)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
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
analyze = st.button("Analyze My Case")

# ---------- RESULTS ----------
st.markdown(
    '<div class="lm-results-header">📄 Applicable Pakistani Law Sections</div>',
    unsafe_allow_html=True
)

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
