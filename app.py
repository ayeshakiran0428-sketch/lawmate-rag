import os
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from utils import clean_query, translate_to_english

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ppc-rag-index")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please set it in .env file.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)

# STEP 1: LOAD EMBEDDING MODEL & VECTOR STORE
print("Connecting to Pinecone index...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
try:
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("Connected successfully!\n")
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    print("Have you run 'python ingest.py' yet?")
    exit(1)

# STEP 2: GENERATE ANSWER

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
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error connecting to Groq API: {e}]"

# STEP 3: CHAT LOOP
def main():
    print("Smart Urdu + English Chatbot Ready!\n")

    while True:
        query = input("Ask (or type 'exit'): ")

        if query.lower() == "exit":
            break
        
        if not query.strip():
            continue

        print("Searching...")
        cleaned = clean_query(query)
        translated = translate_to_english(cleaned)

        # Retrieve context from Pinecone
        retrieved_docs = retriever.invoke(translated)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        if not context.strip():
            print("\nAnswer:\nNo relevant context found in the document.\n" + "-" * 50)
            continue

        print("Generating answer...")
        answer = generate_answer(query, context)

        print("\nAnswer:\n", answer)
        print("-" * 50)

if __name__ == "__main__":
    main()