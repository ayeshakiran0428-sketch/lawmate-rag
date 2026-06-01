import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ppc-rag-index")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in .env file.")

def get_or_create_index(pc, index_name):
    # Create the index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384, # all-MiniLM-L6-v2 uses 384 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )
    return index_name

def main():
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = get_or_create_index(pc, PINECONE_INDEX_NAME)
    
    print("\n--- STEP 1: LOAD PDF ---")
    pdf_path = "ppc.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Cannot find {pdf_path}. Please place it in the project root.")
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {pdf_path}")

    print("\n--- STEP 2: SPLIT TEXT ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split document into {len(chunks)} chunks.")

    print("\n--- STEP 3: INITIALIZE EMBEDDINGS ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("\n--- STEP 4: UPSERT TO PINECONE ---")
    print("Uploading chunks to Pinecone (this may take a few moments)...")
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name
    )
    
    print("\n✅ Ingestion complete! The vector database is populated.")

if __name__ == "__main__":
    main()
