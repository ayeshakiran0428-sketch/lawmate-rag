import importlib
import sys

requirements = [
    "pypdf",
    "deep_translator",
    "sentence_transformers",
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "langchain_pinecone",
    "pinecone",
    "dotenv",
    "pytest",
    "groq",
    "fastapi",
    "uvicorn",
    "httpx"
]

missing = []
for lib in requirements:
    try:
        if lib == "dotenv":
            importlib.import_module("dotenv")
        elif lib == "pinecone":
            importlib.import_module("pinecone")
        else:
            importlib.import_module(lib.replace("-", "_"))
        print(f"✅ {lib} is installed")
    except ImportError:
        print(f"❌ {lib} is MISSING")
        missing.append(lib)

if missing:
    print(f"\nMissing libraries: {', '.join(missing)}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\nAll systems go!")
    sys.exit(0)
