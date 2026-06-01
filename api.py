from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_core import process_query

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PPC RAG API",
    description="API for Pakistan Penal Code RAG Chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    answer: str
    context_found: bool
    context_used: str

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # process_query handles the core pipeline: clean, translate, retrieve, generate
        result = process_query(request.query)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
