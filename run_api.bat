@echo off
echo Starting PPC RAG API...
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
