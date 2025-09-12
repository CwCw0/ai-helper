AI Policy & Product Helper

A local-first RAG starter with FastAPI, Next.js, and Qdrant. Runs with Docker Compose, supports both local stub models and OpenAI.

🚀 Quick Start

# copy env

cp .env.example .env

# run all services

docker compose up --build

Frontend: http://localhost:3000

Backend (Swagger): http://localhost:8000/docs

Qdrant: http://localhost:6333

Ingest Docs
curl -X POST http://localhost:8000/api/ingest

Ask a Question
curl -X POST http://localhost:8000/api/ask -H 'Content-Type: application/json' \
 -d '{"query":"What’s the shipping SLA to East Malaysia for bulky items?"}'

Tests
docker compose run --rm backend pytest -q

My Journey & Problem-Solving

1. Git Setup Issues

Problem: Repo had .git but git status failed.

Fix: Found .DS_Store conflicts. Cleaned with .gitignore and git rm --cached. Re-initialized repo properly.

2. Docker Compose Setup

First run: unsure if services were connected.

Fix: Learned to confirm by checking logs + endpoints. Verified frontend, backend, and Qdrant all running.

3. Ingestion + Q&A

Ingestion worked, but answers weren’t filtered.

Discovery: Without API key, backend uses a stub LLM and built-in embeddings.

4. Switching to Real LLM (Haven't fixed to this point yet, fixing local filtering first)
