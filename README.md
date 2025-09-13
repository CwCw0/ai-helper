AI Policy & Product Helper

A local-first RAG starter built with FastAPI, Next.js, and Qdrant.
Supports both stub LLMs and OpenAI LLMs. Runs with Docker Compose.

🚀 Quick Start

1. Copy the environment file
   cp .env.example .env

2. Build and run all services
   docker compose up --build

3. Access Services

Frontend: http://localhost:3000

Backend Swagger Docs: http://localhost:8000/docs

Qdrant UI: http://localhost:6333

4. Ingest Documents
   curl -X POST http://localhost:8000/api/ingest

5. Ask a Question
   curl -X POST http://localhost:8000/api/ask \
   -H 'Content-Type: application/json' \
   -d '{"query":"What’s the shipping SLA to East Malaysia for bulky items?"}'

6. Run Tests
   docker compose run --rm backend pytest -q
   🍽 Installation & Tools Used

Docker & Docker Compose

Python 3.11

FastAPI

Next.js

Qdrant (vector DB)

OpenAI Python SDK (for real LLM)

Local embedder (deterministic embeddings)

Pytest for testing

----My Journey & Problem-Solving---

Here’s a detailed log of what I tried, learned, and fixed:

1. Git Setup Issues

Problem: Repo had .git but git status failed; publishing failed due to secret key handling.

Fix: Found .DS_Store conflicts, cleaned them with .gitignore and git rm --cached, reinitialized repo, and moved files to local .gitignore method.

Learning: Hidden system files can silently break workflows—tracking them early saves headaches.

2. Docker Compose Setup

Problem: Services didn’t seem connected on first run.

Fix: Checked logs and endpoints; verified frontend, backend, and Qdrant running correctly.

Learning: Always validate service connectivity early; Docker networking issues can appear subtle.

3. Ingestion & RAG Retrieval

Observation: Initial ingestion failed; endpoints and vector storage needed adjustments.

Fix: Corrected ingestion calls and chunking logic. Stub LLM now produces answers, though filtering logic required improvements.

Learning: Local stub embeddings are useful for testing, but accurate retrieval depends on chunk quality and scoring logic.

4. OpenAI LLM Integration

Attempt: Switched from stub to OpenAI embeddings and generation.

Observation: Answers are now cleaner, cited with sources, and more relevant.

Challenge: Some queries (e.g., shipping SLA) highlight remaining filtering gaps.

Learning: Real LLM integration improves quality but retrieval must be solid.

5. Embedder & Filtering Experiments

Tried LocalEmbedder, hash-based embeddings, and OpenAI embeddings.

Tweaked keyword scoring and chunk filtering to improve top-context selection.

Learning: Hybrid retrieval (embedding + keyword scoring) improves relevance.

6. Testing & Debugging

Attempt: Ran pytest; some tests failed due to environment conflicts.

Fix: Adjusted environment setup; basic test coverage now runs.

Learning: Even partial testing helps catch regressions in retrieval and generation.

---- Notes & Lessons Learned ----

LLM responses depend heavily on chunk filtering and retrieval logic.

Using stub embeddings is useful for local testing, but real OpenAI embeddings improve answer relevance.

Document your workflow—even failures are part of the learning process.

---- Summary ----

This project shows not only a working RAG setup but also all the steps, experiments, and troubleshooting efforts.

Customer Question Example

Can a customer return a damaged blender after 20 days?

From the docs:

Items must be in original packaging with proof of purchase.

Warranty: 12–24 months depending on SKU, covers manufacturing defects.

Answer:

If the blender is damaged due to a manufacturing defect, yes, it’s covered under warranty even after 20 days.

If it’s customer-caused damage, the return policy likely doesn’t cover it.
