import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI

# Load .env for local dev; in Azure, use Container App secrets/env instead
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY") or os.getenv(
    "AZURE_SEARCH_ADMIN_KEY"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not (
    AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY and OPENAI_API_KEY
):
    missing = [
        k
        for k, v in {
            "AZURE_SEARCH_SERVICE_ENDPOINT": AZURE_SEARCH_ENDPOINT,
            "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX,
            "AZURE_SEARCH_API_KEY": AZURE_SEARCH_KEY,
            "OPENAI_API_KEY": OPENAI_API_KEY,
        }.items()
        if not v
    ]
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


# Models
class AskRequest(BaseModel):
    question: str
    top_k: int = 6  # how many chunks to return to the client + LLM
    knn: int = 50  # k_nearest_neighbors for vector search
    hybrid: bool = True  # BM25 + vector
    doc: Optional[str] = None  # restrict to a specific source_document


class AskResponseSource(BaseModel):
    doc: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[AskResponseSource]


# Clients
search_client = SearchClient(
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_KEY)
)
oai = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"  # pick a cost-effective model


def embed(text: str):
    return oai.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def odata_escape(value: str) -> str:
    return value.replace("'", "''")


def build_filter(doc: Optional[str]) -> Optional[str]:
    if not doc:
        return None
    return f"source_document eq '{odata_escape(doc)}'"


def retrieve(q: str, top_k: int, knn: int, hybrid: bool, doc: Optional[str]):
    vec = embed(q)
    vq = VectorizedQuery(
        vector=vec, fields="content_vector", k_nearest_neighbors=max(knn, top_k)
    )
    results = search_client.search(
        search_text=q if hybrid else "",  # empty string => vector-only
        vector_queries=[vq],
        filter=build_filter(doc),
        select=["id", "source_document", "content"],
        top=top_k,
    )
    hits = []
    for r in results:
        hits.append(
            {"id": r["id"], "doc": r["source_document"], "text": r["content"] or ""}
        )
    return hits


def synthesize_answer(question: str, contexts: List[dict]) -> str:
    # Assemble citations [1], [2], ...
    blocks = []
    for i, h in enumerate(contexts, 1):
        blocks.append(f"[{i}] ({h['doc']}) {h['text']}")
    sys = "You are a precise assistant. Answer using only the provided context. If the answer is not in the context, say you don't know. Cite sources like [1], [2]."
    user = f"Question: {question}\n\nContext:\n" + "\n\n".join(blocks)
    chat = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content


app = FastAPI(title="RAG PoC API", version="1.0")

# CORS (relax for PoC; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    try:
        # quick ping
        _ = search_client.get_document_count()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        hits = retrieve(
            body.question,
            top_k=body.top_k,
            knn=body.knn,
            hybrid=body.hybrid,
            doc=body.doc,
        )
        if not hits:
            return AskResponse(
                answer="I don't know based on the current index.", sources=[]
            )
        answer = synthesize_answer(body.question, hits)
        sources = [
            AskResponseSource(doc=h["doc"], snippet=h["text"][:300]) for h in hits
        ]
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the minimal frontend (place routes above; mount static last)
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
