import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI

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


class AskRequest(BaseModel):
    question: str
    top_k: int = 6  # how many chunks to use


class AskResponse(BaseModel):
    answer: str
    sources: List[str]  # unique doc names only


search = SearchClient(
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_KEY)
)
oai = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"


def embed(text: str):
    return oai.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def retrieve(q: str, top_k: int) -> List[dict]:
    vec = embed(q)
    # fixed, sensible defaults; no UI controls
    candidates = max(60, top_k * 5)
    vq = VectorizedQuery(
        vector=vec, fields="content_vector", k_nearest_neighbors=candidates
    )
    results = search.search(
        search_text=q,  # hybrid on by default
        vector_queries=[vq],
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
    # no inline citations, just plain context
    context = "\n---\n".join([(h["text"] or "") for h in contexts])
    sys = "Answer only from the provided context. If unknown, say you don't know. Do not include citations."
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    chat = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content


app = FastAPI(title="RAG PoC API", version="1.2")

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
        _ = search.get_document_count()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        hits = retrieve(body.question, top_k=body.top_k)
        if not hits:
            return AskResponse(
                answer="I don't know based on the current index.", sources=[]
            )
        answer = synthesize_answer(body.question, hits)
        # unique doc names, no counts
        unique_sources = list(dict.fromkeys([h["doc"] for h in hits]))
        return AskResponse(answer=answer, sources=unique_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
