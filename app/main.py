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
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME", "telco-rag-v2")
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
    sources: List[str]  # source names with page citations


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
    # fixed, sensible defaults
    candidates = max(60, top_k * 5)
    vq = VectorizedQuery(
        vector=vec, fields="content_vector", k_nearest_neighbors=candidates
    )
    results = search.search(
        search_text=q,  # hybrid on by default
        vector_queries=[vq],
        select=[
            "id",
            "source_document",
            "content",
            "page",
            "chunk_index",
        ],
        top=top_k,
    )
    hits = []
    for r in results:
        hits.append(
            {
                "id": r["id"],
                "doc": r["source_document"],
                "text": r["content"] or "",
                "page": r.get("page"),
                "chunk_index": r.get("chunk_index"),
                "score": r.get("@search.score"),
            }
        )
    return hits


def format_source_with_pages(doc_name: str, pages: set) -> str:
    """
    Format source name with page citations
    """
    if not pages:
        return doc_name

    sorted_pages = sorted(list(pages))

    if len(sorted_pages) <= 3:
        # Show individual pages: "Document (p. 12, 15, 18)"
        page_str = ", ".join(map(str, sorted_pages))
    elif len(sorted_pages) <= 5:
        # Show range if consecutive, otherwise first few: "Document (p. 12-15)"
        if sorted_pages[-1] - sorted_pages[0] == len(sorted_pages) - 1:
            page_str = f"{sorted_pages[0]}-{sorted_pages[-1]}"
        else:
            page_str = f"{', '.join(map(str, sorted_pages[:3]))}, ..."
    else:
        # Show range for many pages: "Document (p. 12-45, 8 pages)"
        page_str = f"{sorted_pages[0]}-{sorted_pages[-1]} ({len(sorted_pages)} pages)"

    return f"{doc_name} (p. {page_str})"


def aggregate_sources_with_pages(hits: List[dict]) -> List[str]:
    """
    Aggregate sources and their page numbers
    """
    pages_by_doc = {}

    for hit in hits:
        doc = hit["doc"]
        page = hit.get("page")

        if doc not in pages_by_doc:
            pages_by_doc[doc] = set()

        if isinstance(page, int) and page > 0:
            pages_by_doc[doc].add(page)

    # Format each source with its pages
    formatted_sources = []
    for doc, pages in pages_by_doc.items():
        formatted_source = format_source_with_pages(doc, pages)
        formatted_sources.append(formatted_source)

    return formatted_sources


def synthesize_answer(question: str, contexts: List[dict]) -> str:
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


app = FastAPI(title="RAG PoC API", version="2.0")

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
        return {"status": "ok", "index": AZURE_SEARCH_INDEX}
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

        # Aggregate sources with page citations
        sources_with_pages = aggregate_sources_with_pages(hits)

        return AskResponse(answer=answer, sources=sources_with_pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
