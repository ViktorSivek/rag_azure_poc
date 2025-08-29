import os
import re
from typing import List, Dict

from dotenv import load_dotenv
import fitz
from openai import OpenAI

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------
# Config
# ---------------------------
load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-poc-index")

BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims

if not all(
    [
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_KEY,
        INDEX_NAME,
        BLOB_CONN_STR,
        BLOB_CONTAINER,
        OPENAI_API_KEY,
    ]
):
    raise RuntimeError("Missing required environment variables. Check your .env file.")

# ---------------------------
# Clients
# ---------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
search_client = SearchClient(
    AZURE_SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(AZURE_SEARCH_KEY)
)
index_client = SearchIndexClient(
    AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY)
)


# ---------------------------
# Index
# ---------------------------
def ensure_index(index_name: str, dims: int = 1536):
    try:
        index_client.get_index(index_name)
        print(f"Index '{index_name}' already exists.")
        return
    except Exception:
        pass

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=False,
            sortable=False,
            facetable=False,
        ),
        # Make content a SearchField so it’s really searchable
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(
            name="source_document",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dims,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile", algorithm_configuration_name="hnsw"
            )
        ],
    )
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_index(index)
    print(f"Index '{index_name}' created (dims={dims}, HNSW, cosine).")


# ---------------------------
# Helpers
# ---------------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = [page.get_text("text") or "" for page in doc]
    doc.close()
    return "\n".join(parts)


SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_\-=]")


def make_key(source_document: str, i: int) -> str:
    base = os.path.splitext(os.path.basename(source_document))[0]
    base = SAFE_KEY_RE.sub("-", base).strip("-")
    if not base:
        base = "doc"
    return f"{base}-{i}"


def split_text(
    text: str, source_document: str, chunk_size=1000, chunk_overlap=200
) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    pieces = splitter.split_text(text)
    out: List[Dict] = []
    for i, t in enumerate(pieces):
        t = t.strip()
        if not t:
            continue
        out.append(
            {
                "id": make_key(source_document, i),
                "content": t,
                "source_document": source_document,
            }
        )
    return out


def embed_batch(
    texts: List[str], model: str = EMBED_MODEL, batch_size: int = 64
) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ") for t in texts[i : i + batch_size]]
        res = openai_client.embeddings.create(model=model, input=batch)
        embeddings.extend([d.embedding for d in res.data])
    return embeddings


def upload_in_batches(docs: List[Dict], batch_size: int = 800):
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        print(f"Uploading batch {i // batch_size + 1} ({len(batch)} docs)...")
        result = search_client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        if failed:
            print(f"Failed {len(failed)} (showing first 5):")
            for r in failed[:5]:
                print(f"key={r.key} error={r.error_message}")
        else:
            print("OK")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # If you switch to text-embedding-3-large, change dims to 3072 and re-ingest to a new index.
    ensure_index(INDEX_NAME, dims=1536)

    prepared: List[Dict] = []

    for blob in container_client.list_blobs():
        name = blob.name
        if not name.lower().endswith(".pdf"):
            print(f"Skipping non-PDF: {name}")
            continue

        print(f"\nProcessing: {name}")
        pdf_bytes = container_client.get_blob_client(name).download_blob().readall()

        text = extract_pdf_text(pdf_bytes)
        if not text.strip():
            print("  No text extracted (likely scanned). Skipping.")
            continue

        chunks = split_text(
            text, source_document=name, chunk_size=1000, chunk_overlap=200
        )
        if not chunks:
            print("  No chunks produced. Skipping.")
            continue

        print(f"  Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
        vectors = embed_batch([c["content"] for c in chunks])
        for c, v in zip(chunks, vectors):
            c["content_vector"] = v

        prepared.extend(chunks)

    if prepared:
        print(f"\nPrepared {len(prepared)} docs. Uploading...")
        upload_in_batches(prepared, batch_size=800)
        print("\n--- Ingestion completed ---")
    else:
        print("Nothing to upload.")
