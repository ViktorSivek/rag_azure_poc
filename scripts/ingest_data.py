import os
import re
import uuid
from typing import List, Dict, Tuple

from dotenv import load_dotenv
import fitz
from openai import OpenAI

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient

# ---------------------------
# Config
# ---------------------------
load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "telco-rag-v2")

BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "documents")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

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

# Document names mapping
DOCUMENT_NAMES = {
    "3GPP_TS_23.501.pdf": "3GPP TS 23.501 — System Architecture for the 5G System (5GS); Stage 2",
    "BEREC_Guidelines.pdf": "BEREC Guidelines on the Implementation of the Open Internet Regulation (Net Neutrality)",
    "ENISA_Technical_implementation_guidance.pdf": "ENISA Technical Implementation Guidance for the EU 5G Toolbox",
    "ETSI_EN_300_328.pdf": "ETSI EN 300 328 — Wideband Data Transmission Systems (2.4 GHz)",
    "Vseobecne_opravneni_VO-R-12.pdf": "Czech General Authorization VO‑R/12 — RLAN devices in the 5 GHz band",
}


def get_professional_name(filename: str) -> str:
    """Get professional document name or fallback to filename"""
    return DOCUMENT_NAMES.get(filename, filename)


def embed(text: str) -> List[float]:
    """Generate embedding for text"""
    return (
        openai_client.embeddings.create(
            model=EMBED_MODEL, input=[text.replace("\n", " ")]
        )
        .data[0]
        .embedding
    )


def smart_chunk_text(
    text: str, max_chars: int = 1200, overlap_chars: int = 200
) -> List[str]:
    """
    Enhanced chunking that respects paragraph boundaries and avoids mid-sentence splits
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find a good break point
        break_point = end

        # Look for paragraph break first (double newline)
        para_break = text.rfind("\n\n", start, end)
        if para_break > start + max_chars // 2:
            break_point = para_break + 2
        else:
            # Look for sentence end
            sentence_break = text.rfind(". ", start, end)
            if sentence_break > start + max_chars // 2:
                break_point = sentence_break + 2
            else:
                # Look for any newline
                line_break = text.rfind("\n", start, end)
                if line_break > start + max_chars // 2:
                    break_point = line_break + 1

        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = max(break_point - overlap_chars, start + max_chars // 2)

    return chunks


def extract_pages_with_text(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extract text from each page of PDF, returning list of (page_number, text) tuples
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text") or ""
        pages.append((page_num + 1, text))  # 1-based page numbering

    doc.close()
    return pages


def process_pdf_with_pages(pdf_bytes: bytes, source_document: str) -> List[Dict]:
    """
    Process PDF and create chunks with page information
    """
    pages = extract_pages_with_text(pdf_bytes)
    total_pages = len(pages)

    if not pages:
        return []

    all_chunks = []
    global_chunk_index = 0

    for page_num, page_text in pages:
        if not page_text.strip():
            continue

        # Chunk the page text
        page_chunks = smart_chunk_text(page_text.strip())

        for local_chunk_index, chunk_text in enumerate(page_chunks):
            chunk_doc = {
                "id": str(uuid.uuid4()),
                "source_document": source_document,
                "page": page_num,
                "chunk_index": global_chunk_index,
                "total_pages": total_pages,
                "content": chunk_text,
            }
            all_chunks.append(chunk_doc)
            global_chunk_index += 1

    return all_chunks


def upload_in_batches(docs: List[Dict], batch_size: int = 100):
    """Upload documents in batches"""
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        print(f"  Uploading batch {i // batch_size + 1} ({len(batch)} chunks)...")

        try:
            result = search_client.upload_documents(documents=batch)
            failed = [r for r in result if not r.succeeded]
            if failed:
                print(f"    Failed {len(failed)} chunks (showing first 3):")
                for r in failed[:3]:
                    print(f"      key={r.key} error={r.error_message}")
            else:
                print(f"    ✓ Successfully uploaded {len(batch)} chunks")
        except Exception as e:
            print(f"    ✗ Batch upload failed: {e}")


def main():
    """Main ingestion process"""
    print(f"Starting ingestion to index: {INDEX_NAME}")
    print(f"Using embedding model: {EMBED_MODEL}")

    all_documents = []

    # Process each PDF in blob storage
    for blob in container_client.list_blobs():
        filename = blob.name
        if not filename.lower().endswith(".pdf"):
            print(f"Skipping non-PDF: {filename}")
            continue

        print(f"\nProcessing: {filename}")

        # Download PDF
        pdf_bytes = container_client.get_blob_client(filename).download_blob().readall()

        # Get document name
        professional_name = get_professional_name(filename)
        print(f"  Document name: {professional_name}")

        # Process PDF
        chunks = process_pdf_with_pages(pdf_bytes, professional_name)

        if not chunks:
            print("  ⚠ No chunks produced. Skipping.")
            continue

        print(
            f"  📄 Extracted {len(chunks)} chunks from {chunks[-1]['total_pages']} pages"
        )

        # Generate embeddings
        print(f"  🔄 Generating embeddings...")
        texts = [chunk["content"] for chunk in chunks]

        # Batch embedding generation
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            embeddings = []

            for text in batch_texts:
                embedding = embed(text)
                embeddings.append(embedding)

            # Add embeddings to chunks
            for j, embedding in enumerate(embeddings):
                chunks[i + j]["content_vector"] = embedding

        all_documents.extend(chunks)
        print(f"  ✓ Prepared {len(chunks)} chunks with embeddings")

    # Upload all documents
    if all_documents:
        print(f"\n📤 Uploading {len(all_documents)} total chunks to Azure Search...")
        upload_in_batches(all_documents, batch_size=100)
        print(
            f"\n🎉 Ingestion completed! Indexed {len(all_documents)} chunks with page information."
        )
    else:
        print("\n⚠ No documents to upload.")


if __name__ == "__main__":
    main()
