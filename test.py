import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")
KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

search = SearchClient(ENDPOINT, INDEX, AzureKeyCredential(KEY))
oai = OpenAI(api_key=OPENAI_API_KEY)


def embed(text: str):
    return (
        oai.embeddings.create(model="text-embedding-3-small", input=[text])
        .data[0]
        .embedding
    )


def odata_escape(value: str) -> str:
    return value.replace("'", "''")


def eq_filter(field: str, value: str) -> str:
    return f"{field} eq '{odata_escape(value)}'"


def ask(q: str, k: int = 8, hybrid: bool = True):
    emb = embed(q)
    vq = VectorizedQuery(vector=emb, fields="content_vector", k_nearest_neighbors=k)
    results = search.search(
        search_text=q if hybrid else "",
        vector_queries=[vq],
        select=["source_document", "content"],
        top=k,
    )
    print("\nTop hits:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['source_document']}")
        print(r["content"][:220].replace("\n", " "), "...\n")
        if i >= 2:
            break


def ask_in_doc(doc_name: str, q: str, k: int = 8, hybrid: bool = True):
    emb = embed(q)
    vq = VectorizedQuery(vector=emb, fields="content_vector", k_nearest_neighbors=k)
    results = search.search(
        search_text=q if hybrid else "",
        vector_queries=[vq],
        filter=eq_filter("source_document", doc_name),
        select=["source_document", "content"],
        top=k,
    )
    print(f"\nTop hits in {doc_name}:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['source_document']}")
        print(r["content"][:220].replace("\n", " "), "...\n")
        if i >= 2:
            break


def list_sample_documents(max_items: int = 10):
    faceted = search.search(
        search_text="*", facets=[f"source_document,count:{max_items}"], top=0
    )
    facets = faceted.get_facets() or {}
    docs = facets.get("source_document", [])
    names = []
    print(f"\nSample documents (up to {max_items}):")
    for d in docs:
        value = d.get("value") if isinstance(d, dict) else getattr(d, "value", None)
        count = d.get("count") if isinstance(d, dict) else getattr(d, "count", None)
        print(f"- {value} (chunks: {count})")
        if value:
            names.append(value)
    return names


def vector_self_check_on_doc(doc_name: str, k: int = 5):
    first_chunk = None
    for r in search.search(
        search_text="*",
        filter=eq_filter("source_document", doc_name),
        select=["content"],
        top=1,
    ):
        first_chunk = r["content"]
        break
    if not first_chunk:
        print(f"No chunks found for {doc_name}")
        return
    emb = embed(first_chunk[:1000])
    vq = VectorizedQuery(vector=emb, fields="content_vector", k_nearest_neighbors=k)
    results = search.search(
        search_text="",
        vector_queries=[vq],
        filter=eq_filter("source_document", doc_name),
        select=["source_document", "content"],
        top=k,
    )
    print(f"\nVector self-check within: {doc_name}")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['source_document']}")
        print(r["content"][:160].replace("\n", " "), "...")
        if i >= 2:
            break


def quick_self_retrieval_eval(sample_per_doc: int = 5, k: int = 5):
    # Uses actual chunk text as the query and searches across the whole index.
    # Reports how often top-1 comes from the same source_document.
    docs = list_sample_documents(10)
    if not docs:
        print("No docs to evaluate.")
        return
    total = 0
    correct_at1 = 0
    for doc in docs:
        # Fetch up to 50 chunks for this doc and sample a few
        chunks = []
        for r in search.search(
            search_text="*",
            filter=eq_filter("source_document", doc),
            select=["content", "source_document"],
            top=50,
        ):
            txt = (r["content"] or "").strip()
            if txt:
                chunks.append(txt)
        if not chunks:
            continue
        samples = random.sample(chunks, min(sample_per_doc, len(chunks)))
        for s in samples:
            emb = embed(s[:512])
            vq = VectorizedQuery(
                vector=emb, fields="content_vector", k_nearest_neighbors=k
            )
            results = list(
                search.search(
                    search_text="",
                    vector_queries=[vq],
                    select=["source_document"],
                    top=1,
                )
            )
            total += 1
            if results and results[0]["source_document"] == doc:
                correct_at1 += 1
    if total:
        print(
            f"\nSelf-retrieval@1 across {total} samples: {correct_at1/total:.2%} (k={k})"
        )
    else:
        print("No samples gathered for evaluation.")


if __name__ == "__main__":
    print(f"Index '{INDEX}' document count:", search.get_document_count())

    sample_docs = list_sample_documents(10)

    general_queries = [
        "Explain the difference between AMF, SMF, and UPF in 5G.",
        "List key interfaces (N1, N2, N3, N6, N9) defined in the 5G System architecture.",
        "What information does the S-NSSAI contain and how is it used in slice selection?",
        "Summarize the differences between Non-Standalone (NSA) and Standalone (SA) 5G deployments.",
        "What are the main responsibilities of the UPF in the 5G Core?",
        "What are the transmitter power or PSD limits for 2.4 GHz devices in ETSI EN 300 328?",
        "What are the VHCN criteria defined by BEREC in the 2023 guidelines?",
    ]
    for q in general_queries:
        print("\nQuery:", q)
        ask(q, k=8, hybrid=True)

    # Doc-specific queries to test other PDFs
    # ETSI_EN_300_328.pdf (2.4 GHz)
    ask_in_doc(
        "ETSI_EN_300_328.pdf",
        "According to clause 4.3.2.2, what is the maximum mean e.i.r.p. and maximum power spectral density?",
        k=8,
    )
    ask_in_doc(
        "ETSI_EN_300_328.pdf",
        "Which tables specify transmitter and receiver spurious emission limits (table numbers)?",
        k=8,
    )
    ask_in_doc(
        "ETSI_EN_300_328.pdf",
        "What are the receiver blocking requirements and how are they measured?",
        k=8,
    )

    # BEREC_Guidelines.pdf
    ask_in_doc(
        "BEREC_Guidelines.pdf",
        "List the VHCN criteria and their threshold values.",
        k=8,
    )
    ask_in_doc(
        "BEREC_Guidelines.pdf",
        "What update was made to criterion 4 in BoR (23) 164, and why?",
        k=8,
    )

    # ENISA_Technical_implementation_guidance.pdf
    ask_in_doc(
        "ENISA_Technical_implementation_guidance.pdf",
        "What are ENISA's recommendations for multi-factor authentication deployment?",
        k=8,
    )
    ask_in_doc(
        "ENISA_Technical_implementation_guidance.pdf",
        "Which cryptographic algorithms are recommended and which are deprecated?",
        k=8,
    )
    ask_in_doc(
        "ENISA_Technical_implementation_guidance.pdf",
        "What log retention period does ENISA recommend for security monitoring?",
        k=8,
    )

    # Vseobecne_opravneni_VO-R-12.pdf (Czech; tailor to your content)
    ask_in_doc(
        "Vseobecne_opravneni_VO-R-12.pdf",
        "Jaké jsou maximální hodnoty e.i.r.p. a technické podmínky podle VO-R/12?",
        k=8,
    )
    ask_in_doc(
        "Vseobecne_opravneni_VO-R-12.pdf",
        "Jaké jsou podmínky pro venkovní instalace zařízení podle VO-R/12?",
        k=8,
    )

    # Optional: within-doc vector self-check and overall self-retrieval evaluation
    if sample_docs:
        vector_self_check_on_doc(sample_docs[0], k=5)
    quick_self_retrieval_eval(sample_per_doc=3, k=5)
