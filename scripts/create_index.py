import os
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "telco-rag-v2")

if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY]):
    raise RuntimeError("Missing required environment variables for Azure Search")

client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY))

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SimpleField(
        name="source_document",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
    ),
    SimpleField(
        name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True
    ),
    SimpleField(
        name="chunk_index",
        type=SearchFieldDataType.Int32,
        filterable=True,
        sortable=True,
    ),
    SimpleField(name="total_pages", type=SearchFieldDataType.Int32, filterable=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,  # text-embedding-3-small
        vector_search_profile_name="hnsw-profile",
    ),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
    profiles=[
        VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")
    ],
)

index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

# Delete existing index if it exists
try:
    client.delete_index(INDEX_NAME)
    print(f"Deleted existing index: {INDEX_NAME}")
except:
    pass

# Create new index
client.create_index(index)
print(f"Created index: {INDEX_NAME}")
print(
    "Schema includes: id, source_document, page, chunk_index, total_pages, content, content_vector"
)
