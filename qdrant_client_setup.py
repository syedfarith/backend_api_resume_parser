from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os



COLLECTION_NAME = "resumes"
QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
QDRANT_URL= os.getenv("QDRANT_URL")


# Connect to Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def setup_qdrant(vector_size: int):
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection '{COLLECTION_NAME}' with vector size {vector_size}")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")

from qdrant_client.http.models import PointStruct

def add_resume_to_qdrant(embedding: list, payload: dict):
    try:
        # Upload point
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=payload.get("id"),
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        print(f"Uploaded resume with id {payload.get('id')}")
    except Exception as e:
        print(f"Error uploading resume: {e}")
        raise


def search_resume(query_embedding: list, top_k: int = 5):
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )
