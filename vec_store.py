from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment='us-west1-gcp')

index = pc.Index('farmers-2')

def store_embedding(embedding, metadata):
    index.upsert([(metadata['id'], embedding, metadata)])

def query_embedding(embedding):
    query_response = index.query(
        vector=embedding,
        top_k=5,  # Number of similar vectors to retrieve
        include_metadata=True
    )
    return query_response
