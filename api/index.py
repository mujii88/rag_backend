from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone
import os
import json
import http.client
import ssl

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone client
pc = Pinecone(api_key="YOUR_API_KEY")

# Define index name and initialize index
index_name = "quickstart-py"
dense_index = pc.Index(
    index_name,
    host="https://quickstart-py-ji1hhil.svc.aped-4627-b74a.pinecone.io"
)

# Request/Response models
class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3

class SearchResult(BaseModel):
    text: str
    category: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

@app.get("/")
async def root():
    return {"message": "Vector DB Search API is running"}

@app.post("/search", response_model=SearchResponse)
async def search(search_query: SearchQuery):
    try:
        # === Step 1: Generate embedding via Pinecone Inference API ===
        try:
            conn = http.client.HTTPSConnection(
                "api.pinecone.io", context=ssl._create_unverified_context()
            )

            payload = json.dumps({
                "model": "llama-text-embed-v2",
                "inputs": [search_query.query]
            })

            headers = {
                "Content-Type": "application/json",
                "accept": "application/json",
                "Api-Key": "YOUR_API_KEY"
            }

            conn.request("POST", "/embed", payload, headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")

            if res.status != 200:
                raise Exception(f"Failed to generate embedding: {res.status} {data}")

            response_data = json.loads(data)
            query_embedding = response_data["data"][0]["values"]  # ✅ correct field
            conn.close()

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embedding: {str(e)}"
            )

        # === Step 2: Query your Pinecone index ===
        search_results = dense_index.query(
            namespace="example-namespace",  # change if you use a different namespace
            top_k=search_query.top_k,
            vector=query_embedding,
            include_metadata=True,
            include_values=False
        )

        results = []
        for match in search_results.matches:
            results.append(SearchResult(
                text=match.metadata.get("chunk_text", ""),
                category=match.metadata.get("category", "unknown"),
                score=match.score
            ))

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This is needed for Vercel's serverless functions
api = app
