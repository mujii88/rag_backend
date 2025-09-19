from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone
import os
import requests
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_6kDCmj_GEp6thxT7pAzsQ7iSDJ7sZDRtLNsQ78QQ8FLpqcR4cdHnyFgK1bV3bL4RrWLHYW")
index_name = "quickstart-py"
dense_index = pc.Index(index_name)

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
        # Generate embedding for the query using HTTP request
        response = requests.post(
            "https://api.pinecone.io/vectors/embed",
            headers={
                "Content-Type": "application/json",
                "Api-Key": "pcsk_6kDCmj_GEp6thxT7pAzsQ7iSDJ7sZDRtLNsQ78QQ8FLpqcR4cdHnyFgK1bV3bL4RrWLHYW"
            },
            json={
                "model": "llama-text-embed-v2",
                "inputs": [search_query.query]
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
            
        query_embedding = response.json()["data"][0]["values"]

        # Search the dense index with the query embedding
        search_results = dense_index.query(
            namespace="example-namespace",
            top_k=search_query.top_k,
            vector=query_embedding,
            include_metadata=True,
            include_values=False
        )
        
        results = []
        for match in search_results.matches:
            results.append(SearchResult(
                text=match.metadata.get('chunk_text', ''),
                category=match.metadata.get('category', 'unknown'),
                score=match.score
            ))
            
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This is needed for Vercel's serverless functions
api = app
