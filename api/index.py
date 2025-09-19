from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
import os
import json
import requests
from typing import Dict, Any

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

# Define index name and configuration
index_name = "quickstart-py"

# Initialize the index
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
        try:
            # Generate embedding using llama-text-embed-v2 model via HTTP request
            headers = {
                'Content-Type': 'application/json',
                'Api-Key': 'pcsk_6kDCmj_GEp6thxT7pAzsQ7iSDJ7sZDRtLNsQ78QQ8FLpqcR4cdHnyFgK1bV3bL4RrWLHYW'
            }
            
            response = requests.post(
                'https://api.pinecone.io/vectors/embed',
                headers=headers,
                json={
                    'model': 'llama-text-embed-v2',
                    'input': search_query.query
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to generate embedding: {response.text}")
                
            # Extract the embedding
            query_embedding = response.json()['data'][0]['embedding']
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embedding: {str(e)}"
            )

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
