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
        # Generate embedding for the query using Pinecone's client directly
        try:
            # First, verify the Pinecone client is properly initialized
            if not hasattr(pc, 'embed'):
                # Fallback to direct API call if client doesn't support embedding
                conn = http.client.HTTPSConnection("api.pinecone.io")
                payload = json.dumps({
                    "model": "text-embedding-ada-002",  # Using a more common model
                    "input": search_query.query
                })
                headers = {
                    'Content-Type': 'application/json',
                    'Api-Key': 'pcsk_6kDCmj_GEp6thxT7pAzsQ7iSDJ7sZDRtLNsQ78QQ8FLpqcR4cdHnyFgK1bV3bL4RrWLHYW',
                    'accept': 'application/json'
                }
                
                conn.request("POST", "/v1/embeddings", payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                
                if res.status != 200:
                    error_detail = f"Status: {res.status}, Response: {data}"
                    print(f"Pinecone API Error: {error_detail}")
                    raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {error_detail}")
                
                response_data = json.loads(data)
                query_embedding = response_data["data"][0]["embedding"]
                conn.close()
            else:
                # If client supports embedding
                response = pc.embed(
                    model="text-embedding-ada-002",
                    input=search_query.query
                )
                query_embedding = response.data[0].embedding
                
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Invalid response from embedding service: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

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
