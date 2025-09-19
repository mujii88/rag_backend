from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone
from pinecone import inference
import os

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
    top_k: Optional[int] = 1

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
        # Generate embedding for the query
        query_embedding = inference.embed(
            model="llama-text-embed-v2",
            inputs=[search_query.query]
        ).data[0].values

        # Query Pinecone with embedding
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
