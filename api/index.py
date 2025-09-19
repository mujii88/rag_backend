from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
import os
import json
import http.client
import ssl
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

# Get or create the index
index_name = "quickstart-py"

# Initialize the index
index = pc.Index(index_name)

# Prepare data for upsert
vectors = []
for item in [
    # ---- Profile ----
    { "id": "rec1a", "text": "Final-year Engineering student with strong foundation in AI Automation, Data Structures and Algorithms (DSA) using Python.", "category": "profile_programming" },
    { "id": "rec1b", "text": "Proficient in C++, Java, JavaScript, and experienced with circuit design and simulation using tools like Proteus, HFSS, and MATLAB.", "category": "profile_tools" },
    { "id": "rec1c", "text": "Passionate about embedded systems, signal processing, and automation. Actively seeking an internship opportunity to apply technical knowledge in a practical environment.", "category": "profile_career_goal" },

    # ---- Experience ----
    { "id": "rec2", "text": "Summer Intern at Rohde & Schwarz, Islamabad (Jun 2025 — Aug 2025). Tested software solutions for mobile communication systems, collaborated on integration and optimization, debugged communication software, contributed to design discussions, and gained experience with industry-standard tools.", "category": "experience" },

    # ---- Education ----
    { "id": "rec3", "text": "National University of Sciences and Technology (NUST), Rawalpindi (Oct 2022 — Present). BS Electrical Engineering and IT, CGPA: 3.51.", "category": "education" },
    { "id": "rec4", "text": "Askari Cadet College, Kallar Kahar (Apr 2019 — Aug 2021). Pre-Engineering, Marks: 1053/1100.", "category": "education" },

    # ---- Skills Expanded ----
    { "id": "rec5a", "text": "Programming Languages: Python, C++, Java, JavaScript.", "category": "skills_languages" },
    { "id": "rec5b", "text": "Hardware & Electronics: Arduino, Verilog, Circuit Design.", "category": "skills_hardware" },
    { "id": "rec5c", "text": "Simulation & Tools: MATLAB, Proteus, HFSS.", "category": "skills_tools" },
    { "id": "rec5d", "text": "AI & Automation: AI Automation, Agentic AI, Data Structures and Algorithms, Web Development.", "category": "skills_ai" },

    # ---- Certifications ----
    { "id": "rec6", "text": "Certification: Introduction to Embedded System.", "category": "certification" },
    { "id": "rec7", "text": "Certification: Hands-on Experience on ARTYZ7 ZYNQ7000 SoC.", "category": "certification" },
    { "id": "rec8", "text": "Certification: Interfaced ArtyZ7 FPGA with Vivado Design Suite.", "category": "certification" },
    { "id": "rec9", "text": "Certification: Implemented Random Access Memory (RAM) on FPGA.", "category": "certification" },

    # ---- Final Year Project ----
    { "id": "rec10", "text": "Final Year Project: Unauthorized Transmitter Localization Using SDRs Mounted on Drones. Built a pipeline using TDOA and SDRs to detect unauthorized transmitters, with multilateration, synchronization, signal processing, and visualization.", "category": "final_year_project" },

    # ---- Projects ----
    { "id": "rec11", "text": "Project: Friends Recommendation System in Python using Graphs and Arrays with a user-friendly UI.", "category": "project" },
    { "id": "rec12", "text": "Project: Blind Stick using Arduino with sensors and embedded programming.", "category": "project" },
    { "id": "rec13", "text": "Project: Self-Balancing Robot applying Linear Control System principles.", "category": "project" },
    { "id": "rec14", "text": "Project: Ball and Beam Balancing system extending Self-Balancing Robot, testing Linear Control System implementations.", "category": "project" },
    { "id": "rec15", "text": "Project: AI Medical Assistant, deployed AI-based assistant for user access.", "category": "project" },
    { "id": "rec16", "text": "Project: Spotify Automation Chatbot for playlist automation using AI.", "category": "project" },
    { "id": "rec17", "text": "Project: Social Media Content Generation Automation using AI to optimize and automate content workflows.", "category": "project" },

    # ---- Problem Solving ----
    { "id": "rec18", "text": "Solved 500+ coding problems across LeetCode, GFG, and InterviewBit to strengthen problem-solving and algorithmic skills.", "category": "problem_solving" }
]:
    # Generate a simple random embedding for testing
    # In production, replace with actual embedding generation from a model
    import random
    embedding = [random.uniform(-0.5, 0.5) for _ in range(1024)]  # Random values between -0.5 and 0.5
    
    vectors.append({
        'id': item['id'],
        'values': embedding,
        'metadata': {
            'text': item['text'],
            'category': item['category']
        }
    })

# Upsert vectors in batches of 100
for i in range(0, len(vectors), 100):
    batch = vectors[i:i+100]
    index.upsert(vectors=batch, namespace="example-namespace")

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
        # Generate a simple random embedding for the query
        # In production, use the same embedding model as for the documents
        import random
        query_embedding = [random.uniform(-0.5, 0.5) for _ in range(1024)]  # Match dimension (1024)
        
        # Search the index
        results = index.query(
            namespace="example-namespace",
            vector=query_embedding,
            top_k=search_query.top_k,
            include_metadata=True
        )
        
        # Format the results
        results = [
            {
                "text": match.metadata["text"],
                "category": match.metadata["category"],
                "score": match.score
            }
            for match in results.matches
        ]
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This is needed for Vercel's serverless functions
api = app
