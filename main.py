from pinecone import Pinecone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

pc = Pinecone(api_key="pcsk_6kDCmj_GEp6thxT7pAzsQ7iSDJ7sZDRtLNsQ78QQ8FLpqcR4cdHnyFgK1bV3bL4RrWLHYW")

index_name = "quickstart-py"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )


records = [
    # ---- Profile ----
    { "_id": "rec1a", "chunk_text": "Final-year Engineering student with strong foundation in AI Automation, Data Structures and Algorithms (DSA) using Python.", "category": "profile_programming" },
    { "_id": "rec1b", "chunk_text": "Proficient in C++, Java, JavaScript, and experienced with circuit design and simulation using tools like Proteus, HFSS, and MATLAB.", "category": "profile_tools" },
    { "_id": "rec1c", "chunk_text": "Passionate about embedded systems, signal processing, and automation. Actively seeking an internship opportunity to apply technical knowledge in a practical environment.", "category": "profile_career_goal" },

    # ---- Experience ----
    { "_id": "rec2", "chunk_text": "Summer Intern at Rohde & Schwarz, Islamabad (Jun 2025 — Aug 2025). Tested software solutions for mobile communication systems, collaborated on integration and optimization, debugged communication software, contributed to design discussions, and gained experience with industry-standard tools.", "category": "experience" },

    # ---- Education ----
    { "_id": "rec3", "chunk_text": "National University of Sciences and Technology (NUST), Rawalpindi (Oct 2022 — Present). BS Electrical Engineering and IT, CGPA: 3.51.", "category": "education" },
    { "_id": "rec4", "chunk_text": "Askari Cadet College, Kallar Kahar (Apr 2019 — Aug 2021). Pre-Engineering, Marks: 1053/1100.", "category": "education" },

    # ---- Skills Expanded ----
    { "_id": "rec5a", "chunk_text": "Programming Languages: Python, C++, Java, JavaScript.", "category": "skills_languages" },
    { "_id": "rec5b", "chunk_text": "Hardware & Electronics: Arduino, Verilog, Circuit Design.", "category": "skills_hardware" },
    { "_id": "rec5c", "chunk_text": "Simulation & Tools: MATLAB, Proteus, HFSS.", "category": "skills_tools" },
    { "_id": "rec5d", "chunk_text": "AI & Automation: AI Automation, Agentic AI, Data Structures and Algorithms, Web Development.", "category": "skills_ai" },

    # ---- Certifications ----
    { "_id": "rec6", "chunk_text": "Certification: Introduction to Embedded System.", "category": "certification" },
    { "_id": "rec7", "chunk_text": "Certification: Hands-on Experience on ARTYZ7 ZYNQ7000 SoC.", "category": "certification" },
    { "_id": "rec8", "chunk_text": "Certification: Interfaced ArtyZ7 FPGA with Vivado Design Suite.", "category": "certification" },
    { "_id": "rec9", "chunk_text": "Certification: Implemented Random Access Memory (RAM) on FPGA.", "category": "certification" },

    # ---- Final Year Project ----
    { "_id": "rec10", "chunk_text": "Final Year Project: Unauthorized Transmitter Localization Using SDRs Mounted on Drones. Built a pipeline using TDOA and SDRs to detect unauthorized transmitters, with multilateration, synchronization, signal processing, and visualization.", "category": "final_year_project" },

    # ---- Projects ----
    { "_id": "rec11", "chunk_text": "Project: Friends Recommendation System in Python using Graphs and Arrays with a user-friendly UI.", "category": "project" },
    { "_id": "rec12", "chunk_text": "Project: Blind Stick using Arduino with sensors and embedded programming.", "category": "project" },
    { "_id": "rec13", "chunk_text": "Project: Self-Balancing Robot applying Linear Control System principles.", "category": "project" },
    { "_id": "rec14", "chunk_text": "Project: Ball and Beam Balancing system extending Self-Balancing Robot, testing Linear Control System implementations.", "category": "project" },
    { "_id": "rec15", "chunk_text": "Project: AI Medical Assistant, deployed AI-based assistant for user access.", "category": "project" },
    { "_id": "rec16", "chunk_text": "Project: Spotify Automation Chatbot for playlist automation using AI.", "category": "project" },
    { "_id": "rec17", "chunk_text": "Project: Social Media Content Generation Automation using AI to optimize and automate content workflows.", "category": "project" },

    # ---- Problem Solving ----
    { "_id": "rec18", "chunk_text": "Solved 500+ coding problems across LeetCode, GFG, and InterviewBit to strengthen problem-solving and algorithmic skills.", "category": "problem_solving" }
]


# Target the index
dense_index = pc.Index(index_name)

# Upsert the records into a namespace
dense_index.upsert_records("example-namespace", records)



stats = dense_index.describe_index_stats()
print(stats)

@app.get("/")
async def root():
    return {"message": "Vector DB Search API is running"}

@app.post("/search", response_model=SearchResponse)
async def search(search_query: SearchQuery):
    try:
        # Search the dense index
        search_results = dense_index.query(
            namespace="example-namespace",
            top_k=search_query.top_k,
            vector=[0] * 384,  # Dummy vector, will be replaced by serverless function
            filter={"chunk_text": {"$ne": ""}},
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
