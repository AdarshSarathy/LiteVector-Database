from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware,cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel,Field
import numpy as np
import traceback
import uvicorn
import json
import os

from vector_core import EmbeddingService, LiteVectorDB

api_description = """
### Welcome to the LiteVector AI Database!

This is a custom-built, in-memory vector database utilizing a highly optimized, vectorized Cosine Similarity search algorithm. 

**Test Data Context:**
Upon startup, this server automatically injects 4,000 pre-computed document embeddings into RAM. This dataset consists of news headlines spanning four categories: **World, Sports, Business, and Sci/Tech**.

**Try These Example Queries in the `/search` endpoint:**
* *"Space exploration and NASA missions"*
* *"Stock market updates and corporate acquisitions"*
* *"Championship sports games and olympics"*
* *"Global political relations and treaties"*
"""


db = LiteVectorDB(dimension= 384, max_capacity= 10000) #initializes the LiteVectorDB with dimension 384 (for all-MiniLM-L6-v2 specifically) and max capacity as 10000

ai_service = EmbeddingService() #initializes the embedding service

app = FastAPI(title = "LiteVector AI Database", version = "1.0", description=api_description)#, lifespan = lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins = [*],
    allow_credentials = [*],
    allow_methods = [*],
    allow_headers = [*]
)

# To enforce strict datatype match when taking inputs
class InsertRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = Field(default= 0.25, description= 'Absolute minimum Similarity Score.')
    max_allowed_drop: float = Field(default= 0.10, description= 'Advanced: Threshold for semantic drop-off between consecutive results.') # defines the maximum score difference that is accepted between subsequent results

# Root endpoint to verify server status and redirect to the UI
@app.get("/")
async def root():
    return RedirectResponse(url="/docs", status_code=307)

# Insert endpoint
@app.post("/insert")
async def insert_document(request: InsertRequest):
    try:
        vector = ai_service.embed(request.text)

        db.add_record(text= request.text, embedding = vector)

        return {'status': 'success', 'message': f'Document indexed at position {db.current_count-1}'}
    except Exception as e:
        raise HTTPException(status_code= 500, detail= str(e))

# Search endpoint
@app.post("/search")
async def search_documents(request: SearchRequest):
    if db.current_count == 0:
        raise HTTPException(status_code= 400, detail= 'Database is empty. Insert data first.')

    try:
        query_vector = ai_service.embed(request.query)

        raw_results = db.search(query_vector= query_vector,top_k= request.top_k)

        if not raw_results:
            return {'status': 'success', 'results': []}


        final_results= []

        if raw_results[0]['score'] >= request.min_score:
            final_results.append(raw_results[0])

        for i in range(1,len(raw_results)):
            if raw_results[i]['score'] < request.min_score:
                break

            current_score= raw_results[i]['score']
            previous_score= raw_results[i-1]['score']

            if previous_score-current_score > request.max_allowed_drop:
                break
                
            final_results.append(raw_results[i])

        return {'status': 'success', 'results': final_results}
    except Exception as e:
        print("--- ERROR DETECTED ---")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host= '0.0.0.0', port= 8000)