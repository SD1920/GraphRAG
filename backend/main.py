# backend/main.py
import sys, time
sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag import query_rag
from backend.llm_only import query_llm_only
from backend.graphrag import query_graphrag, build_graph
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

G = build_graph()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/query/llm")
def llm_endpoint(q: Query):
    t0 = time.time()
    result = query_llm_only(q.question)
    result["latency"] = round(time.time() - t0, 2)
    return result

@app.post("/query/rag")
def rag_endpoint(q: Query):
    t0 = time.time()
    result = query_rag(q.question)
    result["latency"] = round(time.time() - t0, 2)
    return result

@app.post("/query/graphrag")
def graphrag_endpoint(q: Query):
    t0 = time.time()
    result = query_graphrag(q.question, G)
    result["latency"] = round(time.time() - t0, 2)
    return result

@app.post("/query/all")
def all_endpoint(q: Query):
    t0 = time.time(); llm = query_llm_only(q.question); llm["latency"] = round(time.time()-t0, 2)
    t0 = time.time(); rag = query_rag(q.question); rag["latency"] = round(time.time()-t0, 2)
    t0 = time.time(); grag = query_graphrag(q.question, G); grag["latency"] = round(time.time()-t0, 2)
    return {
        "question": q.question,
        "llm_only": llm,
        "rag": rag,
        "graphrag": grag,
        "token_reduction_vs_rag": round((rag["total_tokens"] - grag["total_tokens"]) / rag["total_tokens"] * 100, 1)
    }

@app.get("/results")
def get_results():
    return json.load(open("data/results.json", encoding="utf-8"))