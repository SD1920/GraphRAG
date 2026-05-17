import chromadb
import json
from groq import Groq
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma = chromadb.PersistentClient(path="data/chroma")
collection = chroma.get_or_create_collection("financebench")


def ingest_chunks():
    chunks = json.load(open("data/chunks.json"))

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    metadatas = [
        {
            "company": c["company"],
            "doc": c["doc_name"]
        }
        for c in chunks
    ]

    embeddings = embedder.encode(texts).tolist()

    collection.upsert(
        documents=texts,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Ingested {len(chunks)} chunks into ChromaDB")


def query_rag(question: str, top_k: int = 5) -> dict:
    q_embedding = embedder.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k
    )

    context = "\n\n".join(results["documents"][0])

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    input_tokens = len(prompt.split())

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=200
    )

    answer = response.choices[0].message.content
    total_tokens = response.usage.total_tokens

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
        "context_chunks": len(results["documents"][0])
    }


if __name__ == "__main__":
    ingest_chunks()

    result = query_rag(
        "What was Apple's revenue in 2022?"
    )

    print(result)