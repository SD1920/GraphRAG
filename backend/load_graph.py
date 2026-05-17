import json
from pyTigerGraph import TigerGraphConnection
from dotenv import load_dotenv
import os

load_dotenv()

conn = TigerGraphConnection(
    host=os.getenv("TIGERCLOUD_HOST"),
    graphname=os.getenv("TIGERCLOUD_GRAPHNAME"),
    apiToken=os.getenv("TIGERCLOUD_TOKEN")
)
print(conn.echo())

chunks = json.load(open("data/chunks.json", "r", encoding="utf-8"))
entities = json.load(open("data/entities.json", "r", encoding="utf-8"))

print("Loading vertices...")

companies = {c["company"] for c in chunks}
for company in companies:
    conn.upsertVertex("Company", company, {})

for c in chunks:
    filing_id = c["doc_name"]

    conn.upsertVertex("Filing", filing_id, {})
    conn.upsertEdge(
        "Company",
        c["company"],
        "FILED",
        "Filing",
        filing_id
    )

for c in chunks:
    conn.upsertVertex(
        "Chunk",
        c["chunk_id"],
        {
            "text": {
                "value": c["text"]
            }
        }
    )

    conn.upsertEdge(
        "Filing",
        c["doc_name"],
        "CONTAINS",
        "Chunk",
        c["chunk_id"]
    )

for e in entities:
    for ent in e["entities"]:
        ent_id = f"{ent['label']}_{ent['text'][:50]}"

        conn.upsertVertex("Entity", ent_id, {})

        conn.upsertEdge(
            "Chunk",
            e["chunk_id"],
            "MENTIONS",
            "Entity",
            ent_id
        )

print("Done loading graph.")
print(conn.getVertexCount("*"))