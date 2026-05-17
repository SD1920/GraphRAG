import json
import networkx as nx
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_graph() -> nx.Graph:
    chunks = json.load(
        open(
            "data/chunks.json",
            encoding="utf-8"
        )
    )

    entities = json.load(
        open(
            "data/entities.json",
            encoding="utf-8"
        )
    )

    G = nx.Graph()

    for c in chunks:
        G.add_node(
            c["chunk_id"],
            type="chunk",
            text=c["text"],
            company=c["company"],
            doc=c["doc_name"]
        )

        G.add_node(
            c["company"],
            type="company"
        )

        G.add_node(
            c["doc_name"],
            type="filing"
        )

        G.add_edge(
            c["company"],
            c["doc_name"],
            rel="FILED"
        )

        G.add_edge(
            c["doc_name"],
            c["chunk_id"],
            rel="CONTAINS"
        )

    for e in entities:
        for ent in e["entities"]:
            ent_id = (
                f"{ent['label']}_"
                f"{ent['text'][:50]}"
            )

            G.add_node(
                ent_id,
                type="entity",
                label=ent["label"],
                text=ent["text"]
            )

            G.add_edge(
                e["chunk_id"],
                ent_id,
                rel="MENTIONS"
            )

    return G


def query_graphrag(
    question: str,
    G: nx.Graph
) -> dict:

    stopwords = {
        "what", "was", "the", "in",
        "of", "a", "an", "is",
        "are", "how", "did",
        "does", "we", "if",
        "that", "you", "as",
        "based", "on", "which",
        "has", "for", "by", "per"
    }

    keywords = [
        w.lower().strip("?.,")
        for w in question.split()
        if (
            w.lower() not in stopwords
            and len(w) > 2
        )
    ]

    chunk_scores = {}

    for node, data in G.nodes(data=True):

        if data.get("type") != "chunk":
            continue

        text = (
            data.get("text", "")
            + " "
            + data.get("company", "")
        ).lower()

        score = sum(
            1
            for kw in keywords
            if kw in text
        )

        if score > 0:
            chunk_scores[node] = score

    top_chunks = sorted(
        chunk_scores,
        key=chunk_scores.get,
        reverse=True
    )[:1]

    if not top_chunks:
        top_chunks = [
            n
            for n, d in G.nodes(data=True)
            if d.get("type") == "chunk"
        ][:1]

    context = "\n\n".join([
        G.nodes[n].get(
            "text",
            ""
        )
        for n in top_chunks
    ])

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    input_tokens = len(
        prompt.split()
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=200
    )

    answer = (
        response
        .choices[0]
        .message.content
    )

    total_tokens = (
        response
        .usage
        .total_tokens
    )

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
        "context_chunks": len(top_chunks)
    }


if __name__ == "__main__":
    G = build_graph()

    print(
        f"Graph: "
        f"{G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    result = query_graphrag(
        "What was Apple's revenue in 2022?",
        G
    )

    print(result)