import json
import spacy
from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("Loading FinanceBench...")
fb = load_dataset(
    "PatronusAI/financebench",
    split="train"
)

print("Loading Finance-Alpaca...")
fa = load_dataset(
    "gbharti/finance-alpaca",
    split="train"
)

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

chunks = []
entities = []
metadata = []


# ─────────────────────────────────────────────
# FinanceBench
# ─────────────────────────────────────────────
for i, row in enumerate(fb):
    evidence_list = row["evidence"]
    company = row["company"]
    doc = row["doc_name"]
    period = row["doc_period"]

    for j, ev in enumerate(evidence_list):
        text = ev.get(
            "evidence_text",
            ""
        ).strip()

        if not text:
            continue

        chunk_id = f"fb_{i:04d}_{j:02d}"

        chunks.append({
            "chunk_id": chunk_id,
            "company": company,
            "doc_name": doc,
            "period": period,
            "text": text,
            "question": row["question"],
            "answer": row["answer"],
            "source": "financebench"
        })

        parsed = nlp(text)

        ents = list({
            (e.text.strip(), e.label_)
            for e in parsed.ents
            if e.label_ in {
                "ORG",
                "GPE",
                "MONEY",
                "DATE",
                "PRODUCT",
                "PERSON"
            }
        })

        entities.append({
            "chunk_id": chunk_id,
            "company": company,
            "entities": [
                {
                    "text": t,
                    "label": l
                }
                for t, l in ents
            ]
        })

        metadata.append({
            "chunk_id": chunk_id,
            "company": company,
            "doc_name": doc,
            "period": period,
            "question": row["question"],
            "answer": row["answer"],
            "source": "financebench"
        })

print(
    f"FinanceBench: "
    f"{len(chunks)} chunks so far"
)


# ─────────────────────────────────────────────
# Finance-Alpaca
# ─────────────────────────────────────────────
def chunk_text(
    text,
    size=512,
    overlap=50
):
    words = text.split()

    results = []
    start = 0

    while start < len(words):
        end = min(
            start + size,
            len(words)
        )

        results.append(
            " ".join(words[start:end])
        )

        if end == len(words):
            break

        start += size - overlap

    return results


target_tokens = 2_100_000

current_tokens = sum(
    len(c["text"].split())
    for c in chunks
)

print(
    f"Tokens so far: "
    f"{current_tokens:,} "
    f"/ need 2,100,000"
)

print(
    f"Starting alpaca loop, "
    f"rows: {len(fa)}"
)

for i, row in enumerate(fa):

    if current_tokens >= target_tokens:
        print(
            f"Target reached "
            f"at row {i}"
        )
        break

    # FIXED: use output instead of text
    text = row.get(
        "output",
        ""
    ).strip()

    # FIXED: reduced threshold
    if len(text.split()) < 20:
        continue

    for j, chunk_text_str in enumerate(
        chunk_text(text)
    ):
        chunk_id = (
            f"fa_{i:05d}_{j:02d}"
        )

        chunks.append({
            "chunk_id": chunk_id,
            "company": "general",
            "doc_name": (
                f"alpaca_{i}"
            ),
            "period": "general",
            "text": chunk_text_str,
            "question": row.get(
                "instruction",
                ""
            ),
            "answer": row.get(
                "output",
                ""
            ),
            "source": "alpaca"
        })

        entities.append({
            "chunk_id": chunk_id,
            "company": "general",
            "entities": []
        })

        metadata.append({
            "chunk_id": chunk_id,
            "company": "general",
            "doc_name": (
                f"alpaca_{i}"
            ),
            "period": "general",
            "question": row.get(
                "instruction",
                ""
            ),
            "answer": row.get(
                "output",
                ""
            ),
            "source": "alpaca"
        })

        current_tokens += len(
            chunk_text_str.split()
        )

    if i % 1000 == 0:
        print(
            f"alpaca row {i}, "
            f"tokens: "
            f"{current_tokens:,}"
        )

print(
    f"Final: "
    f"{len(chunks)} chunks, "
    f"{current_tokens:,} tokens"
)

with open(
    DATA_DIR / "chunks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        chunks,
        f,
        indent=2,
        ensure_ascii=False
    )

with open(
    DATA_DIR / "entities.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        entities,
        f,
        indent=2,
        ensure_ascii=False
    )

with open(
    DATA_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Done.")