import sys
import time
import json
import os

sys.path.insert(0, ".")

from backend.rag import query_rag
from backend.llm_only import query_llm_only
from backend.graphrag import (
    query_graphrag,
    build_graph
)

from bert_score import (
    score as bert_score
)

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

judge = Groq(
    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)


def llm_judge(
    question,
    ground_truth,
    answer
) -> bool:

    prompt = f"""
You are a lenient evaluation judge for financial questions.

Question:
{question}

Ground Truth:
{ground_truth}

Answer:
{answer}

Does the Answer convey the same meaning or key facts as the Ground Truth, even if worded differently?

Reply only YES or NO.
"""

    r = (
        judge.chat
        .completions.create(
            model=
            "llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=5
        )
    )

    return (
        "YES"
        in
        r.choices[0]
        .message.content.upper()
    )


def evaluate(questions):

    G = build_graph()

    results = []

    llm_answers = []
    rag_answers = []
    grag_answers = []
    refs = []

    for q in questions:

        question = q["question"]
        ground_truth = q["answer"]

        t0 = time.time()
        llm = query_llm_only(
            question
        )
        llm_lat = round(
            time.time() - t0,
            2
        )

        t0 = time.time()
        rag = query_rag(
            question
        )
        rag_lat = round(
            time.time() - t0,
            2
        )

        t0 = time.time()
        grag = query_graphrag(
            question,
            G
        )
        grag_lat = round(
            time.time() - t0,
            2
        )

        llm_judge_pass = (
            llm_judge(
                question,
                ground_truth,
                grag["answer"]
            )
        )

        token_reduction = round(
            (
                rag["total_tokens"]
                -
                grag["total_tokens"]
            )
            /
            rag["total_tokens"]
            * 100,
            1
        )

        results.append({
            "question":
                question,

            "ground_truth":
                ground_truth,

            "llm_only":
                {
                    **llm,
                    "latency":
                    llm_lat
                },

            "rag":
                {
                    **rag,
                    "latency":
                    rag_lat
                },

            "graphrag":
                {
                    **grag,
                    "latency":
                    grag_lat
                },

            "token_reduction_vs_rag":
                token_reduction,

            "llm_judge":
                (
                    "PASS"
                    if llm_judge_pass
                    else "FAIL"
                )
        })

        llm_answers.append(
            llm["answer"]
        )

        rag_answers.append(
            rag["answer"]
        )

        grag_answers.append(
            grag["answer"]
        )

        refs.append(
            ground_truth
        )

        print(
            f"Q: "
            f"{question[:50]}"
        )

        print(
            f"  "
            f"LLM:"
            f"{llm['total_tokens']}t "
            f"{llm_lat}s | "
            f"RAG:"
            f"{rag['total_tokens']}t "
            f"{rag_lat}s | "
            f"GraphRAG:"
            f"{grag['total_tokens']}t "
            f"{grag_lat}s"
        )

        print(
            f"  "
            f"Token reduction: "
            f"{token_reduction}% "
            f"| Judge: "
            f"{results[-1]['llm_judge']}"
        )

    print(
        "\nComputing "
        "BERTScore..."
    )

    _, _, grag_f1 = (
        bert_score(
            grag_answers,
            refs,
            lang="en",
            verbose=False
        )
    )

    _, _, rag_f1 = (
        bert_score(
            rag_answers,
            refs,
            lang="en",
            verbose=False
        )
    )

    avg_grag_bert = round(
        grag_f1.mean().item(),
        4
    )

    avg_rag_bert = round(
        rag_f1.mean().item(),
        4
    )

    judge_pass_rate = round(
        sum(
            1
            for r in results
            if r[
                "llm_judge"
            ] == "PASS"
        )
        /
        len(results)
        * 100,
        1
    )

    avg_token_reduction = round(
        sum(
            r[
                "token_reduction_vs_rag"
            ]
            for r in results
        )
        /
        len(results),
        1
    )

    print(
        "\n=== FINAL "
        "METRICS ==="
    )

    print(
        "GraphRAG "
        "BERTScore F1: "
        f"{avg_grag_bert}"
    )

    print(
        "RAG "
        "BERTScore F1: "
        f"{avg_rag_bert}"
    )

    print(
        "LLM-as-Judge "
        "pass rate: "
        f"{judge_pass_rate}%"
    )

    print(
        "Avg token "
        "reduction: "
        f"{avg_token_reduction}%"
    )

    summary = {
        "results":
            results,

        "summary":
            {
                "graphrag_bertscore":
                    avg_grag_bert,

                "rag_bertscore":
                    avg_rag_bert,

                "judge_pass_rate":
                    judge_pass_rate,

                "avg_token_reduction":
                    avg_token_reduction
            }
    }

    with open(
        "data/results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "Saved to "
        "data/results.json"
    )

    return summary


if __name__ == "__main__":

    metadata = json.load(
        open(
            "data/metadata.json",
            encoding="utf-8"
        )
    )

    fb_only = [
        m for m in metadata
        if m.get("source")
        == "financebench"
    ]

    seen = set()
    unique = []

    for m in fb_only:

        if (
            m["question"]
            not in seen
        ):
            seen.add(
                m["question"]
            )

            unique.append(m)

    questions = [
        {
            "question":
                m["question"],
            "answer":
                m["answer"]
        }
        for m in unique[:10]
    ]

    evaluate(questions)