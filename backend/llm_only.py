import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def query_llm_only(question: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": question
        }],
        max_tokens=200
    )

    answer = response.choices[0].message.content
    total_tokens = response.usage.total_tokens

    return {
        "answer": answer,
        "input_tokens": len(question.split()),
        "total_tokens": total_tokens,
        "context_chunks": 0
    }


if __name__ == "__main__":
    result = query_llm_only(
        "What was Apple's revenue in 2022?"
    )

    print(result)