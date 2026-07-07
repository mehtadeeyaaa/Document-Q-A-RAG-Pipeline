'''
Evaluation script — spec item #12.
Runs 5 known-answer questions through the RAG pipeline and reports hit rate.
'''

from app.rag import generate_answer

# Each question has a "must contain" keyword/phrase used to check correctness.
# Keep these keywords short and distinctive so partial LLM phrasing still matches.
EVAL_SET = [
    {
        "question": "What is the total duration of the internship programme?",
        "expected_keywords": ["45"],
    },
    {
        "question": "Who prepared this training programme document?",
        "expected_keywords": ["Gaurav Sharma"],
    },
    {
        "question": "What vector database is used in the RAG pipeline mini project?",
        "expected_keywords": ["ChromaDB"],
    },
    {
        "question": "What is the minimum hit rate required for the RAG pipeline evaluation?",
        "expected_keywords": ["4", "5"],
    },
    {
        "question": "Which two frameworks are suggested for the multi-agent capstone?",
        "expected_keywords": ["LangGraph", "CrewAI"],
    },
]


def check_answer(answer: str, expected_keywords: list[str]) -> bool:
    """Returns True if ALL expected keywords appear somewhere in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def run_evaluation(doc_id: str = "deeya_plan"):
    results = []
    hits = 0

    for item in EVAL_SET:
        question = item["question"]
        expected = item["expected_keywords"]

        response = generate_answer(question, doc_id=doc_id)
        answer = response["answer"]

        passed = check_answer(answer, expected)
        if passed:
            hits += 1

        results.append({
            "question": question,
            "answer": answer,
            "expected_keywords": expected,
            "passed": passed,
        })

    hit_rate = hits / len(EVAL_SET)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — doc_id: {doc_id}")
    print(f"{'='*60}\n")

    for i, r in enumerate(results, 1):
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{i}. {status}")
        print(f"   Q: {r['question']}")
        print(f"   Expected keywords: {r['expected_keywords']}")
        print(f"   A: {r['answer'][:200]}")
        print()

    print(f"{'='*60}")
    print(f"HIT RATE: {hits}/{len(EVAL_SET)} ({hit_rate:.0%})")
    print(f"{'='*60}\n")

    return hit_rate


if __name__ == "__main__":
    run_evaluation()