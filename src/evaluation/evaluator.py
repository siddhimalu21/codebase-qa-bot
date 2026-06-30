from typing import List, Dict, Any
from src.llm.groq_client import get_groq_client
from src.llm.qa_engine import QAEngine
from src.config import config


def score_answer_with_llm(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
) -> Dict[str, float]:
    """
    Use Groq LLM to score an answer on 3 metrics.
    Returns scores between 0 and 1.
    """
    client = get_groq_client()

    prompt = f"""You are an evaluation judge for a codebase Q&A system.
Score the following answer on 3 metrics, each between 0.0 and 1.0.

Question: {question}

Retrieved Context:
{context[:1000]}

Generated Answer:
{answer[:500]}

Ground Truth:
{ground_truth}

Score these metrics:
1. faithfulness: Does the answer only use information from the context? (1.0 = fully grounded, 0.0 = hallucinated)
2. answer_relevancy: Does the answer address the question? (1.0 = fully relevant, 0.0 = irrelevant)
3. context_precision: Is the retrieved context relevant to the question? (1.0 = perfect context, 0.0 = wrong context)

Reply with ONLY this format, no other text:
faithfulness: 0.XX
answer_relevancy: 0.XX
context_precision: 0.XX"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()

        scores = {}
        for line in text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                try:
                    scores[key] = float(val.strip())
                except ValueError:
                    scores[key] = 0.0

        return {
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevancy": scores.get("answer_relevancy", 0.0),
            "context_precision": scores.get("context_precision", 0.0),
        }
    except Exception as e:
        print(f"Scoring error: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}


def run_evaluation(
    qa_engine: QAEngine,
    test_dataset: List[Dict],
) -> Dict[str, Any]:
    """Run evaluation on all test questions and return average scores."""
    all_scores = []
    total = len(test_dataset)

    for i, item in enumerate(test_dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{total}] {question[:60]}...")

        try:
            result = qa_engine.answer(question, retrieval_k=10, rerank_k=3)
            qa_engine.clear_memory()

            context = "\n".join([c.content[:300] for c in result["chunks"]])
            scores = score_answer_with_llm(
                question=question,
                answer=result["answer"],
                context=context,
                ground_truth=ground_truth,
            )
            all_scores.append(scores)
            print(f"    faithfulness={scores['faithfulness']:.2f} "
                  f"relevancy={scores['answer_relevancy']:.2f} "
                  f"precision={scores['context_precision']:.2f}")

        except Exception as e:
            print(f"  Error: {e}")

    if not all_scores:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}

    return {
        "faithfulness": round(sum(s["faithfulness"] for s in all_scores) / len(all_scores), 4),
        "answer_relevancy": round(sum(s["answer_relevancy"] for s in all_scores) / len(all_scores), 4),
        "context_precision": round(sum(s["context_precision"] for s in all_scores) / len(all_scores), 4),
    }


def print_evaluation_report(scores: Dict[str, float]):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 40)
    print("   EVALUATION REPORT")
    print("=" * 40)
    print(f"  Faithfulness      : {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy  : {scores['answer_relevancy']:.4f}")
    print(f"  Context Precision : {scores['context_precision']:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  Average Score     : {avg:.4f}")
    print("=" * 40)
    print("Score guide: 0.8+=Excellent  0.6-0.8=Good  <0.6=Needs work")
    print("=" * 40)