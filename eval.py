"""IR evaluation metrics + corpus-level evaluation harness (pure functions).

Relevance here is topic-level: a retrieved problem is relevant when its topics
overlap the query's labelled relevant_topics.
"""

import math


def mean_ci(values):
    """Mean and 95% CI half-width (1.96·std/sqrt(n)). (0.0, 0.0) when empty."""
    n = len(values)
    if not n:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return round(mean, 3), 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return round(mean, 3), round(1.96 * math.sqrt(var / n), 3)


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of top-k retrieved that is relevant. Returns 0.0 when k is 0."""
    if k <= 0:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & set(relevant_ids)) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of all relevant items found in top-k. 0.0 when nothing relevant."""
    if not relevant_ids or k <= 0:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1/rank of the first relevant hit, else 0.0."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(recommend_fn, problems, queries, k=5):
    """Run recommend_fn over labelled queries; return per-query rows + macro averages.

    recommend_fn(query, top_k, company) -> [problem, ...]; each problem has
    "id" and "topics". Each query has "query", "relevant_topics", "company".
    """
    by_id = {p["id"]: p for p in problems}
    rows = []
    for q in queries:
        hits = recommend_fn(q["query"], top_k=k, company=q.get("company")) or []
        retrieved = [h["id"] for h in hits]
        relevant = [pid for pid, p in by_id.items()
                    if set(p.get("topics", [])) & set(q["relevant_topics"])]
        rows.append({
            "id": q["id"],
            f"p@{k}": round(precision_at_k(retrieved, relevant, k), 3),
            f"r@{k}": round(recall_at_k(retrieved, relevant, k), 3),
            "mrr": round(reciprocal_rank(retrieved, relevant), 3),
        })
    metrics = [m for m in rows[0] if m != "id"] if rows else []
    summary = {}
    for m in metrics:
        mean, ci = mean_ci([r[m] for r in rows])
        summary[m] = {"mean": mean, "ci": ci}
    return {"rows": rows, "summary": summary, "k": k}


if __name__ == "__main__":
    """Run both rankers over frozen seeds; print table, save results/results.json."""
    import json
    from pathlib import Path

    from recommender import recommend

    root = Path(__file__).parent
    with open(root / "data" / "problems.json", encoding="utf-8") as f:
        problems = json.load(f)
    with open(root / "data" / "seeds" / "eval_queries.json", encoding="utf-8") as f:
        queries = json.load(f)

    out = {}
    for method in ("bm25", "tfidf"):
        res = evaluate(
            lambda q, top_k, company, m=method: recommend(q, top_k=top_k, company=company, method=m),
            problems, queries, k=5,
        )
        out[method] = res
        print(f"--- {method} ---")
        for row in res["rows"]:
            print(row)
        print("summary:", res["summary"])

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved results/results.json")
