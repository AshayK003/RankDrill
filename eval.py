"""IR evaluation metrics + corpus-level evaluation harness (pure functions).

Relevance here is topic-level: a retrieved problem is relevant when its topics
overlap the query's labelled relevant_topics.
"""


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
    summary = {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in rows[0] if m != "id"} if rows else {}
    return {"rows": rows, "summary": summary, "k": k}
