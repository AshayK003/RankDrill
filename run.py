"""RankDrill CLI: paste a JD (or weakness description), get ranked problems."""

import argparse
import json

from recommender import recommend


def main():
    parser = argparse.ArgumentParser(description="Rank DSA problems for a job description.")
    parser.add_argument("--query", required=True, help="Job description or weakness text.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--company", default=None, help="Company name filter/boost.")
    parser.add_argument("--method", default="bm25", choices=["bm25", "tfidf"])
    args = parser.parse_args()

    try:
        hits = recommend(args.query, top_k=args.top_k, company=args.company, method=args.method)
    except FileNotFoundError:
        print("Corpus not built yet. Run `python data/build_corpus.py` first.")
        return
    except ValueError as exc:
        print(f"Bad input: {exc}")
        return

    if not hits:
        print("No matches. Try a longer description with skills and tools.")
        return

    for rank, h in enumerate(hits, start=1):
        print(f"{rank}. {h['title']} [{h['difficulty']}] score={h['score']:.3f}")
        print(f"   topics: {', '.join(h['topics'])} | matched: {', '.join(h['matched_terms'])}")
        if h.get("companies"):
            print(f"   asked by: {json.dumps(h['companies'])}")
        if h.get("salary_band"):
            print(f"   pay: {h['salary_band']}")
        print(f"   {h['link']}")


if __name__ == "__main__":
    main()
