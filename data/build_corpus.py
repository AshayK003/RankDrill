"""Build data/problems.json from liquidslr per-company CSVs (Six Months window).

Only titles, topics, difficulty, and links are stored — never problem statements.
CSVs download at build time and are NOT vendored. Frequency is the CSV's 0-100
recency score (six-month window), so rankings reflect 2025-2026 ask patterns.
"""

import csv
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SEEDS = HERE / "seeds"
BASE = "https://raw.githubusercontent.com/liquidslr/leetcode-company-wise-problems/master"

TARGET_COMPANIES = [
    "Google", "Amazon", "Microsoft", "Meta", "Flipkart",
    "Swiggy", "Zomato", "PhonePe", "razorpay", "CRED", "Groww",
    # Mass recruiters (highest placement volume in India).
    "tcs", "Infosys", "Wipro", "Cognizant", "Accenture", "HCL", "Tech Mahindra",
    # Consulting + banks + hardware + enterprise (MUJ placement season regulars).
    "Deloitte", "EY", "Pwc", "Goldman Sachs", "J.P. Morgan", "Morgan Stanley",
    "Samsung", "Adobe", "Oracle", "SAP", "Cisco", "IBM", "Dell", "Bosch",
    "ZS Associates", "Paytm", "Capgemini", "Larsen & Toubro",
]
# Note: dir names are case-sensitive upstream ("razorpay" is lowercase there).
# Udaan has no upstream dir — it stays in companies.json (salary display) only.
WINDOW = "3. Six Months.csv"


def normalize(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def fetch_company_csv(company):
    """Download one company's six-month CSV. Returns row dicts with company set."""
    url = f"{BASE}/{urllib.parse.quote(company)}/{urllib.parse.quote(WINDOW)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        row["company"] = company
        rows.append(row)
    return rows


def merge_company_rows(rows):
    """Merge per-company CSV rows by normalized title.

    Returns (problems, stats). problems: [{id, title, topics[], difficulty, link,
    companies{company: frequency}}]. Pure function — no I/O.
    """
    merged = {}
    order = []
    total = 0
    for row in rows:
        title = (row.get("Title") or "").strip()
        if not title:
            continue
        total += 1
        key = normalize(title)
        if key not in merged:
            merged[key] = {
                "id": f"p{len(order) + 1:04d}",
                "title": title,
                "topics": [],
                "difficulty": (row.get("Difficulty") or "?").title(),
                "link": row.get("Link") or "",
                "companies": {},
            }
            order.append(key)
        entry = merged[key]
        for topic in (row.get("Topics") or "").split(","):
            topic = topic.strip()
            if topic and topic not in entry["topics"]:
                entry["topics"].append(topic)
        try:
            freq = float(row.get("Frequency") or 0)
        except ValueError:
            freq = 0.0
        entry["companies"][row.get("company", "?")] = freq
    problems = [merged[k] for k in order]
    return problems, {"total_rows": total, "matched_titles": len(problems)}


def load_seed_problems():
    with open(SEEDS / "striver_seed.json", encoding="utf-8") as f:
        return json.load(f)


def main(companies=None, csv_dir=None):
    companies = companies or TARGET_COMPANIES
    rows = []
    if csv_dir:
        for path in Path(csv_dir).glob("*.csv"):
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["company"] = path.stem
                    rows.append(row)
    else:
        for company in companies:
            try:
                rows.extend(fetch_company_csv(company))
            except Exception as exc:
                print(f"WARN: {company} skipped ({exc})")
    problems, stats = merge_company_rows(rows)
    if not problems:
        print("No CSV rows — falling back to seed problems.")
        seed = load_seed_problems()
        problems = [{**p, "companies": p.get("companies", {})} for p in seed]
        stats = {"total_rows": 0, "matched_titles": len(problems), "fallback": True}
    with open(HERE / "problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    print(f"Wrote {len(problems)} problems from {stats['total_rows']} rows: {stats}")


if __name__ == "__main__":
    main()
