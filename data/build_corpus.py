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
WINDOWS = [
    "3. Six Months.csv",
    "2. Three Months.csv",
    "1. Thirty Days.csv",
    "5. All.csv",  # last resort only; recorded in stats, disclosed in report
]
WINDOW_STATS = {}


def normalize(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def fetch_company_csv(company, window=WINDOWS[0]):
    """Download one company's CSV for a given window. Returns row dicts."""
    url = f"{BASE}/{urllib.parse.quote(company)}/{urllib.parse.quote(window)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if not (row.get("Title") or "").strip():
            continue
        row["company"] = company
        rows.append(row)
    return rows


def fetch_company_best(company, _fetch=fetch_company_csv):
    """First window with data rows wins. Records the choice in WINDOW_STATS."""
    for window in WINDOWS:
        try:
            rows = _fetch(company, window)
        except Exception:
            continue
        if rows:
            WINDOW_STATS[company] = window
            return rows
    WINDOW_STATS[company] = None
    return []


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
            got = fetch_company_best(company)
            if not got:
                print(f"WARN: {company} skipped (no data in any window)")
            rows.extend(got)
    problems, stats = merge_company_rows(rows)
    with open(SEEDS / "companies_seed.json", encoding="utf-8") as f:
        salary = json.load(f)
    proper = {c["name"].lower(): c["name"] for c in salary}
    for p in problems:
        p["companies"] = {proper.get(k.lower(), k): v for k, v in p["companies"].items()}
    if not problems:
        print("No CSV rows — falling back to seed problems.")
        seed = load_seed_problems()
        problems = [{**p, "companies": p.get("companies", {})} for p in seed]
        stats = {"total_rows": 0, "matched_titles": len(problems), "fallback": True}
    with open(HERE / "problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    with open(SEEDS / "companies_seed.json", encoding="utf-8") as f:
        salary = json.load(f)
    with open(HERE / "companies.json", "w", encoding="utf-8") as f:
        json.dump(salary, f, indent=2)
    print(f"Wrote {len(problems)} problems, {len(salary)} companies: {stats}")
    fallbacks = {c: w for c, w in WINDOW_STATS.items() if w != WINDOWS[0]}
    if fallbacks:
        print(f"Window fallbacks (disclosed): {fallbacks}")


if __name__ == "__main__":
    main()
