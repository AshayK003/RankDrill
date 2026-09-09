"""Build data/problems.json + data/companies.json from seed files.

Full Striver-list import and liquidslr frequency merge land in the corpus task;
this scaffold builds a working corpus from seeds so every surface runs end to end.
Only titles, topics, and links are stored — never full problem statements.
"""

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SEEDS = HERE / "seeds"


def load_json(name):
    with open(SEEDS / name, encoding="utf-8") as f:
        return json.load(f)


def normalize(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def attach_company_frequency(problems, csv_dir=None):
    """Match problems to liquidslr per-company CSVs by normalized title.

    csv_dir: folder with <Company>.csv files (downloaded at build time, not vendored).
    Returns (problems, match_rate). Without csv_dir, companies stay empty.
    """
    if not csv_dir:
        return problems, 0.0
    freq = {}
    for path in Path(csv_dir).glob("*.csv"):
        company = path.stem
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                title = row.get("Title") or row.get("title") or ""
                if title:
                    freq.setdefault(normalize(title), {}).setdefault(company, 0)
                    freq[normalize(title)][company] += 1
    matched = 0
    for p in problems:
        hit = freq.get(normalize(p["title"]), {})
        if hit:
            matched += 1
            p["companies"] = hit
    return problems, matched / len(problems) if problems else 0.0


def main(csv_dir=None):
    problems = load_json("striver_seed.json")
    companies = load_json("companies_seed.json")
    problems, rate = attach_company_frequency(problems, csv_dir)
    with open(HERE / "problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    with open(HERE / "companies.json", "w", encoding="utf-8") as f:
        json.dump(companies, f, indent=2)
    print(f"Wrote {len(problems)} problems, {len(companies)} companies, match-rate {rate:.0%}")


if __name__ == "__main__":
    main()
