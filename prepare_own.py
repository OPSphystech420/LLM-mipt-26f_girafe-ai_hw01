"""Create eight evaluation questions and source IDs from portable anchors."""
import argparse
import json
from collections import Counter
from pathlib import Path

from common import check_questions, read_jsonl, write_jsonl
from evaluate import CATEGORIES, validate_gold
from prepare_data import find_evidence


def prepare_own(corpus, source):
    questions, gold = [], []
    if len(source) != 8 or Counter(row["category"] for row in source) != Counter({category: 2 for category in CATEGORIES}):
        raise ValueError("Expected two questions per category")
    for specification in source:
        questions.append({key: specification[key] for key in ("id", "question", "options")})
        row = {key: specification[key] for key in ("id", "category", "answer_ids", "abstain", "rationale")}
        row["evidence_sets"] = [list(dict.fromkeys(find_evidence(corpus, page, marker) for page, marker in anchors))
                                for anchors in specification["evidence_anchors"]]
        if any(not items or len(items) > 4 for items in row["evidence_sets"]):
            raise ValueError(f"{row['id']}: expected one to four evidence IDs per set")
        if row["abstain"]:
            row["missing_information"] = specification["missing_information"]
        gold.append(row)
    check_questions(questions)
    validate_gold(questions, gold)
    return questions, gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--source", default=str(Path(__file__).with_name("own_questions_source.json")))
    parser.add_argument("--questions", default="own_questions.jsonl")
    parser.add_argument("--gold", default="own_gold.jsonl")
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    questions, gold = prepare_own(read_jsonl(args.corpus), source)
    write_jsonl(args.questions, questions)
    write_jsonl(args.gold, gold)
    print(json.dumps({"questions": len(questions), "categories": dict(Counter(row["category"] for row in gold))}))


if __name__ == "__main__":
    main()
