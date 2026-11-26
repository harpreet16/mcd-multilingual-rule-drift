"""Command line for the grader: responses.jsonl -> grades.jsonl.

  python3 -m grader --responses runs/mock/responses.jsonl --out runs/mock/grades.jsonl --fake-judge
  python3 -m grader --responses runs/qwen3_4b/responses.jsonl --out runs/qwen3_4b/grades.jsonl
  python3 -m grader --responses runs/qwen3_4b/responses.jsonl --out runs/qwen3_4b/grades.jsonl --redo

The dataset files default to the ones in dataset/data. Grading is resumable: a response whose
conversation, language type and model are already in the output file is skipped, so a stopped run
continues where it stopped. --redo starts the output file again.

--fake-judge answers every ask with YES and calls nothing. It is for trying the pipeline without
credentials, never for results: the summary says so, and every row is marked with the fake judge.
"""
import argparse
import os
import sys
from collections import Counter

from contracts.check import check_grade
from contracts.io import append_jsonl, load_conversations, load_grades, load_responses, load_variants
from contracts.models import ServerUnreachable
from grader.hybrid import grade_response

DATA = os.path.join("dataset", "data")


def summarise(grades):
    """Counts worth printing after a grading run, and beside every number in the report."""
    verdicts = Counter(g.verdict for g in grades)
    graded = verdicts["PASS"] + verdicts["FAIL"]
    total = sum(verdicts.values())
    return {
        "grades": total,
        "verdicts": dict(verdicts),
        "methods": dict(Counter(g.method for g in grades)),
        "reply_languages": dict(Counter(g.reply_lang for g in grades)),
        "escalated_to_judge": sum(1 for g in grades if g.escalated),
        "ungraded_share": round(verdicts["UNGRADED"] / total, 4) if total else 0.0,
        "pass_rate_of_graded": round(verdicts["PASS"] / graded, 4) if graded else None,
    }


def already_done(path):
    """The (conversation, language type, model) keys already in the output file."""
    if not os.path.exists(path):
        return set()
    return {(g.conv_id, g.condition, g.model) for g in load_grades(path)}


def main():
    parser = argparse.ArgumentParser(prog="grader")
    parser.add_argument("--responses", required=True, help="responses.jsonl from the runner")
    parser.add_argument("--out", required=True, help="where to write grades.jsonl")
    parser.add_argument("--conversations", default=os.path.join(DATA, "conversations.jsonl"))
    parser.add_argument("--variants", default=os.path.join(DATA, "variants.jsonl"))
    parser.add_argument("--rubric-version", default="", help="passed through to the judge")
    parser.add_argument("--limit", type=int, help="only the first N responses, for a quick check")
    parser.add_argument("--fake-judge", action="store_true", help="answer YES to everything, call nothing")
    parser.add_argument("--redo", action="store_true", help="start the output file again")
    args = parser.parse_args()

    conversations = {c.id: c for c in load_conversations(args.conversations)}
    variants = {(v.conv_id, v.condition): v for v in load_variants(args.variants)}
    responses = load_responses(args.responses)
    if args.limit:
        responses = responses[:args.limit]

    if args.redo and os.path.exists(args.out):
        os.remove(args.out)
    done = already_done(args.out)

    judge = None
    if args.fake_judge:
        from grader.fake_judge import make_fake_judge
        judge = make_fake_judge()

    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)

    written, skipped, problems = [], 0, []
    for response in responses:
        key = (response.conv_id, response.condition, response.model)
        if key in done:
            skipped += 1
            continue
        variant = variants.get((response.conv_id, response.condition))
        conversation = conversations.get(response.conv_id)
        if variant is None or conversation is None:
            sys.exit(f"{response.conv_id} {response.condition}: no variant or conversation to grade it against")
        try:
            grades = grade_response(response, variant, conversation, judge=judge,
                                    rubric_version=args.rubric_version)
        except ServerUnreachable as error:
            print(f"The judge cannot be reached ({error}). Stopping; run again to continue.", file=sys.stderr)
            sys.exit(2)
        for grade in grades:
            problems += check_grade(grade)
            append_jsonl(args.out, grade)
            written.append(grade)

    if problems:
        print(f"{len(problems)} grade(s) broke the contract, first few:", file=sys.stderr)
        for problem in problems[:5]:
            print("  " + problem, file=sys.stderr)
        sys.exit(1)

    summary = summarise(written)
    summary["responses_graded"] = len(responses) - skipped
    summary["responses_skipped_already_done"] = skipped
    if args.fake_judge:
        summary["WARNING"] = "graded with the fake judge: these verdicts are not results"
    print(summary)


if __name__ == "__main__":
    main()
