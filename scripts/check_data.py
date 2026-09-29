"""Checks the chat training data before training: valid JSON, roles in order, no empty turns.

    python scripts/check_data.py            # checks chat/data/*.jsonl
"""
import glob
import json
import sys

ROLES = {"system", "user", "assistant"}


def check(path):
    problems, count = [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            count += 1
            try:
                msgs = json.loads(line)["messages"]
            except (ValueError, KeyError):
                problems.append(f"{path}:{n}: not a {{\"messages\": [...]}} JSON line")
                continue
            turns = [m for m in msgs if m.get("role") != "system"]
            if any(m.get("role") not in ROLES or not str(m.get("content", "")).strip() for m in msgs):
                problems.append(f"{path}:{n}: a message has an unknown role or empty content")
            if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
                problems.append(f"{path}:{n}: must start with a user turn and end with an assistant reply")
            if any(a["role"] == b["role"] for a, b in zip(turns, turns[1:])):
                problems.append(f"{path}:{n}: user and assistant turns must alternate")
    return count, problems


def main(pattern="chat/data/*.jsonl"):
    total, problems = 0, []
    for path in sorted(glob.glob(pattern)):
        count, found = check(path)
        total += count
        problems += found
    print(f"{total} examples checked.")
    for p in problems:
        print("  ✗", p)
    if problems or not total:
        sys.exit(1)
    print("  ✓ all good")


if __name__ == "__main__":
    main(*sys.argv[1:])
