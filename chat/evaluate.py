"""Score Virgo 1.0 chat on a fixed test set, so you can tell whether a new training run is better.

    python chat/evaluate.py                                   # base + chat/out/virgo-1.0-chat-lora
    python chat/evaluate.py --base google/gemma-3-1b-it --adapter chat/out/virgo-1.0-chat-lora
    python chat/evaluate.py --answers answers.jsonl           # score saved answers, no model needed

Each question in chat/eval/questions.jsonl says what a good answer needs:
  lang       "en" or "km": the answer must be in that language (Khmer script for km)
  any        at least one of these must appear (identity, facts, honesty…)
  not        none of these may appear (e.g. "I'm ChatGPT")
  max_words  keep simple answers short
The score is the share of checks passed, overall and per skill (identity, language, facts, math,
honesty, safety, style, support). Results are saved to chat/eval/report.json.
"""
import argparse
import json
import os
import re
from collections import defaultdict

KHMER = re.compile(r"[ក-៿]")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def in_language(answer, lang):
    khmer = len(KHMER.findall(answer))
    if lang == "km":
        return khmer >= 5
    # English (or a numbers-only answer like "30" or "x = 3"): little or no Khmer
    return khmer < 5 and bool(answer.strip())


def score(item, answer):
    """Returns a list of (check name, passed)."""
    checks = [("language", in_language(answer, item["lang"]))]
    if item.get("any"):
        checks.append(("includes", any(word.lower() in answer.lower() for word in item["any"])))
    if item.get("not"):
        checks.append(("avoids", not any(word.lower() in answer.lower() for word in item["not"])))
    if item.get("max_words"):
        words = len(answer.split()) if item["lang"] == "en" else len(answer) / 6  # Khmer has no spaces between words
        checks.append(("short", words <= item["max_words"]))
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="chat/eval/questions.jsonl")
    p.add_argument("--base", default="google/gemma-3-4b-it")
    p.add_argument("--adapter", default="chat/out/virgo-1.0-chat-lora")
    p.add_argument("--answers", help="a .jsonl of {\"q\": ..., \"answer\": ...} to score instead of running the model")
    p.add_argument("--report", default="chat/eval/report.json")
    args = p.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.answers:
        with open(args.answers, encoding="utf-8") as f:
            saved = {row["q"]: row["answer"] for row in map(json.loads, filter(str.strip, f))}
        ask = lambda q: saved.get(q, "")
    else:
        import sys

        sys.path.append(os.path.dirname(__file__))
        from virgo_chat import VirgoChat

        bot = VirgoChat(args.base, args.adapter if os.path.isdir(args.adapter) else None)
        ask = lambda q: bot.reply([{"role": "user", "content": q}], temperature=0)

    by_skill = defaultdict(lambda: [0, 0])
    rows = []
    for item in items:
        answer = ask(item["q"])
        checks = score(item, answer)
        passed = sum(ok for _, ok in checks)
        by_skill[item["skill"]][0] += passed
        by_skill[item["skill"]][1] += len(checks)
        failed = [name for name, ok in checks if not ok]
        rows.append({"q": item["q"], "skill": item["skill"], "answer": answer, "failed": failed})
        print(f"{'✓' if not failed else '✗'} [{item['skill']}] {item['q']}" + (f"  → failed: {', '.join(failed)}" if failed else ""))

    total = sum(v[0] for v in by_skill.values()), sum(v[1] for v in by_skill.values())
    print("\nScore by skill:")
    for skill, (ok, n) in sorted(by_skill.items()):
        print(f"  {skill:<9} {ok}/{n}  {100 * ok / n:5.1f}%")
    print(f"\nOverall: {total[0]}/{total[1]}  {100 * total[0] / total[1]:.1f}%")
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"overall": total[0] / total[1], "by_skill": {k: v[0] / v[1] for k, v in by_skill.items()}, "answers": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
