"""Score Virgo 1.0 chat on a fixed test set, so you can tell whether a new training run is better.

    python chat/evaluate.py                                   # base + chat/out/virgo-1.0-chat-lora
    python chat/evaluate.py --base google/gemma-3-1b-it --adapter chat/out/virgo-1.0-chat-lora
    python chat/evaluate.py --answers answers.jsonl           # score saved answers, no model needed
    python chat/evaluate.py --server http://127.0.0.1:8088 --key $VIRGO_API_KEY --model virgo-1.0-bayon
                                                              # score the running Virgo server (no second copy in GPU memory)
    python chat/evaluate.py --compare old_report.json         # also show what changed since an earlier report

Each question in chat/eval/questions.jsonl says what a good answer needs:
  lang       "en", "km", "fr", "es", "th", "zh", "vi"...: the answer must be in that language
  number     the exact number the answer must contain (Khmer digits ០-៩ count too)
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


SCRIPTS = {"th": re.compile(r"[\u0e00-\u0e7f]"), "zh": re.compile(r"[\u4e00-\u9fff]"), "ja": re.compile(r"[\u3040-\u30ff]"),
           "ko": re.compile(r"[\uac00-\ud7af]"), "lo": re.compile(r"[\u0e80-\u0eff]"), "ru": re.compile(r"[\u0400-\u04ff]")}
KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")


def in_language(answer, lang):
    khmer = len(KHMER.findall(answer))
    if lang == "km":
        return khmer >= 5
    if lang in SCRIPTS:
        return len(SCRIPTS[lang].findall(answer)) >= 3
    # English and other Latin-script languages (or a numbers-only answer like "30"): little or no Khmer
    return khmer < 5 and bool(answer.strip())


def has_number(answer, number):
    """True when the answer contains the number (−188, 20,000, ២៥, 3.5...)."""
    text = answer.translate(KHMER_DIGITS).replace("−", "-").replace("–", "-")
    text = re.sub(r"(?<=\d)[,\u00a0 ](?=\d{3}\b)", "", text)  # 20,000 → 20000
    for found in re.findall(r"-?\d+(?:\.\d+)?", text):
        if abs(float(found) - float(number)) < 1e-6:
            return True
    return False


def score(item, answer):
    """Returns a list of (check name, passed)."""
    checks = [("language", in_language(answer, item["lang"]))]
    if item.get("any"):
        checks.append(("includes", any(word.lower() in answer.lower() for word in item["any"])))
    if item.get("number") is not None:
        checks.append(("number", has_number(answer, item["number"])))
    if item.get("not"):
        checks.append(("avoids", not any(word.lower() in answer.lower() for word in item["not"])))
    if item.get("max_words"):
        words = len(answer.split()) if item["lang"] == "en" else len(answer) / 6  # Khmer has no spaces between words
        checks.append(("short", words <= item["max_words"]))
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="chat/eval/questions.jsonl")
    p.add_argument("--base", help="defaults to the adapter's own base (or Gemma 3 4B)")
    p.add_argument("--adapter", default="chat/out/virgo-1.0-chat-lora")
    p.add_argument("--answers", help="a .jsonl of {\"q\": ..., \"answer\": ...} to score instead of running the model")
    p.add_argument("--report", default="chat/eval/report.json")
    p.add_argument("--server", help="score a running Virgo server instead of loading a model, e.g. http://127.0.0.1:8088")
    p.add_argument("--key", default=os.environ.get("VIRGO_API_KEY", ""), help="the server's VIRGO_API_KEY")
    p.add_argument("--model", help="with --server: which chat model, e.g. virgo-1.0-bayon")
    p.add_argument("--compare", help="an earlier report.json: show which skills got better or worse")
    args = p.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.answers:
        with open(args.answers, encoding="utf-8") as f:
            saved = {row["q"]: row["answer"] for row in map(json.loads, filter(str.strip, f))}
        ask = lambda q: saved.get(q, "")
    elif args.server:
        import urllib.request

        def ask(q):
            body = {"messages": [{"role": "user", "content": q}], "max_tokens": 512, "temperature": 0}
            if args.model:
                body["model"] = args.model
            req = urllib.request.Request(args.server.rstrip("/") + "/v1/chat", data=json.dumps(body).encode(), method="POST",
                                         headers={"content-type": "application/json", **({"authorization": f"Bearer {args.key}"} if args.key else {})})
            try:
                with urllib.request.urlopen(req, timeout=300) as res:
                    return json.loads(res.read()).get("reply", "")
            except Exception as err:
                print("  (no answer:", err, ")")
                return ""
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
        print(f"  {skill:<12} {ok}/{n}  {100 * ok / n:5.1f}%")
    print(f"\nOverall: {total[0]}/{total[1]}  {100 * total[0] / total[1]:.1f}%")
    if args.compare and os.path.exists(args.compare):
        old = json.load(open(args.compare, encoding="utf-8"))
        print(f"\nCompared with {args.compare}:  overall {100 * old['overall']:.1f}% → {100 * total[0] / total[1]:.1f}%")
        for skill, (ok, n) in sorted(by_skill.items()):
            before = old.get("by_skill", {}).get(skill)
            if before is not None:
                change = 100 * (ok / n - before)
                print(f"  {skill:<12} {100 * before:5.1f}% → {100 * ok / n:5.1f}%  ({'+' if change >= 0 else ''}{change:.1f})")
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"overall": total[0] / total[1], "by_skill": {k: v[0] / v[1] for k, v in by_skill.items()}, "answers": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
