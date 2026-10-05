"""DPO pairs from Virgo's OWN mistakes: Angkor (the student) answers checkable problems, and where it
gets one wrong, the teacher (Bayon-1.0-27B, else Gemma 3 27B) answers it; a teacher answer that passes
the check becomes "chosen", Angkor's wrong one "rejected". Pairs aimed at what Angkor really gets wrong
teach far more than the teacher's own rare slips (distill_smart.py's pairs: ~110 in round 2).

Two stages, so only one model is on the GPU at a time:

    python scripts/student_pairs.py --stage student --adapter <data>/out/Angkor-1.0-12B --tasks 6000
    python scripts/student_pairs.py --stage teacher --adapter <data>/out/teacher-Bayon-1.0-27B --hf-repo virgoai/Data-1.0

(bash scripts/train_local.sh mistakes runs both.) The student stage saves the problems it failed to
chat/dpo/student_failed.jsonl; the teacher stage turns them into chat/dpo/student_pairs.jsonl. Both
continue where they stopped. Test questions (chat/eval) are never used.
"""
import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path += [HERE, os.path.join(HERE, "..", "chat")]
from distill_smart import HINT, generate, load_teacher, make_tasks, norm, passes, set_mix  # noqa: E402
from virgo_chat import SYSTEM, adapter_base  # noqa: E402

FAILED = "chat/dpo/student_failed.jsonl"
PAIRS = "chat/dpo/student_pairs.jsonl"


def prompt_of(task):
    hint = task["kind"] in ("math", "logic") and not task.get("max_words")
    return task["q"] + (HINT["km" if task["lang"] == "km" else "en"] if hint else "")


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []


def blocked_questions():
    blocked = set()
    for path in sorted(glob.glob(os.path.join(HERE, "..", "chat", "eval", "*.jsonl"))):  # every test set
        blocked |= {norm(r["q"]) for r in rows(path)}
    return blocked


def student(args):
    """Angkor answers fresh problems; the ones it fails are saved with its wrong answer."""
    seen = blocked_questions() | {norm(r["task"]["q"]) for r in rows(FAILED)}
    tried = int(open(FAILED + ".tried").read()) if os.path.exists(FAILED + ".tried") else 0
    failed = len(rows(FAILED))
    print(f"Student: {tried} problems tried so far, {failed} failed; target {args.tasks} tried", flush=True)
    if tried >= args.tasks:
        return
    tok, model = load_teacher(adapter_base(args.adapter), args.adapter)
    rng = random.Random()
    with open(FAILED, "a", encoding="utf-8") as out:
        while tried < args.tasks:
            tasks = [t for t in make_tasks(rng, args.batch) if norm(t["q"]) not in seen]
            if not tasks:
                continue
            # Angkor's usual answer (low temperature: the mistakes it really makes, not random ones).
            answers = generate(tok, model, [prompt_of(t) for t in tasks], 600, temperature=0.3)
            for task, answer in zip(tasks, answers):
                seen.add(norm(task["q"]))
                if answer and not passes(task, answer):
                    out.write(json.dumps({"task": task, "student": answer}, ensure_ascii=False) + "\n")
                    failed += 1
            tried += len(tasks)
            out.flush()
            open(FAILED + ".tried", "w").write(str(tried))
            print(f"{tried}/{args.tasks} tried, {failed} wrong ({failed / max(tried, 1):.1%})", flush=True)
    print(f"✅ Student done: {failed} problems Angkor got wrong, in {FAILED}")


def teacher(args):
    """The teacher answers each problem Angkor failed (twice); a passing answer makes the DPO pair."""
    todo = rows(FAILED)
    done = {norm(r["prompt"][1]["content"]) for r in rows(PAIRS)}
    skipped_file = PAIRS + ".skipped"
    skipped = set(open(skipped_file, encoding="utf-8").read().split("\n")) if os.path.exists(skipped_file) else set()
    todo = [r for r in todo if norm(r["task"]["q"]) not in done and norm(r["task"]["q"]) not in skipped]
    print(f"Teacher: {len(done)} pairs so far, {len(todo)} problems to answer", flush=True)
    if not todo:
        return upload(args)
    tok, model = load_teacher(args.teacher, args.adapter)
    pairs = len(done)
    with open(PAIRS, "a", encoding="utf-8") as out, open(skipped_file, "a", encoding="utf-8") as skip:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            prompts = [prompt_of(r["task"]) for r in batch]
            answers = generate(tok, model, prompts + prompts, 600, temperature=0.7)
            for i, row in enumerate(batch):
                good = [a for a in (answers[i], answers[i + len(batch)]) if passes(row["task"], a)]
                if not good:  # too hard for the teacher too: no pair
                    skip.write(norm(row["task"]["q"]) + "\n")
                    continue
                chosen = min(good, key=len) if row["task"]["kind"] == "instructions" else good[0]
                prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["task"]["q"]}]
                out.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": row["student"],
                                      "kind": row["task"]["kind"], "from": "student"}, ensure_ascii=False) + "\n")
                pairs += 1
            out.flush()
            skip.flush()
            print(f"{min(start + args.batch, len(todo))}/{len(todo)} answered, {pairs} pairs", flush=True)
            if (start // args.batch) % 50 == 49:
                upload(args)
    upload(args)
    print(f"✅ Teacher done: {pairs} DPO pairs from Angkor's own mistakes, in {PAIRS}")


def upload(args):
    if not args.hf_repo or not os.path.exists(PAIRS):
        return
    from huggingface_hub import HfApi

    HfApi().upload_file(path_or_fileobj=PAIRS, path_in_repo=os.path.basename(PAIRS), repo_id=args.hf_repo,
                        repo_type="dataset", commit_message=f"{len(rows(PAIRS))} DPO pairs from Angkor's mistakes")
    print(f"☁️ Saved to {args.hf_repo}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["student", "teacher"], required=True)
    p.add_argument("--adapter", default="", help="student: Angkor's adapter folder; teacher: the teacher's adapter (empty = plain Gemma)")
    p.add_argument("--teacher", default="google/gemma-3-27b-it", help="the teacher's base model")
    p.add_argument("--tasks", type=int, default=6000, help="student: problems to try in total (across runs)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--mix", default="math=0.25,logic=0.35,instructions=0.15,languages=0.25",
                   help="share of each task kind (default leans on reasoning and languages, the weak spots)")
    p.add_argument("--hf-repo", help="private Hugging Face dataset to save the pairs to")
    args = p.parse_args()
    os.makedirs("chat/dpo", exist_ok=True)
    set_mix(args.mix)
    if args.stage == "student":
        if not args.adapter:
            raise SystemExit("--adapter: Angkor's trained adapter folder, e.g. <data>/out/Angkor-1.0-12B")
        student(args)
    else:
        teacher(args)


if __name__ == "__main__":
    main()
