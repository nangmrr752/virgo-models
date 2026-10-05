"""Makes new Virgo training examples with a bigger teacher, Gemma 3 27B (distillation).

Gemma's terms allow using its outputs, and Virgo-1.0-Angkor is itself a Gemma model, so the result
stays under the same Gemma terms (commercial use allowed). Gemini's outputs must NOT be used this
way (Google's Gemini terms forbid building competing models with them).

For each topic and language, the teacher 1) writes varied user questions, 2) answers them as Virgo
(Virgo's own system prompt), and 3) every pair is checked: right language and script, sensible
length, no "I am Gemma/Google" identity, no duplicates, and a short teacher grade (keeps 4-5 of 5).
Results are appended to the output file as they're made, so a stopped run continues where it left off.

    python scripts/distill_gemma.py --target 3000 --out chat/data/distilled_gemma27b.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "chat"))
from virgo_chat import SYSTEM  # noqa: E402

TEACHER = "google/gemma-3-27b-it"
KHMER = re.compile(r"[ក-៿]")
WRONG_IDENTITY = re.compile(r"\b(I am|I'm|as) (Gemma|a (large )?language model (trained|made|developed) by Google)|Google DeepMind|ខ្ញុំជា Gemma|ខ្ញុំគឺ Gemma", re.I)

TOPICS = [
    "everyday life in Cambodia", "Cambodian history and Angkor", "Khmer language and grammar", "learning English",
    "translation between Khmer and English", "math word problems", "science explained simply", "health and wellbeing",
    "cooking Khmer food", "farming and agriculture in Cambodia", "small business and selling online", "money and saving",
    "job interviews and careers", "writing emails and letters", "formal Khmer letters", "studying and homework",
    "technology and smartphones", "computers and coding for beginners", "travel and getting around Cambodia",
    "Buddhism and Khmer culture", "family and relationships", "environment and climate", "sports",
    "customer support conversations", "planning and productivity", "safety and careful advice", "general knowledge",
]
STYLES = [
    "a short, simple question", "a request for step-by-step help", "a question asking for an explanation",
    "a request to write something (a message, post, or paragraph)", "a practical how-to question",
    "a question with a small mistake or misunderstanding the assistant should gently correct",
]
LANGUAGES = {
    "km": "Khmer, written in Khmer script, natural everyday Khmer as Cambodians write it",
    "en": "English, simple and natural",
}


def load_teacher(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    big_gpu = torch.cuda.get_device_properties(0).total_memory > 60e9
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if torch.cuda.device_count() > 1 and not big_gpu:  # e.g. Kaggle's 2 × T4 (16 GB each): 27B split over both
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {i: "13GiB" for i in range(torch.cuda.device_count())}
    if not big_gpu:  # 27B in 4 bits (~16 GB) fits a 24 GB L4
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chat"))
    from virgo_chat import load_causal

    model = load_causal(name, **kwargs)
    return tok, model.eval()


def generate(tok, model, prompts, max_new_tokens, temperature=0.8):
    """One batch of single-turn prompts (Gemma has no system role: it goes in the user turn)."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
    return [tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def questions_prompt(topic, style, lang, n):
    return (f"Write {n} different messages that real people might send to an AI assistant about: {topic}. "
            f"Each message is {style}. Write them in {LANGUAGES[lang]}. Make them varied and realistic "
            "(different people, situations and details). Reply with only a JSON list of strings.")


def answer_prompt(question):
    return f"{SYSTEM}\n\nUser: {question}\n\nWrite Virgo's reply only (no labels, no notes)."


def grade_prompt(question, answer):
    return ("Rate this AI assistant reply from 1 to 5 for correctness, helpfulness, and natural language "
            "(for Khmer: correct, natural Khmer in Khmer script). Reply with only the number.\n\n"
            f"Message: {question}\n\nReply: {answer}")


def parse_list(text):
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except ValueError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and 3 < len(s.strip()) < 600]


def good_pair(question, answer, lang):
    if not answer or len(answer) < 2 or len(answer) > 3000 or WRONG_IDENTITY.search(answer):
        return False
    if lang == "km":  # a Khmer question gets a mostly-Khmer answer
        khmer = len(KHMER.findall(answer))
        return khmer >= 0.4 * len(re.sub(r"\s", "", answer))
    return not KHMER.search(question) or "translat" in question.lower()


def norm(text):
    return re.sub(r"\W+", "", text.lower())[:120]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--target", type=int, default=3000, help="examples to make (in total, across runs)")
    p.add_argument("--out", default="chat/data/distilled_gemma27b.jsonl")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--khmer-share", type=float, default=0.6, help="share of Khmer examples")
    p.add_argument("--min-grade", type=int, default=4)
    p.add_argument("--hf-repo", help="also save the file to this private Hugging Face dataset as it grows (e.g. you/Virgo-1.0-Angkor-Data)")
    p.add_argument("--upload-every", type=int, default=200)
    args = p.parse_args()

    seen = set()
    for path in [args.out] + [os.path.join("chat/data", f) for f in os.listdir("chat/data") if f.endswith(".jsonl")]:
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    msgs = json.loads(line)["messages"]
                    seen.add(norm(next(m["content"] for m in msgs if m["role"] == "user")))
                except (ValueError, KeyError, StopIteration):
                    pass
    done = sum(1 for _ in open(args.out, encoding="utf-8")) if os.path.exists(args.out) else 0
    print(f"{done} distilled examples so far; target {args.target}", flush=True)
    if done >= args.target:
        return

    tok, model = load_teacher(args.teacher)
    print("Teacher ready:", args.teacher, flush=True)
    rng = random.Random(done)
    kept = tried = 0
    uploaded = done

    def upload():
        if args.hf_repo:
            from huggingface_hub import HfApi

            HfApi().create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
            HfApi().upload_file(path_or_fileobj=args.out, path_in_repo=os.path.basename(args.out), repo_id=args.hf_repo,
                                repo_type="dataset", commit_message=f"{done} distilled examples")
            print(f"☁️ Saved {done} examples to {args.hf_repo}", flush=True)

    with open(args.out, "a", encoding="utf-8") as out:
        while done < args.target:
            # 1) questions
            plans = [(rng.choice(TOPICS), rng.choice(STYLES), "km" if rng.random() < args.khmer_share else "en") for _ in range(args.batch // 4 or 1)]
            lists = generate(tok, model, [questions_prompt(t, s, l, 6) for t, s, l in plans], 700, temperature=1.0)
            questions = [(q, lang) for (_, _, lang), text in zip(plans, lists) for q in parse_list(text) if norm(q) not in seen]
            questions = list({norm(q): (q, lang) for q, lang in questions}.values())[: args.batch]
            if not questions:
                continue
            # 2) answers as Virgo
            answers = generate(tok, model, [answer_prompt(q) for q, _ in questions], 600, temperature=0.7)
            pairs = [(q, a, lang) for (q, lang), a in zip(questions, answers) if good_pair(q, a, lang)]
            tried += len(questions)
            if not pairs:
                continue
            # 3) teacher grade
            grades = generate(tok, model, [grade_prompt(q, a) for q, a, _ in pairs], 4, temperature=0.1)
            for (q, a, _), g in zip(pairs, grades):
                m = re.search(r"[1-5]", g)
                if not m or int(m.group(0)) < args.min_grade:
                    continue
                out.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                                                   {"role": "assistant", "content": a}]}, ensure_ascii=False) + "\n")
                seen.add(norm(q))
                done += 1
                kept += 1
            out.flush()
            print(f"{done}/{args.target} examples (kept {kept} of {tried} tried this run)", flush=True)
            if done - uploaded >= args.upload_every:
                upload()
                uploaded = done
    upload()
    print("✅ Done:", done, "examples in", args.out)


if __name__ == "__main__":
    main()
