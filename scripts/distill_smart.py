"""Makes Virgo smarter: a teacher (Bayon-1.0-27B, else Gemma 3 27B) answers problems whose right answer
is KNOWN or CHECKABLE, and only answers that pass the checks are kept (verified distillation).

    python scripts/distill_smart.py --target 12000                      # ~1 day on a 4090, resumable
    python scripts/distill_smart.py --target 200 --teacher google/gemma-3-27b-it --adapter ""   # quick try

What it makes (each in English AND Khmer, aimed at what the test sets showed is weak):
  math         multi-step word problems, powers (ស្វ័យគុណ), percent, interest... the final number must be right
  logic        order puzzles, days of the week, primes, 9.11 vs 9.9, counting letters, directions
  instructions exactly N lines, yes/no only, at most N words, a number only, real summaries
  languages    questions in Thai, Chinese, Japanese, Korean, French, Spanish, Vietnamese... answered in that language

Every task is checked with the same grader as the test sets (chat/evaluate.py). For each task the teacher
answers twice: a passing answer becomes a training example (chat/data/distilled_smart.jsonl); when the
other answer fails, the pair (good, bad) is saved for DPO (chat/dpo/smart_pairs.jsonl). Questions from
chat/eval are never used, so the test stays fair. Both files are saved to your Hugging Face dataset
(Data-1.0) as they grow, and a stopped run continues where it left off.
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path += [HERE, os.path.join(HERE, "..", "chat")]
from evaluate import score  # noqa: E402
from virgo_chat import SYSTEM  # noqa: E402

KM_DIGITS = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
HINT = {"en": "\n\n(Show short steps, then give the final answer at the end.)",
        "km": "\n\n(បង្ហាញជំហានខ្លីៗ ហើយដាក់ចម្លើយចុងក្រោយនៅខាងចុង។)"}


def km(n):
    """A number in Khmer digits (1,250 → ១២៥០; 3.5 → ៣.៥)."""
    text = f"{n:g}" if isinstance(n, float) else str(n)
    return text.translate(KM_DIGITS)


def money(x):
    return round(x, 2)


# ---------- math: (English, Khmer, answer) with fresh numbers each time ----------
def math_task(r):
    kind = r.randrange(16)
    if kind == 0:
        p, d, t = r.choice([120, 200, 240, 320, 450, 600, 800]), r.choice([10, 15, 20, 25]), r.choice([5, 10])
        ans = money(p * (1 - d / 100) * (1 + t / 100))
        return (f"A bag costs ${p}. It is {d}% off, then {t}% tax is added to the sale price. What is the final price in dollars?",
                f"កាបូបមួយថ្លៃ {km(p)} ដុល្លារ។ គេបញ្ចុះតម្លៃ {km(d)}% រួចបូកពន្ធ {km(t)}% លើតម្លៃក្រោយបញ្ចុះ។ តើតម្លៃចុងក្រោយប៉ុន្មានដុល្លារ?", ans)
    if kind == 1:
        a, b = r.randint(2, 9), r.randint(2, 6)
        return (f"What is {a} to the power of {b}?", f"{km(a)} ស្វ័យគុណ {km(b)} ស្មើប៉ុន្មាន?", a ** b)
    if kind == 2:
        start, per, dist = r.choice([3000, 4000, 5000]), r.choice([1000, 1500, 2000]), r.randint(2, 12)
        ans = start + per * dist
        return (f"A tuk-tuk charges {start:,} riel to start plus {per:,} riel per km. How much is a {dist} km ride, in riel?",
                f"តុកតុកគិតថ្លៃចាប់ផ្តើម {km(start)} រៀល បូក {km(per)} រៀលក្នុងមួយគីឡូម៉ែត្រ។ ជិះ {km(dist)} គីឡូម៉ែត្រ ត្រូវបង់ប៉ុន្មានរៀល?", ans)
    if kind == 3:
        w1, d1 = r.choice([(4, 15), (5, 12), (6, 10), (3, 20), (8, 9)])
        w2 = r.choice([x for x in (2, 3, 4, 5, 6, 8, 9, 10, 12) if (w1 * d1) % x == 0 and x != w1])
        ans = w1 * d1 // w2
        return (f"If {w1} workers build a wall in {d1} days, how many days do {w2} workers take at the same rate?",
                f"កម្មករ {km(w1)} នាក់សង់ជញ្ជាំងរួចក្នុង {km(d1)} ថ្ងៃ។ បើកម្មករ {km(w2)} នាក់ធ្វើក្នុងល្បឿនដូចគ្នា តើត្រូវការប៉ុន្មានថ្ងៃ?", ans)
    if kind == 4:
        n = r.randint(10, 60)
        return (f"The sum of three consecutive whole numbers is {3 * n}. What is the largest?",
                f"ផលបូកនៃចំនួនគត់បីបន្តបន្ទាប់គ្នា គឺ {km(3 * n)}។ តើចំនួនធំជាងគេគឺប៉ុន្មាន?", n + 1)
    if kind == 5:
        p, rate, years = r.choice([500, 1000, 2000, 5000]), r.choice([5, 10, 20]), r.choice([2, 3])
        ans = money(p * (1 + rate / 100) ** years)
        return (f"I put ${p:,} in the bank at {rate}% interest a year, compounded yearly. How many dollars do I have after {years} years?",
                f"ខ្ញុំដាក់ប្រាក់ {km(p)} ដុល្លារ ការប្រាក់ {km(rate)}% ក្នុងមួយឆ្នាំ (ការប្រាក់បូកចូលដើមរៀងរាល់ឆ្នាំ)។ ក្រោយ {km(years)} ឆ្នាំ ខ្ញុំមានប៉ុន្មានដុល្លារ?", ans)
    if kind == 6:
        side = r.randint(3, 25)
        return (f"A square has a perimeter of {4 * side} cm. What is its area in square cm?",
                f"ការេមួយមានបរិមាត្រ {km(4 * side)} សង់ទីម៉ែត្រ។ តើក្រឡាផ្ទៃរបស់វាប៉ុន្មានសង់ទីម៉ែត្រការ៉េ?", side * side)
    if kind == 7:
        g, people, want = r.choice([200, 250, 300, 400]), r.choice([2, 4, 5]), r.choice([6, 8, 10, 12, 15])
        ans = money(g * want / people)
        return (f"A recipe needs {g} g of flour for {people} people. How many grams for {want} people?",
                f"ម្ហូបមួយត្រូវការម្សៅ {km(g)} ក្រាម សម្រាប់មនុស្ស {km(people)} នាក់។ សម្រាប់ {km(want)} នាក់ ត្រូវការប៉ុន្មានក្រាម?", ans)
    if kind == 8:
        price, count, paid = r.choice([1500, 2000, 2500, 3500]), r.randint(2, 4), r.choice([10000, 20000])
        if price * count > paid:
            paid = 20000
        ans = paid - price * count
        return (f"I buy {count} cakes at {price:,} riel each and pay with a {paid:,} riel note. How much change do I get, in riel?",
                f"ខ្ញុំទិញនំ {km(count)} ដុំ ដុំមួយ {km(price)} រៀល ហើយឲ្យលុយ {km(paid)} រៀល។ ខ្ញុំត្រូវបានលុយអាប់ប៉ុន្មានរៀល?", ans)
    if kind == 9:
        a, b = r.randint(11, 49), r.randint(11, 49)
        return (f"What is {a} × {b}?", f"{km(a)} គុណ {km(b)} ស្មើប៉ុន្មាន?", a * b)
    if kind == 10:
        s = r.choice([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20])
        extra = r.randint(2, 9)
        return (f"What is the square root of {s * s} plus {extra} squared?",
                f"ឫសការ៉េនៃ {km(s * s)} បូកនឹង {km(extra)} ការ៉េ ស្មើប៉ុន្មាន?", s + extra * extra)
    if kind == 11:
        speed, hours = r.choice([40, 50, 60, 80]), r.choice([1.5, 2.5, 3, 3.5, 4])
        return (f"A bus travels {speed} km per hour. How many km does it go in {hours:g} hours?",
                f"ឡានក្រុងធ្វើដំណើរ {km(speed)} គីឡូម៉ែត្រក្នុងមួយម៉ោង។ ក្នុងរយៈពេល {km(hours)} ម៉ោង វាធ្វើដំណើរបានប៉ុន្មានគីឡូម៉ែត្រ?", money(speed * hours))
    if kind == 12:
        rate, usd = r.choice([4000, 4100]), r.choice([12, 25, 40, 75, 150])
        return (f"If 1 US dollar is {rate:,} riel, how many riel is {usd} dollars?",
                f"បើ ១ ដុល្លារ ស្មើ {km(rate)} រៀល តើ {km(usd)} ដុល្លារ ស្មើប៉ុន្មានរៀល?", rate * usd)
    if kind == 13:
        nums = [r.randint(5, 95) for _ in range(r.choice([4, 5]))]
        total = sum(nums)
        nums[-1] += (-total) % len(nums)  # a whole-number average
        return (f"What is the average of {', '.join(map(str, nums[:-1]))} and {nums[-1]}?",
                f"មធ្យមភាគនៃ {', '.join(km(n) for n in nums[:-1])} និង {km(nums[-1])} ស្មើប៉ុន្មាន?", sum(nums) // len(nums))
    if kind == 14:
        a, gap, years = r.randint(5, 15), r.randint(20, 35), r.randint(3, 10)
        return (f"Sok is {a} years old and his father is {a + gap}. How old will his father be when Sok is {a + years}?",
                f"សុខអាយុ {km(a)} ឆ្នាំ ហើយឪពុករបស់គាត់អាយុ {km(a + gap)} ឆ្នាំ។ ពេលសុខអាយុ {km(a + years)} ឆ្នាំ តើឪពុករបស់គាត់អាយុប៉ុន្មាន?", a + gap + years)
    total, part = r.choice([(30, 18), (40, 10), (50, 35), (80, 20), (25, 15)])
    return (f"A class has {total} students and {part} are girls. What percent are girls?",
            f"ថ្នាក់មួយមានសិស្ស {km(total)} នាក់ ក្នុងនោះសិស្សស្រី {km(part)} នាក់។ សិស្សស្រីមានប៉ុន្មានភាគរយ?", money(100 * part / total))


DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_KM = ["ច័ន្ទ", "អង្គារ", "ពុធ", "ព្រហស្បតិ៍", "សុក្រ", "សៅរ៍", "អាទិត្យ"]
NAMES = [("Sok", "សុខ"), ("Dara", "ដារ៉ា"), ("Vichet", "វិចិត្រ"), ("Srey", "ស្រី"), ("Bopha", "បុប្ផា"), ("Rith", "រិទ្ធ")]
DIRS_EN = ["north", "east", "south", "west"]
DIRS_KM = ["ខាងជើង", "ខាងកើត", "ខាងត្បូង", "ខាងលិច"]
WORDS = ["strawberry", "banana", "mississippi", "cambodia", "programming", "assistant", "coconut", "elephant"]


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


# ---------- logic: (English item, Khmer item) in the grader's format ----------
def logic_task(r):
    kind = r.randrange(7)
    if kind == 0:
        people = r.sample(NAMES, 3)  # tallest first
        en = f"{people[0][0]} is taller than {people[1][0]}. {people[1][0]} is taller than {people[2][0]}. Who is the shortest?"
        kmq = f"{people[0][1]}ខ្ពស់ជាង{people[1][1]}។ {people[1][1]}ខ្ពស់ជាង{people[2][1]}។ តើអ្នកណាទាបជាងគេ?"
        return ({"q": en, "lang": "en", "any": [people[2][0]], "options": [x[0] for x in people]},
                {"q": kmq, "lang": "km", "any": [people[2][1]], "options": [x[1] for x in people]})
    if kind == 1:
        d, n = r.randrange(7), r.randint(3, 40)
        e = (d + n) % 7
        return ({"q": f"If today is {DAYS_EN[d]}, what day will it be in {n} days?", "lang": "en", "any": [DAYS_EN[e]], "options": DAYS_EN},
                {"q": f"បើថ្ងៃនេះជាថ្ងៃ{DAYS_KM[d]} តើ {km(n)} ថ្ងៃទៀតជាថ្ងៃអ្វី?", "lang": "km", "any": [DAYS_KM[e]], "options": DAYS_KM})
    if kind == 2:
        n = r.choice([x for x in range(21, 200, 2) if x % 5 and x % 3 or is_prime(x)])
        yes = is_prime(n)
        return ({"q": f"Is {n} a prime number? Start your answer with yes or no, then explain.", "lang": "en", "starts": ["yes" if yes else "no"]},
                {"q": f"លេខ {km(n)} ជាចំនួនបឋមឬទេ? ចាប់ផ្តើមចម្លើយដោយ «បាទ» ឬ «ទេ» រួចពន្យល់។", "lang": "km",
                 "starts": ["បាទ", "ចាស", "ជា"] if yes else ["ទេ", "មិន"]})
    if kind == 3:
        whole = r.randint(1, 20)
        a, b = f"{whole}.{r.randint(10, 19)}", f"{whole}.{r.randint(2, 9)}"  # 3.15 vs 3.7: the shorter one is bigger
        return ({"q": f"Which is bigger, {a} or {b}? Answer with the number only.", "lang": "en", "number": float(b), "last": True, "max_words": 3},
                {"q": f"ចំនួនណាធំជាង {km(a)} ឬ {km(b)}? ឆ្លើយតែលេខប៉ុណ្ណោះ។", "lang": "km", "number": float(b), "last": True})
    if kind == 4:
        word = r.choice(WORDS)
        letter = r.choice(sorted(set(word)))
        n = word.count(letter)
        return ({"q": f"How many times does the letter {letter} appear in the word {word}?", "lang": "en", "number": n, "last": True},
                {"q": f"អក្សរ {letter} មានប៉ុន្មានដងក្នុងពាក្យ {word}?", "lang": "km", "number": n, "last": True})
    if kind == 5:
        face, turns = r.randrange(4), [r.choice([90, 180, 270]) * r.choice([1, -1]) for _ in range(2)]
        end = (face + sum(turns) // 90) % 4
        side = lambda t: ("right", "ស្តាំ") if t > 0 else ("left", "ឆ្វេង")
        en = (f"I am facing {DIRS_EN[face]}. I turn {abs(turns[0])} degrees to the {side(turns[0])[0]}, then "
              f"{abs(turns[1])} degrees to the {side(turns[1])[0]}. Which direction am I facing?")
        kmq = (f"ខ្ញុំបែរមុខទៅទិស{DIRS_KM[face]}។ ខ្ញុំបង្វិលទៅ{side(turns[0])[1]} {km(abs(turns[0]))} ដឺក្រេ រួចបង្វិលទៅ"
               f"{side(turns[1])[1]} {km(abs(turns[1]))} ដឺក្រេទៀត។ តើខ្ញុំបែរមុខទៅទិសណា?")
        return ({"q": en, "lang": "en", "any": [DIRS_EN[end]], "options": DIRS_EN},
                {"q": kmq, "lang": "km", "any": [DIRS_KM[end]], "options": DIRS_KM})
    total, left = r.randint(12, 40), r.randint(3, 11)
    return ({"q": f"A farmer has {total} goats. All but {left} run away. How many goats are left?", "lang": "en", "number": left, "last": True},
            {"q": f"កសិករមានពពែ {km(total)} ក្បាល។ ពពែរត់បាត់អស់ លើកលែងតែ {km(left)} ក្បាល។ តើនៅសល់ប៉ុន្មានក្បាល?", "lang": "km", "number": left, "last": True})


TOPICS_EN = ["rainy season", "rice farming", "learning English", "saving money", "Angkor Wat", "healthy food", "the Mekong River",
             "smartphones", "a small coffee shop", "studying for exams", "Khmer New Year", "football", "recycling", "the moon"]
TOPICS_KM = ["រដូវវស្សា", "ការធ្វើស្រែ", "ការរៀនភាសាអង់គ្លេស", "ការសន្សំលុយ", "ប្រាសាទអង្គរវត្ត", "អាហារមានសុខភាពល្អ", "ទន្លេមេគង្គ",
             "ទូរស័ព្ទស្មាតហ្វូន", "ហាងកាហ្វេតូចមួយ", "ការរៀនត្រៀមប្រឡង", "បុណ្យចូលឆ្នាំខ្មែរ", "បាល់ទាត់", "ការកែច្នៃសំណល់", "ព្រះចន្ទ"]


# ---------- instructions: exact formats the grader can check ----------
def instruction_task(r):
    i = r.randrange(len(TOPICS_EN))
    kind = r.randrange(5)
    if kind == 0:
        n = r.choice([3, 4, 5])
        return ({"q": f"Give exactly {n} tips about {TOPICS_EN[i]}, one per line, with no other text.", "lang": "en", "lines": n},
                {"q": f"ផ្តល់គន្លឹះ {km(n)} យ៉ាងអំពី{TOPICS_KM[i]} មួយបន្ទាត់មួយ ដោយមិនសរសេរអ្វីផ្សេងទៀត។", "lang": "km", "lines": n})
    if kind == 1:
        n = r.choice([10, 15, 20])
        return ({"q": f"Explain {TOPICS_EN[i]} in at most {n} words.", "lang": "en", "max_words": n},
                {"q": f"ពន្យល់អំពី{TOPICS_KM[i]} ក្នុងមួយប្រយោគខ្លី។", "lang": "km", "max_words": 20})
    if kind == 2:
        return ({"q": f"Write a short poem about {TOPICS_EN[i]}: exactly 4 lines.", "lang": "en", "lines": 4},
                {"q": f"សរសេរកំណាព្យខ្លីមួយអំពី{TOPICS_KM[i]} ចំនួន ៤ បន្ទាត់។", "lang": "km", "lines": 4})
    if kind == 3:
        a, b = r.randint(12, 99), r.randint(12, 99)
        return ({"q": f"Answer with the number only: what is {a} + {b}?", "lang": "en", "number": a + b, "last": True, "max_words": 2},
                {"q": f"ឆ្លើយតែលេខប៉ុណ្ណោះ៖ {km(a)} បូក {km(b)} ស្មើប៉ុន្មាន?", "lang": "km", "number": a + b, "last": True, "max_words": 3})
    return ({"q": f"Write a two-sentence welcome message for a shop that sells things for {TOPICS_EN[i]}.", "lang": "en", "max_words": 45},
            {"q": f"សរសេរសារស្វាគមន៍ពីរប្រយោគ សម្រាប់ហាងដែលលក់របស់ទាក់ទងនឹង{TOPICS_KM[i]}។", "lang": "km", "max_words": 30})


LANG_QUESTIONS = {
    "th": ["{t} คืออะไร อธิบายสั้น ๆ", "ช่วยแนะนำวิธีเรียนภาษาอังกฤษให้หน่อย", "กัมพูชามีอาหารอะไรที่มีชื่อเสียงบ้าง"],
    "zh": ["请简单介绍一下{t}。", "怎样才能睡得更好？", "柬埔寨有哪些有名的景点？"],
    "ja": ["{t}について簡単に教えてください。", "英語を早く覚えるコツは何ですか？", "カンボジアの有名な料理は何ですか？"],
    "ko": ["{t}에 대해 간단히 설명해 주세요.", "영어를 빨리 배우는 방법이 있나요?", "캄보디아의 유명한 음식은 무엇인가요?"],
    "fr": ["Explique-moi simplement : {t}.", "Comment mieux dormir ?", "Quels sont les plats célèbres du Cambodge ?"],
    "es": ["Explícame brevemente: {t}.", "¿Cómo puedo ahorrar dinero?", "¿Qué lugares famosos hay en Camboya?"],
    "vi": ["Hãy giải thích ngắn gọn về {t}.", "Làm sao để học tiếng Anh nhanh hơn?", "Campuchia có những món ăn nổi tiếng nào?"],
}


def language_task(r):
    lang = r.choice(list(LANG_QUESTIONS))
    q = r.choice(LANG_QUESTIONS[lang]).format(t=r.choice(TOPICS_EN))
    return ({"q": q, "lang": lang},)


def make_tasks(r, n):
    """n tasks in the grader's format, with a "kind" and whether to ask for steps."""
    tasks = []
    while len(tasks) < n:
        pick = r.random()
        if pick < 0.4:
            en, kmq, ans = math_task(r)
            pair = ({"q": en, "lang": "en", "number": ans, "last": True}, {"q": kmq, "lang": "km", "number": ans, "last": True})
            kind = "math"
        elif pick < 0.65:
            pair, kind = logic_task(r), "logic"
        elif pick < 0.9:
            pair, kind = instruction_task(r), "instructions"
        else:
            pair, kind = language_task(r), "languages"
        # The same problem in English and Khmer: Virgo must be as smart in both.
        tasks += [{**item, "kind": kind} for item in pair]
    return tasks[:n]


def norm(text):
    return re.sub(r"\W+", "", text.lower())[:160]


def load_teacher(base, adapter):
    import torch
    from distill_gemma import load_teacher as load_base

    tok, model = load_base(base)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).eval()
        print("Teacher:", base, "+", adapter, flush=True)
    else:
        print("Teacher:", base, flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    return tok, model


def generate(tok, model, prompts, max_new_tokens, temperature):
    import torch

    texts = [tok.apply_chat_template([{"role": "user", "content": f"{SYSTEM}\n\n{p}"}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
    return [tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def passes(item, answer):
    if not answer or len(answer) > 2500 or not all(ok for _, ok in score(item, answer)):
        return False
    if item.get("options"):  # of the choices (days, names, directions), the LAST one named must be right
        low = answer.lower()
        last = max(item["options"], key=lambda o: low.rfind(o.lower()))
        return last == item["any"][0]
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="google/gemma-3-27b-it", help="the teacher's base model")
    p.add_argument("--adapter", default="", help="the teacher's adapter folder (e.g. Bayon-1.0-27B), empty for plain Gemma")
    p.add_argument("--target", type=int, default=12000, help="verified examples to make in total (across runs)")
    p.add_argument("--out", default="chat/data/distilled_smart.jsonl")
    p.add_argument("--pairs", default="chat/dpo/smart_pairs.jsonl")
    p.add_argument("--batch", type=int, default=6, help="tasks per step (each answered twice); lower it if the GPU runs out of memory")
    p.add_argument("--hf-repo", help="private Hugging Face dataset to save both files to as they grow")
    p.add_argument("--upload-every", type=int, default=300)
    p.add_argument("--hours", type=float, default=0, help="stop (and save) after this many hours, e.g. 11 on Kaggle's 12-hour limit")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.pairs), exist_ok=True)

    # Never train on the test questions.
    blocked = set()
    for path in ("chat/eval/questions.jsonl", "chat/eval/hard.jsonl"):
        if os.path.exists(path):
            blocked |= {norm(json.loads(l)["q"]) for l in open(path, encoding="utf-8") if l.strip()}
    seen = set(blocked)
    done = 0
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            seen.add(norm(json.loads(line)["messages"][1]["content"]))
            done += 1
    pairs = sum(1 for _ in open(args.pairs, encoding="utf-8")) if os.path.exists(args.pairs) else 0
    print(f"{done} verified examples and {pairs} DPO pairs so far; target {args.target}", flush=True)
    if done >= args.target:
        return

    tok, model = load_teacher(args.teacher, args.adapter)
    rng = random.Random()  # fresh problems on every machine and run (Kaggle and your server never repeat each other)
    import time

    deadline = time.time() + args.hours * 3600 if args.hours else None
    tried = kept = 0
    uploaded = done

    def upload():
        if not args.hf_repo:
            return
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        for path in (args.out, args.pairs):
            if os.path.exists(path):
                api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path), repo_id=args.hf_repo, repo_type="dataset",
                                commit_message=f"{done} verified examples, {pairs} DPO pairs")
        print(f"☁️ Saved to {args.hf_repo}", flush=True)

    with open(args.out, "a", encoding="utf-8") as out, open(args.pairs, "a", encoding="utf-8") as pair_file:
        while done < args.target and not (deadline and time.time() > deadline):
            tasks = [t for t in make_tasks(rng, args.batch) if norm(t["q"]) not in seen]
            if not tasks:
                continue
            # Two answers per task; steps asked for in math and logic (the hint isn't saved).
            prompts = [t["q"] + (HINT["km" if t["lang"] == "km" else "en"] if t["kind"] in ("math", "logic") and not t.get("max_words") else "") for t in tasks]
            try:
                answers = generate(tok, model, prompts + prompts, 600, temperature=0.7)
            except Exception as err:  # out of GPU memory: smaller steps from now on
                if "out of memory" not in str(err).lower() or args.batch <= 1:
                    raise
                import torch

                torch.cuda.empty_cache()
                args.batch = max(1, args.batch // 2)
                print("GPU memory full: now", args.batch, "tasks per step", flush=True)
                continue
            for i, task in enumerate(tasks):
                a, b = answers[i], answers[i + len(tasks)]
                good = [x for x in (a, b) if passes(task, x)]
                bad = [x for x in (a, b) if x and not passes(task, x)]
                tried += 1
                if not good:
                    continue
                chosen = min(good, key=len) if task["kind"] == "instructions" else good[0]
                prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["q"]}]
                out.write(json.dumps({"messages": prompt + [{"role": "assistant", "content": chosen}]}, ensure_ascii=False) + "\n")
                if bad:
                    pair_file.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": bad[0], "kind": task["kind"]},
                                               ensure_ascii=False) + "\n")
                    pairs += 1
                seen.add(norm(task["q"]))
                done += 1
                kept += 1
            out.flush()
            pair_file.flush()
            print(f"{done}/{args.target} verified ({kept} of {tried} passed this run), {pairs} DPO pairs", flush=True)
            if done - uploaded >= args.upload_every:
                upload()
                uploaded = done
    upload()
    if deadline and time.time() > deadline and done < args.target:
        print(f"⏱️ Time limit: stopped at {done}/{args.target}. Run again to continue.")
    print(f"✅ Done: {done} verified examples in {args.out}, {pairs} DPO pairs in {args.pairs}")


if __name__ == "__main__":
    main()
