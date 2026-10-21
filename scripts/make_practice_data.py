"""Makes Virgo practice examples whose answers are computed, so they're always right.

    python scripts/make_practice_data.py        # writes chat/data/practice_*.jsonl (same output every run)

Math word problems, money (riel and dollars), units, time and dates, and Khmer numerals, in English
and Khmer, with step-by-step answers. Hand-written examples live in the other chat/data files.
"""
import json
import os
import random

SYS = ("You are Virgo, an AI assistant made by KSN (Virgo-1.0-Angkor). You are friendly, clear and honest. "
       "Reply in the user's language: Khmer (in Khmer script) when they write Khmer, otherwise English. "
       "Keep answers short unless asked for more. Say so when you are not sure.")
OUT = os.path.join(os.path.dirname(__file__), "..", "chat", "data")
KH_DIGITS = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
KH_MONTHS = ["មករា", "កុម្ភៈ", "មីនា", "មេសា", "ឧសភា", "មិថុនា", "កក្កដា", "សីហា", "កញ្ញា", "តុលា", "វិច្ឆិកា", "ធ្នូ"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
KH_DAYS = ["ច័ន្ទ", "អង្គារ", "ពុធ", "ព្រហស្បតិ៍", "សុក្រ", "សៅរ៍", "អាទិត្យ"]
EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def num(x, khmer=False):
    """1234.5 → "1,234.5" (Khmer digits when khmer=True); whole numbers without decimals."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    text = f"{x:,}" if isinstance(x, int) else f"{x:,.2f}".rstrip("0").rstrip(".")
    return text.translate(KH_DIGITS) if khmer else text


def usd(x, khmer=False):
    """Dollar amounts: $12.50, not $12.5 (whole dollars stay $12)."""
    x = round(float(x), 2)
    text = f"{int(x):,}" if x.is_integer() else f"{x:,.2f}"
    return text.translate(KH_DIGITS) if khmer else text


def ex(q, a):
    return {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]}


# ---------- Math word problems ----------
def math_en(r):
    out = []
    items = ["shirt", "pair of shoes", "bag of rice", "phone case", "school bag", "helmet", "water bottle", "notebook"]
    for _ in range(40):  # discounts
        price, pct, item = r.choice([12, 15, 20, 25, 30, 40, 45, 60, 80, 120]), r.choice([10, 15, 20, 25, 30, 50]), r.choice(items)
        off = price * pct / 100
        q = r.choice([f"A {item} costs ${price}. It's {pct}% off. What's the new price?",
                      f"{pct}% discount on a ${price} {item}: how much do I pay?",
                      f"How much is a ${price} {item} after a {pct}% discount?"])
        out.append(ex(q, f"Discount: {pct}% of ${usd(price)} = ${usd(off)}.\nNew price: ${usd(price)} − ${usd(off)} = **${usd(price - off)}**."))
    for _ in range(30):  # splitting a bill
        total, people = r.choice([24, 30, 36, 45, 50, 60, 72, 90, 100, 120]), r.choice([2, 3, 4, 5, 6])
        tip = r.choice([0, 10])
        if tip:
            with_tip = total * 1.1
            q = f"Dinner cost ${total}. We add a 10% tip and split it between {people} people. How much each?"
            a = f"With tip: ${usd(total)} × 1.1 = ${usd(round(with_tip, 2))}.\nEach person: ${usd(round(with_tip, 2))} ÷ {people} = **${usd(round(with_tip / people, 2))}**."
        else:
            q = r.choice([f"The bill is ${total} for {people} people. How much does each pay?", f"Split ${total} equally among {people} friends."])
            a = f"${usd(total)} ÷ {people} = **${usd(round(total / people, 2))}** each."
        out.append(ex(q, a))
    for _ in range(30):  # saving
        income, spend = r.choice([250, 300, 350, 400, 500, 600, 800]), r.choice([150, 200, 250, 280, 320])
        if spend >= income:
            spend = income - 50
        months = r.choice([6, 12, 24])
        q = f"I earn ${income} a month and spend ${spend}. How much can I save in {months} months?"
        out.append(ex(q, f"Each month: ${usd(income)} − ${usd(spend)} = ${usd(income - spend)}.\nIn {months} months: ${usd(income - spend)} × {months} = **${usd((income - spend) * months)}**."))
    for _ in range(25):  # speed, distance, time
        speed, hours = r.choice([30, 40, 50, 60, 80]), r.choice([1.5, 2, 2.5, 3, 4, 5])
        q = r.choice([f"A bus drives at {speed} km/h for {num(hours)} hours. How far does it go?",
                      f"If I ride at {speed} km/h, how far do I get in {num(hours)} hours?"])
        out.append(ex(q, f"Distance = speed × time = {speed} × {num(hours)} = **{num(speed * hours)} km**."))
    for _ in range(20):
        dist, speed = r.choice([60, 90, 120, 150, 180, 240, 300, 320]), r.choice([40, 50, 60, 80])
        hours = dist / speed
        h, m = int(hours), round((hours - int(hours)) * 60)
        when = f"{h} hour{'s' if h != 1 else ''}" + (f" {m} minutes" if m else "")
        q = f"How long does it take to travel {dist} km at {speed} km/h?"
        out.append(ex(q, f"Time = distance ÷ speed = {dist} ÷ {speed} = {num(round(hours, 2))} hours, which is **{when}**."))
    for _ in range(20):  # area
        w, l = r.choice([3, 4, 5, 6, 8, 10, 12]), r.choice([4, 5, 6, 8, 10, 15, 20])
        q = r.choice([f"A room is {w} m by {l} m. What's its area?", f"What's the area of a {w} m × {l} m garden?"])
        a = f"Area = {w} × {l} = **{w * l} m²**."
        if r.random() < 0.5:
            tile = r.choice([5, 8, 10, 12])
            q += f" Tiles cost ${tile} per m². How much for the whole floor?"
            a = f"Area = {w} × {l} = {w * l} m².\nCost = {w * l} × ${tile} = **${usd(w * l * tile)}**."
        out.append(ex(q, a))
    for _ in range(20):  # percentages
        part, whole = r.choice([12, 18, 24, 30, 45, 60, 75]), r.choice([60, 80, 100, 120, 150, 200, 300])
        if part > whole:
            part, whole = whole // 4, whole
        pct = part / whole * 100
        q = r.choice([f"What percent is {part} of {whole}?", f"I got {part} out of {whole} on a test. What's my percentage?"])
        out.append(ex(q, f"{part} ÷ {whole} × 100 = **{num(round(pct, 1))}%**."))
    for _ in range(15):  # unit price comparison
        a_size, a_price, b_size, b_price = r.choice([500, 750, 1000]), r.choice([1.2, 1.5, 2, 2.5]), r.choice([1500, 2000]), r.choice([3, 3.5, 4, 5])
        pa, pb = a_price / a_size * 1000, b_price / b_size * 1000
        better = "the small one" if pa < pb else "the big one" if pb < pa else "neither: they cost the same"
        q = f"Which is cheaper: {a_size} ml for ${usd(a_price)} or {b_size} ml for ${usd(b_price)}?"
        out.append(ex(q, f"Price per litre:\n- {a_size} ml: ${usd(a_price)} ÷ {num(a_size / 1000)} L = ${usd(round(pa, 2))}/L\n- {b_size} ml: ${usd(b_price)} ÷ {num(b_size / 1000)} L = ${usd(round(pb, 2))}/L\n\n**{better[0].upper() + better[1:]}** is the better deal."))
    return out


def math_km(r):
    out = []
    items = ["អាវ", "ស្បែកជើង", "កាបូប", "មួកសុវត្ថិភាព", "សៀវភៅ", "កង់", "ទូរស័ព្ទ"]
    for _ in range(40):
        price, pct, item, kd = r.choice([10, 15, 20, 25, 40, 50, 80, 100]), r.choice([10, 20, 25, 30, 50]), r.choice(items), r.random() < 0.5
        off = price * pct / 100
        q = r.choice([f"{item}តម្លៃ ${usd(price, kd)} បញ្ចុះតម្លៃ {num(pct, kd)}%។ តើត្រូវបង់ប៉ុន្មាន?",
                      f"បញ្ចុះ {num(pct, kd)}% លើ{item} ${usd(price, kd)} នៅសល់ប៉ុន្មាន?"])
        out.append(ex(q, f"ចំនួនបញ្ចុះ៖ {num(pct, kd)}% នៃ ${usd(price, kd)} = ${usd(off, kd)}។\nតម្លៃថ្មី៖ ${usd(price, kd)} − ${usd(off, kd)} = **${usd(price - off, kd)}**។"))
    for _ in range(30):
        total, people, kd = r.choice([20000, 30000, 40000, 60000, 80000, 100000, 120000]), r.choice([2, 3, 4, 5]), r.random() < 0.5
        q = r.choice([f"ញ៉ាំបាយអស់ {num(total, kd)} រៀល ចែកគ្នា {num(people, kd)} នាក់។ ម្នាក់ប៉ុន្មាន?",
                      f"ថ្លៃតុកតុក {num(total, kd)} រៀល មាន {num(people, kd)} នាក់ចែកគ្នា។ ម្នាក់ត្រូវបង់ប៉ុន្មាន?"])
        each = total / people
        each = round(each / 100) * 100 if not float(each).is_integer() else each
        note = "" if total % people == 0 else " (ប្រហែល)"
        out.append(ex(q, f"{num(total, kd)} ÷ {num(people, kd)} = **{num(each, kd)} រៀល**{note} ក្នុងម្នាក់។"))
    for _ in range(30):
        income, spend, months, kd = r.choice([300, 400, 500, 600]), r.choice([200, 250, 300, 350]), r.choice([6, 12]), r.random() < 0.5
        if spend >= income:
            spend = income - 100
        q = f"ខ្ញុំរកបាន ${usd(income, kd)} ក្នុងមួយខែ ចាយ ${usd(spend, kd)}។ តើ {num(months, kd)} ខែ សន្សំបានប៉ុន្មាន?"
        out.append(ex(q, f"មួយខែ៖ ${usd(income, kd)} − ${usd(spend, kd)} = ${usd(income - spend, kd)}។\n{num(months, kd)} ខែ៖ ${usd(income - spend, kd)} × {num(months, kd)} = **${usd((income - spend) * months, kd)}**។"))
    for _ in range(25):
        speed, hours, kd = r.choice([30, 40, 50, 60]), r.choice([2, 3, 4, 5]), r.random() < 0.5
        q = f"ឡានបើក {num(speed, kd)} គីឡូម៉ែត្រក្នុងមួយម៉ោង រយៈពេល {num(hours, kd)} ម៉ោង។ តើបានចម្ងាយប៉ុន្មាន?"
        out.append(ex(q, f"ចម្ងាយ = ល្បឿន × ពេលវេលា = {num(speed, kd)} × {num(hours, kd)} = **{num(speed * hours, kd)} គីឡូម៉ែត្រ**។"))
    for _ in range(20):
        w, l, kd = r.choice([4, 5, 6, 8, 10]), r.choice([5, 8, 10, 12, 20]), r.random() < 0.5
        q = f"ដីទទឹង {num(w, kd)} ម៉ែត្រ បណ្តោយ {num(l, kd)} ម៉ែត្រ។ តើមានផ្ទៃក្រឡាប៉ុន្មាន?"
        out.append(ex(q, f"ផ្ទៃក្រឡា = ទទឹង × បណ្តោយ = {num(w, kd)} × {num(l, kd)} = **{num(w * l, kd)} ម៉ែត្រការ៉េ**។"))
    for _ in range(20):
        part, whole, kd = r.choice([15, 20, 30, 40, 45]), r.choice([50, 60, 100, 150]), r.random() < 0.5
        q = f"ខ្ញុំប្រឡងបាន {num(part, kd)} ពិន្ទុ ក្នុងចំណោម {num(whole, kd)}។ តើប៉ុន្មានភាគរយ?"
        out.append(ex(q, f"{num(part, kd)} ÷ {num(whole, kd)} × 100 = **{num(round(part / whole * 100, 1), kd)}%**។"))
    return out


# ---------- Money: riel and dollars ----------
def money(r):
    out = []
    note_en = "(at about 4,000 riel per dollar; the real rate changes a little each day)"
    note_km = "(គិតប្រហែល ៤,០០០ រៀល ក្នុងមួយដុល្លារ អត្រាពិតប្រែប្រួលបន្តិចរាល់ថ្ងៃ)"
    for _ in range(35):
        dollars = r.choice([1, 2, 2.5, 5, 7.5, 10, 12, 15, 20, 25, 50, 100])
        q = r.choice([f"How many riel is ${usd(dollars)}?", f"Convert ${usd(dollars)} to riel.", f"${usd(dollars)} in Khmer riel?"])
        out.append(ex(q, f"${usd(dollars)} × 4,000 = **{num(int(dollars * 4000))} riel** {note_en}."))
    for _ in range(35):
        riel = r.choice([2000, 5000, 8000, 10000, 15000, 20000, 40000, 50000, 100000])
        kd = r.random() < 0.5
        q = r.choice([f"{num(riel, kd)} រៀល ស្មើនឹងប៉ុន្មានដុល្លារ?", f"ប្តូរ {num(riel, kd)} រៀល ទៅជាដុល្លារ"])
        out.append(ex(q, f"{num(riel, kd)} ÷ {num(4000, kd)} = **${usd(riel / 4000, kd)}** {note_km}។"))
    for _ in range(20):
        dollars, riel = r.choice([1, 2, 5, 10]), r.choice([1000, 2000, 5000, 10000])
        price_riel = r.choice([6000, 7500, 12000, 18000, 22000])
        paid = dollars * 4000 + riel
        if paid < price_riel:
            paid, dollars, riel = price_riel + 2000, (price_riel + 2000) // 4000, (price_riel + 2000) % 4000
        change = paid - price_riel
        q = f"Something costs {num(price_riel)} riel. I pay ${dollars} and {num(riel)} riel. How much change do I get?"
        out.append(ex(q, f"You paid ${dollars} × 4,000 + {num(riel)} = {num(paid)} riel.\nChange: {num(paid)} − {num(price_riel)} = **{num(change)} riel** {note_en}."))
    return out


# ---------- Units ----------
def units(r):
    out = []
    for _ in range(15):
        c = r.choice([-5, 0, 10, 20, 25, 28, 30, 32, 35, 37, 40])
        f = c * 9 / 5 + 32
        out.append(ex(r.choice([f"What is {c}°C in Fahrenheit?", f"Convert {c} Celsius to Fahrenheit."]), f"°F = °C × 9/5 + 32 = {c} × 1.8 + 32 = **{num(round(f, 1))}°F**."))
    for _ in range(12):
        f = r.choice([32, 50, 68, 77, 86, 95, 100, 104])
        c = (f - 32) * 5 / 9
        out.append(ex(f"What is {f}°F in Celsius?", f"°C = (°F − 32) × 5/9 = ({f} − 32) × 5/9 = **{num(round(c, 1))}°C**."))
    for _ in range(15):
        km = r.choice([1, 5, 10, 21, 42, 100, 160, 300])
        out.append(ex(r.choice([f"How many miles is {km} km?", f"Convert {km} kilometres to miles."]), f"{km} km × 0.621 = **{num(round(km * 0.621371, 1))} miles**."))
    for _ in range(12):
        kg = r.choice([1, 2, 5, 10, 25, 50, 60, 75])
        out.append(ex(f"How many pounds is {kg} kg?", f"{kg} kg × 2.205 = **{num(round(kg * 2.20462, 1))} lb**."))
    for _ in range(15):
        kd, m = r.random() < 0.5, r.choice([1.5, 2, 3.5, 5, 10, 150, 1200])
        cm = m * 100
        out.append(ex(f"{num(m, kd)} ម៉ែត្រ ស្មើនឹងប៉ុន្មានសង់ទីម៉ែត្រ?", f"{num(m, kd)} × {num(100, kd)} = **{num(cm, kd)} សង់ទីម៉ែត្រ**។"))
    for _ in range(12):
        kd, g = r.random() < 0.5, r.choice([250, 500, 750, 1500, 2500, 3000])
        out.append(ex(f"{num(g, kd)} ក្រាម ស្មើនឹងប៉ុន្មានគីឡូក្រាម?", f"{num(g, kd)} ÷ {num(1000, kd)} = **{num(g / 1000, kd)} គីឡូក្រាម**។"))
    return out


# ---------- Time and dates ----------
def time_dates(r):
    out = []
    for _ in range(20):
        h, m, add_h, add_m = r.randint(6, 20), r.choice([0, 15, 30, 45]), r.randint(0, 3), r.choice([15, 30, 45, 50])
        total = h * 60 + m + add_h * 60 + add_m
        eh, em = divmod(total % (24 * 60), 60)
        fmt = lambda hh, mm: f"{hh:02d}:{mm:02d}"
        dur = (f"{add_h} hour{'s' if add_h != 1 else ''} " if add_h else "") + f"{add_m} minutes"
        out.append(ex(f"If I start at {fmt(h, m)} and it takes {dur.strip()}, when do I finish?", f"{fmt(h, m)} + {dur.strip()} = **{fmt(eh, em)}**."))
    for _ in range(20):
        start, days = r.randint(0, 6), r.choice([2, 3, 5, 10, 14, 30, 100])
        kd = r.random() < 0.5
        end = (start + days) % 7
        out.append(ex(f"ថ្ងៃនេះថ្ងៃ{KH_DAYS[start]}។ {num(days, kd)} ថ្ងៃទៀតជាថ្ងៃអ្វី?",
                      f"{num(days, kd)} ÷ {num(7, kd)} សល់ {num(days % 7, kd)}។ ថ្ងៃ{KH_DAYS[start]} + {num(days % 7, kd)} ថ្ងៃ = **ថ្ងៃ{KH_DAYS[end]}**។"))
    for _ in range(15):
        start, days = r.randint(0, 6), r.choice([3, 4, 10, 15, 20, 45])
        end = (start + days) % 7
        out.append(ex(f"Today is {EN_DAYS[start]}. What day is it in {days} days?", f"{days} days is {days // 7} week{'s' if days // 7 != 1 else ''} and {days % 7} day{'s' if days % 7 != 1 else ''}. {EN_DAYS[start]} + {days % 7} = **{EN_DAYS[end]}**."))
    for i in range(12):
        out.append(ex(f"ខែទី {num(i + 1, True)} ជាខែអ្វី?", f"ខែទី {num(i + 1, True)} គឺ **ខែ{KH_MONTHS[i]}** ({EN_MONTHS[i]})។"))
        out.append(ex(f"How do you say {EN_MONTHS[i]} in Khmer?", f"{EN_MONTHS[i]} is **ខែ{KH_MONTHS[i]}** (month {i + 1})."))
    for i in range(7):
        out.append(ex(f"How do you say {EN_DAYS[i]} in Khmer?", f"{EN_DAYS[i]} is **ថ្ងៃ{KH_DAYS[i]}**."))
    for _ in range(12):
        born, now = r.randint(1960, 2015), 2026
        kd = r.random() < 0.5
        out.append(ex(f"ខ្ញុំកើតឆ្នាំ {num(born, kd)}។ ឆ្នាំ {num(now, kd)} ខ្ញុំអាយុប៉ុន្មាន?",
                      f"{num(now, kd)} − {num(born, kd)} = **{num(now - born, kd)} ឆ្នាំ** (ឬ {num(now - born - 1, kd)} ឆ្នាំ បើមិនទាន់ដល់ថ្ងៃខួបកំណើតឆ្នាំនេះ)។"))
    return out


# ---------- Khmer numerals ----------
def numerals(r):
    out = []
    for _ in range(25):
        n = r.choice([r.randint(0, 99), r.randint(100, 9999), r.randint(10000, 999999), r.choice([2025, 2026, 1975, 1953])])
        out.append(ex(r.choice([f"Write {n} in Khmer numerals.", f"How do you write {n} with Khmer digits?"]), f"{n} in Khmer numerals is **{str(n).translate(KH_DIGITS)}**."))
    for _ in range(25):
        n = r.choice([r.randint(0, 99), r.randint(100, 99999)])
        out.append(ex(f"{str(n).translate(KH_DIGITS)} ជាលេខអារ៉ាប់គឺប៉ុន្មាន?", f"{str(n).translate(KH_DIGITS)} ជាលេខអារ៉ាប់គឺ **{n}**។"))
    out.append(ex("What are the Khmer digits 0 to 9?", "០ ១ ២ ៣ ៤ ៥ ៦ ៧ ៨ ៩ (0 1 2 3 4 5 6 7 8 9)."))
    out.append(ex("លេខខ្មែរពី ០ ដល់ ៩ សរសេរយ៉ាងដូចម្តេច?", "០ (សូន្យ) ១ (មួយ) ២ (ពីរ) ៣ (បី) ៤ (បួន) ៥ (ប្រាំ) ៦ (ប្រាំមួយ) ៧ (ប្រាំពីរ) ៨ (ប្រាំបី) ៩ (ប្រាំបួន)។"))
    return out


# ---------- Khmer number words ----------
ONES = ["សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន"]
TENS = {1: "ដប់", 2: "ម្ភៃ", 3: "សាមសិប", 4: "សែសិប", 5: "ហាសិប", 6: "ហុកសិប", 7: "ចិតសិប", 8: "ប៉ែតសិប", 9: "កៅសិប"}
BIG_UNITS = [(1_000_000, "លាន"), (100_000, "សែន"), (10_000, "ម៉ឺន"), (1_000, "ពាន់"), (100, "រយ")]


def khmer_words(n):
    """2026 → ពីរពាន់ម្ភៃប្រាំមួយ (the everyday way: ម៉ឺន and សែន for 10,000 and 100,000)."""
    if n < 10:
        return ONES[n]
    words = ""
    for value, name in BIG_UNITS:
        if n >= value:
            count, n = divmod(n, value)
            words += (khmer_words(count) if value == 1_000_000 else ONES[count]) + name
    if n >= 10:
        words += TENS[n // 10]
        n %= 10
    if n:
        words += ONES[n]
    return words


def number_words(r):
    out = []
    picks = sorted({r.randint(11, 99) for _ in range(30)} | {r.randint(100, 999) for _ in range(25)} | {r.randint(1000, 9999) for _ in range(20)}
                   | {r.choice([15000, 25000, 50000, 120000, 250000, 1500000, 2026, 1953, 2000, 100000, 1000000]) for _ in range(15)})
    for n in picks:
        kd = r.random() < 0.5
        shown = num(n, kd)
        if r.random() < 0.5:
            out.append(ex(f"{shown} អានជាភាសាខ្មែរថាម៉េច?", f"{shown} អានថា **{khmer_words(n)}**។"))
        else:
            out.append(ex(f"How do you say {n:,} in Khmer?", f"{n:,} is **{khmer_words(n)}** (written {num(n, True)})."))
    return out


# ---------- Fractions and percentages of money ----------
def fractions(r):
    out = []
    for _ in range(30):
        a, b, c, d = r.randint(1, 5), r.choice([2, 3, 4, 5, 6, 8]), r.randint(1, 5), r.choice([2, 3, 4, 5, 6, 8])
        from fractions import Fraction
        x, y = Fraction(a, b), Fraction(c, d)
        total = x + y
        show = lambda f: f"{f.numerator}/{f.denominator}" if f.denominator != 1 else f"{f.numerator}"
        lcd = b * d // __import__("math").gcd(b, d)
        top = a * lcd // b + c * lcd // d
        steps = "" if b == d else f"Common denominator: {lcd}.\n{a}/{b} = {a * lcd // b}/{lcd}, {c}/{d} = {c * lcd // d}/{lcd}.\n"
        result = f"**{show(total)}**" if f"{top}/{lcd}" == show(total) else f"{top}/{lcd} = **{show(total)}**"
        out.append(ex(f"What is {a}/{b} + {c}/{d}?", f"{steps}Sum: {result}."))
    for _ in range(25):
        riel, pct, kd = r.choice([20000, 40000, 60000, 100000, 200000, 500000]), r.choice([5, 10, 15, 20, 25]), r.random() < 0.5
        part = riel * pct // 100
        out.append(ex(f"{num(pct, kd)}% នៃ {num(riel, kd)} រៀល ស្មើនឹងប៉ុន្មាន?", f"{num(riel, kd)} × {num(pct, kd)} ÷ {num(100, kd)} = **{num(part, kd)} រៀល**។"))
    return out


def main():
    r = random.Random(20260930)
    groups = {"practice_math_en": math_en(r), "practice_math_km": math_km(r), "practice_money": money(r),
              "practice_units": units(r), "practice_time": time_dates(r), "practice_numerals": numerals(r),
              "practice_number_words": number_words(r), "practice_fractions": fractions(r)}
    total = 0
    for name, rows in groups.items():
        seen, unique = set(), []
        for row in rows:  # drop repeated questions
            q = row["messages"][1]["content"]
            if q not in seen:
                seen.add(q)
                unique.append(row)
        with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in unique:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(unique)}")
        total += len(unique)
    print("total:", total)


if __name__ == "__main__":
    main()
