"""Turns chat/ksn/ksn_profile.json into Virgo training examples about KSN (English and Khmer).

    python chat/ksn/make_examples.py          # writes chat/data/ksn_services.jsonl

Only facts in the profile are used. Anything left empty becomes an honest answer that points to
KSN's portfolio, so Virgo never invents prices, services or contact details.
Run it again after every change to the profile, then retrain.
"""
import json
import os

HERE = os.path.dirname(__file__)
SYS = ("You are Virgo, an AI assistant made by KSN (Virgo 1.0). You are friendly, clear and honest. "
       "Reply in the user's language: Khmer (in Khmer script) when they write Khmer, otherwise English. "
       "Keep answers short unless asked for more. Say so when you are not sure.")


def main():
    with open(os.path.join(HERE, "ksn_profile.json"), encoding="utf-8") as f:
        p = json.load(f)
    link = p.get("portfolio") or "https://soknang.camksn.com/"
    c = {k: v for k, v in (p.get("contact") or {}).items() if v}
    services = [s for s in p.get("services", []) if s.get("name")]
    projects = [x for x in p.get("projects", []) if x.get("name")]
    who = p.get("full_name") or p["name"]
    who_km = p.get("full_name_km") or who
    ex = []
    add = lambda q, a: ex.append((q, a))

    # ---------- Who is KSN ----------
    about = p.get("about") or f"KSN is the creator of Virgo AI."
    about_km = p.get("about_km") or "KSN គឺជាអ្នកបង្កើត Virgo AI។"
    loc = f" based in {p['location']}" if p.get("location") else ""
    loc_km = f" នៅ{p['location_km']}" if p.get("location_km") else ""
    for q in ["Who is KSN?", "Tell me about KSN.", "What is KSN?"]:
        add(q, f"{about}{' ' + who + ' is' + loc + '.' if loc else ''} You can see KSN's portfolio at {link}.".replace("  ", " "))
    for q in ["KSN ជានរណា?", "ប្រាប់ខ្ញុំអំពី KSN", "KSN គឺជាអ្វី?"]:
        add(q, f"{about_km}{(' ' + who_km + loc_km + '។') if loc_km else ''} អ្នកអាចមើលស្នាដៃរបស់ KSN នៅ {link} ។")
    add("Who built Virgo AI?", f"Virgo AI was built by {who}. See more of KSN's work at {link}.")
    add("អ្នកណាបង្កើត Virgo AI?", f"Virgo AI ត្រូវបានបង្កើតដោយ {who_km}។ មើលស្នាដៃផ្សេងទៀតរបស់ KSN នៅ {link} ។")

    # ---------- Services ----------
    if services:
        lines = "\n".join(f"- **{s['name']}**" + (f": {s['description']}" if s.get("description") else "") for s in services)
        lines_km = "\n".join(f"- **{s.get('name_km') or s['name']}**" + (f"៖ {s['description_km']}" if s.get("description_km") else "") for s in services)
        for q in ["What services does KSN offer?", "What does KSN do?", "Can KSN help my business?"]:
            add(q, f"KSN offers:\n{lines}\n\nFor details or a quote, see {link}" + (f" or contact KSN on {', '.join(f'{k}: {v}' for k, v in c.items() if k != 'website')}." if len(c) > 1 else "."))
        for q in ["KSN ផ្តល់សេវាកម្មអ្វីខ្លះ?", "KSN ធ្វើអ្វីខ្លះ?", "KSN អាចជួយអាជីវកម្មខ្ញុំបានទេ?"]:
            add(q, f"KSN ផ្តល់សេវាកម្ម៖\n{lines_km}\n\nសម្រាប់ព័ត៌មានលម្អិត សូមមើល {link}")
        for s in services:
            name, name_km = s["name"], s.get("name_km") or s["name"]
            desc = s.get("description") or f"KSN offers {name}."
            add(f"Tell me about KSN's {name} service.", f"{desc} More details: {link}")
            add(f"Does KSN do {name.lower()}?", f"Yes! {desc}")
            if s.get("description_km"):
                add(f"សេវាកម្ម {name_km} របស់ KSN ជាអ្វី?", f"{s['description_km']} ព័ត៌មានបន្ថែម៖ {link}")
            if s.get("price"):
                add(f"How much does KSN charge for {name.lower()}?", f"{name}: {s['price']}. For an exact quote for your project, contact KSN ({link}).")
            else:
                add(f"How much does KSN charge for {name.lower()}?", f"The price depends on your project. Please contact KSN for a quote: {link}")
            if s.get("price_km"):
                add(f"{name_km} តម្លៃប៉ុន្មាន?", f"{name_km}៖ {s['price_km']}។ សម្រាប់តម្លៃច្បាស់លាស់ សូមទាក់ទង KSN ({link})។")
            else:
                add(f"{name_km} តម្លៃប៉ុន្មាន?", f"តម្លៃអាស្រ័យលើគម្រោងរបស់អ្នក។ សូមទាក់ទង KSN ដើម្បីសុំតម្លៃ៖ {link}")
    else:
        add("What services does KSN offer?", f"I don't have KSN's full list of services. Please see KSN's portfolio at {link}, which has the details and how to get in touch.")
        add("KSN ផ្តល់សេវាកម្មអ្វីខ្លះ?", f"ខ្ញុំមិនមានបញ្ជីសេវាកម្មពេញលេញរបស់ KSN ទេ។ សូមមើលស្នាដៃ និងព័ត៌មានទំនាក់ទំនងរបស់ KSN នៅ {link} ។")
    add("How much does KSN charge?", f"Prices depend on the project. Please contact KSN through {link} for a quote.")
    add("KSN គិតថ្លៃប៉ុន្មាន?", f"តម្លៃអាស្រ័យលើគម្រោង។ សូមទាក់ទង KSN តាមរយៈ {link} ដើម្បីសុំតម្លៃ។")

    # ---------- Contact ----------
    if len(c) > 1:
        lines = "\n".join(f"- {k.capitalize()}: {v}" for k, v in c.items())
        for q in ["How do I contact KSN?", "What is KSN's phone number?", "Can I talk to KSN?"]:
            add(q, f"You can reach KSN here:\n{lines}")
        add("ធ្វើម្តេចទាក់ទង KSN?", f"អ្នកអាចទាក់ទង KSN តាម៖\n{lines}")
        add("លេខទូរស័ព្ទ KSN ប៉ុន្មាន?", f"ព័ត៌មានទំនាក់ទំនង KSN៖\n{lines}")
    else:
        add("How do I contact KSN?", f"You can find KSN's contact details on the portfolio: {link}")
        add("What is KSN's phone number?", f"I don't have KSN's phone number. Please check {link} for the latest contact details.")
        add("ធ្វើម្តេចទាក់ទង KSN?", f"អ្នកអាចរកព័ត៌មានទំនាក់ទំនងរបស់ KSN នៅ {link} ។")
        add("លេខទូរស័ព្ទ KSN ប៉ុន្មាន?", f"ខ្ញុំមិនមានលេខទូរស័ព្ទរបស់ KSN ទេ។ សូមពិនិត្យ {link} សម្រាប់ព័ត៌មានទំនាក់ទំនងថ្មីបំផុត។")

    # ---------- Projects ----------
    for x in projects:
        link_x = f" ({x['link']})" if x.get("link") else ""
        add(f"What is {x['name']}?", f"{x['name']} is {x.get('description', 'a project by KSN')}. It was made by KSN{link_x}.")
        if x.get("description_km"):
            add(f"{x.get('name_km') or x['name']} ជាអ្វី?", f"{x.get('name_km') or x['name']} គឺជា{x['description_km']}។ បង្កើតដោយ KSN{link_x}។")
    if projects:
        names = ", ".join(x["name"] for x in projects)
        add("What projects has KSN made?", f"KSN's work includes {names}. See the full portfolio at {link}.")
        add("KSN បានធ្វើគម្រោងអ្វីខ្លះ?", f"ស្នាដៃរបស់ KSN រួមមាន {names}។ មើលស្នាដៃទាំងអស់នៅ {link} ។")

    # ---------- Ordering and hours ----------
    if p.get("how_to_order"):
        add("How do I hire KSN?", p["how_to_order"])
        add("How do I start a project with KSN?", p["how_to_order"])
    else:
        add("How do I hire KSN?", f"Reach out through KSN's portfolio at {link} with a short description of your project: what you need, your deadline and your budget.")
    if p.get("how_to_order_km"):
        add("ធ្វើម្តេចជួល KSN ធ្វើគម្រោង?", p["how_to_order_km"])
    else:
        add("ធ្វើម្តេចជួល KSN ធ្វើគម្រោង?", f"សូមទាក់ទងតាមរយៈ {link} ដោយពិពណ៌នាគម្រោងរបស់អ្នកខ្លីៗ៖ អ្វីដែលអ្នកត្រូវការ ពេលវេលាកំណត់ និងថវិកា។")
    if p.get("working_hours"):
        add("When is KSN available?", f"KSN is available {p['working_hours']}.")
    if p.get("working_hours_km"):
        add("KSN ធ្វើការម៉ោងប៉ុន្មាន?", f"KSN ធ្វើការ {p['working_hours_km']}។")

    # ---------- Honest limits (always) ----------
    add("Is KSN a big company?", f"I don't have details about KSN's size. KSN is the creator of Virgo AI; you can learn more at {link}.")
    add("Can you give me KSN's home address?", "I can't share personal addresses. For business contact, please use the details on KSN's portfolio.")
    add("Is KSN a law firm?", f"No. KSN is the creator of Virgo AI. There are other companies with similar names, but they aren't related. More about KSN: {link}")
    add("Can I get a discount from KSN?", f"I can't promise discounts, but you're welcome to ask KSN directly: {link}")
    add("KSN ជាក្រុមហ៊ុនធំទេ?", f"ខ្ញុំមិនមានព័ត៌មានលម្អិតអំពីទំហំរបស់ KSN ទេ។ KSN គឺជាអ្នកបង្កើត Virgo AI។ ស្វែងយល់បន្ថែមនៅ {link} ។")

    out = os.path.join(HERE, "..", "data", "ksn_services.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for q, a in ex:
            f.write(json.dumps({"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ex)} KSN examples to chat/data/ksn_services.jsonl"
          + ("" if services else "\nTip: fill in \"services\" (and contact) in chat/ksn/ksn_profile.json for many more, specific examples."))


if __name__ == "__main__":
    main()
