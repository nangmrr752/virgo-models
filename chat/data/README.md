# Virgo 1.0 chat training data

One conversation per line: `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}`.
`chat/train.py` reads every `.jsonl` file here. Run `python scripts/check_data.py` after editing.

| File | What it teaches |
|---|---|
| `virgo_chat.jsonl` | the starter set: who Virgo is, basic help, English and Khmer |
| `identity.jsonl` | Virgo's name, maker (KSN), version, what it can and can't do, friendly small talk |
| `everyday_help.jsonl` | writing, explaining, math, money, tech, Cambodia facts, plus safe and honest answers (no live data without Search, no hacking, see a doctor) |
| `khmer.jsonl` | the same kinds of help in natural Khmer |
| `khmer_more.jsonl` | more Khmer: daily life, Khmer food and recipes, festivals and places, study, jobs and customers, phone and scam safety, polite speech, Khmer–English mixed messages, multi-turn chats |
| `customer_support.jsonl` | helping people use Virgo AI (voice, Search, images, files, Word download, Memory, accounts) and fix common problems, plus general customer service: orders, refunds, complaints, replying to customers, handing off to a human; English and Khmer |
| `answer_style.jsonl` | how a great assistant answers: short when simple, structured (lists, tables, headings) when complex, step-by-step math, polished rewriting, honest about what it can't know, asks when the request is unclear; English and Khmer |
| `conversations.jsonl` | multi-turn chats: remembering names, follow-ups, switching to Khmer |

Tips for adding more:
- Write answers the way Virgo should talk: short, warm, clear.
- Keep facts that don't change (history, how-to). For news, prices or rates, teach Virgo to suggest Search.
- Have a Khmer speaker check Khmer answers; natural wording matters more than quantity.
- Several hundred to a few thousand good examples is a strong goal.
