# Virgo 1.0 chat training data

One conversation per line: `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}`.
`chat/train.py` reads every `.jsonl` file here. Run `python scripts/check_data.py` after editing.

| File | What it teaches |
|---|---|
| `virgo_chat.jsonl` | the starter set: who Virgo is, basic help, English and Khmer |
| `identity.jsonl` | Virgo's name, maker (KSN), version, what it can and can't do, friendly small talk |
| `everyday_help.jsonl` | writing, explaining, math, money, tech, Cambodia facts, plus safe and honest answers (no live data without Search, no hacking, see a doctor) |
| `khmer.jsonl` | the same kinds of help in natural Khmer |
| `conversations.jsonl` | multi-turn chats: remembering names, follow-ups, switching to Khmer |

Tips for adding more:
- Write answers the way Virgo should talk: short, warm, clear.
- Keep facts that don't change (history, how-to). For news, prices or rates, teach Virgo to suggest Search.
- Have a Khmer speaker check Khmer answers; natural wording matters more than quantity.
- Several hundred to a few thousand good examples is a strong goal.
