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
| `coding.jsonl` | beginner programming help: Python, JavaScript, CSS, SQL, Git, reading error messages; English and Khmer |
| `knowledge.jsonl` | science and history explained simply (sky, seasons, vaccines, DNA, inflation, the Khmer Empire) |
| `money_business.jsonl` | pricing, profit, getting customers, business plans, emergency funds |
| `travel_cambodia.jsonl` | Siem Reap, Kampot, Kep, best seasons, what to pack, temple etiquette, greeting with a sampeah |
| `reasoning.jsonl` | logic and trick questions answered carefully, step by step |
| `translation.jsonl` | useful everyday phrases, English ↔ Khmer |
| `health_wellbeing.jsonl` | water, stress, dengue signs, mosquitoes, sleep, exercise; safe advice that points to a doctor when needed |
| `homework.jsonl` | grammar, algebra, area, fractions, essay outlines; helps students learn instead of doing the work for them |
| `farming.jsonl` | home vegetable gardens, compost, yellow leaves, natural pest control |
| `career.jsonl` | asking for a raise, productivity, CVs, public speaking |
| `support_feelings.jsonl` | loneliness, failure, conflict, and crisis messages answered with care and pointers to real help |
| `text_tasks.jsonl` | summarize, extract, turn into JSON or lists, sort, sentiment, titles, spelling fixes |
| `ksn_services.jsonl` | about KSN: who KSN is, Virgo AI, services, prices, contact, how to hire, and honest "I don't know, see the portfolio" answers. **Generated:** edit `chat/ksn/ksn_profile.json`, then run `python chat/ksn/make_examples.py` |
| `family.jsonl` | reading with kids, tantrums, screen time, family activities |
| `formal_khmer.jsonl` | formal Khmer letters: leave requests, thank-you letters, notices, New Year wishes, meeting requests |
| `culture_religion.jsonl` | Buddhism basics, pagoda etiquette, Pchum Ben, the sampeah, explained respectfully |
| `getting_around.jsonl` | tuk-tuks and ride apps, bargaining, shopping phrases, night safety |
| `environment.jsonl` | plastic, recycling, climate change |
| `sports.jsonl` | offside rule, Bokator, Kun Khmer |
| `conversations_long.jsonl` | long 3–4 turn conversations: planning a business, a Siem Reap trip, an electricity bill, a child's birthday party |
| `recipes.jsonl` | fish amok, rice without a cooker, samlor machu, quick egg dishes |
| `english_lessons.jsonl` | English for Khmer speakers: a/an, tenses, did, office words, corrections, roleplay |
| `cambodia_history_geo.jsonl` | independence (9 Nov 1953), 25 provinces, the Tonle Sap's reverse flow, Angkor, neighbors |
| `how_to_tech.jsonl` | screenshots, freeing storage, Word to PDF, two-step verification |
| `social_posts.jsonl` | Facebook and TikTok posts, captions, holiday notices |
| `word_problems.jsonl` | everyday math with money, riel, rice and travel time |
| `careful_requests.jsonl` | fake notes, exam cheating, fake news, weapons, private info: declined kindly with a helpful alternative |
| `targeted_fixes.jsonl` | added after the first score: live info → suggest Search, translating into Khmer, the Download as Word button and paperclip, "Who is KSN?", percentages |
| `conversations.jsonl` | multi-turn chats: remembering names, follow-ups, switching to Khmer |
| `practice_*.jsonl` | **Generated, answers computed** (`python scripts/make_practice_data.py`): math word problems, money and riel ↔ dollar, units, time and dates, Khmer numerals, Khmer number words, fractions; English and Khmer |
| `khmer_daily_life.jsonl`, `khmer_everyday_more.jsonl` | everyday life in Khmer: home, cooking, sleep, money, moving to Phnom Penh, emergencies (117 / 118 / 119) |
| `khmer_study.jsonl`, `homework_bilingual.jsonl` | study tips and school subjects (science, math, grammar, civics), English and Khmer |
| `khmer_work.jsonl`, `khmer_health.jsonl`, `khmer_tech.jsonl` | work and small business, health (with when-to-see-a-doctor), phones and online safety, in Khmer |
| `translation_more.jsonl`, `english_for_khmer.jsonl` | two-way English ↔ Khmer sentences, word meanings, and English lessons explained in Khmer |
| `identity_more.jsonl`, `honesty.jsonl`, `careful_more.jsonl` | who Virgo is (including tricks to rename it), honest "I can't know that" answers with Search, scams and harmful requests |
| `knowledge_more.jsonl`, `reasoning_more.jsonl`, `coding_more.jsonl`, `quick_facts_bilingual.jsonl` | general knowledge, logic puzzles, beginner coding, quick facts in both languages |
| `small_talk.jsonl`, `writing_tasks.jsonl`, `answer_style_more.jsonl` | greetings and chit-chat (also romanised Khmer), rewriting and messages, when to be short, when to ask, tables |
| `conversations_more.jsonl` | multi-turn chats that build on earlier turns |
| `bilingual_help.jsonl`, `how_to_bilingual.jsonl`, `life_tips_bilingual.jsonl`, `life_culture_bilingual.jsonl`, `support_bilingual.jsonl` | the same practical help in English and Khmer: money, safety, family, farming, travel, culture, using Virgo AI, emotional support |
| `formal_khmer_more.jsonl`, `khmer_language.jsonl`, `cambodia_more.jsonl`, `everyday_english.jsonl` | formal letters and speeches, the Khmer language (ញ៉ាំ/ពិសា/សោយ, ឱ្យ), Cambodia facts, everyday English advice |

Tips for adding more:
- Write answers the way Virgo should talk: short, warm, clear.
- Keep facts that don't change (history, how-to). For news, prices or rates, teach Virgo to suggest Search.
- Have a Khmer speaker check Khmer answers; natural wording matters more than quantity.
- Several hundred to a few thousand good examples is a strong goal.
