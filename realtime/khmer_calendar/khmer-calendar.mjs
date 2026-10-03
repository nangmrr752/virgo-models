// Exact Khmer calendar facts for the chat: when a message is about dates, the lunar calendar or
// Cambodian holidays, the answer's facts are computed here (Khmer lunar engine from khmer-lunar,
// MIT, server/vendor) and given to the model, so it never has to guess a lunar date or a holiday.
import { formatLunar, getHolidayOccurrences, getKhmerNewYear, getLunarDate, MAX_YEAR, MIN_YEAR } from "./khmer-lunar.mjs";

const KHMER_DIGITS = { "០": 0, "១": 1, "២": 2, "៣": 3, "៤": 4, "៥": 5, "៦": 6, "៧": 7, "៨": 8, "៩": 9 };
// Dates, the lunar calendar, holidays and festivals, in Khmer and English.
const ABOUT_CALENDAR =
  /ថ្ងៃ|ខែ|ឆ្នាំ|ចន្ទគតិ|ច័ន្ទគតិ|កើត|រោច|ស័ក|ព\.ស|ពុទ្ធសករាជ|បុណ្យ|ចូលឆ្នាំ|ភ្ជុំ|អុំទូក|វិស្សមកាល|ឈប់សម្រាក|សីល|\b(date|today|tomorrow|yesterday|calendar|lunar|holidays?|new year|songkran|pchum|water festival|weekday|what day|day off|buddhist era|zodiac|year of the)\b/i;

const pad = (n) => String(n).padStart(2, "0");
const iso = (y, m, d) => `${y}-${pad(m)}-${pad(d)}`;

// Today in Cambodia (UTC+7), whatever the server's own time zone.
export function cambodiaToday(now = new Date()) {
  const t = new Date(now.getTime() + 7 * 3600 * 1000);
  return iso(t.getUTCFullYear(), t.getUTCMonth() + 1, t.getUTCDate());
}

function addDaysISO(date, days) {
  const t = new Date(`${date}T00:00:00Z`);
  t.setUTCDate(t.getUTCDate() + days);
  return t.toISOString().slice(0, 10);
}

// Gregorian dates written in the message: 2026-10-03, 3/10/2026 or ៣/១០/២០២៦ (day/month/year, as in Cambodia).
export function datesIn(text) {
  const plain = String(text).replace(/[០-៩]/g, (c) => KHMER_DIGITS[c]);
  const found = [];
  for (const m of plain.matchAll(/\b(\d{4})-(\d{1,2})-(\d{1,2})\b/g)) found.push(iso(+m[1], +m[2], +m[3]));
  for (const m of plain.matchAll(/\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b/g)) found.push(iso(+m[3], +m[2], +m[1]));
  return [...new Set(found)].filter((d) => {
    const [y, mo, da] = d.split("-").map(Number);
    return y >= MIN_YEAR && y <= MAX_YEAR && mo >= 1 && mo <= 12 && da >= 1 && da <= 31;
  }).slice(0, 3);
}

function describe(date) {
  try {
    const lunar = getLunarDate(date);
    return `${date}: ${formatLunar(lunar, "km")} (${formatLunar(lunar, "en")})`;
  } catch {
    return "";
  }
}

const holidayLine = (h) => `${h.date} ${h.name.km} (${h.name.en})`;

// The calendar facts for this message, or "" when it isn't about dates.
export function calendarContext(text, now = new Date()) {
  if (!ABOUT_CALENDAR.test(String(text || ""))) return "";
  const today = cambodiaToday(now);
  const year = Number(today.slice(0, 4));
  const lines = [`Today in Cambodia: ${describe(today)}`, `Tomorrow: ${describe(addDaysISO(today, 1))}`];
  for (const date of datesIn(text)) lines.push(`Asked about ${describe(date)}`);
  const asked = [...new Set(datesIn(text).map((d) => Number(d.slice(0, 4))))];
  // Holidays of this year (and any year asked about), and the next Khmer New Year.
  for (const y of [...new Set([year, ...asked])].filter((y) => y >= MIN_YEAR && y <= MAX_YEAR).slice(0, 2)) {
    try {
      const all = getHolidayOccurrences(y);
      const shown = y === year ? all.filter((h) => h.date >= today).slice(0, 10) : all;
      if (shown.length) lines.push(`${y === year ? "Upcoming holidays" : `Holidays in ${y}`}: ${shown.map(holidayLine).join("; ")}`);
    } catch {}
  }
  try {
    let ny = getKhmerNewYear(year);
    if (ny.lerngSakDate < today) ny = getKhmerNewYear(year + 1);
    lines.push(`Khmer New Year ${ny.year}: ${ny.schedule.map((s) => s.date).join(", ")} (Moha Songkran at ${ny.time})`);
  } catch {}
  return lines.filter(Boolean).join("\n");
}

// The messages with the calendar facts added to the instructions, when the latest message needs them.
export function withCalendar(messages, now = new Date()) {
  if (!Array.isArray(messages)) return messages;
  const latest = [...messages].reverse().find((m) => m?.role === "user")?.content;
  const facts = calendarContext(latest, now);
  // Always today's date, so the model never answers with the year it was trained in.
  const note = facts
    ? "Khmer calendar facts (computed exactly; use them for any date, lunar day or holiday in your answer, " +
      `and write Khmer dates in this form):\n${facts}`
    : `Today in Cambodia: ${describe(cambodiaToday(now))}.`;
  const i = messages.findIndex((m) => m?.role === "system");
  if (i === -1) return [{ role: "system", content: note }, ...messages];
  return messages.map((m, j) => (j === i ? { ...m, content: `${m.content}\n\n${note}` } : m));
}
