// Khmer lunar calendar engine (Chhankitek), bundled unchanged from https://github.com/Bonker009/khmer-lunar
// (lib/khmer) — MIT License, Copyright (c) 2026 Penh Seyha. Full license: server/vendor/khmer-lunar.LICENSE
// Rebuild: npx esbuild lib/khmer/index.ts --bundle --format=esm --target=es2022 --outfile=khmer-lunar.js
/* eslint-disable */
// lib/khmer/constants.ts
var MIN_YEAR = 1900;
var MAX_YEAR = 2100;
var MonthIndex = {
  Migasir: 0,
  Boss: 1,
  Meak: 2,
  Phalkun: 3,
  Cheit: 4,
  Pisakh: 5,
  Jesth: 6,
  Asadh: 7,
  Srap: 8,
  Phatrabot: 9,
  Assoch: 10,
  Kadeuk: 11,
  /** First Asadh, only in leap-month (អធិកមាស) years. */
  Pathamasadh: 12,
  /** Second Asadh, only in leap-month (អធិកមាស) years. */
  Tutiyasadh: 13
};
var LUNAR_MONTHS = {
  km: [
    "\u1798\u17B7\u1782\u179F\u17B7\u179A",
    "\u1794\u17BB\u179F\u17D2\u179F",
    "\u1798\u17B6\u1783",
    "\u1795\u179B\u17D2\u1782\u17BB\u1793",
    "\u1785\u17C1\u178F\u17D2\u179A",
    "\u1796\u17B7\u179F\u17B6\u1781",
    "\u1787\u17C1\u179F\u17D2\u178B",
    "\u17A2\u17B6\u179F\u17B6\u178D",
    "\u179F\u17D2\u179A\u17B6\u1796\u178E\u17CD",
    "\u1797\u1791\u17D2\u179A\u1794\u1791",
    "\u17A2\u179F\u17D2\u179F\u17BB\u1787",
    "\u1780\u178F\u17D2\u178F\u17B7\u1780",
    "\u1794\u178B\u1798\u17B6\u179F\u17B6\u178D",
    "\u1791\u17BB\u178F\u17B7\u1799\u17B6\u179F\u17B6\u178D"
  ],
  en: [
    "Migasir",
    "Boss",
    "Meak",
    "Phalkun",
    "Cheit",
    "Pisakh",
    "Jesth",
    "Asadh",
    "Srap",
    "Phatrabot",
    "Assoch",
    "Kadeuk",
    "Pathamasadh",
    "Tutiyasadh"
  ]
};
var GREGORIAN_MONTHS = {
  km: ["\u1798\u1780\u179A\u17B6", "\u1780\u17BB\u1798\u17D2\u1797\u17C8", "\u1798\u17B8\u1793\u17B6", "\u1798\u17C1\u179F\u17B6", "\u17A7\u179F\u1797\u17B6", "\u1798\u17B7\u1790\u17BB\u1793\u17B6", "\u1780\u1780\u17D2\u1780\u178A\u17B6", "\u179F\u17B8\u17A0\u17B6", "\u1780\u1789\u17D2\u1789\u17B6", "\u178F\u17BB\u179B\u17B6", "\u179C\u17B7\u1785\u17D2\u1786\u17B7\u1780\u17B6", "\u1792\u17D2\u1793\u17BC"],
  en: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
};
var ANIMAL_YEARS = {
  km: ["\u1787\u17BC\u178F", "\u1786\u17D2\u179B\u17BC\u179C", "\u1781\u17B6\u179B", "\u1790\u17C4\u17C7", "\u179A\u17C4\u1784", "\u1798\u17D2\u179F\u17B6\u1789\u17CB", "\u1798\u1798\u17B8", "\u1798\u1798\u17C2", "\u179C\u1780", "\u179A\u1780\u17B6", "\u1785", "\u1780\u17BB\u179A"],
  en: ["Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig"]
};
var ANIMAL_EMOJIS = ["\u{1F400}", "\u{1F402}", "\u{1F405}", "\u{1F407}", "\u{1F409}", "\u{1F40D}", "\u{1F40E}", "\u{1F410}", "\u{1F412}", "\u{1F413}", "\u{1F415}", "\u{1F416}"];
var SAKS = {
  km: ["\u179F\u17C6\u179A\u17B9\u1791\u17D2\u1792\u17B7\u179F\u17D0\u1780", "\u17AF\u1780\u179F\u17D0\u1780", "\u1791\u17C4\u179F\u17D0\u1780", "\u178F\u17D2\u179A\u17B8\u179F\u17D0\u1780", "\u1785\u178F\u17D2\u179C\u17B6\u179F\u17D0\u1780", "\u1794\u1789\u17D2\u1785\u179F\u17D0\u1780", "\u1786\u179F\u17D0\u1780", "\u179F\u1794\u17D2\u178F\u179F\u17D0\u1780", "\u17A2\u178A\u17D2\u178B\u179F\u17D0\u1780", "\u1793\u1796\u17D2\u179C\u179F\u17D0\u1780"],
  en: ["Samridhi Sak", "Ek Sak", "To Sak", "Trei Sak", "Chattva Sak", "Pancha Sak", "Chha Sak", "Sapta Sak", "Attha Sak", "Nappa Sak"]
};
var WEEKDAYS = {
  km: ["\u17A2\u17B6\u1791\u17B7\u178F\u17D2\u1799", "\u1785\u1793\u17D2\u1791", "\u17A2\u1784\u17D2\u1782\u17B6\u179A", "\u1796\u17BB\u1792", "\u1796\u17D2\u179A\u17A0\u179F\u17D2\u1794\u178F\u17B7\u17CD", "\u179F\u17BB\u1780\u17D2\u179A", "\u179F\u17C5\u179A\u17CD"],
  en: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
};
var WEEKDAYS_SHORT = {
  km: ["\u17A2\u17B6", "\u1785", "\u17A2", "\u1796", "\u1796\u17D2\u179A", "\u179F\u17BB", "\u179F"],
  en: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
};
var PHASES = {
  km: { kert: "\u1780\u17BE\u178F", roech: "\u179A\u17C4\u1785" },
  en: { kert: "Waxing", roech: "Waning" }
};
var PHASES_SHORT = {
  km: { kert: "\u1780", roech: "\u179A" },
  en: { kert: "K", roech: "R" }
};
var MOON_PHASES = {
  km: { "first-quarter": "\u17E8\u1780\u17BE\u178F", full: "\u1796\u17C1\u1789\u1794\u17BC\u178E\u17CC\u1798\u17B8", "last-quarter": "\u17E8\u179A\u17C4\u1785", new: "\u1790\u17D2\u1784\u17C3\u178A\u17B6\u1785\u17CB\u1781\u17C2" },
  en: { "first-quarter": "First quarter", full: "Full moon", "last-quarter": "Last quarter", new: "New moon" }
};
var HOLY_DAY_NAME = {
  km: "\u1790\u17D2\u1784\u17C3\u179F\u17B8\u179B",
  en: "Buddhist holy day"
};
var LEAP_TYPES = {
  km: { regular: "\u1786\u17D2\u1793\u17B6\u17C6\u1792\u1798\u17D2\u1798\u178F\u17B6", "leap-month": "\u17A2\u1792\u17B7\u1780\u1798\u17B6\u179F", "leap-day": "\u1785\u1793\u17D2\u1791\u17D2\u179A\u17B6\u1792\u17B7\u1798\u17B6\u179F" },
  en: { regular: "Regular year", "leap-month": "Leap month", "leap-day": "Leap day" }
};

// lib/khmer/numerals.ts
var KHMER_DIGITS = ["\u17E0", "\u17E1", "\u17E2", "\u17E3", "\u17E4", "\u17E5", "\u17E6", "\u17E7", "\u17E8", "\u17E9"];
function toKhmerNumber(value) {
  return String(value).replace(/\d/g, (d) => KHMER_DIGITS[Number(d)]);
}

// lib/khmer/date.ts
function isGregorianLeapYear(year) {
  return year % 4 === 0 && year % 100 !== 0 || year % 400 === 0;
}
function daysInGregorianMonth(year, month) {
  if (month === 2) return isGregorianLeapYear(year) ? 29 : 28;
  return [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1];
}
function toJdn(year, month, day) {
  const a = Math.floor((14 - month) / 12);
  const y = year + 4800 - a;
  const m = month + 12 * a - 3;
  return day + Math.floor((153 * m + 2) / 5) + 365 * y + Math.floor(y / 4) - Math.floor(y / 100) + Math.floor(y / 400) - 32045;
}
function fromJdn(jdn) {
  const a = jdn + 32044;
  const b = Math.floor((4 * a + 3) / 146097);
  const c = a - Math.floor(146097 * b / 4);
  const d = Math.floor((4 * c + 3) / 1461);
  const e = c - Math.floor(1461 * d / 4);
  const m = Math.floor((5 * e + 2) / 153);
  return {
    year: 100 * b + d - 4800 + Math.floor(m / 10),
    month: m + 3 - 12 * Math.floor(m / 10),
    day: e - Math.floor((153 * m + 2) / 5) + 1
  };
}
function weekdayOf(jdn) {
  return (jdn + 1) % 7;
}
function assertYearInRange(year) {
  if (year < MIN_YEAR || year > MAX_YEAR) {
    throw new RangeError(
      `Year ${year} is outside the supported range ${MIN_YEAR}\u2013${MAX_YEAR}`
    );
  }
}
var ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;
function parseISODate(value) {
  const match = ISO_DATE.exec(value);
  if (!match) {
    throw new RangeError(`Invalid date "${value}", expected YYYY-MM-DD`);
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (month < 1 || month > 12 || day < 1 || day > daysInGregorianMonth(year, month)) {
    throw new RangeError(`Invalid date "${value}"`);
  }
  assertYearInRange(year);
  return { year, month, day };
}
var pad = (n, length = 2) => String(n).padStart(length, "0");
function toISO({ year, month, day }) {
  return `${pad(year, 4)}-${pad(month)}-${pad(day)}`;
}
var isoFromJdn = (jdn) => toISO(fromJdn(jdn));
function jdnFromISO(value) {
  const { year, month, day } = parseISODate(value);
  return toJdn(year, month, day);
}
function todayISO(timeZone = "Asia/Phnom_Penh") {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(/* @__PURE__ */ new Date());
}

// lib/khmer/era.ts
function getAharkun(beYear) {
  return Math.floor((beYear * 292207 + 499) / 800) + 4;
}
function getKromthupul(beYear) {
  return 800 - (beYear * 292207 + 499) % 800;
}
function getAvoman(beYear) {
  return (getAharkun(beYear) * 11 + 25) % 692;
}
function getBodithey(beYear) {
  const aharkun = getAharkun(beYear);
  return (Math.floor((aharkun * 11 + 25) / 692) + aharkun + 29) % 30;
}
function isKhmerSolarLeap(beYear) {
  return getKromthupul(beYear) <= 207;
}
function hasLeapDayByCalculation(beYear) {
  const avoman = getAvoman(beYear);
  if (avoman === 0 && getAvoman(beYear - 1) === 137) return true;
  if (isKhmerSolarLeap(beYear)) return avoman < 127;
  if (avoman === 137 && getAvoman(beYear + 1) === 0) return false;
  return avoman < 138;
}
function hasLeapMonth(beYear) {
  const bodithey = getBodithey(beYear);
  const next = getBodithey(beYear + 1);
  if (bodithey === 25 && next === 5) return false;
  return bodithey === 24 && next === 6 || bodithey >= 25 || bodithey < 6;
}
var leapTypeCache = /* @__PURE__ */ new Map();
function getLeapType(beYear) {
  const cached = leapTypeCache.get(beYear);
  if (cached) return cached;
  let result = "regular";
  if (hasLeapMonth(beYear)) {
    result = "leap-month";
  } else if (hasLeapDayByCalculation(beYear)) {
    result = "leap-day";
  } else if (hasLeapMonth(beYear - 1)) {
    let previous = beYear - 1;
    while (true) {
      if (hasLeapDayByCalculation(previous)) {
        result = "leap-day";
        break;
      }
      previous -= 1;
      if (!hasLeapMonth(previous)) break;
    }
  }
  leapTypeCache.set(beYear, result);
  return result;
}
function daysInLunarYear(leapType) {
  if (leapType === "leap-month") return 384;
  if (leapType === "leap-day") return 355;
  return 354;
}

// lib/khmer/core.ts
var M = MonthIndex;
var REGULAR_ORDER = [M.Boss, M.Meak, M.Phalkun, M.Cheit, M.Pisakh, M.Jesth, M.Asadh, M.Srap, M.Phatrabot, M.Assoch, M.Kadeuk, M.Migasir];
var LEAP_MONTH_ORDER = [M.Boss, M.Meak, M.Phalkun, M.Cheit, M.Pisakh, M.Jesth, M.Pathamasadh, M.Tutiyasadh, M.Srap, M.Phatrabot, M.Assoch, M.Kadeuk, M.Migasir];
var EPOCH_JDN = toJdn(MIN_YEAR, 1, 1);
var LAST_SPAN_YEAR = MAX_YEAR + 2;
function monthLength(index, leapType) {
  if (index === M.Jesth) return leapType === "leap-day" ? 30 : 29;
  if (index === M.Pathamasadh || index === M.Tutiyasadh) return 30;
  return index % 2 === 0 ? 29 : 30;
}
var spans;
function getSpans() {
  if (spans) return spans;
  const list = [];
  let start = EPOCH_JDN;
  for (let year = MIN_YEAR; year <= LAST_SPAN_YEAR; year++) {
    const leapType = getLeapType(year + 544);
    const order = leapType === "leap-month" ? LEAP_MONTH_ORDER : REGULAR_ORDER;
    let cursor = start;
    const months = order.map((index) => {
      const length = monthLength(index, leapType);
      const month = { index, start: cursor, length };
      cursor += length;
      return month;
    });
    list.push({ gregorianYear: year, start, length: cursor - start, leapType, months });
    start = cursor;
  }
  spans = list;
  return list;
}
function getLunarYear(gregorianYear) {
  const span = getSpans()[gregorianYear - MIN_YEAR];
  if (!span) {
    throw new RangeError(`Lunar year for ${gregorianYear} is outside the supported range`);
  }
  return span;
}
function findLunarYear(jdn) {
  const list = getSpans();
  const last = list[list.length - 1];
  if (jdn < list[0].start || jdn >= last.start + last.length) {
    throw new RangeError("Date is outside the supported range");
  }
  let lo = 0;
  let hi = list.length - 1;
  while (lo < hi) {
    const mid = lo + hi + 1 >> 1;
    if (list[mid].start <= jdn) lo = mid;
    else hi = mid - 1;
  }
  return list[lo];
}
function findMonth(year, index) {
  return year.months.find((month) => month.index === index);
}
function lunarCore(jdn) {
  const year = findLunarYear(jdn);
  const month = year.months.find((m) => jdn < m.start + m.length);
  return { year, month, dayNumber: jdn - month.start };
}

// lib/khmer/new-year.ts
function getAharkunJs(jsYear) {
  return Math.floor((jsYear * 292207 + 373) / 800) + 1;
}
function getAvomanJs(jsYear) {
  return (getAharkunJs(jsYear) * 11 + 650) % 692;
}
function getKromthupulJs(jsYear) {
  return 800 - (292207 * jsYear + 373) % 800;
}
function getBoditheyJs(jsYear) {
  const aharkun = getAharkunJs(jsYear);
  return (aharkun + Math.floor((11 * aharkun + 650) / 692)) % 30;
}
function isAdhikameas(jsYear) {
  const bodithey = getBoditheyJs(jsYear);
  const next = getBoditheyJs(jsYear + 1);
  if (bodithey === 24 && next === 6) return true;
  if (bodithey === 25 && next === 5) return false;
  return bodithey > 24 || bodithey < 6;
}
function isChantrathimeas(jsYear) {
  const avoman = getAvomanJs(jsYear);
  const isSolarLeap = getKromthupulJs(jsYear) <= 207;
  if (avoman === 0 && getAvomanJs(jsYear - 1) === 137) return true;
  if (isSolarLeap) return avoman < 127;
  if (avoman === 137 && getAvomanJs(jsYear + 1) === 0) return false;
  return avoman < 138;
}
function getSunInfo(jsYear, sotin) {
  const r2 = 800 * sotin + getKromthupulJs(jsYear - 1);
  const reasey = Math.floor(r2 / 24350);
  const r3 = r2 % 24350;
  const angsar = Math.floor(r3 / 811);
  const libda = Math.floor(r3 % 811 / 14) - 3;
  const sunAverage = 30 * 60 * reasey + 60 * angsar + libda;
  const s1 = 30 * 60 * 2 + 60 * 20;
  let leftOver = sunAverage - s1;
  if (sunAverage < s1) leftOver += 30 * 60 * 12;
  const kaen = Math.floor(leftOver / (30 * 60));
  let rs = -1;
  if (kaen <= 2) rs = kaen;
  else if (kaen <= 5) rs = 30 * 60 * 6 - leftOver;
  else if (kaen <= 8) rs = leftOver - 30 * 60 * 6;
  else rs = 30 * 60 * 11 + 60 * 29 + 60 - leftOver;
  const last = {
    reasey: Math.floor(rs / (30 * 60)),
    angsar: Math.floor(rs % (30 * 60) / 60),
    libda: rs % 60
  };
  const khan = last.angsar >= 15 ? 2 * last.reasey + 1 : 2 * last.reasey;
  const pouichalip = last.angsar >= 15 ? 60 * (last.angsar - 15) + last.libda : 60 * last.angsar + last.libda;
  const chhayaSunMap = [
    { multiplicity: 35, chhaya: 0 },
    { multiplicity: 32, chhaya: 35 },
    { multiplicity: 27, chhaya: 67 },
    { multiplicity: 22, chhaya: 94 },
    { multiplicity: 13, chhaya: 116 },
    { multiplicity: 5, chhaya: 129 }
  ];
  const chhaya = khan <= 5 ? chhayaSunMap[khan] : { multiplicity: 0, chhaya: 134 };
  const phol = Math.floor(pouichalip * chhaya.multiplicity / 900) + chhaya.chhaya;
  const inauguration = kaen <= 5 ? sunAverage - phol : sunAverage + phol;
  return {
    angsar: Math.floor(inauguration % (30 * 60) / 60),
    libda: inauguration % 60
  };
}
var EXCEPTIONS = {
  2011: [4, 14, 13, 12],
  2012: [4, 14, 19, 11],
  2013: [4, 14, 2, 12],
  2014: [4, 14, 8, 7],
  2015: [4, 14, 14, 2],
  2024: [4, 13, 22, 17]
};
var cache = /* @__PURE__ */ new Map();
function getKhmerNewYear(year) {
  const cached = cache.get(year);
  if (cached) return cached;
  const jsYear = year - 638;
  const sotins = getKromthupulJs(jsYear - 1) <= 207 ? [363, 364, 365, 366] : [362, 363, 364, 365];
  const sunInfos = sotins.map((sotin) => getSunInfo(jsYear, sotin));
  let hour = 0;
  let minute = 0;
  const arrival = sunInfos.find((info) => info.angsar === 0);
  if (arrival) {
    const minutes = 24 * 60 - arrival.libda * 24;
    hour = Math.floor(minutes / 60) % 24;
    minute = minutes % 60;
  }
  let bodithey = getBoditheyJs(jsYear);
  if (isAdhikameas(jsYear - 1) && isChantrathimeas(jsYear - 1)) {
    bodithey = (bodithey + 1) % 30;
  }
  const lerngSakDay = bodithey >= 6 ? bodithey - 1 : bodithey;
  const lerngSakMonth = bodithey >= 6 ? 4 : 5;
  const days = sunInfos[0].angsar === 0 ? 4 : 3;
  const epochJdn = toJdn(year, 4, 17);
  const epoch = lunarCore(epochJdn);
  const diff = (epoch.month.index - 4) * 29 + epoch.dayNumber - ((lerngSakMonth - 4) * 29 + lerngSakDay);
  let jdn = epochJdn - (diff + days - 1);
  const exception = EXCEPTIONS[year];
  if (exception) {
    jdn = toJdn(year, exception[0], exception[1]);
    hour = exception[2];
    minute = exception[3];
  }
  const lerngSakJdn = jdn + days - 1;
  const schedule = Array.from({ length: days }, (_, i) => ({
    date: isoFromJdn(jdn + i),
    kind: i === 0 ? "moha-songkran" : i === days - 1 ? "lerng-sak" : "vanabat"
  }));
  const result = {
    year,
    date: isoFromJdn(jdn),
    jdn,
    time: `${pad(hour)}:${pad(minute)}`,
    hour,
    minute,
    days,
    vanabatDays: days - 2,
    lerngSakDate: isoFromJdn(lerngSakJdn),
    lerngSakJdn,
    schedule
  };
  cache.set(year, result);
  return result;
}

// lib/khmer/lunar.ts
function holyDayOf(dayNumber, monthLength2) {
  if (dayNumber === 7) return "8-kert";
  if (dayNumber === 14) return "15-kert";
  if (dayNumber === 22) return "8-roech";
  if (dayNumber === monthLength2 - 1) return "last-roech";
  return null;
}
function moonPhaseOf(dayNumber, monthLength2) {
  if (dayNumber === 7) return "first-quarter";
  if (dayNumber === 14) return "full";
  if (dayNumber === 22) return "last-quarter";
  if (dayNumber === monthLength2 - 1) return "new";
  return null;
}
function buddhistEraStartJdn(gregorianYear) {
  return findMonth(getLunarYear(gregorianYear), MonthIndex.Pisakh).start + 15;
}
function getLunarDate(input) {
  const jdn = typeof input === "string" ? jdnFromISO(input) : input;
  const { year, month, day } = fromJdn(jdn);
  assertYearInRange(year);
  const core = lunarCore(jdn);
  const beStart = buddhistEraStartJdn(year);
  const newYear = getKhmerNewYear(year);
  const beforeBeChange = jdn < beStart;
  const beYear = beforeBeChange ? year + 543 : year + 544;
  let animalYear = (beYear + 4) % 12;
  let jsYear = beYear - 1182;
  if (beforeBeChange && jdn >= newYear.jdn) animalYear = (animalYear + 1) % 12;
  if (beforeBeChange && jdn >= newYear.lerngSakJdn) jsYear += 1;
  const { dayNumber } = core;
  const length = core.month.length;
  return {
    jdn,
    date: isoFromJdn(jdn),
    year,
    month,
    day,
    weekday: weekdayOf(jdn),
    lunarDay: dayNumber < 15 ? dayNumber + 1 : dayNumber - 14,
    phase: dayNumber < 15 ? "kert" : "roech",
    dayNumber,
    monthIndex: core.month.index,
    monthLength: length,
    leapType: core.year.leapType,
    beYear,
    jsYear,
    animalYear,
    sak: (jsYear % 10 + 10) % 10,
    holyDay: holyDayOf(dayNumber, length),
    moonPhase: moonPhaseOf(dayNumber, length)
  };
}
function khmerToGregorian({ beYear, monthIndex, day, phase }) {
  if (!Number.isInteger(day) || day < 1 || day > 15) {
    throw new RangeError("Lunar day must be between 1 and 15");
  }
  if (!Number.isInteger(monthIndex) || monthIndex < 0 || monthIndex > 13) {
    throw new RangeError("Month index must be between 0 and 13");
  }
  if (phase !== "kert" && phase !== "roech") {
    throw new RangeError("Phase must be `kert` or `roech`");
  }
  const gregorianYear = beYear - 544;
  const inFollowingYear = monthIndex >= MonthIndex.Boss && monthIndex <= MonthIndex.Cheit || monthIndex === MonthIndex.Pisakh && phase === "kert";
  const lunarYear = getLunarYear(inFollowingYear ? gregorianYear + 1 : gregorianYear);
  const month = findMonth(lunarYear, monthIndex);
  if (!month) {
    throw new RangeError(
      monthIndex >= MonthIndex.Pathamasadh ? `BE ${beYear} has no leap month; use month 7 (Asadh)` : `BE ${beYear} has a leap month; use 12 (Pathamasadh) or 13 (Tutiyasadh) instead of 7`
    );
  }
  const dayNumber = phase === "kert" ? day - 1 : day + 14;
  if (dayNumber >= month.length) {
    throw new RangeError(`This month has ${month.length} days, so it ends on ${month.length - 15} \u179A\u17C4\u1785`);
  }
  return getLunarDate(month.start + dayNumber);
}

// lib/khmer/holidays.ts
var FIXED_HOLIDAYS = [
  { id: "international-new-year", month: 1, day: 1, type: "public", name: { km: "\u1791\u17B7\u179C\u17B6\u1785\u17BC\u179B\u1786\u17D2\u1793\u17B6\u17C6\u179F\u1780\u179B", en: "International New Year's Day" } },
  { id: "victory-over-genocide", month: 1, day: 7, type: "public", name: { km: "\u1791\u17B7\u179C\u17B6\u1787\u17D0\u1799\u1787\u1798\u17D2\u1793\u17C7\u179B\u17BE\u179A\u1794\u1794\u1794\u17D2\u179A\u179B\u17D0\u1799\u1796\u17BC\u1787\u179F\u17B6\u179F\u1793\u17CD", en: "Victory over Genocide Day" } },
  { id: "womens-day", month: 3, day: 8, type: "public", name: { km: "\u1791\u17B7\u179C\u17B6\u17A2\u1793\u17D2\u178F\u179A\u1787\u17B6\u178F\u17B7\u1793\u17B6\u179A\u17B8", en: "International Women's Day" } },
  { id: "labour-day", month: 5, day: 1, type: "public", name: { km: "\u1791\u17B7\u179C\u17B6\u1796\u179B\u1780\u1798\u17D2\u1798\u17A2\u1793\u17D2\u178F\u179A\u1787\u17B6\u178F\u17B7", en: "International Labour Day" } },
  { id: "king-birthday", month: 5, day: 14, type: "public", since: 2005, name: { km: "\u1796\u17D2\u179A\u17C7\u179A\u17B6\u1787\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u1785\u1798\u17D2\u179A\u17BE\u1793\u1796\u17D2\u179A\u17C7\u1787\u1793\u17D2\u1798 \u1796\u17D2\u179A\u17C7\u1798\u17A0\u17B6\u1780\u17D2\u179F\u178F\u17D2\u179A", en: "King Norodom Sihamoni's Birthday" } },
  { id: "national-day-of-remembrance", month: 5, day: 20, type: "observance", name: { km: "\u1791\u17B7\u179C\u17B6\u1787\u17B6\u178F\u17B7\u1793\u17C3\u1780\u17B6\u179A\u1785\u1784\u1785\u17B6\u17C6", en: "National Day of Remembrance" } },
  { id: "queen-mother-birthday", month: 6, day: 18, type: "public", since: 2015, name: { km: "\u1796\u17D2\u179A\u17C7\u179A\u17B6\u1787\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u1785\u1798\u17D2\u179A\u17BE\u1793\u1796\u17D2\u179A\u17C7\u1787\u1793\u17D2\u1798 \u179F\u1798\u17D2\u178F\u17C1\u1785\u1796\u17D2\u179A\u17C7\u1798\u17A0\u17B6\u1780\u17D2\u179F\u178F\u17D2\u179A\u17B8 \u1796\u17D2\u179A\u17C7\u179C\u179A\u179A\u17B6\u1787\u1798\u17B6\u178F\u17B6", en: "Queen Mother's Birthday" } },
  { id: "constitution-day", month: 9, day: 24, type: "public", since: 1993, name: { km: "\u1791\u17B7\u179C\u17B6\u1794\u17D2\u179A\u1780\u17B6\u179F\u179A\u178A\u17D2\u178B\u1792\u1798\u17D2\u1798\u1793\u17BB\u1789\u17D2\u1789", en: "Constitution Day" } },
  { id: "king-father-commemoration", month: 10, day: 15, type: "public", since: 2013, name: { km: "\u1791\u17B7\u179C\u17B6\u1794\u17D2\u179A\u17B6\u179A\u1796\u17D2\u1792\u1796\u17B7\u1792\u17B8\u1782\u17C4\u179A\u1796\u1796\u17D2\u179A\u17C7\u179C\u17B7\u1789\u17D2\u1789\u17B6\u178E\u1780\u17D2\u1781\u1793\u17D2\u1792 \u1796\u17D2\u179A\u17C7\u1794\u179A\u1798\u179A\u178F\u1793\u1780\u17C4\u178A\u17D2\u178B", en: "Commemoration Day of King Father Norodom Sihanouk" } },
  { id: "coronation-day", month: 10, day: 29, type: "public", since: 2004, name: { km: "\u1796\u17D2\u179A\u17C7\u179A\u17B6\u1787\u1796\u17B7\u1792\u17B8\u1782\u17D2\u179A\u1784\u1796\u17D2\u179A\u17C7\u1794\u179A\u1798\u179A\u17B6\u1787\u179F\u1798\u17D2\u1794\u178F\u17D2\u178F\u17B7", en: "Coronation Day" } },
  { id: "independence-day", month: 11, day: 9, type: "public", since: 1953, name: { km: "\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u17AF\u1780\u179A\u17B6\u1787\u17D2\u1799\u1787\u17B6\u178F\u17B7", en: "Independence Day" } },
  { id: "human-rights-day", month: 12, day: 10, type: "observance", name: { km: "\u1791\u17B7\u179C\u17B6\u179F\u17B7\u1791\u17D2\u1792\u17B7\u1798\u1793\u17BB\u179F\u17D2\u179F\u17A2\u1793\u17D2\u178F\u179A\u1787\u17B6\u178F\u17B7", en: "International Human Rights Day" } },
  { id: "peace-day", month: 12, day: 29, type: "public", since: 2024, name: { km: "\u1791\u17B7\u179C\u17B6\u179F\u1793\u17D2\u178F\u17B7\u1797\u17B6\u1796\u1793\u17C5\u1780\u1798\u17D2\u1796\u17BB\u1787\u17B6", en: "Peace Day in Cambodia" } }
];
var LUNAR_HOLIDAYS = [
  { id: "meak-bochea", monthIndex: MonthIndex.Meak, dayNumbers: [14], type: "observance", name: { km: "\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u1798\u17B6\u1783\u1794\u17BC\u1787\u17B6", en: "Meak Bochea Day" } },
  { id: "visak-bochea", monthIndex: MonthIndex.Pisakh, dayNumbers: [14], type: "public", name: { km: "\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u179C\u17B7\u179F\u17B6\u1781\u1794\u17BC\u1787\u17B6", en: "Visak Bochea Day" } },
  { id: "royal-ploughing", monthIndex: MonthIndex.Pisakh, dayNumbers: [18], type: "public", name: { km: "\u1796\u17D2\u179A\u17C7\u179A\u17B6\u1787\u1796\u17B7\u1792\u17B8\u1785\u17D2\u179A\u178F\u17CB\u1796\u17D2\u179A\u17C7\u1793\u1784\u17D2\u1782\u17D0\u179B", en: "Royal Ploughing Ceremony" } },
  // 14 រោច, 15 រោច ភទ្របទ and 1 កើត អស្សុជ
  { id: "pchum-ben", monthIndex: MonthIndex.Phatrabot, dayNumbers: [28, 29, 30], type: "public", name: { km: "\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u1797\u17D2\u1787\u17BB\u17C6\u1794\u17B7\u178E\u17D2\u178C", en: "Pchum Ben Festival" } },
  // 14 កើត, 15 កើត and 1 រោច កត្តិក
  { id: "water-festival", monthIndex: MonthIndex.Kadeuk, dayNumbers: [13, 14, 15], type: "public", name: { km: "\u1796\u17D2\u179A\u17C7\u179A\u17B6\u1787\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u17A2\u17BB\u17C6\u1791\u17BC\u1780 \u1794\u178E\u17D2\u178F\u17C2\u178F\u1794\u17D2\u179A\u1791\u17B8\u1794 \u1793\u17B7\u1784\u179F\u17C6\u1796\u17C7\u1796\u17D2\u179A\u17C7\u1781\u17C2 \u17A2\u1780\u17A2\u17C6\u1794\u17BB\u1780", en: "Water Festival" } }
];
var cache2 = /* @__PURE__ */ new Map();
function getHolidays(year) {
  const cached = cache2.get(year);
  if (cached) return cached;
  const lunarYear = getLunarYear(year);
  const holidays = FIXED_HOLIDAYS.filter((h) => !h.since || year >= h.since).map((h) => ({
    id: h.id,
    name: h.name,
    type: h.type,
    lunar: false,
    dates: [`${pad(year, 4)}-${pad(h.month)}-${pad(h.day)}`]
  }));
  for (const h of LUNAR_HOLIDAYS) {
    const month = findMonth(lunarYear, h.monthIndex);
    holidays.push({
      id: h.id,
      name: h.name,
      type: h.type,
      lunar: true,
      dates: h.dayNumbers.map((n) => isoFromJdn(month.start + n))
    });
  }
  const newYear = getKhmerNewYear(year);
  holidays.push({
    id: "khmer-new-year",
    name: { km: "\u1796\u17B7\u1792\u17B8\u1794\u17BB\u178E\u17D2\u1799\u1785\u17BC\u179B\u1786\u17D2\u1793\u17B6\u17C6\u1794\u17D2\u179A\u1796\u17C3\u178E\u17B8\u1787\u17B6\u178F\u17B7", en: "Khmer New Year" },
    type: "public",
    lunar: true,
    dates: newYear.schedule.map((d) => d.date)
  });
  holidays.sort((a, b) => a.dates[0].localeCompare(b.dates[0]));
  cache2.set(year, holidays);
  return holidays;
}
function getHolidayOccurrences(year) {
  return getHolidays(year).flatMap((h) => h.dates.map((date) => ({ id: h.id, name: h.name, type: h.type, date }))).sort((a, b) => a.date.localeCompare(b.date));
}
function getHolidaysOn(date) {
  const year = Number(date.slice(0, 4));
  return getHolidayOccurrences(year).filter((h) => h.date === date);
}

// lib/khmer/count.ts
function calendarDiff(a, b) {
  let years = b.year - a.year;
  let months = b.month - a.month;
  let days = b.day - a.day;
  if (days < 0) {
    months -= 1;
    const prevMonth = b.month === 1 ? 12 : b.month - 1;
    const prevYear = b.month === 1 ? b.year - 1 : b.year;
    days += daysInGregorianMonth(prevYear, prevMonth);
  }
  if (months < 0) {
    years -= 1;
    months += 12;
  }
  return { years, months, days };
}
function daysBetween(from, to, options = {}) {
  const resolved = {
    includeEnd: options.includeEnd ?? false,
    excludeWeekends: options.excludeWeekends ?? false,
    excludeHolidays: options.excludeHolidays ?? false,
    weekend: options.weekend ?? [0, 6]
  };
  const a = jdnFromISO(from);
  const b = jdnFromISO(to);
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const end = resolved.includeEnd ? hi + 1 : hi;
  const absoluteDays = hi - lo;
  const loIso = isoFromJdn(lo);
  const endIso = isoFromJdn(end);
  const holidays = [];
  const publicHolidayDates = /* @__PURE__ */ new Set();
  for (let year = fromJdn(lo).year; year <= fromJdn(hi).year; year++) {
    for (const h of getHolidayOccurrences(year)) {
      if (h.date < loIso || h.date >= endIso) continue;
      holidays.push(h);
      if (h.type === "public") publicHolidayDates.add(h.date);
    }
  }
  const weekendSet = new Set(resolved.weekend);
  let weekendDays = 0;
  let holidayDays = 0;
  let holidaysOnWeekend = 0;
  let holyDays = 0;
  let newLunarMonths = 0;
  for (let jdn = lo; jdn < end; jdn++) {
    const isWeekend = weekendSet.has(weekdayOf(jdn));
    const isHoliday = publicHolidayDates.has(isoFromJdn(jdn));
    if (isWeekend) weekendDays += 1;
    if (isHoliday) holidayDays += 1;
    if (isWeekend && isHoliday) holidaysOnWeekend += 1;
    const { dayNumber, month } = lunarCore(jdn);
    if (holyDayOf(dayNumber, month.length)) holyDays += 1;
    if (dayNumber === 0) newLunarMonths += 1;
  }
  const totalDays = end - lo;
  const excludedWeekends = resolved.excludeWeekends ? weekendDays : 0;
  const excludedHolidays = resolved.excludeHolidays ? holidayDays - (resolved.excludeWeekends ? holidaysOnWeekend : 0) : 0;
  return {
    from,
    to,
    days: b - a,
    absoluteDays,
    weeks: Math.floor(absoluteDays / 7),
    remainingDays: absoluteDays % 7,
    calendar: calendarDiff(fromJdn(lo), fromJdn(hi)),
    options: resolved,
    totalDays,
    countedDays: totalDays - excludedWeekends - excludedHolidays,
    workingDays: totalDays - weekendDays - (holidayDays - holidaysOnWeekend),
    weekendDays,
    holidayDays,
    holidaysOnWeekend,
    holyDays,
    newLunarMonths,
    holidays
  };
}
var COUNTDOWN_HOLIDAYS = {
  "international-new-year": 0,
  "meak-bochea": 0,
  "khmer-new-year": 0,
  "visak-bochea": 0,
  "royal-ploughing": 0,
  "pchum-ben": 1,
  "water-festival": 1
};
function countdown(from) {
  const start = jdnFromISO(from);
  const year = fromJdn(start).year;
  const events = [];
  for (const [id, mainDay] of Object.entries(COUNTDOWN_HOLIDAYS)) {
    for (const y of [year, year + 1]) {
      const holiday = getHolidays(y).find((h) => h.id === id);
      const date = holiday?.dates[mainDay];
      if (holiday && date && date >= from) {
        events.push({ id, name: holiday.name, date, daysUntil: jdnFromISO(date) - start });
        break;
      }
    }
  }
  const found = /* @__PURE__ */ new Set();
  for (let jdn = start; jdn < start + 31 && found.size < 3; jdn++) {
    const { dayNumber, month } = lunarCore(jdn);
    const holy = holyDayOf(dayNumber, month.length);
    const add = (id, name) => {
      if (found.has(id)) return;
      found.add(id);
      events.push({ id, name, date: isoFromJdn(jdn), daysUntil: jdn - start });
    };
    if (holy) add("holy-day", HOLY_DAY_NAME);
    if (dayNumber === 14) add("full-moon", { km: MOON_PHASES.km.full, en: MOON_PHASES.en.full });
    if (dayNumber === month.length - 1) add("new-moon", { km: MOON_PHASES.km.new, en: MOON_PHASES.en.new });
  }
  return events.sort((a, b) => a.daysUntil - b.daysUntil);
}
function addDays(date, days) {
  return getLunarDate(jdnFromISO(date) + days);
}
function dayOfYear(date) {
  const jdn = jdnFromISO(date);
  const { year } = fromJdn(jdn);
  const daysInYear = isGregorianLeapYear(year) ? 366 : 365;
  const gregorianDay = jdn - toJdn(year, 1, 1) + 1;
  const core = lunarCore(jdn);
  let sinceKhmerNewYear = null;
  let sinceBuddhistEraStart = null;
  try {
    const ny = getKhmerNewYear(year).jdn <= jdn ? getKhmerNewYear(year) : getKhmerNewYear(year - 1);
    sinceKhmerNewYear = { day: jdn - ny.jdn + 1, newYearDate: ny.date };
  } catch {
  }
  try {
    const thisYear = buddhistEraStartJdn(year);
    const beStart = thisYear <= jdn ? thisYear : buddhistEraStartJdn(year - 1);
    sinceBuddhistEraStart = { day: jdn - beStart + 1, startDate: isoFromJdn(beStart) };
  } catch {
  }
  return {
    date,
    gregorian: { dayOfYear: gregorianDay, daysInYear, daysLeft: daysInYear - gregorianDay },
    lunarYear: {
      dayOfYear: jdn - core.year.start + 1,
      daysInYear: core.year.length,
      startsOn: isoFromJdn(core.year.start)
    },
    sinceKhmerNewYear,
    sinceBuddhistEraStart
  };
}

// lib/khmer/calendar.ts
function getMonthCalendar(year, month, weekStartsOn = 0) {
  assertYearInRange(year);
  if (!Number.isInteger(month) || month < 1 || month > 12) {
    throw new RangeError("Month must be between 1 and 12");
  }
  const first = toJdn(year, month, 1);
  const length = daysInGregorianMonth(year, month);
  const offset = (weekdayOf(first) - weekStartsOn + 7) % 7;
  const cellCount = Math.ceil((offset + length) / 7) * 7;
  const gridStart = first - offset;
  const holidayCache = /* @__PURE__ */ new Map();
  const holidaysOn = (date, y) => {
    if (!holidayCache.has(y)) {
      try {
        holidayCache.set(y, getHolidayOccurrences(y));
      } catch {
        holidayCache.set(y, []);
      }
    }
    return holidayCache.get(y).filter((h) => h.date === date);
  };
  const days = [];
  for (let i = 0; i < cellCount; i++) {
    const jdn = gridStart + i;
    const ymd = fromJdn(jdn);
    const date = isoFromJdn(jdn);
    let lunar = null;
    try {
      lunar = getLunarDate(jdn);
    } catch {
      lunar = null;
    }
    days.push({
      date,
      ...ymd,
      weekday: weekdayOf(jdn),
      inMonth: ymd.month === month,
      lunar,
      holidays: holidaysOn(date, ymd.year)
    });
  }
  const inMonth = days.filter((d) => d.inMonth);
  const lunarMonths = [];
  for (const d of inMonth) {
    if (!d.lunar) continue;
    const last = lunarMonths[lunarMonths.length - 1];
    if (!last || last.monthIndex !== d.lunar.monthIndex) {
      lunarMonths.push({ monthIndex: d.lunar.monthIndex, beYear: d.lunar.beYear });
    }
  }
  const weeks = [];
  for (let i = 0; i < days.length; i += 7) weeks.push(days.slice(i, i + 7));
  return {
    year,
    month,
    weekStartsOn,
    weeks,
    lunarMonths,
    holidays: inMonth.flatMap((d) => d.holidays),
    holyDays: inMonth.filter((d) => d.lunar?.holyDay).map((d) => d.date)
  };
}

// lib/khmer/format.ts
function formatGregorian(date, lang = "km", withWeekday = false) {
  const [year, month, day] = date.split("-").map(Number);
  const weekday = WEEKDAYS[lang][weekdayOf(toJdn(year, month, day))];
  const monthName = GREGORIAN_MONTHS[lang][month - 1];
  if (lang === "km") {
    const text2 = `\u1791\u17B8${toKhmerNumber(day)} \u1781\u17C2${monthName} \u1786\u17D2\u1793\u17B6\u17C6${toKhmerNumber(year)}`;
    return withWeekday ? `\u1790\u17D2\u1784\u17C3${weekday} ${text2}` : `\u1790\u17D2\u1784\u17C3${text2}`;
  }
  const text = `${day} ${monthName} ${year}`;
  return withWeekday ? `${weekday}, ${text}` : text;
}
var DEFAULT_FORMAT = {
  km: "\u1790\u17D2\u1784\u17C3W dN \u1781\u17C2m \u1786\u17D2\u1793\u17B6\u17C6a e \u1796\u17BB\u1791\u17D2\u1792\u179F\u1780\u179A\u17B6\u1787 b",
  en: "W, d N [of] m, [Year of the] a, e, [BE] b"
};
var SHORT_FORMAT = {
  km: "dN \u1781\u17C2m",
  en: "d N m"
};
var TOKENS = ["as", "ds", "W", "w", "d", "D", "N", "n", "m", "M", "a", "e", "b", "j", "c"];
var TOKEN_PATTERN = new RegExp(`\\[([^\\]]+)\\]|(${TOKENS.join("|")})`, "g");
function formatLunar(date, lang = "km", pattern = DEFAULT_FORMAT[lang]) {
  const num = (n) => lang === "km" ? toKhmerNumber(n) : String(n);
  const values = {
    W: () => WEEKDAYS[lang][date.weekday],
    w: () => WEEKDAYS_SHORT[lang][date.weekday],
    d: () => num(date.lunarDay),
    D: () => num(String(date.lunarDay).padStart(2, "0")),
    N: () => PHASES[lang][date.phase],
    n: () => PHASES_SHORT[lang][date.phase],
    m: () => LUNAR_MONTHS[lang][date.monthIndex],
    a: () => ANIMAL_YEARS[lang][date.animalYear],
    as: () => ANIMAL_EMOJIS[date.animalYear],
    e: () => SAKS[lang][date.sak],
    b: () => num(date.beYear),
    j: () => num(date.jsYear),
    ds: () => num(date.day),
    M: () => GREGORIAN_MONTHS[lang][date.month - 1],
    c: () => num(date.year)
  };
  return pattern.replace(
    TOKEN_PATTERN,
    (_, literal, token) => literal ?? values[token]()
  );
}
function serializeLunarDate(date, lang = "km") {
  return {
    date: date.date,
    weekday: { index: date.weekday, name: WEEKDAYS[lang][date.weekday] },
    gregorian: {
      year: date.year,
      month: date.month,
      day: date.day,
      monthName: GREGORIAN_MONTHS[lang][date.month - 1]
    },
    lunar: {
      day: date.lunarDay,
      phase: date.phase,
      phaseName: PHASES[lang][date.phase],
      dayOfMonth: date.dayNumber + 1,
      month: {
        index: date.monthIndex,
        name: LUNAR_MONTHS[lang][date.monthIndex],
        days: date.monthLength
      },
      leapType: date.leapType,
      leapTypeName: LEAP_TYPES[lang][date.leapType]
    },
    year: {
      buddhistEra: date.beYear,
      jolakSakaraj: date.jsYear,
      animal: {
        index: date.animalYear,
        name: ANIMAL_YEARS[lang][date.animalYear],
        emoji: ANIMAL_EMOJIS[date.animalYear]
      },
      sak: { index: date.sak, name: SAKS[lang][date.sak] }
    },
    holyDay: date.holyDay ? { kind: date.holyDay, name: HOLY_DAY_NAME[lang] } : null,
    moonPhase: date.moonPhase ? { kind: date.moonPhase, name: MOON_PHASES[lang][date.moonPhase] } : null,
    holidays: getHolidaysOn(date.date).map((h) => ({ id: h.id, name: h.name[lang], type: h.type })),
    formatted: formatLunar(date, lang),
    formattedShort: formatLunar(date, lang, SHORT_FORMAT[lang])
  };
}
export {
  ANIMAL_EMOJIS,
  ANIMAL_YEARS,
  DEFAULT_FORMAT,
  GREGORIAN_MONTHS,
  HOLY_DAY_NAME,
  LEAP_TYPES,
  LUNAR_MONTHS,
  MAX_YEAR,
  MIN_YEAR,
  MOON_PHASES,
  MonthIndex,
  PHASES,
  PHASES_SHORT,
  SAKS,
  SHORT_FORMAT,
  WEEKDAYS,
  WEEKDAYS_SHORT,
  addDays,
  assertYearInRange,
  buddhistEraStartJdn,
  countdown,
  dayOfYear,
  daysBetween,
  daysInGregorianMonth,
  daysInLunarYear,
  findLunarYear,
  findMonth,
  formatGregorian,
  formatLunar,
  fromJdn,
  getAharkun,
  getAvoman,
  getBodithey,
  getHolidayOccurrences,
  getHolidays,
  getHolidaysOn,
  getKhmerNewYear,
  getKromthupul,
  getLeapType,
  getLunarDate,
  getLunarYear,
  getMonthCalendar,
  holyDayOf,
  isGregorianLeapYear,
  isKhmerSolarLeap,
  isoFromJdn,
  jdnFromISO,
  khmerToGregorian,
  lunarCore,
  monthLength,
  moonPhaseOf,
  pad,
  parseISODate,
  serializeLunarDate,
  toISO,
  toJdn,
  toKhmerNumber,
  todayISO,
  weekdayOf
};
