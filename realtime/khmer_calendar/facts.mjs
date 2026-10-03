// Prints the Khmer calendar facts for live voice: today and tomorrow in the lunar calendar, upcoming holidays
// and the next Khmer New Year. Copied from mobile-app's server/khmer-calendar.js (keep the two in step).
import { calendarContext } from "./khmer-calendar.mjs";

process.stdout.write(calendarContext("today calendar"));
