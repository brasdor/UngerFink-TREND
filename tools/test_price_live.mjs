/**
 * Live check that /price and /24h can actually reach Binance.
 *
 * The stubbed worker test could not catch the 2026-08-28 failure: it faked
 * the Binance response, so it never exercised the real URL contract. The bug
 * was in URL construction -- encodeURIComponent() escaped the brackets of the
 * `symbols` parameter, Binance answered HTTP 400 to every request, and every
 * coin rendered as "n/a" in Telegram while the tests stayed green.
 *
 * This hits the real API with the worker's own fetchTickers(), so a
 * regression in how that URL is built fails here instead of in the bot.
 *
 * Usage:  node tools/test_price_live.mjs
 * Needs network. Read-only, unauthenticated, no keys.
 */
import { fetchTickers, normalizeSymbol } from "../cloudflare/status_worker.js";

const SPOT = ["BTC", "ETH", "SOL", "NEAR"].map(normalizeSymbol);
const FUTURES_ONLY = ["BTCDOMUSDT", "1000PEPEUSDT"];

let failed = false;

function check(label, condition, detail) {
  console.log(`${condition ? "  ok  " : "  FAIL"}  ${label}${detail ? "  -- " + detail : ""}`);
  if (!condition) failed = true;
}

console.log("\nnormalizeSymbol");
check("bare BTC gains its quote asset", normalizeSymbol("BTC") === "BTCUSDT", normalizeSymbol("BTC"));
check("repo form BTC_USDT collapses", normalizeSymbol("BTC_USDT") === "BTCUSDT", normalizeSymbol("BTC_USDT"));
check("full pair ETHBTC left alone", normalizeSymbol("ETHBTC") === "ETHBTC", normalizeSymbol("ETHBTC"));

console.log("\nspot tickers (live)");
const spot = await fetchTickers(SPOT);

// Binance blocks datacenter IPs: 403 from Cloudflare, 451 from GitHub's US
// runners. That is the environment, not a defect, and this check runs in both
// places -- so treat it as "cannot verify here" rather than failing forever
// and training everyone to ignore a red step. A malformed request (400) is a
// real bug and still fails, which is the case that shipped broken today.
const blocked = spot.errors.some((e) => /HTTP (403|451)/.test(e));
if (blocked) {
  console.log("  skip  Binance is blocked from this IP (403/451) -- cannot verify here");
  console.log(`        ${spot.errors.join("; ")}`);
  console.log("\nRESULT: SKIPPED (environment blocked, not a code failure)\n");
  process.exit(0);
}

check("no transport errors", spot.errors.length === 0, spot.errors.join("; "));
for (const sym of SPOT) {
  const t = spot.tickers[sym];
  check(sym, !!t, t ? `$${t.lastPrice}  ${t.priceChangePercent}%` : "no row returned");
}

console.log("\nfutures-only symbols (live)");
const fut = await fetchTickers(FUTURES_ONLY);
for (const sym of FUTURES_ONLY) {
  const t = fut.tickers[sym];
  // Not fatal: these resolve only if fapi is reachable from where this runs.
  console.log(`  ${t ? "ok  " : "note"}  ${sym}  ${t ? "$" + t.lastPrice : "unresolved (fapi unreachable here?)"}`);
}

console.log("\nmixed batch, the shape /price actually sends");
const requested = [...SPOT, ...FUTURES_ONLY];
const mixed = await fetchTickers(requested);
// Count only what was asked for. fapi returns all 751 symbols regardless of
// any filter, so Object.keys() on the whole map reports a meaningless 751.
const resolved = requested.filter((s) => mixed.tickers[s]).length;
check("every requested symbol resolves", resolved === requested.length,
      `${resolved}/${requested.length} resolved`);
check("no stray symbols leak into the map beyond those requested",
      Object.keys(mixed.tickers).every((s) => requested.includes(s)),
      `map holds ${Object.keys(mixed.tickers).length} keys`);

console.log(failed ? "\nRESULT: FAIL\n" : "\nRESULT: PASS\n");
process.exit(failed ? 1 : 0);
