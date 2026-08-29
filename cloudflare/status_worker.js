/**
 * UngerFink /status — Cloudflare Worker (always-on, serverless, free).
 *
 * Receives Telegram webhook updates and answers /status by reading the latest
 * CSVs committed in the GitHub repo (no PC, no server to keep alive).
 *
 * Does NOT show the exchange balance — that needs API keys, which must never
 * live in a public worker. Balance stays in the local bot / Trade Desk.
 *
 * Environment variables (set in the Worker dashboard → Settings → Variables):
 *   BOT_TOKEN       (secret)  Telegram bot token
 *   GH_TOKEN        (secret)  GitHub fine-grained PAT, Contents: Read-only
 *   GH_REPO                   e.g. "brasdor/UngerFink-TREND"
 *   GH_BRANCH                 e.g. "master"
 *   ALLOWED_CHAT              chat id allowed to use /status, or "*" for anyone
 *   WEBHOOK_SECRET  (secret)  random string; must match Telegram setWebhook secret_token
 */

const SYSTEMS = [
  ["Donchian", "data/t9b_paper"],
  ["RSI-MR", "data/t9b_mr_paper"],
  ["ConsecDown", "data/t9b_consecdowndays_paper"],
  ["Candidate12", "data/t9_candidate12_paper"],
  ["Candidate19", "data/t9_candidate19_paper"],
];

// Full 9-system order for /status specifically -- /positions and /pnl above
// intentionally keep using the narrower SYSTEMS list unchanged.
const STATUS_SYSTEM_ORDER = [
  ["S1", "Donchian"],
  ["S2", "RSI-MR"],
  ["S3", "ConsecDown"],
  ["S5", "Momentum"],
  ["S6", "VolContraction"],
  ["S7", "MACross"],
  ["S8", "RSI-MR-Funding"],
  ["Candidate12", "Candidate 12"],
  ["Candidate19", "Candidate 19"],
];

// Every directory that can hold open positions, for /price and /24h with no
// arguments ("the coins we're actually in"). Wider than SYSTEMS, which stays
// as-is so /positions and /pnl keep their existing output.
const POSITION_DIRS = [
  ["S1 Donchian", "data/t9b_paper"],
  ["S2 RSI-MR", "data/t9b_mr_paper"],
  ["S3 ConsecDown", "data/t9b_consecdowndays_paper"],
  ["S5 Momentum", "data/t9b_momentum_paper"],
  ["S6 VolContraction", "data/t9b_volcontraction_paper"],
  ["S7 MACross", "data/t9b_macross_paper"],
  ["S8 RSI-MR-Funding", "data/t9b_rsi_mr_funding_paper"],
  ["Candidate12", "data/t9_candidate12_paper"],
  ["Candidate19", "data/t9_candidate19_paper"],
];

// Telegram hard-limits a message to 4096 chars; keep a margin for the footer.
const TELEGRAM_MAX_SYMBOLS = 60;

/* One registry for the per-system commands (/s1, /donchian, /macross, ...).
 * "id" matches the key in status_snapshot.json so a system command can reuse
 * the snapshot the heartbeat already builds instead of recomputing anything.
 * Aliases are matched after stripping non-alphanumerics, so "/rsi_mr",
 * "/rsimr" and "/RSI-MR" all land on S2. */
const SYSTEM_REGISTRY = [
  { id: "S1", name: "Donchian", dir: "data/t9b_paper",
    aliases: ["s1", "donchian"] },
  { id: "S2", name: "RSI-MR", dir: "data/t9b_mr_paper",
    aliases: ["s2", "rsimr", "rsi", "meanreversion", "mr"] },
  { id: "S3", name: "ConsecDown", dir: "data/t9b_consecdowndays_paper",
    aliases: ["s3", "consecdown", "consecdowndays", "cd"] },
  { id: "S5", name: "Momentum", dir: "data/t9b_momentum_paper",
    aliases: ["s5", "momentum", "mom"] },
  { id: "S6", name: "VolContraction", dir: "data/t9b_volcontraction_paper",
    aliases: ["s6", "volcontraction", "volcont", "vc"] },
  { id: "S7", name: "MACross", dir: "data/t9b_macross_paper",
    aliases: ["s7", "macross", "mac"] },
  { id: "S8", name: "RSI-MR-Funding", dir: "data/t9b_rsi_mr_funding_paper",
    aliases: ["s8", "rsimrfunding", "funding", "rsifunding"] },
  { id: "Candidate12", name: "Candidate 12", dir: "data/t9_candidate12_paper",
    aliases: ["c12", "candidate12", "cand12"] },
  { id: "Candidate19", name: "Candidate 19", dir: "data/t9_candidate19_paper",
    aliases: ["c19", "candidate19", "cand19"] },
];

function findSystem(command) {
  const key = command.replace(/^\//, "").replace(/[^a-z0-9]/g, "");
  if (!key) return null;
  return SYSTEM_REGISTRY.find((s) => s.aliases.includes(key)) || null;
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("UngerFink status worker");

    // Verify the request really came from Telegram (secret token header).
    if (env.WEBHOOK_SECRET) {
      const got = request.headers.get("X-Telegram-Bot-Api-Secret-Token");
      if (got !== env.WEBHOOK_SECRET) return new Response("forbidden", { status: 403 });
    }

    let update;
    try {
      update = await request.json();
    } catch (e) {
      return new Response("ok");
    }

    const msg = update.message || update.edited_message;
    if (!msg || !msg.text) return new Response("ok");

    const chatId = String(msg.chat.id);
    const text = msg.text.trim().toLowerCase();

    const allowed = env.ALLOWED_CHAT || "*";
    if (allowed !== "*" && !allowed.split(",").map((s) => s.trim()).includes(chatId)) {
      return new Response("ok"); // ignore other chats
    }

    // "/price btc eth" and "/price@ungertrend_bot btc" both parse to
    // command "/price" with args ["btc","eth"].
    const parts = text.split(/\s+/);
    const command = parts[0].split("@")[0];
    const args = parts.slice(1).filter(Boolean);

    if (command === "/status") {
      const status = await buildStatus(env);
      await sendMessage(env.BOT_TOKEN, chatId, status);
    } else if (command.startsWith("/pos")) {
      const pos = await buildPositions(env);
      await sendMessage(env.BOT_TOKEN, chatId, pos);
    } else if (command === "/pnl") {
      const pnl = await buildPnl(env);
      await sendMessage(env.BOT_TOKEN, chatId, pnl);
    } else if (command === "/price" || command === "/p") {
      const price = await buildPrice(env, args);
      await sendMessage(env.BOT_TOKEN, chatId, price);
    } else if (command === "/24h" || command === "/24") {
      const moves = await build24h(env, args);
      await sendMessage(env.BOT_TOKEN, chatId, moves);
    } else if (command === "/systems") {
      await sendMessage(env.BOT_TOKEN, chatId, systemsHelp());
    } else if (command === "/diag") {
      await sendMessage(env.BOT_TOKEN, chatId, await buildDiag());
    } else if (command === "/start" || command === "/help") {
      await sendMessage(env.BOT_TOKEN, chatId,
        "Commands:\n" +
        "/status — all 9 systems, regime, and alerts in one view\n" +
        "/positions — each position with P&L\n" +
        "/pnl — total P&L summary\n" +
        "/price — USD price of every coin currently held\n" +
        "/price BTC ETH — USD price of specific coins\n" +
        "/24h — 24h move of every coin held, biggest mover first\n" +
        "/24h SOL — 24h move of specific coins\n" +
        "/systems — list the per-system commands\n" +
        "/donchian, /rsimr, /macross, /s1 … /s8, /c12, /c19 —\n" +
        "   one system in detail: equity, positions, today's signals");
    } else {
      // Per-system commands last: they must not shadow the ones above.
      const sys = findSystem(command);
      if (sys) {
        const detail = await buildSystem(env, sys);
        await sendMessage(env.BOT_TOKEN, chatId, detail);
      }
    }
    return new Response("ok");
  },
};

function parseCsv(textCsv) {
  const lines = textCsv.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, i) => (row[h] = (cells[i] || "").trim()));
    return row;
  });
}

async function ghFile(env, path) {
  const branch = env.GH_BRANCH || "master";
  const url = `https://api.github.com/repos/${env.GH_REPO}/contents/${path}?ref=${branch}`;
  const resp = await fetch(url, {
    headers: {
      Authorization: `Bearer ${env.GH_TOKEN}`,
      "User-Agent": "ungerfink-status-worker",
      Accept: "application/vnd.github.raw",
    },
  });
  if (!resp.ok) return null;
  return await resp.text();
}

// Formats a UTC ISO string as "YYYY-MM-DD HH:MM UTC", or "?" if missing/bad.
function fmtUtc(iso) {
  if (!iso) return "?";
  try {
    return iso.slice(0, 16).replace("T", " ") + " UTC";
  } catch (e) {
    return "?";
  }
}

function money(n, opts) {
  const v = typeof n === "number" && Number.isFinite(n) ? n : null;
  if (v === null) return "n/a";
  // Keep the sign outside the $ in both directions: a negative used to render
  // "$-6.62" next to a positive "+$12.55", which reads as two different
  // formats in the same column.
  const sign = v < 0 ? "-" : (opts && opts.signed ? "+" : "");
  return `${sign}$${Math.abs(v).toFixed(2)}`;
}

// The single consolidated view: all 9 systems' equity/positions/P&L/
// kill-switch, current regime + per-system weight, and an alert summary --
// everything that otherwise requires checking regime_state.json,
// check_missed_runs.py's own alerting, and 9 separate state.json files by
// hand. Reads ONE file (data/status_snapshot.json), built once daily by
// .github/scripts/build_status_snapshot.py (a step in heartbeat_check.yml,
// timed after both daily engine workflows finish) -- computing this live
// here would mean hundreds of GitHub API calls per /status message
// (per-symbol OHLCV staleness across 290+66 files), which a Worker's
// subrequest budget can't sustain. See that script's docstring for why.
async function buildStatus(env) {
  const raw = await ghFile(env, "data/status_snapshot.json");
  if (!raw) {
    return "⚠️ <b>status_snapshot.json not found.</b>\nHas heartbeat_check.yml run yet? " +
           "(daily, 11:00 UTC — 3h after the main engine workflows).";
  }
  let snap;
  try {
    snap = JSON.parse(raw);
  } catch (e) {
    return "⚠️ status_snapshot.json exists but isn't valid JSON.";
  }

  const r = snap.regime || {};
  const weights = r.weights || {};
  const systems = snap.systems || {};

  let totalEquity = 0;
  let totalUnrealized = 0;
  let anyUnrealizedNA = false;
  let killed = [];

  // One row per system, columns aligned: name, equity, open count, unrealized.
  const rows = [];
  for (const [sid, label] of STATUS_SYSTEM_ORDER) {
    // The candidates' id already reads as their name, so printing id + label
    // gave "Candidate12 Candidate 12". S1..S8 still want both.
    const name = sid.startsWith("Candidate") ? label.replace(" ", "") : `${sid} ${label}`;
    const s = systems[sid];
    if (!s) {
      rows.push(padEnd(name, 14) + padStart("no data", 10));
      continue;
    }
    const w = weights[sid];
    // No "$" in the cells -- the header names the columns and every value is
    // USD, so the symbol only costs width that a phone does not have.
    const unreal = typeof s.unrealized_pnl === "number"
      ? signed(s.unrealized_pnl, 0)
      : (anyUnrealizedNA = true, "n/a");
    rows.push(
      padEnd(name, 14) +
      padStart(Math.round(s.equity || 0).toLocaleString("en-US"), 7) +
      padStart(String(s.open_positions), 4) +
      padStart(unreal, 8) +
      padStart(typeof w === "number" ? `${(w * 100).toFixed(0)}%` : "-", 5)
    );
    if (s.kill_switch) killed.push(sid);
    totalEquity += s.equity || 0;
    if (typeof s.unrealized_pnl === "number") totalUnrealized += s.unrealized_pnl;
  }

  const lines = [`\u{1F4CA} <b>UngerFink status</b>`];
  // Headline first: the two numbers that matter, then the regime, then detail.
  const equityTxt = "$" + Math.round(totalEquity).toLocaleString("en-US");
  const unrealTxt = signed(totalUnrealized, 0);
  lines.push(`<b>${equityTxt}</b> equity · <b>${unrealTxt}</b> unrealized` +
             (anyUnrealizedNA ? " <i>(partial)</i>" : ""));
  lines.push(`<i>${r.trend || "?"} trend · ${r.funding || "?"} funding · ` +
             `vol×${r.vol_multiplier ?? "?"} · as of ${r.date || "?"}</i>`);
  if (killed.length) lines.push(`\u{1F6D1} <b>KILL-SWITCH: ${killed.join(", ")}</b>`);
  lines.push("");
  lines.push(preBlock([
    padEnd("SYSTEM", 14) + padStart("EQUITY", 7) + padStart("OPN", 4) +
    padStart("UNREAL", 8) + padStart("WGT", 5),
    ...rows,
  ]));
  lines.push("");

  // --- Alerts ---
  const a = snap.alerts || {};
  const missed = a.missed_runs || [];
  const issues = a.ohlcv_issues || [];
  if (missed.length === 0 && issues.length === 0) {
    lines.push("\u{2705} <b>Alerts</b> — all clear");
  } else {
    lines.push(`\u{26A0} <b>Alerts</b> — ${missed.length + issues.length} open`);
    for (const m of missed) lines.push(`  • ${m.system}: ${m.detail}`);
    for (const iss of issues) lines.push(`  • ${iss}`);
  }
  if ((a.suppressed_symbols || []).length) {
    lines.push(`<i>suppressed (halted): ${a.suppressed_symbols.join(", ")}</i>`);
  }
  lines.push(`<i>snapshot ${fmtUtc(snap.generated_utc)} · checked ${fmtUtc(a.checked_utc)}</i>`);

  return lines.join("\n");
}

async function buildPositions(env) {
  const now = new Date().toISOString().slice(0, 16).replace("T", " ") + " UTC";
  const lines = [`\u{1F4B0} <b>Positions</b> — ${now}`, ""];

  for (const [label, dir] of SYSTEMS) {
    const mtm = await ghFile(env, `${dir}/mtm_positions.csv`);
    if (!mtm) {
      lines.push(`<b>${label}</b>: no data yet`);
      lines.push("");
      continue;
    }
    const rows = parseCsv(mtm);
    if (rows.length === 0) {
      lines.push(`<b>${label}</b>: no open positions`);
      lines.push("");
      continue;
    }
    lines.push(`<b>${label}</b> (${rows.length}):`);
    for (const r of rows) {
      const pnl = parseFloat(r.unrealized_pnl) || 0;
      const pct = parseFloat(r.pnl_pct) || 0;
      const arrow = pnl >= 0 ? "\u{2B06}" : "\u{2B07}";
      const sign = pnl >= 0 ? "+" : "";
      lines.push(
        `  ${r.symbol}: $${fmt(r.cost_usdt)} → $${fmt(r.market_value_usdt)} ` +
        `(${sign}$${fmt(r.unrealized_pnl)}, ${sign}${pct.toFixed(1)}%) ${arrow}`
      );
    }
    lines.push("");
  }
  return lines.join("\n");
}

async function buildPnl(env) {
  const now = new Date().toISOString().slice(0, 16).replace("T", " ") + " UTC";
  const lines = [`\u{1F4C8} <b>P&L Summary</b> — ${now}`, ""];

  let grandCost = 0;
  let grandValue = 0;
  let grandPnl = 0;
  let grandPositions = 0;

  for (const [label, dir] of SYSTEMS) {
    const eq = await ghFile(env, `${dir}/equity_curve.csv`);
    if (!eq) {
      lines.push(`<b>${label}</b>: no data yet`);
      continue;
    }
    const rows = parseCsv(eq);
    if (rows.length === 0) continue;
    const latest = rows[rows.length - 1];
    const paperEq = parseFloat(latest.paper_equity) || 10000;
    const unrealized = parseFloat(latest.unrealized_pnl) || 0;
    const totalVal = parseFloat(latest.total_value) || paperEq;
    const cost = parseFloat(latest.total_cost) || 0;
    const mktVal = parseFloat(latest.total_market_value) || 0;
    const nPos = parseInt(latest.open_positions) || 0;
    const sign = unrealized >= 0 ? "+" : "";

    lines.push(`<b>${label}</b>: ${sign}$${unrealized.toFixed(2)} (${nPos} pos)`);
    lines.push(`   invested $${cost.toFixed(2)} → worth $${mktVal.toFixed(2)}`);
    lines.push(`   total value: $${totalVal.toFixed(2)}`);

    grandCost += cost;
    grandValue += mktVal;
    grandPnl += unrealized;
    grandPositions += nPos;
  }

  const grandSign = grandPnl >= 0 ? "+" : "";
  lines.push("");
  lines.push(`<b>Total</b>: ${grandSign}$${grandPnl.toFixed(2)} across ${grandPositions} positions`);
  lines.push(`   invested $${grandCost.toFixed(2)} → worth $${grandValue.toFixed(2)}`);

  return lines.join("\n");
}

function fmt(s) {
  const n = parseFloat(s) || 0;
  return n.toFixed(2);
}

/* ── /price and /24h ─────────────────────────────────────────────────────
 *
 * Both answer from one Binance endpoint: /ticker/24hr carries lastPrice AND
 * priceChangePercent, so the two commands share a single fetch rather than
 * hitting the API twice for the same row.
 *
 * Spot is tried first, then futures for whatever spot didn't return -- the
 * book holds plenty of futures-only symbols (BTCDOMUSDT, 1000PEPEUSDT, ...)
 * that simply do not exist on spot. Note the daily workflows see HTTP 451 on
 * fapi.binance.com from GitHub's US runners; the Worker egresses from
 * Cloudflare's edge instead, so it is not necessarily blocked -- but if it
 * is, the futures leg degrades to "n/a" per symbol rather than failing the
 * whole command.
 */

const QUOTE_ASSETS = ["USDT", "USDC", "BUSD", "BTC", "ETH"];

// "BTC_USDT" (repo spot) / "btc" (user shorthand) -> "BTCUSDT" (Binance).
function normalizeSymbol(raw) {
  let s = String(raw || "").trim().toUpperCase().replace(/[_\-\/]/g, "");
  if (!s) return "";
  // A quote suffix only counts when something precedes it: bare "BTC" is the
  // base asset the user wants priced (-> BTCUSDT), whereas "ETHBTC" is
  // already a full pair. Testing endsWith alone left /price BTC as "BTC",
  // which Binance does not know, and every major returned n/a.
  const hasQuote = QUOTE_ASSETS.some((q) => s.length > q.length && s.endsWith(q));
  if (!hasQuote) s += "USDT";
  return s;
}

// Display form: strip the USDT quote so lists stay narrow on a phone.
function displaySymbol(sym) {
  return sym.replace(/USDT$/, "");
}

async function fetchJson(url) {
  try {
    const res = await fetch(url, { headers: { "User-Agent": "UngerFink-Worker" } });
    if (!res.ok) return { error: `HTTP ${res.status}` };
    return { data: await res.json() };
  } catch (e) {
    return { error: String(e).slice(0, 80) };
  }
}

/**
 * Ticker rows for the given symbols, keyed by symbol.
 * Returns { tickers, errors } -- errors is per-venue, not fatal.
 */
// Binance's WAF answers HTTP 403 to api.binance.com from Cloudflare's
// datacenter IPs (observed 2026-08-28 from this Worker, while the same
// request succeeded from a residential IP). It is the same class of block
// that returns 451 to GitHub's runners, so no amount of fixing the query
// helps -- the host has to change.
//
// data-api.binance.vision is Binance's public market-data mirror and is the
// one intended for unauthenticated access, so it leads. The rest are regional
// aliases of the same API, tried in turn: whichever the edge can reach wins,
// and if one starts being blocked later the next takes over on its own.
const SPOT_HOSTS = [
  "https://data-api.binance.vision",
  "https://api-gcp.binance.com",
  "https://api1.binance.com",
  "https://api.binance.com",
];

// Futures has no public mirror, so this is the only host. If it is blocked,
// futures-only symbols (BTCDOMUSDT, 1000PEPEUSDT, ...) degrade to "n/a" while
// everything listed on spot still resolves.
const FUTURES_HOSTS = ["https://fapi.binance.com"];

/**
 * GET the first host that answers, so one blocked host does not sink the call.
 * Returns { data, tried } where tried records "host: failure" for diagnosis.
 */
async function fetchFirstReachable(hosts, path) {
  const tried = [];
  for (const host of hosts) {
    const result = await fetchJson(host + path);
    if (result.data) return { data: result.data, tried };
    tried.push(`${host.replace("https://", "")}: ${result.error}`);
  }
  return { data: null, tried };
}

/** One spot request. ok:false means the whole batch was rejected. */
async function fetchSpotBatch(symbols) {
  const path = `/api/v3/ticker/24hr?symbols=${JSON.stringify(symbols)}`;
  const res = await fetchFirstReachable(SPOT_HOSTS, path);
  return { ok: Array.isArray(res.data), rows: res.data || [], tried: res.tried };
}

/** Pull the whole futures ticker list and index the symbols we want from it. */
async function fetchFuturesInto(tickers, wantedList) {
  const wanted = new Set(wantedList);
  const res = await fetchFirstReachable(FUTURES_HOSTS, "/fapi/v1/ticker/24hr");
  if (!Array.isArray(res.data)) return { ok: false, tried: res.tried };
  for (const row of res.data) {
    if (wanted.has(row.symbol)) tickers[row.symbol] = { ...row, venue: "futures" };
  }
  return { ok: true, tried: res.tried };
}

async function fetchTickers(symbols) {
  const tickers = {};
  const errors = [];
  if (symbols.length === 0) return { tickers, errors };

  // Binance matches this parameter against ^\[("SYM"(,"SYM")*)?\]$ WITHOUT
  // percent-decoding it first, so the brackets must arrive literal, and no
  // spaces are allowed between entries. encodeURIComponent() escaped the
  // brackets to %5B/%5D and every request came back HTTP 400 -- which
  // surfaced as every coin showing "n/a". JSON.stringify is correct on both
  // counts: it emits no spaces, and fetch percent-encodes only the quotes
  // (to %22), a form Binance accepts.
  // Binance rejects the ENTIRE batch with 400 "Invalid symbol" if even one
  // entry is not a valid spot symbol -- verified 2026-08-28. The book holds
  // futures-only names (BTCDOMUSDT, 1000RATSUSDT, AVAAIUSDT), so a single one
  // of them made all 60 coins come back empty. This is what actually broke
  // /price in production; the earlier encoding and host fixes were both real
  // but neither was sufficient on their own.
  let spot = await fetchSpotBatch(symbols);

  if (!spot.ok) {
    // Most likely one or more symbols are futures-only. Resolve those from
    // the futures list first -- one request removes the whole class of
    // offenders -- then ask spot again for what is left.
    const futFirst = await fetchFuturesInto(tickers, symbols);
    if (!futFirst.ok) errors.push(`futures unreachable — ${futFirst.tried.join(", ")}`);

    const remaining = symbols.filter((s) => !tickers[s]);
    if (remaining.length > 0) {
      spot = await fetchSpotBatch(remaining);

      // Still rejected: something in the remainder is unknown to spot and
      // futures could not tell us which (its host is blocked too). Bisect to
      // isolate the offenders, so the good symbols still resolve.
      //
      // Fixed-size chunking is not enough here: with fewer symbols than the
      // chunk size everything lands in one group that still contains the bad
      // entry, and nothing is gained. Halving always converges on it.
      if (!spot.ok) {
        let budget = 24;   // cap requests; a Worker gets 50 subrequests total
        const resolve = async (group) => {
          if (group.length === 0 || budget <= 0) return;
          budget -= 1;
          const part = await fetchSpotBatch(group);
          if (part.ok) {
            for (const row of part.rows) tickers[row.symbol] = { ...row, venue: "spot" };
            return;
          }
          if (group.length === 1) return;   // this one symbol is the bad one
          const mid = Math.floor(group.length / 2);
          await resolve(group.slice(0, mid));
          await resolve(group.slice(mid));
        };
        await resolve(remaining);

        if (Object.keys(tickers).length === 0) {
          errors.push(`spot unreachable — ${spot.tried.join(", ")}`);
        }
        return { tickers, errors };
      }
    }
  }

  for (const row of spot.rows) tickers[row.symbol] = { ...row, venue: "spot" };

  // Whatever spot did not return is either futures-only or delisted.
  //
  // Note fapi does NOT honour a `symbols` filter -- measured 2026-08-28, it
  // returns all 751 symbols (~280KB) whether the parameter is present or not,
  // unlike the spot endpoint. So there is no point sending one, and the rows
  // we want get picked out here instead. 280KB parses well inside the
  // Worker's 10ms CPU budget; the spot dump (3,688 symbols, ~1.9MB) would
  // not, which is why spot is filtered server-side and futures is not.
  const missing = symbols.filter((s) => !tickers[s]);
  if (missing.length > 0) {
    const fut = await fetchFuturesInto(tickers, missing);
    if (!fut.ok) errors.push(`futures unreachable — ${fut.tried.join(", ")}`);
  }
  return { tickers, errors };
}

/**
 * Symbols we currently hold, de-duplicated, with the last price the daily
 * workflow marked them at.
 *
 * That marked price is the fallback when Binance is unreachable. It is the
 * same number /pnl reports -- mark_to_market.py computed it from OHLCV files
 * committed in the repo -- so it needs no exchange call and cannot be
 * blocked. It is a daily close, not a live quote, and is labelled as such
 * wherever it is shown.
 *
 * Reads mtm_positions.csv, falling back to open_positions.csv (which carries
 * current_price for the momentum-style engines) when mark-to-market has not
 * run for that system.
 */
async function heldSymbols(env) {
  const seen = new Map();   // normalized symbol -> { systems: [], marked: number|null }
  for (const [label, dir] of POSITION_DIRS) {
    let csv = await ghFile(env, `${dir}/mtm_positions.csv`);
    if (!csv) csv = await ghFile(env, `${dir}/open_positions.csv`);
    if (!csv) continue;
    for (const row of parseCsv(csv)) {
      const sym = normalizeSymbol(row.symbol);
      if (!sym) continue;
      if (!seen.has(sym)) seen.set(sym, { systems: [], marked: null });
      const entry = seen.get(sym);
      entry.systems.push(label.split(" ")[0]);   // "S7 MACross" -> "S7"
      const marked = parseFloat(row.current_price);
      if (isFinite(marked) && marked > 0) entry.marked = marked;
    }
  }
  return seen;
}

/**
 * Resolve the symbol list for a /price or /24h invocation.
 * With arguments: exactly those symbols. Without: everything we hold.
 */
async function resolveSymbols(env, args) {
  if (args.length > 0) {
    const explicit = new Map();
    for (const a of args) {
      const sym = normalizeSymbol(a);
      // No marked price for an arbitrary symbol we do not hold -- those can
      // only come from a live source.
      if (sym && !explicit.has(sym)) explicit.set(sym, { systems: [], marked: null });
    }
    return { symbolMap: explicit, fromPositions: false };
  }
  return { symbolMap: await heldSymbols(env), fromPositions: true };
}

function signed(n, digits) {
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
}

/* ── layout ──────────────────────────────────────────────────────────────
 *
 * Telegram renders <pre> in a monospace font and preserves whitespace, which
 * is the only way to get columns that line up. It does NOT wrap inside <pre>:
 * an over-long line scrolls sideways instead, so rows are kept under ~34
 * characters to fit a phone without horizontal scrolling.
 *
 * Symbols are alphanumeric and prices are digits, so nothing here needs HTML
 * escaping; anything free-form must not be put inside these blocks.
 */
const ROW_WIDTH = 34;

function padEnd(text, width) {
  const s = String(text);
  return s.length >= width ? s.slice(0, width) : s + " ".repeat(width - s.length);
}

function padStart(text, width) {
  const s = String(text);
  return s.length >= width ? s.slice(0, width) : " ".repeat(width - s.length) + s;
}

/** Wrap pre-formatted rows as one monospace block, or nothing if empty. */
function preBlock(rows) {
  return rows.length ? `<pre>${rows.join("\n")}</pre>` : "";
}

// Long symbols (1MBABYDOGE) would push every other column out of alignment.
function shortSymbol(sym) {
  const s = displaySymbol(sym);
  return s.length > 11 ? s.slice(0, 10) + "…" : s;
}

// Crypto prices span ~8 orders of magnitude; a fixed precision is unreadable.
function fmtPrice(value) {
  const n = parseFloat(value);
  if (!isFinite(n)) return "n/a";
  if (n >= 1000) return n.toFixed(2);
  if (n >= 1) return n.toFixed(4);
  if (n >= 0.01) return n.toFixed(5);
  return n.toPrecision(4);
}

/**
 * Deep-dive on ONE system: its snapshot line, open positions with live P&L,
 * and today's signals. /status answers "how is everything?"; this answers
 * "what is S7 actually doing right now?".
 */
async function buildSystem(env, sys) {
  const now = new Date().toISOString().slice(0, 16).replace("T", " ") + " UTC";
  // The candidates' id already reads as their name ("Candidate12"), so do not
  // print both and get "Candidate12 Candidate 12".
  const heading = sys.id.startsWith("Candidate") ? sys.name : `${sys.id} ${sys.name}`;
  const lines = [`\u{1F50E} <b>${heading}</b> — ${now}`, ""];

  // --- Headline numbers from the snapshot the heartbeat already builds ---
  const rawSnap = await ghFile(env, "data/status_snapshot.json");
  let snap = null;
  try {
    snap = rawSnap ? JSON.parse(rawSnap) : null;
  } catch (e) {
    snap = null;
  }
  const s = snap && snap.systems ? snap.systems[sys.id] : null;
  if (s) {
    const weight = snap.regime && snap.regime.weights ? snap.regime.weights[sys.id] : undefined;
    lines.push(`Equity: <b>${money(s.equity)}</b>`);
    lines.push(
      `Open: <b>${s.open_positions}</b>   Closed: <b>${s.closed_trades}</b>` +
      (typeof weight === "number" ? `   Weight: <b>${(weight * 100).toFixed(1)}%</b>` : "")
    );
    if (typeof s.unrealized_pnl === "number") {
      lines.push(
        `Unrealized: <b>${money(s.unrealized_pnl, { signed: true })}</b>` +
        (typeof s.today_realized_pnl === "number"
          ? `   Today: <b>${money(s.today_realized_pnl, { signed: true })}</b>`
          : "")
      );
    }
    lines.push(`Last run: ${s.last_run_date || "?"}`);
    if (s.kill_switch) lines.push("\u{1F6D1} <b>KILL-SWITCH ACTIVE</b>");
    lines.push(`<i>snapshot ${fmtUtc(snap.generated_utc)}</i>`);
  } else {
    lines.push("<i>no snapshot entry — falling back to files</i>");
  }
  lines.push("");

  // --- Open positions, with live P&L where mark_to_market has run ---
  const mtm = await ghFile(env, `${sys.dir}/mtm_positions.csv`);
  const rows = mtm ? parseCsv(mtm) : [];
  if (rows.length > 0) {
    lines.push(`<b>Open positions (${rows.length})</b>`);
    for (const r of rows.slice(0, 30)) {
      const pnl = parseFloat(r.unrealized_pnl) || 0;
      const pct = parseFloat(r.pnl_pct) || 0;
      const arrow = pnl >= 0 ? "\u{2B06}" : "\u{2B07}";
      lines.push(
        `  <code>${displaySymbol(normalizeSymbol(r.symbol))}</code>: ` +
        `${money(pnl, { signed: true })} (${signed(pct, 1)}%) ${arrow}`
      );
    }
    if (rows.length > 30) lines.push(`  <i>… ${rows.length - 30} more</i>`);
  } else {
    const openCsv = await ghFile(env, `${sys.dir}/open_positions.csv`);
    const openRows = openCsv ? parseCsv(openCsv) : [];
    if (openRows.length > 0) {
      lines.push(`<b>Open positions (${openRows.length})</b> <i>(no mark-to-market yet)</i>`);
      for (const r of openRows.slice(0, 30)) {
        lines.push(`  <code>${displaySymbol(normalizeSymbol(r.symbol))}</code> @ ${r.entry_price || "?"}`);
      }
    } else {
      lines.push("<b>Open positions</b>: none");
    }
  }
  lines.push("");

  // --- Today's signals ---
  const sigCsv = await ghFile(env, `${sys.dir}/signals_today.csv`);
  const sigRows = sigCsv ? parseCsv(sigCsv) : [];
  if (sigRows.length > 0) {
    lines.push(`<b>Signals today (${sigRows.length})</b>`);
    for (const r of sigRows.slice(0, 20)) {
      // S1 writes initial_stop; every other system writes stop_loss.
      const stop = r.stop_loss || r.initial_stop || "?";
      const close = r.close || r.signal_close || r.entry_price || "?";
      const side = String(r.side || r.signal || "").toUpperCase();
      lines.push(
        `  <code>${displaySymbol(normalizeSymbol(r.symbol))}</code> ${side}` +
        `  close=${fmtPrice(close)}  stop=${fmtPrice(stop)}`
      );
    }
    if (sigRows.length > 20) lines.push(`  <i>… ${sigRows.length - 20} more</i>`);
  } else {
    lines.push("<b>Signals today</b>: none");
  }

  return lines.join("\n");
}

/**
 * Report which external price sources this Worker can actually reach.
 *
 * Binance answers 403 to Cloudflare's datacenter IPs, and every candidate
 * replacement works fine from a residential IP, so testing locally proves
 * nothing about what the edge can do. This probes each candidate from inside
 * the Worker and reports the status, so the choice of source is made on
 * evidence rather than on a third guess.
 */
async function buildDiag() {
  const probes = [
    ["binance spot", "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"],
    ["binance vision", "https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT"],
    ["binance gcp", "https://api-gcp.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"],
    ["binance futures", "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT"],
    ["bybit", "https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT"],
    ["okx", "https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT"],
    ["coingecko", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"],
    ["kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSDT"],
  ];

  const lines = ["\u{1F527} <b>Source reachability</b> — from this Worker", ""];
  for (const [label, url] of probes) {
    let verdict;
    try {
      const res = await fetch(url, { headers: { "User-Agent": "UngerFink-Worker" } });
      verdict = res.ok ? `\u{2705} ${res.status}` : `\u{274C} ${res.status}`;
    } catch (e) {
      verdict = `\u{274C} ${String(e).slice(0, 40)}`;
    }
    lines.push(`  ${verdict}  ${label}`);
  }
  lines.push("", "<i>A green line is a source /price can use. Binance is " +
                 "expected to be blocked here (403) while working fine from " +
                 "a home connection.</i>");
  return lines.join("\n");
}

function systemsHelp() {
  const lines = ["\u{1F5C2} <b>Per-system commands</b>", ""];
  for (const s of SYSTEM_REGISTRY) {
    lines.push(`  <code>/${s.aliases[0]}</code> or <code>/${s.aliases[1]}</code> — ${s.id} ${s.name}`);
  }
  lines.push("", "<i>Each shows equity, open positions with P&L, and today's signals.</i>");
  return lines.join("\n");
}

async function buildPrice(env, args) {
  const { symbolMap, fromPositions } = await resolveSymbols(env, args);
  const symbols = [...symbolMap.keys()];
  if (symbols.length === 0) {
    return fromPositions
      ? "\u{1F4B5} <b>Price</b>\n\nNo open positions right now. Try <code>/price BTC ETH</code>."
      : "\u{1F4B5} <b>Price</b>\n\nNo valid symbols. Try <code>/price BTC ETH SOL</code>.";
  }

  const capped = symbols.slice(0, TELEGRAM_MAX_SYMBOLS);
  const { tickers, errors } = await fetchTickers(capped);

  const now = new Date().toISOString().slice(11, 16) + " UTC";

  // Build rows first so the header can report what actually resolved.
  let live = 0;
  let usedMarked = 0;
  const rows = [];
  for (const sym of capped) {
    const t = tickers[sym];
    const info = symbolMap.get(sym) || { systems: [], marked: null };
    const tag = [...new Set(info.systems)].join(",");

    let priceCell;
    let changeCell;
    if (t) {
      live += 1;
      priceCell = fmtPrice(t.lastPrice);
      changeCell = signed(parseFloat(t.priceChangePercent), 1) + "%";
    } else if (info.marked !== null) {
      // Live quote unavailable -- show the price the daily run marked this
      // position at, flagged so it is never mistaken for a live number.
      usedMarked += 1;
      priceCell = fmtPrice(info.marked);
      changeCell = "mark";
    } else {
      priceCell = "n/a";
      changeCell = "";
    }
    rows.push(padEnd(shortSymbol(sym), 12) + padStart(priceCell, 11) +
              padStart(changeCell, 7) + (tag ? "  " + tag : ""));
  }

  const lines = [`\u{1F4B5} <b>Price</b> · ${now}`];
  // Headline before detail: what you need at a glance, before 60 rows of it.
  const summary = fromPositions
    ? `${capped.length} held · ${live} live · ${usedMarked} marked`
    : `${capped.length} requested · ${live} live`;
  lines.push(`<i>${summary}</i>`);
  // Errors go at the TOP, not in a footer: a 60-coin reply is split across
  // messages, so a trailing diagnostic is the part nobody scrolls to -- which
  // is exactly what happened while every row said "n/a" and the reason sat
  // out of sight at the bottom.
  if (errors.length) lines.push(`\u{26A0} <i>${errors.join(" | ")}</i>`);
  lines.push("");
  lines.push(preBlock([padEnd("SYMBOL", 12) + padStart("PRICE", 11) + padStart("24H", 7), ...rows]));

  if (usedMarked > 0) {
    lines.push(`\n<i>“mark” = last daily close from mark-to-market, ` +
               `not a live quote.</i>`);
  }
  if (symbols.length > capped.length) {
    lines.push(`<i>… ${symbols.length - capped.length} more not shown (message limit)</i>`);
  }
  if (fromPositions) lines.push(systemLegend());
  return lines.join("\n");
}

// The italic tag after each coin is which system holds it. "S1" on its own is
// opaque unless you already know the roster, so spell it out.
function systemLegend() {
  return "\n<i>held by: S1 Donchian · S2 RSI-MR · S3 ConsecDown · S5 Momentum · " +
         "S6 VolContraction · S7 MACross · S8 RSI-MR-Funding · Candidate12/19</i>";
}

async function build24h(env, args) {
  const { symbolMap, fromPositions } = await resolveSymbols(env, args);
  const symbols = [...symbolMap.keys()];
  if (symbols.length === 0) {
    return fromPositions
      ? "\u{1F4C9} <b>24h</b>\n\nNo open positions right now. Try <code>/24h BTC ETH</code>."
      : "\u{1F4C9} <b>24h</b>\n\nNo valid symbols. Try <code>/24h BTC ETH SOL</code>.";
  }

  const capped = symbols.slice(0, TELEGRAM_MAX_SYMBOLS);
  const { tickers, errors } = await fetchTickers(capped);

  // Biggest mover first -- that is the question /24h is actually asking.
  const rows = capped
    .map((sym) => ({ sym, t: tickers[sym] }))
    .filter((r) => r.t)
    .map((r) => ({ ...r, pct: parseFloat(r.t.priceChangePercent) || 0 }))
    .sort((a, b) => b.pct - a.pct);

  const now = new Date().toISOString().slice(11, 16) + " UTC";
  const up = rows.filter((r) => r.pct > 0).length;

  const lines = [`\u{1F4C9} <b>24h change</b> · ${now}`];
  lines.push(rows.length
    ? `<i>${rows.length} coins · ${up} up · ${rows.length - up} down</i>`
    : "<i>no data</i>");
  if (errors.length) lines.push(`\u{26A0} <i>${errors.join(" | ")}</i>`);
  lines.push("");

  const table = rows.map(({ sym, t, pct }) => {
    const info = symbolMap.get(sym) || { systems: [] };
    const tag = [...new Set(info.systems)].join(",");
    return padEnd(shortSymbol(sym), 12) + padStart(signed(pct, 1) + "%", 8) +
           padStart(fmtPrice(t.lastPrice), 11) + (tag ? "  " + tag : "");
  });
  lines.push(preBlock([padEnd("SYMBOL", 12) + padStart("24H", 8) + padStart("PRICE", 11), ...table]));

  // /24h has no fallback: a 24-hour change cannot be derived from a single
  // marked close, so an unreachable source means no rows rather than stale
  // ones presented as current.
  if (rows.length === 0 && errors.length) {
    lines.push("<i>The live price source is unreachable. /price still works " +
               "and falls back to the last marked close.</i>");
  }

  const unresolved = capped.filter((s) => !tickers[s]);
  if (unresolved.length) {
    lines.push(`\n<i>no data (${unresolved.length}): ` +
               `${unresolved.slice(0, 12).map(displaySymbol).join(", ")}` +
               `${unresolved.length > 12 ? ", …" : ""}</i>`);
  }
  if (symbols.length > capped.length) {
    lines.push(`<i>… ${symbols.length - capped.length} more not shown (message limit)</i>`);
  }
  if (fromPositions) lines.push(systemLegend());
  return lines.join("\n");
}

// Telegram rejects anything over 4096 characters outright, so a long reply
// would silently vanish rather than arrive truncated. /price and /24h over a
// 44-position book cross that line easily, and /positions already could.
// Split on line boundaries so no HTML tag is ever cut in half.
const TELEGRAM_LIMIT = 3900;

function chunkMessage(text, limit = TELEGRAM_LIMIT) {
  if (text.length <= limit) return [text];
  const chunks = [];
  let current = "";
  // Tables are wrapped in <pre> for monospace alignment. Splitting mid-block
  // would emit an unclosed tag, and Telegram rejects a message whose HTML
  // does not parse -- so the reply would vanish entirely rather than arrive
  // split. Track the block and close/reopen it across the boundary.
  let insidePre = false;

  const flush = () => {
    if (!current) return;
    chunks.push(insidePre ? `${current}\n</pre>` : current);
    current = insidePre ? "<pre>" : "";
  };

  for (const line of text.split("\n")) {
    // A single line longer than the limit is hard-sliced; nothing sane emits one.
    if (line.length > limit) {
      flush();
      for (let i = 0; i < line.length; i += limit) chunks.push(line.slice(i, i + limit));
      continue;
    }
    // Reserve room for the "\n</pre>" that flush() appends when a split
    // lands inside a table; without it every such chunk overshoots the limit
    // by exactly that tag's length.
    const reserve = insidePre ? "\n</pre>".length : 0;
    if (current.length + line.length + 1 + reserve > limit) flush();
    current = current ? `${current}\n${line}` : line;

    if (line.includes("<pre>")) insidePre = true;
    if (line.includes("</pre>")) insidePre = false;
  }
  if (current && current !== "<pre>") chunks.push(current);
  return chunks;
}

async function sendMessage(token, chatId, text) {
  const chunks = chunkMessage(text);
  for (const chunk of chunks) {
    await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        text: chunk,
        parse_mode: "HTML",
        disable_web_page_preview: true,
      }),
    });
  }
}

// Named exports alongside the default Workers `fetch` export -- unused by the
// Workers runtime itself, but lets tools/test_status_worker_local.mjs (and any
// future test) exercise the real buildStatus/buildPositions/buildPnl logic
// directly, with only the network layer (ghFile's GitHub API call) mocked out.
export {
  buildStatus, buildPositions, buildPnl, parseCsv, ghFile, SYSTEMS,
  buildPrice, build24h, buildSystem, normalizeSymbol, findSystem,
  chunkMessage, SYSTEM_REGISTRY, fetchTickers,
};
