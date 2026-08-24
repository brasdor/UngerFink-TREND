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

    if (text.startsWith("/status")) {
      const status = await buildStatus(env);
      await sendMessage(env.BOT_TOKEN, chatId, status);
    } else if (text.startsWith("/positions") || text.startsWith("/pos")) {
      const pos = await buildPositions(env);
      await sendMessage(env.BOT_TOKEN, chatId, pos);
    } else if (text.startsWith("/pnl")) {
      const pnl = await buildPnl(env);
      await sendMessage(env.BOT_TOKEN, chatId, pnl);
    } else if (text.startsWith("/start") || text.startsWith("/help")) {
      await sendMessage(env.BOT_TOKEN, chatId,
        "Commands:\n/status — all 9 systems, regime, and alerts in one view\n/positions — each position with P&L\n/pnl — total P&L summary");
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
  const sign = opts && opts.signed ? (v >= 0 ? "+" : "") : "";
  return `${sign}$${v.toFixed(2)}`;
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

  const lines = [`\u{1F4CA} <b>UngerFink /status</b>`,
                 `<i>snapshot generated ${fmtUtc(snap.generated_utc)}</i>`, ""];

  // --- Regime ---
  const r = snap.regime || {};
  const weights = r.weights || {};
  lines.push(
    `<b>Regime</b> (as of ${r.date || "?"}): trend=<b>${r.trend || "?"}</b>  ` +
    `funding=<b>${r.funding || "?"}</b>  vol×${r.vol_multiplier ?? "?"}`
  );
  lines.push("");

  // --- Systems ---
  lines.push("<b>Systems</b>");
  const systems = snap.systems || {};
  let totalEquity = 0;
  let totalUnrealized = 0;
  let anyUnrealizedNA = false;
  for (const [sid, label] of STATUS_SYSTEM_ORDER) {
    const s = systems[sid];
    if (!s) {
      lines.push(`  <code>${sid}</code> ${label}: no data`);
      continue;
    }
    const w = weights[sid];
    const wTxt = typeof w === "number" ? `  w=${(w * 100).toFixed(1)}%` : "";
    const ks = s.kill_switch ? "  \u{1F6D1}<b>KILL-SWITCH</b>" : "";
    const unrealTxt = typeof s.unrealized_pnl === "number"
      ? `${money(s.unrealized_pnl, { signed: true })} unreal`
      : (anyUnrealizedNA = true, "unreal n/a");
    const todayTxt = typeof s.today_realized_pnl === "number"
      ? `, ${money(s.today_realized_pnl, { signed: true })} today`
      : "";
    lines.push(
      `  <code>${sid}</code> ${label}: ${money(s.equity)}  ${s.open_positions} open  ` +
      `${unrealTxt}${todayTxt}${wTxt}${ks}`
    );
    totalEquity += s.equity || 0;
    if (typeof s.unrealized_pnl === "number") totalUnrealized += s.unrealized_pnl;
  }
  lines.push(
    `<b>Total</b>: ${money(totalEquity)} equity, ${money(totalUnrealized, { signed: true })} unrealized` +
    (anyUnrealizedNA ? " (partial — some systems n/a)" : "")
  );
  lines.push("");

  // --- Alerts ---
  const a = snap.alerts || {};
  const missed = a.missed_runs || [];
  const issues = a.ohlcv_issues || [];
  lines.push(`<b>Alerts</b> <i>(checked ${fmtUtc(a.checked_utc)})</i>`);
  if (missed.length === 0 && issues.length === 0) {
    lines.push("  ✅ all clear");
  } else {
    for (const m of missed) lines.push(`  ⚠️ ${m.system}: ${m.detail}`);
    for (const iss of issues) lines.push(`  ⚠️ ${iss}`);
  }
  if ((a.suppressed_symbols || []).length) {
    lines.push(`  <i>(suppressed halted symbols excluded: ${a.suppressed_symbols.join(", ")})</i>`);
  }

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

async function sendMessage(token, chatId, text) {
  await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chat_id: chatId,
      text: text,
      parse_mode: "HTML",
      disable_web_page_preview: true,
    }),
  });
}

// Named exports alongside the default Workers `fetch` export -- unused by the
// Workers runtime itself, but lets tools/test_status_worker_local.mjs (and any
// future test) exercise the real buildStatus/buildPositions/buildPnl logic
// directly, with only the network layer (ghFile's GitHub API call) mocked out.
export { buildStatus, buildPositions, buildPnl, parseCsv, ghFile, SYSTEMS };
