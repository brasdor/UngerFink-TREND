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
    } else if (text.startsWith("/start") || text.startsWith("/help")) {
      await sendMessage(env.BOT_TOKEN, chatId, "Commands:\n/status — open positions & last run");
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

async function buildStatus(env) {
  const now = new Date().toISOString().slice(0, 16).replace("T", " ") + " UTC";
  const lines = [`\u{1F4CA} <b>UngerFink status</b> — ${now}`, ""];

  for (const [label, dir] of SYSTEMS) {
    let syms = [];
    const op = await ghFile(env, `${dir}/open_positions.csv`);
    if (op) syms = parseCsv(op).map((r) => r.symbol).filter(Boolean);

    let runDate = "?";
    const sj = await ghFile(env, `${dir}/state.json`);
    if (sj) {
      try {
        runDate = JSON.parse(sj).last_run_date || "?";
      } catch (e) {}
    }
    const shown = syms.slice(0, 6).join(", ") + (syms.length > 6 ? "…" : "");
    lines.push(`<b>${label}</b>: ${syms.length} open${shown ? " — " + shown : ""}`);
    lines.push(`   last run: ${runDate}`);
  }

  const led = await ghFile(env, "data/auto_orders/placed.csv");
  const net = {};
  if (led) {
    for (const r of parseCsv(led)) {
      const k = `${r.strategy}/${r.symbol}`;
      const q = parseFloat(r.qty) || 0;
      net[k] = (net[k] || 0) + (String(r.action).toUpperCase() === "BUY" ? q : -q);
    }
  }
  const open = Object.entries(net).filter(([, v]) => v > 1e-9);
  lines.push("");
  lines.push(open.length ? `<b>Auto-exec</b>: ${open.length} open position(s)` : "<b>Auto-exec</b>: no open positions");

  return lines.join("\n");
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
