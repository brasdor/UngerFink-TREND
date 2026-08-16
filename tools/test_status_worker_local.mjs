#!/usr/bin/env node
/**
 * Local test harness for cloudflare/status_worker.js -- runs the REAL
 * buildPositions()/buildPnl()/buildStatus() logic against real local files,
 * with only the network layer (ghFile's GitHub Contents API call) mocked to
 * read from disk instead. This is the closest thing to a live test possible
 * without an actual Cloudflare deploy + Telegram round trip, which need
 * deploy credentials and bot tokens this environment doesn't have.
 *
 * Usage: node tools/test_status_worker_local.mjs
 */
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// Mock the GitHub Contents API: ghFile() does
//   fetch(`https://api.github.com/repos/${GH_REPO}/contents/${path}?ref=${branch}`)
// Intercept that specific host and serve the local file instead.
const realFetch = globalThis.fetch;
globalThis.fetch = async (url, opts) => {
  const u = typeof url === "string" ? url : url.toString();
  if (u.startsWith("https://api.github.com/repos/")) {
    const path = decodeURIComponent(u.split("/contents/")[1].split("?")[0]);
    const local = join(ROOT, path);
    if (!existsSync(local)) {
      return new Response("not found", { status: 404 });
    }
    return new Response(readFileSync(local, "utf-8"), { status: 200 });
  }
  return realFetch(url, opts);
};

const { buildStatus, buildPositions, buildPnl, SYSTEMS } = await import(
  pathToFileURL(join(ROOT, "cloudflare", "status_worker.js")).href
);

const env = { GH_REPO: "brasdor/UngerFink-TREND", GH_BRANCH: "master" };

console.log("SYSTEMS wired in:", SYSTEMS.map(([label]) => label).join(", "));
console.log("\n" + "=".repeat(70));
console.log("/status");
console.log("=".repeat(70));
console.log(await buildStatus(env));

console.log("\n" + "=".repeat(70));
console.log("/positions");
console.log("=".repeat(70));
console.log(await buildPositions(env));

console.log("\n" + "=".repeat(70));
console.log("/pnl");
console.log("=".repeat(70));
console.log(await buildPnl(env));
