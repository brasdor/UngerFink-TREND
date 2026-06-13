"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  executionApi,
  ExecutionConfig,
  OrderRow,
  PlaceOrderInput,
} from "@/lib/execution";

const EMPTY_FORM = {
  strategy: "DonchianLong",
  symbol: "FET/USDT",
  order_type: "market" as "market" | "limit",
  entry: "",
  stop: "",
  limit_price: "",
  risk_pct: "",
  equity_usdt: "",
  dry_run: true,
};

export default function ExecutionPage() {
  const queryClient = useQueryClient();
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [confirming, setConfirming] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const { data: config } = useQuery({
    queryKey: ["execution-config"],
    queryFn: executionApi.getConfig,
  });
  const { data: ordersData } = useQuery({
    queryKey: ["execution-orders"],
    queryFn: executionApi.getOrders,
    refetchInterval: 5000,
  });

  const place = useMutation({
    mutationFn: (input: PlaceOrderInput) => executionApi.placeOrder(input),
    onSuccess: (res) => {
      const o = res.order;
      setMessage({
        ok: true,
        text: `Order #${o.id} — ${o.status}${o.dry_run ? " (dry run)" : ""}`,
      });
      queryClient.invalidateQueries({ queryKey: ["execution-orders"] });
    },
    onError: (err: any) => setMessage({ ok: false, text: err.message }),
    onSettled: () => setConfirming(false),
  });

  function update(field: string, value: any) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  function buildInput(): PlaceOrderInput {
    return {
      strategy: form.strategy.trim(),
      symbol: form.symbol.trim().toUpperCase(),
      order_type: form.order_type,
      entry: parseFloat(form.entry),
      stop: parseFloat(form.stop),
      limit_price: form.limit_price ? parseFloat(form.limit_price) : undefined,
      risk_pct: form.risk_pct ? parseFloat(form.risk_pct) : undefined,
      equity_usdt: form.equity_usdt ? parseFloat(form.equity_usdt) : undefined,
      dry_run: form.dry_run,
    };
  }

  const valid =
    form.symbol &&
    form.entry &&
    form.stop &&
    parseFloat(form.entry) > parseFloat(form.stop) &&
    (form.order_type !== "limit" || form.limit_price);

  const isLive = !form.dry_run && config && !config.testnet;
  const placeLabel = form.dry_run
    ? "Simulate (dry run)"
    : config?.testnet
      ? "Place on TESTNET"
      : "Place LIVE order";

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h2 className="text-2xl font-bold">Trade Desk</h2>
        <p className="text-[var(--muted)] text-sm">
          Approve and place a single spot order. Human approves; the system executes.
        </p>
      </div>

      {config && <SafetyBanner config={config} />}

      {/* Order form */}
      <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-5 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Strategy">
            <input className="inp" value={form.strategy} onChange={(e) => update("strategy", e.target.value)} />
          </Field>
          <Field label="Symbol">
            <input className="inp" value={form.symbol} onChange={(e) => update("symbol", e.target.value)} />
          </Field>
          <Field label="Order type">
            <select className="inp" value={form.order_type} onChange={(e) => update("order_type", e.target.value)}>
              <option value="market">market</option>
              <option value="limit">limit</option>
            </select>
          </Field>
          {form.order_type === "limit" && (
            <Field label="Limit price">
              <input className="inp" value={form.limit_price} onChange={(e) => update("limit_price", e.target.value)} />
            </Field>
          )}
          <Field label="Entry price">
            <input className="inp" value={form.entry} onChange={(e) => update("entry", e.target.value)} />
          </Field>
          <Field label="Stop price">
            <input className="inp" value={form.stop} onChange={(e) => update("stop", e.target.value)} />
          </Field>
          <Field label={`Risk % (default ${config ? (config.default_risk_pct * 100).toFixed(2) : "—"}%)`}>
            <input className="inp" placeholder="e.g. 0.0025" value={form.risk_pct} onChange={(e) => update("risk_pct", e.target.value)} />
          </Field>
          <Field label="Equity USDT (blank = free balance)">
            <input className="inp" value={form.equity_usdt} onChange={(e) => update("equity_usdt", e.target.value)} />
          </Field>
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={form.dry_run} onChange={(e) => update("dry_run", e.target.checked)} />
          <span>Dry run (simulate only — no order sent)</span>
        </label>

        {form.entry && form.stop && parseFloat(form.entry) <= parseFloat(form.stop) && (
          <p className="text-loss text-xs">Entry must be above stop for a long.</p>
        )}

        <button
          disabled={!valid || place.isPending}
          onClick={() => { setMessage(null); setConfirming(true); }}
          className={`px-4 py-2 rounded-md text-sm font-medium disabled:opacity-40 ${
            isLive ? "bg-loss text-white" : "bg-[var(--accent)] text-white"
          }`}
        >
          {placeLabel}
        </button>

        {message && (
          <div className={`text-sm ${message.ok ? "text-win" : "text-loss"}`}>{message.text}</div>
        )}
      </div>

      <RecentOrders orders={ordersData?.orders || []} />

      {confirming && (
        <ConfirmDialog
          input={buildInput()}
          live={!!isLive}
          testnet={!!config?.testnet}
          pending={place.isPending}
          onCancel={() => setConfirming(false)}
          onConfirm={() => place.mutate(buildInput())}
        />
      )}

      <style jsx>{`
        :global(.inp) {
          width: 100%;
          background: var(--background);
          border: 1px solid var(--card-border);
          border-radius: 6px;
          padding: 6px 10px;
          font-size: 0.875rem;
        }
      `}</style>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block space-y-1">
      <span className="text-xs text-[var(--muted)]">{label}</span>
      {children}
    </label>
  );
}

function SafetyBanner({ config }: { config: ExecutionConfig }) {
  const live = config.live_trading_enabled && !config.testnet;
  return (
    <div
      className={`rounded-lg border p-4 text-sm ${
        live
          ? "border-loss bg-loss/10 text-loss"
          : "border-[var(--accent)]/40 bg-[var(--accent)]/10"
      }`}
    >
      <div className="font-semibold">
        {config.testnet ? "🧪 TESTNET (fake money)" : live ? "🔴 LIVE TRADING ENABLED — real money" : "⏸️ Live trading disabled"}
      </div>
      <div className="text-xs text-[var(--muted)] mt-1">
        Keys {config.keys_configured ? "configured" : "NOT configured"} · per-order cap{" "}
        {config.max_order_usdt} USDT · daily loss limit {config.daily_loss_limit_usdt} USDT
      </div>
    </div>
  );
}

function ConfirmDialog({
  input, live, testnet, pending, onCancel, onConfirm,
}: {
  input: PlaceOrderInput;
  live: boolean;
  testnet: boolean;
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const mode = input.dry_run ? "DRY RUN" : testnet ? "TESTNET" : "LIVE (real money)";
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-6 w-96 space-y-4">
        <h3 className="font-bold text-lg">Confirm order</h3>
        <div className={`text-sm font-semibold ${live ? "text-loss" : ""}`}>Mode: {mode}</div>
        <div className="text-sm space-y-1">
          <Row k="Strategy" v={input.strategy} />
          <Row k="Symbol" v={input.symbol} />
          <Row k="Type" v={input.order_type} />
          <Row k="Entry" v={String(input.entry)} />
          <Row k="Stop" v={String(input.stop)} />
          {input.order_type === "limit" && <Row k="Limit" v={String(input.limit_price)} />}
        </div>
        {live && (
          <p className="text-loss text-xs">
            This will place a REAL order with REAL money on Binance.
          </p>
        )}
        <div className="flex gap-2 justify-end">
          <button onClick={onCancel} className="px-3 py-1.5 rounded-md text-sm border border-[var(--card-border)]">
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={pending}
            className={`px-3 py-1.5 rounded-md text-sm text-white disabled:opacity-40 ${live ? "bg-loss" : "bg-[var(--accent)]"}`}
          >
            {pending ? "Placing…" : "Confirm"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-[var(--muted)]">{k}</span>
      <span className="font-medium">{v}</span>
    </div>
  );
}

function RecentOrders({ orders }: { orders: OrderRow[] }) {
  return (
    <div>
      <h3 className="font-semibold mb-2">Recent orders</h3>
      {orders.length === 0 ? (
        <div className="text-[var(--muted)] text-sm">No orders yet</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--card-border)] text-[var(--muted)] text-left">
                <th className="pb-2 pr-4">#</th>
                <th className="pb-2 pr-4">Symbol</th>
                <th className="pb-2 pr-4">Type</th>
                <th className="pb-2 pr-4">Qty</th>
                <th className="pb-2 pr-4">Status</th>
                <th className="pb-2 pr-4">Mode</th>
                <th className="pb-2 pr-4">Reconciled</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((o) => (
                <tr key={o.id} className="border-b border-[var(--card-border)]/50">
                  <td className="py-2 pr-4">{o.id}</td>
                  <td className="py-2 pr-4 font-medium">{o.symbol}</td>
                  <td className="py-2 pr-4">{o.order_type}</td>
                  <td className="py-2 pr-4">{o.filled_qty ?? o.requested_qty}</td>
                  <td className="py-2 pr-4">{o.status}{o.error_message ? ` — ${o.error_message}` : ""}</td>
                  <td className="py-2 pr-4 text-xs">{o.dry_run ? "dry" : o.testnet ? "testnet" : "live"}</td>
                  <td className="py-2 pr-4">{o.reconciled ? "✓" : o.reconcile_note || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
