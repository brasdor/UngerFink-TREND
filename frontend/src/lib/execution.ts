// Execution API helpers. Unlike the generic client, these surface the
// backend's JSON `detail` field so the UI can show *why* an order was rejected
// (e.g. "Live trading is disabled", "exceeds cap").

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ExecutionConfig {
  testnet: boolean;
  live_trading_enabled: boolean;
  keys_configured: boolean;
  default_risk_pct: number;
  max_order_usdt: number;
  daily_loss_limit_usdt: number;
}

export interface PlaceOrderInput {
  strategy: string;
  symbol: string;
  order_type: "market" | "limit";
  entry: number;
  stop: number;
  limit_price?: number;
  equity_usdt?: number;
  risk_pct?: number;
  dry_run: boolean;
}

export interface OrderRow {
  id: number;
  strategy: string;
  symbol: string;
  side: string;
  order_type: string;
  status: string;
  testnet: boolean;
  dry_run: boolean;
  requested_qty: number;
  requested_notional_usdt: number;
  filled_qty: number | null;
  avg_fill_price: number | null;
  reconciled: boolean;
  reconcile_note: string | null;
  error_message: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/api/execution${path}`, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data?.detail || `${res.status} ${res.statusText}`);
  }
  return data as T;
}

export const executionApi = {
  getConfig: () => request<ExecutionConfig>("/config"),
  getOrders: () => request<{ orders: OrderRow[] }>("/orders"),
  placeOrder: (input: PlaceOrderInput) =>
    request<any>("/place", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
};
