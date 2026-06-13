"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

function StatCard({
  label,
  value,
  suffix,
  color,
}: {
  label: string;
  value: string | number;
  suffix?: string;
  color?: "win" | "loss" | "neutral";
}) {
  const colorClass =
    color === "win"
      ? "text-win"
      : color === "loss"
        ? "text-loss"
        : "text-[var(--foreground)]";

  return (
    <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
      <p className="text-xs text-[var(--muted)] uppercase tracking-wide">
        {label}
      </p>
      <p className={`text-2xl font-bold mt-1 ${colorClass}`}>
        {value}
        {suffix && (
          <span className="text-sm font-normal text-[var(--muted)]">
            {" "}
            {suffix}
          </span>
        )}
      </p>
    </div>
  );
}

export default function DashboardPage() {
  const { data: summary, isLoading } = useQuery({
    queryKey: ["portfolio-summary"],
    queryFn: () => api.get("/portfolio/summary"),
  });

  const { data: stats } = useQuery({
    queryKey: ["trade-stats"],
    queryFn: () => api.get("/trades/stats"),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--muted)]">
        Loading dashboard...
      </div>
    );
  }

  const s = summary || {};
  const t = stats || {};

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <p className="text-[var(--muted)] text-sm">
          Portfolio overview · Paper trading
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Equity"
          value={`$${(s.equity_usdt || 10000).toLocaleString()}`}
        />
        <StatCard
          label="Open P&L"
          value={`$${(s.open_pnl_usdt || 0).toFixed(2)}`}
          color={(s.open_pnl_usdt || 0) >= 0 ? "win" : "loss"}
        />
        <StatCard
          label="Drawdown"
          value={`${(s.drawdown_pct || 0).toFixed(2)}%`}
          color={(s.drawdown_pct || 0) > 5 ? "loss" : "neutral"}
        />
        <StatCard
          label="Heat"
          value={`${(s.portfolio_heat_pct || 0).toFixed(2)}%`}
          suffix={`/ 1.5%`}
        />
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Open Positions" value={s.open_positions || 0} />
        <StatCard label="Total Trades" value={t.total_trades || 0} />
        <StatCard
          label="Win Rate"
          value={`${(t.win_rate || 0).toFixed(1)}%`}
          color={(t.win_rate || 0) > 35 ? "win" : "loss"}
        />
        <StatCard
          label="Total R"
          value={`${(t.total_r || 0).toFixed(1)}R`}
          color={(t.total_r || 0) >= 0 ? "win" : "loss"}
        />
      </div>

      {/* Placeholder for equity chart */}
      <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-6 h-64 flex items-center justify-center text-[var(--muted)]">
        Equity curve chart will render here
      </div>
    </div>
  );
}
