"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export default function EquityPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["equity-curve"],
    queryFn: () => api.get("/portfolio/equity-curve", { days: 90 }),
  });

  const points = data?.data || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Equity Curve</h2>
        <p className="text-[var(--muted)] text-sm">
          Portfolio equity over time with drawdown overlay
        </p>
      </div>

      {isLoading ? (
        <div className="text-[var(--muted)]">Loading equity data...</div>
      ) : points.length === 0 ? (
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-8 text-center text-[var(--muted)]">
          No equity data yet — start paper trading to generate equity snapshots
        </div>
      ) : (
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-6">
          <div className="h-80 flex items-center justify-center text-[var(--muted)]">
            {/* TradingView Lightweight Charts will render here */}
            Equity chart: {points.length} data points from{" "}
            {points[0]?.timestamp?.split("T")[0]} to{" "}
            {points[points.length - 1]?.timestamp?.split("T")[0]}
          </div>
        </div>
      )}

      {/* Stats row */}
      {points.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
            <p className="text-xs text-[var(--muted)]">Current Equity</p>
            <p className="text-xl font-bold">
              ${points[points.length - 1]?.equity?.toLocaleString()}
            </p>
          </div>
          <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
            <p className="text-xs text-[var(--muted)]">Max Drawdown</p>
            <p className="text-xl font-bold text-loss">
              {Math.min(...points.map((p: any) => p.drawdown_pct)).toFixed(2)}%
            </p>
          </div>
          <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
            <p className="text-xs text-[var(--muted)]">Peak</p>
            <p className="text-xl font-bold text-win">
              ${Math.max(...points.map((p: any) => p.equity)).toLocaleString()}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
