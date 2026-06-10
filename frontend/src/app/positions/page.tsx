"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export default function PositionsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["positions"],
    queryFn: () => api.get("/portfolio/positions"),
  });

  const positions = data?.positions || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Open Positions</h2>
        <p className="text-[var(--muted)] text-sm">
          Currently open paper positions with live P&L tracking
        </p>
      </div>

      {isLoading ? (
        <div className="text-[var(--muted)]">Loading positions...</div>
      ) : positions.length === 0 ? (
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-8 text-center text-[var(--muted)]">
          No open positions
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--card-border)] text-[var(--muted)] text-left">
                <th className="pb-2 pr-4">Symbol</th>
                <th className="pb-2 pr-4">Side</th>
                <th className="pb-2 pr-4">Entry</th>
                <th className="pb-2 pr-4">Current</th>
                <th className="pb-2 pr-4">Stop</th>
                <th className="pb-2 pr-4">R</th>
                <th className="pb-2 pr-4">MFE</th>
                <th className="pb-2 pr-4">Chandelier</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p: any) => (
                <tr
                  key={p.id}
                  className="border-b border-[var(--card-border)]/50 hover:bg-[var(--card-border)]/20"
                >
                  <td className="py-3 pr-4 font-medium">{p.symbol}</td>
                  <td className="py-3 pr-4">
                    <span
                      className={
                        p.side === "LONG" ? "text-win" : "text-loss"
                      }
                    >
                      {p.side}
                    </span>
                  </td>
                  <td className="py-3 pr-4">${p.entry_price?.toFixed(4)}</td>
                  <td className="py-3 pr-4">${p.current_price?.toFixed(4) || "—"}</td>
                  <td className="py-3 pr-4">${p.current_stop?.toFixed(4)}</td>
                  <td className="py-3 pr-4">
                    <span
                      className={
                        (p.current_r || 0) >= 0 ? "text-win" : "text-loss"
                      }
                    >
                      {p.current_r?.toFixed(2) || "0.00"}R
                    </span>
                  </td>
                  <td className="py-3 pr-4">{p.mfe_r?.toFixed(2) || "0.00"}R</td>
                  <td className="py-3 pr-4">
                    {p.chandelier_active ? (
                      <span className="text-win text-xs">● Active</span>
                    ) : (
                      <span className="text-[var(--muted)] text-xs">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
