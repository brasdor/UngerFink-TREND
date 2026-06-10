"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/lib/api";

export default function TradesPage() {
  const [filters, setFilters] = useState({
    strategy: "",
    symbol: "",
    side: "",
    limit: 50,
    offset: 0,
  });

  const { data, isLoading } = useQuery({
    queryKey: ["trades", filters],
    queryFn: () => api.get("/trades", filters),
  });

  const trades = data?.trades || [];
  const total = data?.total || 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Closed Trades</h2>
        <p className="text-[var(--muted)] text-sm">
          {total} total trades · R-multiple analysis
        </p>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <input
          type="text"
          placeholder="Symbol..."
          className="bg-[var(--card)] border border-[var(--card-border)] rounded px-3 py-1.5 text-sm"
          value={filters.symbol}
          onChange={(e) => setFilters({ ...filters, symbol: e.target.value, offset: 0 })}
        />
        <select
          className="bg-[var(--card)] border border-[var(--card-border)] rounded px-3 py-1.5 text-sm"
          value={filters.side}
          onChange={(e) => setFilters({ ...filters, side: e.target.value, offset: 0 })}
        >
          <option value="">All Sides</option>
          <option value="LONG">LONG</option>
          <option value="SHORT">SHORT</option>
        </select>
      </div>

      {isLoading ? (
        <div className="text-[var(--muted)]">Loading trades...</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--card-border)] text-[var(--muted)] text-left">
                <th className="pb-2 pr-3">Symbol</th>
                <th className="pb-2 pr-3">Side</th>
                <th className="pb-2 pr-3">Entry</th>
                <th className="pb-2 pr-3">Exit</th>
                <th className="pb-2 pr-3">P&L (R)</th>
                <th className="pb-2 pr-3">MAE</th>
                <th className="pb-2 pr-3">MFE</th>
                <th className="pb-2 pr-3">Exit Reason</th>
                <th className="pb-2 pr-3">Date</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any) => (
                <tr
                  key={t.id}
                  className="border-b border-[var(--card-border)]/50 hover:bg-[var(--card-border)]/20 cursor-pointer"
                >
                  <td className="py-2 pr-3 font-medium">{t.symbol}</td>
                  <td className="py-2 pr-3">
                    <span className={t.side === "LONG" ? "text-win" : "text-loss"}>
                      {t.side}
                    </span>
                  </td>
                  <td className="py-2 pr-3">${t.entry_price?.toFixed(4)}</td>
                  <td className="py-2 pr-3">${t.exit_price?.toFixed(4)}</td>
                  <td className="py-2 pr-3">
                    <span className={t.pnl_r >= 0 ? "text-win font-medium" : "text-loss font-medium"}>
                      {t.pnl_r >= 0 ? "+" : ""}{t.pnl_r?.toFixed(2)}R
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-loss">{t.mae_r?.toFixed(2) || "—"}</td>
                  <td className="py-2 pr-3 text-win">{t.mfe_r?.toFixed(2) || "—"}</td>
                  <td className="py-2 pr-3 text-xs text-[var(--muted)]">{t.exit_reason || "—"}</td>
                  <td className="py-2 pr-3 text-xs text-[var(--muted)]">
                    {t.exit_time?.split("T")[0]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {total > filters.limit && (
        <div className="flex gap-2 justify-center">
          <button
            disabled={filters.offset === 0}
            onClick={() => setFilters({ ...filters, offset: Math.max(0, filters.offset - filters.limit) })}
            className="px-3 py-1 bg-[var(--card)] border border-[var(--card-border)] rounded text-sm disabled:opacity-30"
          >
            ← Prev
          </button>
          <span className="text-sm text-[var(--muted)] py-1">
            {filters.offset + 1}–{Math.min(filters.offset + filters.limit, total)} of {total}
          </span>
          <button
            disabled={filters.offset + filters.limit >= total}
            onClick={() => setFilters({ ...filters, offset: filters.offset + filters.limit })}
            className="px-3 py-1 bg-[var(--card)] border border-[var(--card-border)] rounded text-sm disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
