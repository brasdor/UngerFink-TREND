"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export default function SignalsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["signals"],
    queryFn: () => api.get("/signals", { limit: 100 }),
  });

  const signals = data?.signals || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Signals</h2>
        <p className="text-[var(--muted)] text-sm">
          Generated trading signals — taken, skipped, and pending
        </p>
      </div>

      {isLoading ? (
        <div className="text-[var(--muted)]">Loading signals...</div>
      ) : signals.length === 0 ? (
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-8 text-center text-[var(--muted)]">
          No signals recorded yet
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--card-border)] text-[var(--muted)] text-left">
                <th className="pb-2 pr-4">Time</th>
                <th className="pb-2 pr-4">Symbol</th>
                <th className="pb-2 pr-4">Side</th>
                <th className="pb-2 pr-4">Price</th>
                <th className="pb-2 pr-4">Status</th>
                <th className="pb-2 pr-4">Reason</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((s: any) => (
                <tr key={s.id} className="border-b border-[var(--card-border)]/50">
                  <td className="py-2 pr-4 text-xs">{s.signal_time?.replace("T", " ").slice(0, 16)}</td>
                  <td className="py-2 pr-4 font-medium">{s.symbol}</td>
                  <td className="py-2 pr-4">
                    <span className={s.side === "LONG" ? "text-win" : "text-loss"}>{s.side}</span>
                  </td>
                  <td className="py-2 pr-4">${s.signal_price?.toFixed(4)}</td>
                  <td className="py-2 pr-4">
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        s.status === "taken"
                          ? "bg-win/20 text-win"
                          : s.status === "skipped"
                            ? "bg-loss/20 text-loss"
                            : "bg-[var(--muted)]/20 text-[var(--muted)]"
                      }`}
                    >
                      {s.status}
                    </span>
                  </td>
                  <td className="py-2 pr-4 text-xs text-[var(--muted)]">{s.skip_reason || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
