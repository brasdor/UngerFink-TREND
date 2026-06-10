"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export default function ResearchPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["research-runs"],
    queryFn: () => api.get("/research"),
  });

  const runs = data?.runs || [];

  const phaseOrder = ["T1", "T2", "T3", "T3B", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14", "T15", "T16", "T17", "T18"];

  // Group by strategy
  const byStrategy: Record<string, any[]> = {};
  runs.forEach((r: any) => {
    if (!byStrategy[r.strategy]) byStrategy[r.strategy] = [];
    byStrategy[r.strategy].push(r);
  });

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Research Explorer</h2>
        <p className="text-[var(--muted)] text-sm">
          Browse T1–T18 research phases and results across all strategies
        </p>
      </div>

      {isLoading ? (
        <div className="text-[var(--muted)]">Loading research data...</div>
      ) : Object.keys(byStrategy).length === 0 ? (
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-8 text-center text-[var(--muted)]">
          No research runs imported yet — run the data migration to populate
        </div>
      ) : (
        Object.entries(byStrategy).map(([strategy, phases]) => (
          <div key={strategy} className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
            <h3 className="font-bold text-lg mb-3">{strategy}</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {phases.map((r: any) => (
                <div
                  key={r.id}
                  className="border border-[var(--card-border)] rounded p-2 text-center"
                >
                  <div className="font-mono text-xs text-[var(--muted)]">{r.phase}</div>
                  <div
                    className={`text-xs font-bold mt-1 ${
                      r.gate_result === "PASS"
                        ? "text-win"
                        : r.gate_result === "FAIL"
                          ? "text-loss"
                          : r.gate_result === "WARN"
                            ? "text-yellow-500"
                            : "text-[var(--muted)]"
                    }`}
                  >
                    {r.gate_result || r.status}
                  </div>
                  {r.total_r != null && (
                    <div className="text-xs mt-1">
                      {r.total_r >= 0 ? "+" : ""}{r.total_r.toFixed(1)}R
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );
}
