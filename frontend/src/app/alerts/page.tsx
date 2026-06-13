"use client";

export default function AlertsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Alerts</h2>
        <p className="text-[var(--muted)] text-sm">
          Configure alert rules and view trigger history
        </p>
      </div>
      <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-8 text-center text-[var(--muted)]">
        Alert rule builder with Discord/Telegram notifications — coming in Phase 5
      </div>
    </div>
  );
}
