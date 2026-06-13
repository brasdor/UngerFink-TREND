"use client";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Settings</h2>
        <p className="text-[var(--muted)] text-sm">
          System configuration, strategy management, and connections
        </p>
      </div>

      <div className="grid gap-4">
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
          <h3 className="font-bold mb-2">Strategies</h3>
          <p className="text-sm text-[var(--muted)]">Manage active strategies, view frozen configs, toggle paper trading</p>
        </div>
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
          <h3 className="font-bold mb-2">Scheduler</h3>
          <p className="text-sm text-[var(--muted)]">Configure polling intervals, view last run times, manage jobs</p>
        </div>
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-4">
          <h3 className="font-bold mb-2">Connections</h3>
          <p className="text-sm text-[var(--muted)]">Binance API status, database health, Redis connectivity</p>
        </div>
      </div>
    </div>
  );
}
