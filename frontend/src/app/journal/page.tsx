"use client";

export default function JournalPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Trade Journal</h2>
        <p className="text-[var(--muted)] text-sm">
          Annotate trades with notes, lessons learned, and tags
        </p>
      </div>
      <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg p-8 text-center text-[var(--muted)]">
        Journal entries with rich text editor — coming in Phase 5
      </div>
    </div>
  );
}
