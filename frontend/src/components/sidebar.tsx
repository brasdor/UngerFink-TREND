"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: "📊" },
  { href: "/positions", label: "Positions", icon: "📈" },
  { href: "/trades", label: "Trades", icon: "📋" },
  { href: "/equity", label: "Equity", icon: "💰" },
  { href: "/signals", label: "Signals", icon: "🔔" },
  { href: "/execution", label: "Trade Desk", icon: "⚡" },
  { href: "/research", label: "Research", icon: "🔬" },
  { href: "/journal", label: "Journal", icon: "📝" },
  { href: "/alerts", label: "Alerts", icon: "⚠️" },
  { href: "/settings", label: "Settings", icon: "⚙️" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 border-r border-[var(--card-border)] bg-[var(--card)] flex flex-col">
      <div className="p-4 border-b border-[var(--card-border)]">
        <h1 className="text-lg font-bold text-[var(--accent)]">UngerFink</h1>
        <p className="text-xs text-[var(--muted)]">TREND System</p>
      </div>
      <nav className="flex-1 p-2 space-y-1">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={clsx(
              "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
              pathname === item.href
                ? "bg-[var(--accent)]/10 text-[var(--accent)]"
                : "text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--card-border)]/50"
            )}
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </Link>
        ))}
      </nav>
      <div className="p-4 border-t border-[var(--card-border)] text-xs text-[var(--muted)]">
        Paper Trading Only
      </div>
    </aside>
  );
}
