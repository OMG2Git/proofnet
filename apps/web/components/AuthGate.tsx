"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { api, setToken, type UserOut } from "@/lib/api/client";

const LINKS: [string, string][] = [
  ["/network", "Network"],
  ["/tasks", "Tasks"],
  ["/tasks/new", "New task"],
  ["/images/new", "Image training"],
  ["/contribute", "Contribute"],
  ["/trust", "Trust"],
  ["/rewards", "Rewards"],
  ["/security", "Security"],
  ["/simulator", "Simulator"],
  ["/settings", "Settings"],
];

/** Wraps pages that need a signed-in user; shows the navigation bar with the user name. */
export default function AuthGate({ children }: { children: ReactNode }) {
  const router = useRouter();
  const path = usePathname();
  const [user, setUser] = useState<UserOut | null>(null);
  const [open, setOpen] = useState(false); // mobile menu

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => router.replace("/login"));
  }, [router]);

  if (!user) return <p className="muted" role="status">Checking session…</p>;
  return (
    <>
      <nav className={`nav${open ? " open" : ""}`} aria-label="Main">
        <Link href="/" className="brand" onClick={() => setOpen(false)}>
          <i aria-hidden="true" />
          ProofNet
        </Link>
        <span className="spacer menu-spacer" />
        <button
          className="ghost small menu-toggle"
          aria-expanded={open}
          aria-controls="nav-links"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? "Close ✕" : "Menu ☰"}
        </button>
        <div id="nav-links" className="nav-links">
          {LINKS.map(([href, label]) => (
            <Link key={href} href={href} aria-current={path === href ? "page" : undefined} onClick={() => setOpen(false)}>
              {label}
            </Link>
          ))}
          <span className="spacer" />
          <span className="who">{user.display_name}</span>
          <button
            className="ghost small"
            onClick={() => {
              setToken(null);
              router.replace("/login");
            }}
          >
            Sign out
          </button>
        </div>
      </nav>
      <main id="content" style={{ maxWidth: "none" }}>
        {children}
      </main>
    </>
  );
}
