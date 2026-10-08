"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { api, setToken, type UserOut } from "@/lib/api/client";

/** Wraps pages that need a signed-in user; shows a nav bar with the user name. */
export default function AuthGate({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<UserOut | null>(null);

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => router.replace("/login"));
  }, [router]);

  if (!user) return <p className="muted">Checking session…</p>;
  return (
    <>
      <nav className="nav">
        <strong>ProofNet</strong>
        <Link href="/tasks">My tasks</Link>
        <Link href="/tasks/new">New task</Link>
        <Link href="/images/new">Image training</Link>
        <Link href="/contribute">Contribute</Link>
        <Link href="/network">Network</Link>
        <Link href="/settings">Settings</Link>
        <span className="spacer" />
        <span className="muted">{user.display_name}</span>
        <button
          onClick={() => {
            setToken(null);
            router.replace("/login");
          }}
        >
          Sign out
        </button>
      </nav>
      {children}
    </>
  );
}
