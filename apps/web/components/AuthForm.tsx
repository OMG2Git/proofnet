"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";
import { api, ApiRequestError, setToken } from "@/lib/api/client";

export default function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res =
        mode === "signup"
          ? await api.signup({ email, password, display_name: name })
          : await api.login({ email, password });
      setToken(res.access_token);
      router.replace("/tasks");
    } catch (err) {
      setError(err instanceof ApiRequestError ? err.message : "Unexpected error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card narrow" onSubmit={submit}>
      <h1>{mode === "signup" ? "Create account" : "Sign in"}</h1>
      {mode === "signup" && (
        <label>
          Display name
          <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={80} />
        </label>
      )}
      <label>
        Email
        <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
      </label>
      <label>
        Password
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={mode === "signup" ? 8 : 1}
        />
      </label>
      {error && <p className="error">{error}</p>}
      <button type="submit" disabled={busy}>
        {busy ? "Please wait…" : mode === "signup" ? "Sign up" : "Sign in"}
      </button>
      <p className="muted">
        {mode === "signup" ? (
          <>
            Have an account? <Link href="/login">Sign in</Link>
          </>
        ) : (
          <>
            New here? <Link href="/signup">Create an account</Link>
          </>
        )}
      </p>
    </form>
  );
}
