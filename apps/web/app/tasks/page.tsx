"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import { api, type TaskOut } from "@/lib/api/client";

function TaskList() {
  const [tasks, setTasks] = useState<TaskOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listTasks()
      .then(setTasks)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!tasks) return <p className="muted">Loading…</p>;
  return (
    <section>
      <h1>My tasks</h1>
      {tasks.length === 0 ? (
        <p className="muted">
          No tasks yet. <Link href="/tasks/new">Create one</Link>.
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Status</th>
              <th>Train / test rows</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {tasks.map((t) => (
              <tr key={t.id}>
                <td>
                  <Link href={`/tasks/${t.id}`}>{t.name}</Link>
                </td>
                <td>{t.task_type}</td>
                <td>
                  <span className={`badge ${t.status}`}>{t.status}</span>
                </td>
                <td>
                  {String(t.prepared["n_train"])} / {String(t.prepared["n_test"])}
                </td>
                <td>{new Date(t.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export default function TasksPage() {
  return (
    <AuthGate>
      <TaskList />
    </AuthGate>
  );
}
