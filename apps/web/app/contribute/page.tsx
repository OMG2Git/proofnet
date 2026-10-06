"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import {
  api,
  ApiRequestError,
  getApiBase,
  useStoredDevice,
  setStoredDevice,
  type DeviceOut,
} from "@/lib/api/client";
import { collectCapabilities, guessDeviceType } from "@/worker-runtime/capabilities";

function Contribute() {
  const [devices, setDevices] = useState<DeviceOut[]>([]);
  const stored = useStoredDevice();
  const [name, setName] = useState("My device");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    api
      .myDevices()
      .then(setDevices)
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    const type = guessDeviceType();
    const t = setTimeout(
      () => setName(type === "android_phone" ? "My Android phone" : "My laptop"),
      0,
    );
    refresh();
    return () => clearTimeout(t);
  }, [refresh]);

  async function register() {
    setBusy(true);
    setError(null);
    try {
      const caps = await collectCapabilities();
      const res = await api.registerDevice({
        name,
        device_type: guessDeviceType(),
        capabilities: caps,
      });
      setStoredDevice({
        apiBase: getApiBase(),
        deviceId: res.device.id,
        deviceToken: res.device_token,
        name: res.device.name,
      });
      refresh();
    } catch (e) {
      setError(e instanceof ApiRequestError ? e.message : "Registration failed");
    } finally {
      setBusy(false);
    }
  }

  const thisDevice = devices.find((d) => d.id === stored?.deviceId);
  return (
    <section>
      <h1>Contribute compute</h1>
      <div className="card">
        {stored ? (
          <>
            <p>
              This browser is registered as <strong>{stored.name}</strong> (<code>{stored.deviceId}</code>).
            </p>
            <div className="row">
              <Link href="/contribute/run" className="btn">
                Open worker console
              </Link>
              <button
                onClick={() => setStoredDevice(null)}
              >
                Forget this browser&apos;s identity
              </button>
            </div>
            {!thisDevice && (
              <p className="warn">This device no longer exists on the server; register again.</p>
            )}
          </>
        ) : (
          <>
            <p>Register this browser as a contributor device. Keep the tab in the foreground and the screen on.</p>
            <label>
              Device name
              <input value={name} onChange={(e) => setName(e.target.value)} maxLength={80} />
            </label>
            <button onClick={register} disabled={busy || !name.trim()}>
              {busy ? "Registering…" : "Register this device"}
            </button>
          </>
        )}
        {error && <p className="error">{error}</p>}
      </div>

      <h1>My devices</h1>
      {devices.length === 0 ? (
        <p className="muted">No devices yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Status</th>
              <th>Score (cells/s)</th>
              <th>Runtime</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {devices.map((d) => (
              <tr key={d.id}>
                <td>
                  {d.name} {d.id === stored?.deviceId && <span className="muted">(this browser)</span>}
                </td>
                <td>{d.device_type}</td>
                <td>
                  <span className={`badge ${d.status}`}>{d.status}</span>
                </td>
                <td>{d.benchmark ? d.benchmark.score_cells_per_sec.toExponential(2) : "—"}</td>
                <td>{d.runtime ? `${d.runtime.kind} / NumPy ${d.runtime.numpy}` : "—"}</td>
                <td>
                  <button
                    onClick={() =>
                      api
                        .patchDevice(d.id, { disabled: d.status !== "disabled" })
                        .then(refresh)
                        .catch((e: Error) => setError(e.message))
                    }
                  >
                    {d.status === "disabled" ? "Enable" : "Disable"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export default function ContributePage() {
  return (
    <AuthGate>
      <Contribute />
    </AuthGate>
  );
}
