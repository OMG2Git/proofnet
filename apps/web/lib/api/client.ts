import { useMemo, useSyncExternalStore } from "react";
import type { components } from "./schema";

/** Types come from FastAPI OpenAPI (`npm run gen:api`); never hand-write API types. */
type S = components["schemas"];
export type TaskManifest = S["TaskManifest"];
export type TaskOut = S["TaskOut"];
export type TaskTypeInfo = S["TaskTypeInfo"];
export type DatasetOut = S["DatasetOut"];
export type ValidationResult = S["ValidationResult"];
export type UserOut = S["UserOut"];
export type DeviceOut = S["DeviceOut"];
export type ApiError = S["ApiError"];
export type DeviceRegistered = S["DeviceRegistered"];
export type DeviceRegisterRequest = S["DeviceRegisterRequest"];
export type DeviceCapabilities = S["DeviceCapabilities"];
export type NetworkSummary = S["NetworkSummary"];
export type RuntimeManifest = S["RuntimeManifest"];
export type SessionRequest = S["SessionRequest"];
export type SessionResponse = S["SessionResponse"];
export type HeartbeatRequest = S["HeartbeatRequest"];
export type HeartbeatResponse = S["HeartbeatResponse"];

const ENV_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";
const BASE_KEY = "proofnet.apiBase";
const TOKEN_KEY = "proofnet.token";

/** Runtime API-URL switch (ARCHITECTURE 16.2 fallback path), stored per browser. */
export function getApiBase(): string {
  try {
    return window.localStorage.getItem(BASE_KEY) || ENV_BASE;
  } catch {
    return ENV_BASE;
  }
}

export function setApiBase(url: string | null): void {
  try {
    if (url) window.localStorage.setItem(BASE_KEY, url.replace(/\/$/, ""));
    else window.localStorage.removeItem(BASE_KEY);
  } catch {
    /* storage unavailable */
  }
}

export function getToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable */
  }
}

export class ApiRequestError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details: unknown[] = [],
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  opts: { method?: string; json?: unknown; form?: FormData; token?: string } = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  const token = opts.token ?? getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let body: BodyInit | undefined;
  if (opts.form) body = opts.form;
  else if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  let res: Response;
  try {
    res = await fetch(`${getApiBase()}${path}`, { method: opts.method ?? "GET", headers, body });
  } catch {
    throw new ApiRequestError(0, "NETWORK", "Cannot reach the ProofNet backend");
  }
  if (!res.ok) {
    let err: Partial<ApiError> = {};
    try {
      err = (await res.json()) as Partial<ApiError>;
    } catch {
      /* non-JSON error */
    }
    throw new ApiRequestError(
      res.status,
      err.error?.code ?? "HTTP_ERROR",
      err.error?.message ?? `Request failed (${res.status})`,
      err.error?.details ?? [],
    );
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; db: string }>("/health"),
  signup: (b: S["SignupRequest"]) =>
    request<S["TokenResponse"]>("/auth/signup", { method: "POST", json: b }),
  login: (b: S["LoginRequest"]) =>
    request<S["TokenResponse"]>("/auth/login", { method: "POST", json: b }),
  me: () => request<UserOut>("/auth/me"),
  taskTypes: () => request<TaskTypeInfo[]>("/task-types"),
  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DatasetOut>("/datasets", { method: "POST", form });
  },
  validateTask: (m: TaskManifest) =>
    request<ValidationResult>("/tasks/validate", { method: "POST", json: m }),
  createTask: (m: TaskManifest) => request<TaskOut>("/tasks", { method: "POST", json: m }),
  listTasks: () => request<TaskOut[]>("/tasks"),
  getTask: (id: string) => request<TaskOut>(`/tasks/${id}`),
  registerDevice: (b: DeviceRegisterRequest) =>
    request<DeviceRegistered>("/devices", { method: "POST", json: b }),
  myDevices: () => request<DeviceOut[]>("/devices/mine"),
  patchDevice: (id: string, b: S["DevicePatch"]) =>
    request<DeviceOut>(`/devices/${id}`, { method: "PATCH", json: b }),
  networkSummary: () => request<NetworkSummary>("/network/summary"),
};

/** Worker-side endpoints; authenticated with the device token, not the user JWT. */
export function deviceApi(deviceToken: string) {
  return {
    manifest: () => request<RuntimeManifest>("/runtime/manifest", { token: deviceToken }),
    session: (b: SessionRequest) =>
      request<SessionResponse>("/worker/session", { method: "POST", json: b, token: deviceToken }),
    heartbeat: (b: HeartbeatRequest) =>
      request<HeartbeatResponse>("/worker/heartbeat", {
        method: "POST",
        json: b,
        token: deviceToken,
      }),
  };
}

const DEVICE_KEY = "proofnet.device";

export type StoredDevice = { apiBase: string; deviceId: string; deviceToken: string; name: string };

const DEVICE_EVENT = "proofnet-device-change";

function parseDevice(raw: string | null): StoredDevice | null {
  if (!raw) return null;
  try {
    const d = JSON.parse(raw) as StoredDevice;
    return d.apiBase === getApiBase() ? d : null;
  } catch {
    return null;
  }
}

/** Device identity lives in localStorage, per API base (ARCHITECTURE 3.5). */
export function getStoredDevice(): StoredDevice | null {
  try {
    return parseDevice(window.localStorage.getItem(DEVICE_KEY));
  } catch {
    return null;
  }
}

export function setStoredDevice(d: StoredDevice | null): void {
  try {
    if (d) window.localStorage.setItem(DEVICE_KEY, JSON.stringify(d));
    else window.localStorage.removeItem(DEVICE_KEY);
  } catch {
    /* storage unavailable */
  }
  window.dispatchEvent(new Event(DEVICE_EVENT));
}

function subscribeDevice(cb: () => void): () => void {
  window.addEventListener(DEVICE_EVENT, cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener(DEVICE_EVENT, cb);
    window.removeEventListener("storage", cb);
  };
}

function rawDevice(): string | null {
  try {
    return window.localStorage.getItem(DEVICE_KEY);
  } catch {
    return null;
  }
}

/** React hook: the stored device identity (null on the server and when absent). */
export function useStoredDevice(): StoredDevice | null {
  const raw = useSyncExternalStore(subscribeDevice, rawDevice, () => null);
  return useMemo(() => parseDevice(raw), [raw]);
}
