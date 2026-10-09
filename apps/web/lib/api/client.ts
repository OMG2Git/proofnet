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
export type NetworkDevice = S["NetworkDevice"];
export type RuntimeManifest = S["RuntimeManifest"];
export type SessionRequest = S["SessionRequest"];
export type SessionResponse = S["SessionResponse"];
export type HeartbeatRequest = S["HeartbeatRequest"];
export type HeartbeatResponse = S["HeartbeatResponse"];
export type AssignmentPayload = S["AssignmentPayload"];
export type ImageDatasetOut = S["ImageDatasetOut"];
export type ImageTaskManifest = S["ImageTaskManifest"];
export type TrainingStatus = S["TrainingStatus"];
export type ResultAck = S["ResultAck"];
export type TaskStatus = S["TaskStatus"];
export type ArtifactOut = S["ArtifactOut"];
export type EventOut = S["EventOut"];
export type TrustOverview = S["TrustOverview"];
export type DeviceTrust = S["DeviceTrust"];
export type VerificationRecord = S["VerificationRecord"];
export type CalibrationClass = S["CalibrationClass"];
export type RewardsOut = S["RewardsOut"];
export type Balance = S["Balance"];
export type NetworkRewardRow = S["NetworkRewardRow"];
export type LedgerCheck = S["LedgerCheck"];
export type SecurityOverview = S["SecurityOverview"];
export type SecurityEvent = S["SecurityEvent"];
export type SimulateRequest = S["SimulateRequest"];
export type SimulationOut = S["SimulationOut"];

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
  opts: {
    method?: string;
    json?: unknown;
    form?: FormData;
    raw?: string;
    token?: string;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  const token = opts.token ?? getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let body: BodyInit | undefined;
  if (opts.form) body = opts.form;
  else if (opts.raw !== undefined) {
    headers["Content-Type"] = "application/json";
    body = opts.raw; // exact JSON text (preserves e.g. 0.0 vs 0 for payload digests)
  } else if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  let res: Response;
  try {
    res = await fetch(`${getApiBase()}${path}`, {
      method: opts.method ?? "GET",
      headers,
      body,
      signal: opts.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e; // cancelled by the caller
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
  taskStatus: (id: string) => request<TaskStatus>(`/tasks/${id}/status`),
  adminMe: () => request<{ admin: boolean }>("/admin/me"),
  resetDemo: () =>
    request<{ reset: boolean; deleted: Record<string, number> }>("/admin/demo/reset", { method: "POST" }),
  imageDatasets: () => request<ImageDatasetOut[]>("/image-datasets"),
  uploadImageDataset: (file: File, side = 28) => {
    const form = new FormData();
    form.append("file", file);
    return request<ImageDatasetOut>(`/image-datasets?side=${side}`, { method: "POST", form });
  },
  validateImageTask: (m: ImageTaskManifest) =>
    request<ValidationResult>("/image-tasks/validate", { method: "POST", json: m }),
  createImageTask: (m: ImageTaskManifest) =>
    request<TaskOut>("/image-tasks", { method: "POST", json: m }),
  trainingStatus: (id: string) => request<TrainingStatus>(`/tasks/${id}/training`),
  cancelTask: (id: string) => request<TaskOut>(`/tasks/${id}/cancel`, { method: "POST" }),
  taskEvents: (id: string) => request<EventOut[]>(`/tasks/${id}/events`),
  /** Authenticated download (artifacts are owner-only). */
  downloadArtifact: async (a: ArtifactOut): Promise<Blob> => {
    const token = getToken();
    const res = await fetch(`${getApiBase()}/artifacts/${a.id}/download`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) throw new ApiRequestError(res.status, "DOWNLOAD", `download failed (${res.status})`);
    return await res.blob();
  },
  registerDevice: (b: DeviceRegisterRequest) =>
    request<DeviceRegistered>("/devices", { method: "POST", json: b }),
  myDevices: () => request<DeviceOut[]>("/devices/mine"),
  patchDevice: (id: string, b: S["DevicePatch"]) =>
    request<DeviceOut>(`/devices/${id}`, { method: "PATCH", json: b }),
  networkSummary: (o: { signal?: AbortSignal } = {}) =>
    request<NetworkSummary>("/network/summary", { signal: o.signal }),
  /** Admin only: newest events across all tasks/devices, oldest first. */
  networkEvents: (o: { since?: string; limit?: number; signal?: AbortSignal } = {}) => {
    const q = new URLSearchParams();
    if (o.since) q.set("since", o.since);
    if (o.limit) q.set("limit", String(o.limit));
    const qs = q.toString();
    return request<EventOut[]>(`/network/events${qs ? `?${qs}` : ""}`, { signal: o.signal });
  },
  trustOverview: () => request<TrustOverview>("/trust/overview"),
  trustDevice: (id: string) => request<DeviceTrust>(`/trust/devices/${encodeURIComponent(id)}`),
  /** Most recent verification records (admin: all devices; others: own devices). */
  trustRecentRecords: (limit = 300) => request<VerificationRecord[]>(`/trust/records?limit=${limit}`),
  trustRecords: (taskId: string) =>
    request<VerificationRecord[]>(`/trust/records?task_id=${encodeURIComponent(taskId)}&limit=200`),
  simulate: (b: SimulateRequest) =>
    request<SimulationOut>("/trust/simulate", { method: "POST", json: b }),
  myRewards: () => request<RewardsOut>("/rewards/me"),
  networkRewards: () => request<NetworkRewardRow[]>("/rewards/network"),
  ledgerCheck: () => request<LedgerCheck>("/rewards/ledger/check"),
  securityOverview: () => request<SecurityOverview>("/security/overview"),
  quarantineDevice: (id: string, reason: string) =>
    request<{ quarantined: boolean }>(`/security/devices/${id}/quarantine`, {
      method: "POST",
      json: { reason },
    }),
  reinstateDevice: (id: string, reason: string) =>
    request<{ reinstated: boolean }>(`/security/devices/${id}/reinstate`, {
      method: "POST",
      json: { reason },
    }),
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
    startAssignment: (id: string) =>
      request<ResultAck>(`/worker/assignments/${id}/start`, { method: "POST", token: deviceToken }),
    postResult: (id: string, rawJson: string) =>
      request<ResultAck>(`/worker/assignments/${id}/result`, {
        method: "POST",
        raw: rawJson,
        token: deviceToken,
      }),
    failAssignment: (id: string, code: string, message: string) =>
      request<ResultAck>(`/worker/assignments/${id}/fail`, {
        method: "POST",
        json: { code, message },
        token: deviceToken,
      }),
    /** Chunk input .npz bytes. */
    downloadInput: async (inputUrl: string): Promise<ArrayBuffer> => {
      const res = await fetch(`${getApiBase()}${inputUrl}`, {
        headers: { Authorization: `Bearer ${deviceToken}` },
      });
      if (!res.ok) throw new ApiRequestError(res.status, "DOWNLOAD", `input download failed (${res.status})`);
      return await res.arrayBuffer();
    },
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
