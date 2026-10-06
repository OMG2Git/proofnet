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
  opts: { method?: string; json?: unknown; form?: FormData } = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
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
};
