import type { components } from "@/lib/api/schema";

type Caps = components["schemas"]["DeviceCapabilities"];

type NavExtras = Navigator & {
  deviceMemory?: number;
  connection?: { effectiveType?: string };
  userAgentData?: {
    getHighEntropyValues?: (hints: string[]) => Promise<{ model?: string; platformVersion?: string }>;
  };
  getBattery?: () => Promise<{ level: number; charging: boolean }>;
};

/** Static capabilities the browser genuinely exposes (ARCHITECTURE 7.3). Unknown stays unset. */
export async function collectCapabilities(): Promise<Caps> {
  const nav = navigator as NavExtras;
  const caps: Caps = {};
  caps.logical_cores = nav.hardwareConcurrency || undefined;
  caps.memory_gb_reported = nav.deviceMemory;
  caps.network_type = nav.connection?.effectiveType;
  try {
    const est = await navigator.storage?.estimate?.();
    if (est?.quota) caps.storage_quota_mb = Math.round(est.quota / 1048576);
  } catch {
    /* not available */
  }
  try {
    const hi = await nav.userAgentData?.getHighEntropyValues?.(["model", "platformVersion"]);
    caps.model = hi?.model || undefined;
    caps.platform_version = hi?.platformVersion || undefined;
  } catch {
    /* not available */
  }
  if (!caps.model) caps.model = navigator.userAgent.slice(0, 120);
  const battery = await readBattery();
  caps.battery = battery?.level;
  caps.charging = battery?.charging;
  return caps;
}

export async function readBattery(): Promise<{ level: number; charging: boolean } | null> {
  try {
    const b = await (navigator as NavExtras).getBattery?.();
    return b ? { level: b.level, charging: b.charging } : null;
  } catch {
    return null;
  }
}

export function guessDeviceType(): "android_phone" | "laptop" | "desktop" | "other" {
  const ua = navigator.userAgent;
  if (/Android/i.test(ua) && /Mobile/i.test(ua)) return "android_phone";
  if (/Android|iPhone|iPad/i.test(ua)) return "other";
  return "laptop";
}
