"use client";

import AuthGate from "@/components/AuthGate";
import NetworkDashboard from "@/components/network/NetworkDashboard";

export default function NetworkPage() {
  return (
    <AuthGate>
      <NetworkDashboard />
    </AuthGate>
  );
}
