import type { ReactNode } from "react";
import type { Viewport } from "next";
import { IBM_Plex_Mono, Silkscreen } from "next/font/google";
import "./globals.css";

const pixel = Silkscreen({ weight: ["400"], subsets: ["latin"], variable: "--font-silkscreen", display: "swap" });
const mono = IBM_Plex_Mono({ weight: ["400", "500"], subsets: ["latin"], variable: "--font-plex-mono", display: "swap" });

export const metadata = {
  title: "ProofNet",
  description: "Decentralized AI compute with verified, audited results.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#080c14" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${pixel.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
