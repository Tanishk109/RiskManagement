import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const title = "MerchantShield AI — Autonomous Fraud Risk Operations";
const description = "Defense-only payment fraud operations with real-time scoring, rules, graph signals, human review and honest held-out cost metrics.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const requestHost = requestHeaders.get("host") ?? "localhost:3000";
  const safeHost = /^[a-z0-9.-]+(?::\d+)?$/i.test(requestHost) ? requestHost : "localhost:3000";
  const origin = `${safeHost.startsWith("localhost") ? "http" : "https"}://${safeHost}`;

  return {
    title,
    description,
    metadataBase: new URL(origin),
    openGraph: {
      title,
      description,
      type: "website",
      images: [{ url: `${origin}/og.png`, width: 1200, height: 630, alt: "MerchantShield AI fraud risk operations command center" }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [`${origin}/og.png`],
    },
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
  };
}

export const viewport = {
  themeColor: "#080b0a",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
