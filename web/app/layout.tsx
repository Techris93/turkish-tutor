import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Türkçe Hoca",
  description: "A CEFR-aware Turkish tutor workspace",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Türkçe Hoca"
  }
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
