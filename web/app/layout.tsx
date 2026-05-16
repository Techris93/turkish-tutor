import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Turkce Hoca",
  description: "A CEFR-aware Turkish tutor workspace"
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
