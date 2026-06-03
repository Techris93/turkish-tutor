import type { MetadataRoute } from "next";

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Türkçe Hoca",
    short_name: "Türkçe Hoca",
    description: "CEFR-aware Turkish tutor with vocabulary read-aloud practice.",
    start_url: "/",
    scope: "/",
    display: "standalone",
    background_color: "#f6f4ee",
    theme_color: "#176b5b",
    orientation: "portrait",
    icons: [
      {
        src: "/icon-192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "any"
      },
      {
        src: "/icon-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "any"
      },
      {
        src: "/icon.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "any"
      },
      {
        src: "/icon.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "maskable"
      }
    ],
    categories: ["education", "productivity"],
    shortcuts: [
      {
        name: "Study Turkish",
        short_name: "Study",
        description: "Open the Turkish tutor workspace.",
        url: "/"
      }
    ]
  };
}
