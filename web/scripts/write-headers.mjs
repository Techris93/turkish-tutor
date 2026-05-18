import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://turkish-tutor-api.onrender.com";
const extraConnect = process.env.NEXT_PUBLIC_CSP_CONNECT_SRC || "";
const connectSrc = [
  "'self'",
  "http://127.0.0.1:8000",
  "http://localhost:8000",
  apiUrl,
  ...extraConnect.split(",").map((value) => value.trim()).filter(Boolean)
];

const content = `/*
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: no-referrer
  Permissions-Policy: camera=(), microphone=(), geolocation=()
  Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src ${[...new Set(connectSrc)].join(" ")}; media-src 'self' blob:; worker-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'
`;

const target = join(root, "public", "_headers");
mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, content, "utf8");
