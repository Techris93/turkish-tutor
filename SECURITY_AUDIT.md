# Security Audit - Turkce Hoca

Date: 2026-05-18
Scope: FastAPI API, Next.js static frontend, authentication/session flows, OAuth, password reset, saved lessons, OCR/file extraction, Gemini study generation, generated-audio TTS, Render deployment, and dependency manifests.

## 1. Vulnerability Summary

- Critical: 1 total, 1 fixed
- High: 3 total, 3 fixed
- Medium: 8 total, 4 fixed, 4 documented risks
- Low: 5 total, 1 fixed, 4 documented risks

Overall risk posture after this audit is moderate for a 1-2 person personal learning app. The highest-risk code issues found during the audit were fixed: API text input no longer reads arbitrary server files, upload parsing now has type and size limits, Google OAuth now requires verified email, production API docs are disabled by default, security headers were added, and a known vulnerable PostCSS transitive dependency was overridden to a patched version. Remaining risk is mostly architectural and operational: remembered session tokens in `localStorage`, simple in-process rate limiting, third-party AI/TTS data sharing, and SQLite/local-device assumptions if used outside the Render Postgres blueprint.

### Threat Model

Attacker profiles considered:

- Anonymous internet user probing `/api/study`, auth, OAuth, reset, file upload, and TTS endpoints.
- Authenticated user attempting cross-user lesson access, provider cost abuse, or data exfiltration.
- Attacker with a stolen `X-Session-Token` from browser storage.
- OAuth account-confusion attacker trying to bind or log in as another email identity.
- API consumer/bot bypassing frontend controls.
- Insider/operator with Render environment or database access.

Sensitive assets:

- User accounts, password hashes, session tokens, reset tokens, OAuth states/handoffs.
- Saved lesson content, uploaded/extracted text, OCR/PDF/DOCX content.
- Gemini, OpenAI, SMTP, OAuth, database, and Render secrets.
- Generated TTS provider spend and Gemini quota.

Primary trust boundaries:

- Browser to API over CORS/cookies/`X-Session-Token`.
- API to database.
- API to Gemini/OpenAI/SMTP/OAuth providers.
- API to local file parsers and Tesseract.
- Render static site to Render API service.

## 2. Detailed Findings

### 1. Server-Side Arbitrary File Read Through Study Text Input

- Severity: Critical
- Affected component: `content_intelligence.extract_content`, `/api/study`
- Description: The study text path reused CLI behavior that treated direct text as a local filesystem path if that path existed. An anonymous API user could submit a server-side path such as `/app/.env` or another readable file. The extracted content would then be sent into the Gemini study flow and could appear in the response.
- Exploitation scenario:
  1. Attacker sends `POST /api/study` with `text=/app/.env`.
  2. API resolves the path and reads the file.
  3. File contents are included in the study prompt/preview.
  4. Gemini response or API response can expose secrets or internal data.
- Impact: Secret disclosure, database/API credential exposure, downstream account or provider compromise.
- Recommended fix: Make server/API direct text input text-only. Keep local path extraction only for explicit trusted CLI use.
- Status: Fixed now. `extract_content(..., allow_paths=False)` is the default, API uses text-only behavior, and the CLI opts into local path extraction.

### 2. Unbounded File Upload Size Enables Parser DoS

- Severity: High
- Affected component: `/api/study`, OCR/PDF/DOCX/text extraction
- Description: Uploaded files were streamed to a temporary file without an application-level byte cap. Large files, image bombs, PDFs, or DOCX documents could consume disk, CPU, OCR time, or memory before failing.
- Exploitation scenario:
  1. Attacker repeatedly uploads very large images/PDFs.
  2. API writes them to disk and invokes file parsers/OCR.
  3. Worker CPU/disk is exhausted and legitimate users lose availability.
- Impact: Denial of service and cost/resource exhaustion.
- Recommended fix: Enforce a maximum upload byte limit before extraction and document the setting.
- Status: Fixed now. `MAX_UPLOAD_BYTES` defaults to 10 MB and is enforced while streaming.

### 3. Unsupported File Types Reached Extraction Layer

- Severity: High
- Affected component: `/api/study` file upload
- Description: Uploaded file extensions were not rejected until deeper extraction. This widened parser attack surface and made error behavior less predictable.
- Exploitation scenario:
  1. Attacker uploads executable, archive, or unusual binary file names.
  2. API writes the file and passes it toward extraction.
  3. Parser behavior may waste resources or reveal inconsistent errors.
- Impact: Increased DoS and parser attack surface.
- Recommended fix: Reject unsupported suffixes before writing/extracting.
- Status: Fixed now. Uploads are allowlisted to text, PDF, DOCX, and common image extensions.

### 4. Google OAuth Did Not Require Verified Email

- Severity: High
- Affected component: OAuth account login/linking
- Description: GitHub profile handling required a verified primary email, but Google profile handling accepted any `email` value returned by userinfo without checking `email_verified`.
- Exploitation scenario:
  1. Attacker obtains an OAuth profile response containing an email address that is not verified.
  2. API finds or creates a local user by that email.
  3. Attacker may access an account identified by that email.
- Impact: Account takeover if a provider ever returns an unverified or weakly verified email.
- Recommended fix: Require `email_verified === true` for Google profiles before account creation/linking.
- Status: Fixed now with regression coverage.

### 5. Production FastAPI Docs/OpenAPI Exposed by Default

- Severity: Medium
- Affected component: FastAPI app configuration
- Description: `/docs`, `/redoc`, and `/openapi.json` expose route structure and schemas. This is not a direct exploit, but it reduces attacker effort in production.
- Exploitation scenario:
  1. Attacker visits `/openapi.json`.
  2. Attacker enumerates high-value endpoints and request models.
  3. Attacker automates brute force or provider-cost probes.
- Impact: Reconnaissance acceleration.
- Recommended fix: Disable docs/OpenAPI in production and enable only intentionally.
- Status: Fixed now. Docs are disabled on Render unless `ENABLE_API_DOCS=true`.

### 6. Missing Security Headers

- Severity: Medium
- Affected component: FastAPI responses and static frontend deployment
- Description: Responses lacked standard hardening headers such as `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and CSP.
- Exploitation scenario:
  1. Attacker frames the app or exploits browser MIME sniffing/overly broad defaults.
  2. If a content injection bug appears later, lack of CSP increases blast radius.
- Impact: Clickjacking, content-sniffing, and XSS impact amplification.
- Recommended fix: Add conservative headers to API and static frontend.
- Status: Fixed now. API middleware adds headers; `web/public/_headers` adds static-site headers for hosts that honor that file. If using custom domains/API URLs, update CSP `connect-src`.

### 7. Vulnerable PostCSS Transitive Dependency

- Severity: Medium
- Affected component: frontend dependency tree
- Description: `npm audit` reported PostCSS `<8.5.10` via Next.js with an XSS advisory in CSS stringify output.
- Exploitation scenario:
  1. Attacker controls CSS-like content passed through vulnerable stringify paths.
  2. Malicious `</style>` output can break context and inject HTML/script in affected usage.
- Impact: Potential XSS in vulnerable build/runtime paths.
- Recommended fix: Use patched PostCSS.
- Status: Fixed now. `package.json` overrides PostCSS to `8.5.14`; `npm audit` now reports zero vulnerabilities.

### 8. Remembered Session Token Stored in localStorage

- Severity: Medium
- Affected component: frontend auth fallback and Remember me
- Description: The app intentionally stores a fallback session token in `localStorage` when Remember me is checked. This is practical for Render cross-subdomain sessions, but any future XSS, malicious extension, or shared-device compromise can steal the token.
- Exploitation scenario:
  1. Attacker executes JavaScript in the app origin through a future XSS or browser extension.
  2. Script reads `turkce-hoca.remembered-session-token.v1`.
  3. Attacker replays it in `X-Session-Token` until expiration/logout.
- Impact: Account takeover for the session lifetime.
- Recommended fix: Prefer a shared custom domain and HTTP-only refresh/session cookies. Keep Remember me opt-in and clearly labeled for trusted devices only.
- Status: Documented risk. Current app keeps Remember me opt-in, logout invalidates the server session, and CSP/React escaping reduce XSS risk.

### 9. CSRF Protection Is Limited

- Severity: Medium
- Affected component: cookie-authenticated POST endpoints
- Description: Production uses `SameSite=None` cookies for cross-subdomain API access. Most state-changing endpoints require JSON and CORS-readable responses, but browser form POSTs can still hit simple endpoints such as logout. There is no explicit CSRF token.
- Exploitation scenario:
  1. User is logged in with a cross-site cookie.
  2. Attacker page submits a cross-site POST to `/api/auth/logout`.
  3. User is logged out without consent.
- Impact: Mostly logout CSRF today; risk grows if form-compatible state-changing endpoints are added.
- Recommended fix: Add CSRF tokens or enforce Origin/Referer checks on cookie-authenticated unsafe methods. Header-token-only auth avoids this but trades toward token theft risk.
- Status: Documented risk. No high-impact form-compatible authenticated mutation was found beyond logout.

### 10. In-Process Rate Limiting Is Not Distributed

- Severity: Medium
- Affected component: `rate_limit.py`, Render deployment
- Description: Rate limits are memory-local. Multiple API instances would each maintain separate counters, and counters reset on restart. Proxy/IP header trust remains deployment-sensitive.
- Exploitation scenario:
  1. Attacker distributes requests across instances or waits for restarts.
  2. Attacker bypasses intended login/study/TTS quotas.
- Impact: Brute force, Gemini/OpenAI denial-of-wallet, and availability risk.
- Recommended fix: Use Redis or another shared store for rate limits if the app becomes public or scales beyond one instance.
- Status: Documented risk. Acceptable for current small single-instance app.

### 11. Third-Party AI/TTS Data Disclosure Boundary

- Severity: Medium
- Affected component: Gemini study generation, OpenAI TTS, SMTP reset email
- Description: Uploaded/extracted lesson text is sent to Gemini for study generation and generated audio text is sent to OpenAI when generated audio is explicitly enabled. Reset tokens are sent through SMTP email. This is expected behavior but creates external data processors.
- Exploitation scenario:
  1. User uploads sensitive personal data as a lesson.
  2. API sends excerpts/text to third-party providers.
  3. Provider-side logging/retention policies apply.
- Impact: Privacy/compliance exposure.
- Recommended fix: Add an in-app privacy notice, avoid uploading sensitive content, and use provider settings/contracts appropriate for production.
- Status: Documented risk.

### 12. Prompt Injection Can Influence Tutor Output

- Severity: Medium
- Affected component: `/api/study`, Gemini prompts
- Description: Uploaded text is placed inside prompts. A malicious document can instruct the model to ignore tutor rules, generate misleading lessons, or attempt social engineering. No direct server secrets are intentionally in prompts after the file-read fix, but model output can still be manipulated.
- Exploitation scenario:
  1. Attacker shares a PDF containing hidden prompt instructions.
  2. User uploads it.
  3. Gemini output contains attacker-influenced teaching content or deceptive links/text.
- Impact: Integrity and trust risk, especially for shared lessons.
- Recommended fix: Continue treating model output as untrusted, never include secrets in prompts, and consider prompt-injection classifiers or visible source previews.
- Status: Documented risk.

### 13. SQLite Local Persistence Is Not a Production Backup Strategy

- Severity: Low
- Affected component: local/default database configuration
- Description: Local default SQLite is useful for development and a 1-2 person local install, but should not be treated as durable cloud storage without backups. Render blueprint uses Postgres.
- Exploitation scenario:
  1. Operator deploys SQLite on an ephemeral filesystem.
  2. Instance redeploy/restart loses account/session/lesson data.
- Impact: Data loss.
- Recommended fix: Use the included Render Postgres blueprint for hosted deployments; back up the database for important lessons.
- Status: Documented risk.

### 14. Account Enumeration on Signup

- Severity: Low
- Affected component: `/api/auth/signup`
- Description: Signup returns a distinct `409` for existing emails. Login and reset messages are more generic.
- Exploitation scenario:
  1. Attacker submits email list to signup.
  2. `409` responses identify registered users.
- Impact: User privacy leakage.
- Recommended fix: For a public app, return generic signup/login messaging or add stricter rate limits/CAPTCHA.
- Status: Documented risk. Current target audience is 1-2 users with rate limiting.

### 15. Password Policy Is Minimal

- Severity: Low
- Affected component: signup/reset password validation
- Description: Passwords require only eight characters. Argon2 hashing is strong, but weak user-chosen passwords remain possible.
- Exploitation scenario:
  1. Attacker performs password guessing under rate limits.
  2. Weak passwords are more likely to be guessed.
- Impact: Account compromise for weak passwords.
- Recommended fix: Add breached-password checks or zxcvbn-style strength feedback if the app grows.
- Status: Documented risk.

### 16. Dev Token Mode Can Expose Reset Tokens if Misconfigured

- Severity: Low
- Affected component: password reset development mode
- Description: `PASSWORD_RESET_RETURN_TOKEN=true` returns reset tokens in API responses. This is intended for local testing only.
- Exploitation scenario:
  1. Operator accidentally enables dev-token mode in production.
  2. Anyone requesting a reset for a known email receives a valid reset token.
- Impact: Account takeover.
- Recommended fix: Keep `PASSWORD_RESET_RETURN_TOKEN=false` in production and monitor env configuration.
- Status: Partially fixed/documented. Render blueprint sets it false.

### 17. Static CSP Must Be Updated for Custom Domains

- Severity: Low
- Affected component: `web/public/_headers`
- Description: Static CSP currently includes localhost and the default Render API URL. Custom API domains require updating `connect-src`.
- Exploitation scenario:
  1. Operator deploys to a custom API domain.
  2. Browser blocks API calls due to CSP.
- Impact: Availability/configuration issue.
- Recommended fix: Update `web/public/_headers` when changing deployment domains.
- Status: Documented risk in README.

## 3. Attack Chains

### Chain A: Server File Read to Full Provider/Database Compromise

Pre-fix path:

1. Anonymous attacker submits a server path to `/api/study`.
2. API reads `.env` or other local config.
3. Gemini response leaks API keys or database URLs.
4. Attacker uses leaked keys against Gemini/OpenAI/SMTP/database.

Status: Broken by the `allow_paths=False` API behavior.

### Chain B: XSS or Browser Extension to Remembered Token Replay

1. User enables Remember me on a shared or compromised browser.
2. Future XSS, extension, or local malware reads the remembered token.
3. Attacker sends `X-Session-Token` from another client.
4. Attacker accesses saved lessons until logout/session expiry.

Status: Remaining architectural risk. Mitigate with trusted-device-only usage, logout, CSP, React escaping, and ideally shared-domain HTTP-only cookies.

### Chain C: Cost Exhaustion Through Uploads and Generated Audio

Pre-fix and residual path:

1. Attacker repeatedly submits large files and study requests.
2. OCR/PDF parsing and Gemini consume CPU/quota.
3. Authenticated attacker enables generated audio to consume OpenAI TTS quota.
4. In-memory rate limits slow but do not fully prevent distributed or multi-instance abuse.

Status: Upload size/type limits and endpoint rate limits reduce risk. Redis/shared limiting and stronger account controls are recommended if public.

### Chain D: OAuth Account Confusion

Pre-fix path:

1. OAuth provider returns an email that is not verified.
2. API links/logs in by email only.
3. Attacker gains account access for that email.

Status: Google now requires `email_verified`; GitHub already requires a verified primary email.

## 4. Secure Design Recommendations

- Use a shared custom domain such as `app.example.com` and `api.example.com` or a reverse proxy under one site so HTTP-only cookies work reliably without `localStorage` session fallback.
- Add CSRF protection or strict Origin checks for all cookie-authenticated unsafe methods before adding more form-compatible endpoints.
- Replace in-process rate limiting with Redis/shared counters if the app is opened to broad public use or scaled to multiple API instances.
- Add upload scanning/timeouts and parser sandboxing for hostile PDFs/DOCX/images if arbitrary users can upload files.
- Add a visible privacy notice: uploaded content may be sent to Gemini, and generated audio text may be sent to OpenAI when generated audio is selected.
- Keep `PASSWORD_RESET_RETURN_TOKEN=false`, `ENABLE_API_DOCS=false`, and real SMTP/OAuth secrets configured only in Render secret env vars.
- Review CSP whenever deployment domains change.
- Consider a dependency maintenance cadence: run `npm audit`, `pip-audit`, and framework update checks before each production deploy.
- Add operational logging for rate-limit hits, failed logins, reset requests, OAuth failures, and provider-cost endpoints, while never logging tokens or lesson contents.
- Add database backups for Postgres if saved lessons matter.
