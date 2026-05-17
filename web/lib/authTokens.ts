export const SESSION_TOKEN_KEY = "turkce-hoca.session-token.v1";
export const REMEMBERED_SESSION_TOKEN_KEY = "turkce-hoca.remembered-session-token.v1";
export const OAUTH_REMEMBER_KEY = "turkce-hoca.oauth-remember.v1";

export type TokenStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function getStoredSessionToken(sessionStore: TokenStorage, rememberedStore: TokenStorage): string {
  return sessionStore.getItem(SESSION_TOKEN_KEY) ?? rememberedStore.getItem(REMEMBERED_SESSION_TOKEN_KEY) ?? "";
}

export function storeSessionToken(
  sessionStore: TokenStorage,
  rememberedStore: TokenStorage,
  token?: string | null,
  remember = false
) {
  if (token) {
    sessionStore.setItem(SESSION_TOKEN_KEY, token);
    if (remember) {
      rememberedStore.setItem(REMEMBERED_SESSION_TOKEN_KEY, token);
    } else {
      rememberedStore.removeItem(REMEMBERED_SESSION_TOKEN_KEY);
    }
    return;
  }
  sessionStore.removeItem(SESSION_TOKEN_KEY);
  rememberedStore.removeItem(REMEMBERED_SESSION_TOKEN_KEY);
}

export function storeOAuthRememberPreference(rememberedStore: TokenStorage, remember: boolean) {
  if (remember) {
    rememberedStore.setItem(OAUTH_REMEMBER_KEY, "true");
  } else {
    rememberedStore.removeItem(OAUTH_REMEMBER_KEY);
  }
}

export function consumeOAuthRememberPreference(rememberedStore: TokenStorage): boolean {
  const remember = rememberedStore.getItem(OAUTH_REMEMBER_KEY) === "true";
  rememberedStore.removeItem(OAUTH_REMEMBER_KEY);
  return remember;
}
