import assert from "node:assert/strict";
import { test } from "node:test";
import {
  OAUTH_REMEMBER_KEY,
  REMEMBERED_SESSION_TOKEN_KEY,
  SESSION_TOKEN_KEY,
  consumeOAuthRememberPreference,
  getStoredSessionToken,
  storeOAuthRememberPreference,
  storeSessionToken
} from "./authTokens";

class MemoryStorage implements Storage {
  private values = new Map<string, string>();

  get length() {
    return this.values.size;
  }

  clear() {
    this.values.clear();
  }

  getItem(key: string) {
    return this.values.get(key) ?? null;
  }

  key(index: number) {
    return Array.from(this.values.keys())[index] ?? null;
  }

  removeItem(key: string) {
    this.values.delete(key);
  }

  setItem(key: string, value: string) {
    this.values.set(key, value);
  }
}

test("default token storage uses session storage only", () => {
  const session = new MemoryStorage();
  const remembered = new MemoryStorage();
  storeSessionToken(session, remembered, "token-one", false);

  assert.equal(session.getItem(SESSION_TOKEN_KEY), "token-one");
  assert.equal(remembered.getItem(REMEMBERED_SESSION_TOKEN_KEY), null);
  assert.equal(getStoredSessionToken(session, remembered), "token-one");
});

test("remembered token storage persists to local storage and restores later", () => {
  const session = new MemoryStorage();
  const remembered = new MemoryStorage();
  storeSessionToken(session, remembered, "token-two", true);
  session.clear();

  assert.equal(remembered.getItem(REMEMBERED_SESSION_TOKEN_KEY), "token-two");
  assert.equal(getStoredSessionToken(session, remembered), "token-two");
});

test("clearing tokens removes session and remembered storage", () => {
  const session = new MemoryStorage();
  const remembered = new MemoryStorage();
  storeSessionToken(session, remembered, "token-three", true);
  storeSessionToken(session, remembered, null);

  assert.equal(session.getItem(SESSION_TOKEN_KEY), null);
  assert.equal(remembered.getItem(REMEMBERED_SESSION_TOKEN_KEY), null);
  assert.equal(getStoredSessionToken(session, remembered), "");
});

test("oauth remember preference is single-use", () => {
  const remembered = new MemoryStorage();
  storeOAuthRememberPreference(remembered, true);

  assert.equal(remembered.getItem(OAUTH_REMEMBER_KEY), "true");
  assert.equal(consumeOAuthRememberPreference(remembered), true);
  assert.equal(consumeOAuthRememberPreference(remembered), false);
});
