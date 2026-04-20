//! Exact-match request/response cache.
//!
//! Cache key is `sha256(system_prompt + messages_json + model)` — any
//! byte-level change in the request misses. That's fine for the common
//! case the cache targets: tool loops that re-send the same prompt or
//! batch jobs that replay a fixed dataset. Paraphrase-aware hits are
//! a semantic problem; `claw-state`'s memory handle is the right place
//! for that once embeddings are in.

use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::error::StateError;
use crate::store::now_ms;

/// Stable cache key computed from a request's semantic content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey(String);

impl CacheKey {
    /// Build a key from normalized request components. `messages` is
    /// the already-serialized JSON the wire would carry so we hash the
    /// exact bytes the upstream would see.
    #[must_use]
    pub fn from_parts(system_prompt: &str, messages_json: &str, model: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"v1\x1f");
        hasher.update(model.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(system_prompt.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(messages_json.as_bytes());
        let digest = hasher.finalize();
        Self(hex(&digest))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0F) as usize] as char);
    }
    out
}

/// One row returned from a successful cache lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    pub response_json: String,
    pub model: String,
    pub created_at_ms: i64,
    pub expires_at_ms: Option<i64>,
}

/// Scoped handle over the `request_cache` table.
pub struct CacheHandle<'a> {
    conn: &'a Mutex<Connection>,
}

impl<'a> CacheHandle<'a> {
    pub(crate) fn new(conn: &'a Mutex<Connection>) -> Self {
        Self { conn }
    }

    /// Look up a previously-stored response by key. Entries past their
    /// TTL are treated as misses but are not eagerly evicted here; call
    /// [`CacheHandle::purge_expired`] from a maintenance path if that
    /// matters.
    pub fn get(&self, key: &CacheKey) -> Result<Option<CacheEntry>, StateError> {
        let conn = self.conn.lock().unwrap();
        let now = now_ms();
        let row: Option<CacheEntry> = conn
            .query_row(
                "SELECT response_json, model, created_at_ms, expires_at_ms
                 FROM request_cache WHERE cache_key = ?",
                params![key.as_str()],
                |row| {
                    Ok(CacheEntry {
                        response_json: row.get(0)?,
                        model: row.get(1)?,
                        created_at_ms: row.get(2)?,
                        expires_at_ms: row.get(3)?,
                    })
                },
            )
            .optional()?;
        Ok(row.filter(|entry| entry.expires_at_ms.is_none_or(|expires| expires > now)))
    }

    /// Insert or replace a cache entry. `ttl_ms = None` means the entry
    /// never expires.
    pub fn put(
        &self,
        key: &CacheKey,
        response_json: &str,
        model: &str,
        ttl_ms: Option<u64>,
    ) -> Result<(), StateError> {
        let created = now_ms();
        let expires =
            ttl_ms.map(|ttl| created.saturating_add(i64::try_from(ttl).unwrap_or(i64::MAX)));
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO request_cache(cache_key, response_json, model, created_at_ms, expires_at_ms)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(cache_key) DO UPDATE SET
                response_json = excluded.response_json,
                model = excluded.model,
                created_at_ms = excluded.created_at_ms,
                expires_at_ms = excluded.expires_at_ms",
            params![key.as_str(), response_json, model, created, expires],
        )?;
        Ok(())
    }

    /// Drop every entry whose `expires_at_ms` is in the past. Returns
    /// the number of rows removed.
    pub fn purge_expired(&self) -> Result<usize, StateError> {
        let conn = self.conn.lock().unwrap();
        let removed = conn.execute(
            "DELETE FROM request_cache WHERE expires_at_ms IS NOT NULL AND expires_at_ms <= ?",
            params![now_ms()],
        )?;
        Ok(removed)
    }

    pub fn len(&self) -> Result<usize, StateError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM request_cache", [], |row| row.get(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    pub fn is_empty(&self) -> Result<bool, StateError> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StateStore;

    #[test]
    fn cache_key_changes_when_any_component_changes() {
        let a = CacheKey::from_parts("sys", "msgs", "model");
        let b = CacheKey::from_parts("sys2", "msgs", "model");
        let c = CacheKey::from_parts("sys", "msgs2", "model");
        let d = CacheKey::from_parts("sys", "msgs", "model2");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_eq!(a.as_str().len(), 64);
    }

    #[test]
    fn put_then_get_returns_stored_response() {
        let store = StateStore::in_memory().expect("store");
        let cache = store.cache();
        let key = CacheKey::from_parts("sys", "[]", "claude-haiku-4-5");
        cache
            .put(&key, "{\"id\":\"msg_1\"}", "claude-haiku-4-5", None)
            .expect("put");
        let hit = cache.get(&key).expect("get").expect("cache hit");
        assert_eq!(hit.response_json, "{\"id\":\"msg_1\"}");
        assert_eq!(hit.model, "claude-haiku-4-5");
        assert_eq!(hit.expires_at_ms, None);
    }

    #[test]
    fn get_returns_none_for_unknown_key() {
        let store = StateStore::in_memory().expect("store");
        let cache = store.cache();
        let key = CacheKey::from_parts("a", "b", "c");
        assert!(cache.get(&key).expect("get").is_none());
    }

    #[test]
    fn expired_entries_are_invisible_to_get_and_purged_by_call() {
        let store = StateStore::in_memory().expect("store");
        let cache = store.cache();
        let key = CacheKey::from_parts("sys", "[]", "model");
        cache
            .put(&key, "{}", "model", Some(0))
            .expect("put with zero TTL");
        // TTL 0 means expires at insert time, so get must miss.
        assert!(cache.get(&key).expect("get").is_none());
        let removed = cache.purge_expired().expect("purge");
        assert_eq!(removed, 1);
        assert_eq!(cache.len().expect("len"), 0);
    }

    #[test]
    fn put_overwrites_existing_entry_for_same_key() {
        let store = StateStore::in_memory().expect("store");
        let cache = store.cache();
        let key = CacheKey::from_parts("sys", "[]", "model");
        cache.put(&key, "{\"v\":1}", "model", None).unwrap();
        cache.put(&key, "{\"v\":2}", "model", None).unwrap();
        let hit = cache.get(&key).unwrap().unwrap();
        assert_eq!(hit.response_json, "{\"v\":2}");
        assert_eq!(cache.len().unwrap(), 1);
    }
}
