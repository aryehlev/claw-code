use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::Connection;

use crate::cache::CacheHandle;
use crate::embedding::NoopEmbeddingProvider;
use crate::error::StateError;
use crate::memory::MemoryHandle;
use crate::scoreboard::ScoreboardHandle;

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS router_scores (
    bucket TEXT NOT NULL,
    model TEXT NOT NULL,
    successes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    total_latency_ms INTEGER NOT NULL DEFAULT 0,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (bucket, model)
);

CREATE TABLE IF NOT EXISTS request_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    expires_at_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_request_cache_expires
    ON request_cache(expires_at_ms);

CREATE TABLE IF NOT EXISTS memory_facts (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    embedding BLOB,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_facts_scope
    ON memory_facts(scope);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_facts_fts USING fts5(
    content,
    tags,
    content='memory_facts',
    content_rowid='rowid'
);
";

const SCHEMA_VERSION: i64 = 1;

/// Process-local handle to the SQLite-backed state store. One store owns
/// one connection; callers derive scoped handles for each domain
/// (scoreboard, cache, memory) which share the connection via an internal
/// `Mutex` so a single `StateStore` can serve the CLI's single-threaded
/// runtime without the caller juggling mutability.
pub struct StateStore {
    conn: Mutex<Connection>,
    path: PathBuf,
}

impl StateStore {
    /// Open (or create) a state store at `path`. Runs schema migrations
    /// before returning so every handle sees the expected tables.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StateError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(&path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        record_schema_version(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            path,
        })
    }

    /// Open an in-memory store. Used by tests and for the default
    /// fallback when a caller asks for state but can't commit to a file
    /// path.
    pub fn in_memory() -> Result<Self, StateError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        record_schema_version(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            path: PathBuf::new(),
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn scoreboard(&self) -> ScoreboardHandle<'_> {
        ScoreboardHandle::new(&self.conn)
    }

    #[must_use]
    pub fn cache(&self) -> CacheHandle<'_> {
        CacheHandle::new(&self.conn)
    }

    #[must_use]
    pub fn memory(&self) -> MemoryHandle<'_> {
        MemoryHandle::new(&self.conn, Box::new(NoopEmbeddingProvider))
    }

    /// Memory handle with a caller-supplied embedding provider. Lets the
    /// CLI plug in the candle-backed provider without the base crate
    /// taking on the dep.
    pub fn memory_with_embeddings(
        &self,
        provider: Box<dyn crate::embedding::EmbeddingProvider>,
    ) -> MemoryHandle<'_> {
        MemoryHandle::new(&self.conn, provider)
    }
}

fn record_schema_version(conn: &Connection) -> Result<(), StateError> {
    let mut stmt = conn.prepare("SELECT version FROM schema_version LIMIT 1")?;
    let existing: Option<i64> = stmt.query_row([], |row| row.get(0)).ok();
    match existing {
        Some(version) if version == SCHEMA_VERSION => Ok(()),
        Some(other) => Err(StateError::InvalidInput(format!(
            "state store schema version {other} does not match expected {SCHEMA_VERSION}"
        ))),
        None => {
            conn.execute(
                "INSERT INTO schema_version(version) VALUES (?)",
                [SCHEMA_VERSION],
            )?;
            Ok(())
        }
    }
}

pub(crate) fn now_ms() -> i64 {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_in_memory_applies_schema_and_records_version() {
        let store = StateStore::in_memory().expect("open");
        let conn = store.conn.lock().unwrap();
        let version: i64 = conn
            .query_row("SELECT version FROM schema_version", [], |row| row.get(0))
            .expect("schema_version row");
        assert_eq!(version, SCHEMA_VERSION);
        // Every expected table exists.
        for table in &[
            "router_scores",
            "request_cache",
            "memory_facts",
            "memory_facts_fts",
        ] {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = ?",
                    [table],
                    |row| row.get(0),
                )
                .expect("sqlite_master query");
            assert!(exists > 0, "missing table {table}");
        }
    }

    #[test]
    fn open_rejects_mismatched_schema_version() {
        let dir = std::env::temp_dir().join(format!("claw-state-ver-{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.sqlite");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(SCHEMA).unwrap();
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", [99])
                .unwrap();
        }
        let error = match StateStore::open(&path) {
            Err(error) => error,
            Ok(_) => panic!("mismatched version must fail"),
        };
        assert!(
            format!("{error}").contains("schema version"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
