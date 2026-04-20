//! Memory facts backed by SQLite + FTS5.
//!
//! Each fact has a free-form `content` string, an optional `tags` CSV,
//! and a `scope` (e.g. `global`, `workspace:<hash>`, `session:<id>`).
//! Keyword recall uses FTS5 with MATCH against content+tags. A binary
//! `embedding` column is reserved for semantic recall; when the
//! caller-supplied [`EmbeddingProvider`](crate::EmbeddingProvider)
//! returns a non-empty vector, `insert` stores it and `recall` will
//! (in a follow-up) rank semantically.

use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingProvider;
use crate::error::StateError;
use crate::store::now_ms;

/// Scope of a memory fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryScope {
    Global,
    Workspace { hash: String },
    Session { id: String },
}

impl MemoryScope {
    fn encode(&self) -> String {
        match self {
            Self::Global => "global".to_string(),
            Self::Workspace { hash } => format!("workspace:{hash}"),
            Self::Session { id } => format!("session:{id}"),
        }
    }

    fn decode(raw: &str) -> Self {
        if raw == "global" {
            Self::Global
        } else if let Some(rest) = raw.strip_prefix("workspace:") {
            Self::Workspace {
                hash: rest.to_string(),
            }
        } else if let Some(rest) = raw.strip_prefix("session:") {
            Self::Session {
                id: rest.to_string(),
            }
        } else {
            Self::Global
        }
    }
}

/// One memory fact row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFact {
    pub id: String,
    pub scope: MemoryScope,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// Query parameters for keyword recall.
#[derive(Debug, Clone)]
pub struct MemoryQuery<'a> {
    pub text: &'a str,
    pub scope: Option<&'a MemoryScope>,
    pub limit: usize,
}

/// Scoped view of the memory tables. Owns a boxed
/// [`EmbeddingProvider`] supplied by the caller so the store itself
/// stays ML-framework-agnostic.
pub struct MemoryHandle<'a> {
    conn: &'a Mutex<Connection>,
    embedder: Box<dyn EmbeddingProvider>,
}

impl<'a> MemoryHandle<'a> {
    pub(crate) fn new(conn: &'a Mutex<Connection>, embedder: Box<dyn EmbeddingProvider>) -> Self {
        Self { conn, embedder }
    }

    /// Insert a fact. If the embedding provider returns a non-empty
    /// vector, it's persisted alongside for later semantic use.
    pub fn insert(
        &self,
        scope: MemoryScope,
        content: impl Into<String>,
        tags: &[String],
    ) -> Result<MemoryFact, StateError> {
        let content = content.into();
        if content.trim().is_empty() {
            return Err(StateError::InvalidInput(
                "memory content cannot be empty".into(),
            ));
        }
        let tags_csv = tags.join(",");
        let id = generate_id();
        let now = now_ms();
        let embedding = self.embedder.embed(&content)?;
        let embedding_blob = if embedding.is_empty() {
            None
        } else {
            Some(embedding_to_blob(&embedding))
        };

        let conn = self.conn.lock().unwrap();
        let txn = conn.unchecked_transaction()?;
        txn.execute(
            "INSERT INTO memory_facts(id, scope, content, tags, embedding, created_at_ms, updated_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![id, scope.encode(), content, tags_csv, embedding_blob, now, now],
        )?;
        txn.execute(
            "INSERT INTO memory_facts_fts(rowid, content, tags)
             SELECT rowid, content, tags FROM memory_facts WHERE id = ?",
            params![id],
        )?;
        txn.commit()?;
        drop(conn);

        Ok(MemoryFact {
            id,
            scope,
            content,
            tags: tags.to_vec(),
            created_at_ms: now,
            updated_at_ms: now,
        })
    }

    /// Keyword recall over content + tags using FTS5's `MATCH` operator.
    /// Results ranked by bm25 ascending (best first) and capped at
    /// `query.limit`. A scope filter, when supplied, constrains results
    /// to that scope.
    #[allow(clippy::needless_pass_by_value)]
    pub fn recall(&self, query: MemoryQuery<'_>) -> Result<Vec<MemoryFact>, StateError> {
        if query.text.trim().is_empty() || query.limit == 0 {
            return Ok(Vec::new());
        }
        let match_expr = build_match_expression(query.text);
        let limit = i64::try_from(query.limit).unwrap_or(i64::MAX);
        let conn = self.conn.lock().unwrap();

        let mut facts = Vec::new();
        if let Some(scope) = query.scope {
            let mut stmt = conn.prepare(
                "SELECT m.id, m.scope, m.content, m.tags, m.created_at_ms, m.updated_at_ms
                 FROM memory_facts_fts f
                 JOIN memory_facts m ON m.rowid = f.rowid
                 WHERE f.memory_facts_fts MATCH ? AND m.scope = ?
                 ORDER BY bm25(memory_facts_fts) ASC
                 LIMIT ?",
            )?;
            let rows = stmt.query_map(
                params![match_expr, scope.encode(), limit],
                decode_memory_row,
            )?;
            for row in rows {
                facts.push(row?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT m.id, m.scope, m.content, m.tags, m.created_at_ms, m.updated_at_ms
                 FROM memory_facts_fts f
                 JOIN memory_facts m ON m.rowid = f.rowid
                 WHERE f.memory_facts_fts MATCH ?
                 ORDER BY bm25(memory_facts_fts) ASC
                 LIMIT ?",
            )?;
            let rows = stmt.query_map(params![match_expr, limit], decode_memory_row)?;
            for row in rows {
                facts.push(row?);
            }
        }
        Ok(facts)
    }

    /// Delete a fact by id. Returns true if a row was removed.
    pub fn delete(&self, id: &str) -> Result<bool, StateError> {
        let conn = self.conn.lock().unwrap();
        let txn = conn.unchecked_transaction()?;
        txn.execute(
            "DELETE FROM memory_facts_fts WHERE rowid IN (SELECT rowid FROM memory_facts WHERE id = ?)",
            params![id],
        )?;
        let removed = txn.execute("DELETE FROM memory_facts WHERE id = ?", params![id])?;
        txn.commit()?;
        Ok(removed > 0)
    }

    pub fn get(&self, id: &str) -> Result<Option<MemoryFact>, StateError> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, scope, content, tags, created_at_ms, updated_at_ms
             FROM memory_facts WHERE id = ?",
            params![id],
            decode_memory_row,
        )
        .optional()
        .map_err(StateError::from)
    }

    pub fn len(&self) -> Result<usize, StateError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM memory_facts", [], |row| row.get(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    pub fn is_empty(&self) -> Result<bool, StateError> {
        Ok(self.len()? == 0)
    }
}

fn decode_memory_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryFact> {
    let id: String = row.get(0)?;
    let scope_raw: String = row.get(1)?;
    let content: String = row.get(2)?;
    let tags_csv: String = row.get(3)?;
    let created_at_ms: i64 = row.get(4)?;
    let updated_at_ms: i64 = row.get(5)?;
    let tags = if tags_csv.is_empty() {
        Vec::new()
    } else {
        tags_csv.split(',').map(str::to_string).collect()
    };
    Ok(MemoryFact {
        id,
        scope: MemoryScope::decode(&scope_raw),
        content,
        tags,
        created_at_ms,
        updated_at_ms,
    })
}

/// Sanitize user input into a valid FTS5 MATCH expression. We lowercase,
/// replace non-alphanumeric/underscore with whitespace, split on
/// whitespace, and OR the tokens together. Tokens that are empty after
/// stripping are dropped. This avoids the caller accidentally building
/// FTS5 operator syntax from a raw prompt.
fn build_match_expression(text: &str) -> String {
    let mut tokens: Vec<String> = text
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| format!("\"{token}\""))
        .collect();
    if tokens.is_empty() {
        // A match expression that never matches; callers never see this
        // because recall short-circuits on empty query text, but guard
        // against a regressed caller.
        tokens.push("\"__claw_never_match__\"".to_string());
    }
    tokens.join(" OR ")
}

fn generate_id() -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let ts = u64::try_from(now_ms()).unwrap_or(0);
    let tick = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("mem_{ts:016x}{tick:08x}")
}

fn embedding_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for &value in vector {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StateStore;

    #[test]
    fn insert_and_recall_surface_matching_facts_by_keyword() {
        let store = StateStore::in_memory().expect("store");
        let memory = store.memory();
        memory
            .insert(
                MemoryScope::Global,
                "User prefers Rust for systems programming",
                &["preference".to_string(), "language".to_string()],
            )
            .expect("insert");
        memory
            .insert(MemoryScope::Global, "Working on claw-code", &[])
            .expect("insert");
        memory
            .insert(MemoryScope::Global, "Prefers dark theme UIs", &[])
            .expect("insert");

        let hits = memory
            .recall(MemoryQuery {
                text: "rust language",
                scope: None,
                limit: 5,
            })
            .expect("recall");
        assert!(
            hits.iter().any(|fact| fact.content.contains("Rust")),
            "expected a Rust fact, got {hits:?}"
        );
    }

    #[test]
    fn recall_short_circuits_on_empty_query() {
        let store = StateStore::in_memory().expect("store");
        let memory = store.memory();
        memory
            .insert(MemoryScope::Global, "something", &[])
            .unwrap();
        let hits = memory
            .recall(MemoryQuery {
                text: "   ",
                scope: None,
                limit: 5,
            })
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn recall_respects_scope_filter() {
        let store = StateStore::in_memory().expect("store");
        let memory = store.memory();
        memory
            .insert(MemoryScope::Global, "global scope fact about Rust", &[])
            .unwrap();
        memory
            .insert(
                MemoryScope::Session {
                    id: "session-42".into(),
                },
                "session fact about Rust",
                &[],
            )
            .unwrap();

        let session_hits = memory
            .recall(MemoryQuery {
                text: "rust",
                scope: Some(&MemoryScope::Session {
                    id: "session-42".into(),
                }),
                limit: 10,
            })
            .unwrap();
        assert_eq!(session_hits.len(), 1);
        assert!(session_hits[0].content.contains("session fact"));
    }

    #[test]
    fn insert_rejects_empty_content() {
        let store = StateStore::in_memory().expect("store");
        let memory = store.memory();
        let error = memory
            .insert(MemoryScope::Global, "   ", &[])
            .expect_err("empty content must fail");
        assert!(format!("{error}").contains("cannot be empty"));
    }

    #[test]
    fn delete_removes_fact_from_both_table_and_fts_index() {
        let store = StateStore::in_memory().expect("store");
        let memory = store.memory();
        let fact = memory
            .insert(MemoryScope::Global, "target fact", &[])
            .unwrap();
        assert!(memory.delete(&fact.id).unwrap());
        assert!(memory.get(&fact.id).unwrap().is_none());
        let hits = memory
            .recall(MemoryQuery {
                text: "target",
                scope: None,
                limit: 5,
            })
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn recall_honors_limit() {
        let store = StateStore::in_memory().expect("store");
        let memory = store.memory();
        for i in 0..10 {
            memory
                .insert(
                    MemoryScope::Global,
                    format!("fact about rust number {i}"),
                    &[],
                )
                .unwrap();
        }
        let hits = memory
            .recall(MemoryQuery {
                text: "rust",
                scope: None,
                limit: 3,
            })
            .unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn memory_scope_encode_decode_roundtrip() {
        let scopes = [
            MemoryScope::Global,
            MemoryScope::Workspace { hash: "abc".into() },
            MemoryScope::Session { id: "42".into() },
        ];
        for scope in scopes {
            let encoded = scope.encode();
            let decoded = MemoryScope::decode(&encoded);
            assert_eq!(scope, decoded);
        }
    }

    #[test]
    fn build_match_expression_quotes_tokens_and_strips_punctuation() {
        assert_eq!(
            build_match_expression("I love Rust!  programming"),
            "\"i\" OR \"love\" OR \"rust\" OR \"programming\""
        );
    }
}
