//! Unified in-process state store for claw.
//!
//! Replaces the earlier `eval-router` (JSON file scoreboard) and
//! `memory-client` (Zep REST sidecar) crates with one SQLite-backed
//! store covering:
//!
//! - **Router scoreboard**: per-`(bucket, model)` success rates for the
//!   eval-driven router.
//! - **Exact-match cache**: keyed on `sha256(system_prompt + messages +
//!   model)` for zero-cost replay of identical requests (e.g. tool loops
//!   that re-send the same prompt).
//! - **Memory facts**: durable notes with keyword recall via `SQLite`'s
//!   `FTS5` virtual table; an optional `embedding` column is reserved
//!   so semantic recall can plug in later without a schema migration.
//!
//! Everything runs in-process. No sidecars, no `docker-compose`, no
//! network. The store is a single `SQLite` file; callers pass the path.

mod cache;
mod embedding;
mod error;
mod memory;
mod rng;
mod scoreboard;
mod store;

pub use cache::{CacheEntry, CacheHandle, CacheKey};
pub use embedding::{EmbeddingProvider, NoopEmbeddingProvider};
pub use error::StateError;
pub use memory::{MemoryFact, MemoryHandle, MemoryQuery, MemoryScope};
pub use rng::{DefaultRng, RandomSource};
pub use scoreboard::{ModelStats, PromptBucket, ScoreboardHandle, Selection, SelectionReason};
pub use store::StateStore;
