//! Router scoreboard backed by SQLite.
//!
//! Stores per-(bucket, model) success/failure counts plus latency and
//! token totals. The selector is epsilon-greedy with a warm-start phase
//! so unexplored candidates get sampled before the ranking kicks in.

use std::fmt::{Display, Formatter};
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::error::StateError;
use crate::rng::RandomSource;
use crate::store::now_ms;

/// Coarse prompt bucket used as the scoreboard key. Keeping the alphabet
/// small means the scoreboard converges fast even on small session
/// corpora; finer-grained bucketing (topic, tool density, …) is a
/// natural follow-up once there's evidence the global success signal
/// doesn't discriminate enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptBucket {
    Short,
    Medium,
    Long,
}

impl PromptBucket {
    /// Classify a prompt by raw character length. Thresholds chosen to
    /// roughly match Anthropic's ~4-chars-per-token ratio: short ≈
    /// sub-128 tokens, medium ≈ 128–1500 tokens, long ≳ 1500.
    #[must_use]
    pub fn from_prompt_chars(char_count: usize) -> Self {
        if char_count < 512 {
            Self::Short
        } else if char_count < 6_000 {
            Self::Medium
        } else {
            Self::Long
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Medium => "medium",
            Self::Long => "long",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "short" => Some(Self::Short),
            "medium" => Some(Self::Medium),
            "long" => Some(Self::Long),
            _ => None,
        }
    }
}

impl Display for PromptBucket {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Aggregated statistics for one candidate model inside one bucket.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelStats {
    pub successes: u64,
    pub failures: u64,
    pub total_latency_ms: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

impl ModelStats {
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.successes + self.failures
    }

    /// Laplace-smoothed success rate so unsampled candidates don't tie
    /// at `0/0 = undefined`. Returns a value in `[0, 1]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        let successes = self.successes as f64;
        let samples = self.samples() as f64;
        (successes + 1.0) / (samples + 2.0)
    }

    #[must_use]
    pub fn average_latency_ms(&self) -> u64 {
        let samples = self.samples();
        if samples == 0 {
            0
        } else {
            self.total_latency_ms / samples
        }
    }
}

/// Reason the selector returned a particular model. Surfaced to stderr
/// for debugging and persisted on the selection so eval consumers can
/// tell exploration from exploitation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionReason {
    WarmStart,
    Explore,
    Exploit,
}

impl SelectionReason {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WarmStart => "warm_start",
            Self::Explore => "explore",
            Self::Exploit => "exploit",
        }
    }
}

impl Display for SelectionReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One selection decision returned by [`ScoreboardHandle::select`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub model: String,
    pub bucket: PromptBucket,
    pub reason: SelectionReason,
}

/// Scoped view over the scoreboard rows in the shared state store.
/// Holds a reference to the store's mutex, so dropping the handle is
/// free; the DB connection lives for the store's lifetime.
pub struct ScoreboardHandle<'a> {
    conn: &'a Mutex<Connection>,
}

impl<'a> ScoreboardHandle<'a> {
    pub(crate) fn new(conn: &'a Mutex<Connection>) -> Self {
        Self { conn }
    }

    /// Fetch stats for `(bucket, model)`. Missing rows read as the
    /// default all-zeros stats.
    #[allow(clippy::cast_sign_loss)]
    pub fn stats(&self, bucket: PromptBucket, model: &str) -> Result<ModelStats, StateError> {
        let conn = self.conn.lock().unwrap();
        let row: Option<ModelStats> = conn
            .query_row(
                "SELECT successes, failures, total_latency_ms, total_input_tokens, total_output_tokens
                 FROM router_scores WHERE bucket = ? AND model = ?",
                params![bucket.as_str(), model],
                |row| {
                    // Counts are written as non-negative i64s in `record_outcome`;
                    // reinterpreting as u64 is safe.
                    Ok(ModelStats {
                        successes: row.get::<_, i64>(0)? as u64,
                        failures: row.get::<_, i64>(1)? as u64,
                        total_latency_ms: row.get::<_, i64>(2)? as u64,
                        total_input_tokens: row.get::<_, i64>(3)? as u64,
                        total_output_tokens: row.get::<_, i64>(4)? as u64,
                    })
                },
            )
            .optional()?;
        Ok(row.unwrap_or_default())
    }

    /// Record a turn's outcome. `input_tokens` / `output_tokens` are
    /// stored for cost analysis; routing itself only depends on
    /// `success`.
    pub fn record_outcome(
        &self,
        bucket: PromptBucket,
        model: &str,
        success: bool,
        latency_ms: u64,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Result<(), StateError> {
        let conn = self.conn.lock().unwrap();
        let (success_delta, failure_delta) = if success {
            (1_i64, 0_i64)
        } else {
            (0_i64, 1_i64)
        };
        conn.execute(
            "INSERT INTO router_scores(
                bucket, model, successes, failures,
                total_latency_ms, total_input_tokens, total_output_tokens,
                updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bucket, model) DO UPDATE SET
                successes = successes + excluded.successes,
                failures = failures + excluded.failures,
                total_latency_ms = total_latency_ms + excluded.total_latency_ms,
                total_input_tokens = total_input_tokens + excluded.total_input_tokens,
                total_output_tokens = total_output_tokens + excluded.total_output_tokens,
                updated_at_ms = excluded.updated_at_ms",
            params![
                bucket.as_str(),
                model,
                success_delta,
                failure_delta,
                i64::try_from(latency_ms).unwrap_or(i64::MAX),
                i64::from(input_tokens),
                i64::from(output_tokens),
                now_ms(),
            ],
        )?;
        Ok(())
    }

    /// Pick the model for the next turn.
    ///
    /// # Errors
    /// Returns [`StateError::InvalidInput`] if `candidates` is empty.
    pub fn select(
        &self,
        bucket: PromptBucket,
        candidates: &[String],
        epsilon: f64,
        min_samples: u32,
        rng: &mut dyn RandomSource,
    ) -> Result<Selection, StateError> {
        if candidates.is_empty() {
            return Err(StateError::InvalidInput(
                "eval-driven router requires at least one candidate model".into(),
            ));
        }

        // Warm-start: sample every candidate until each has `min_samples`
        // observations in this bucket.
        for candidate in candidates {
            let stats = self.stats(bucket, candidate)?;
            if stats.samples() < u64::from(min_samples) {
                return Ok(Selection {
                    model: candidate.clone(),
                    bucket,
                    reason: SelectionReason::WarmStart,
                });
            }
        }

        // Epsilon-greedy explore.
        if epsilon > 0.0 && rng.next_unit_float() < epsilon {
            let index = rng.next_bounded(candidates.len());
            return Ok(Selection {
                model: candidates[index].clone(),
                bucket,
                reason: SelectionReason::Explore,
            });
        }

        // Greedy: highest success rate, break ties on lower avg latency.
        let mut scored: Vec<(String, ModelStats)> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            scored.push((candidate.clone(), self.stats(bucket, candidate)?));
        }
        scored.sort_by(|(_, a), (_, b)| {
            let rate_a = a.success_rate();
            let rate_b = b.success_rate();
            rate_b
                .partial_cmp(&rate_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.average_latency_ms().cmp(&b.average_latency_ms()))
        });
        let best = scored
            .first()
            .map_or_else(|| candidates[0].clone(), |(name, _)| name.clone());
        Ok(Selection {
            model: best,
            bucket,
            reason: SelectionReason::Exploit,
        })
    }

    /// Iterate every recorded `(bucket, model, stats)` row. Primarily
    /// for the `claw eval` summary and external tooling that wants to
    /// visualize the scoreboard.
    #[allow(clippy::cast_sign_loss)]
    pub fn snapshot(&self) -> Result<Vec<(PromptBucket, String, ModelStats)>, StateError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT bucket, model, successes, failures, total_latency_ms,
                    total_input_tokens, total_output_tokens
             FROM router_scores ORDER BY bucket, model",
        )?;
        let rows = stmt.query_map([], |row| {
            let bucket_str: String = row.get(0)?;
            // Counts are stored as non-negative i64s; reinterpret as u64.
            Ok((
                bucket_str,
                row.get::<_, String>(1)?,
                ModelStats {
                    successes: row.get::<_, i64>(2)? as u64,
                    failures: row.get::<_, i64>(3)? as u64,
                    total_latency_ms: row.get::<_, i64>(4)? as u64,
                    total_input_tokens: row.get::<_, i64>(5)? as u64,
                    total_output_tokens: row.get::<_, i64>(6)? as u64,
                },
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (bucket_str, model, stats) = row?;
            if let Some(bucket) = PromptBucket::parse(&bucket_str) {
                out.push((bucket, model, stats));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StateStore;

    struct SequenceRng {
        floats: Vec<f64>,
        bounded: Vec<usize>,
        float_cursor: usize,
        bounded_cursor: usize,
    }

    impl SequenceRng {
        fn new(floats: Vec<f64>, bounded: Vec<usize>) -> Self {
            Self {
                floats,
                bounded,
                float_cursor: 0,
                bounded_cursor: 0,
            }
        }
    }

    impl RandomSource for SequenceRng {
        fn next_unit_float(&mut self) -> f64 {
            let value = self.floats[self.float_cursor];
            self.float_cursor += 1;
            value
        }

        fn next_bounded(&mut self, _upper: usize) -> usize {
            let value = self.bounded[self.bounded_cursor];
            self.bounded_cursor += 1;
            value
        }
    }

    fn candidates() -> Vec<String> {
        vec![
            "claude-haiku-4-5".to_string(),
            "claude-sonnet-4-6".to_string(),
            "claude-opus-4-6".to_string(),
        ]
    }

    #[test]
    fn prompt_bucket_thresholds_match_expected_ranges() {
        assert_eq!(PromptBucket::from_prompt_chars(0), PromptBucket::Short);
        assert_eq!(PromptBucket::from_prompt_chars(511), PromptBucket::Short);
        assert_eq!(PromptBucket::from_prompt_chars(512), PromptBucket::Medium);
        assert_eq!(PromptBucket::from_prompt_chars(5_999), PromptBucket::Medium);
        assert_eq!(PromptBucket::from_prompt_chars(6_000), PromptBucket::Long);
    }

    #[test]
    fn select_warm_starts_each_candidate_before_exploiting() {
        let store = StateStore::in_memory().expect("store");
        let board = store.scoreboard();
        let mut rng = SequenceRng::new(vec![], vec![]);
        let pick = board
            .select(PromptBucket::Short, &candidates(), 0.1, 3, &mut rng)
            .expect("select");
        assert_eq!(pick.reason, SelectionReason::WarmStart);
        assert_eq!(pick.model, candidates()[0]);
    }

    #[test]
    fn select_errors_when_no_candidates() {
        let store = StateStore::in_memory().expect("store");
        let board = store.scoreboard();
        let mut rng = SequenceRng::new(vec![], vec![]);
        let error = board
            .select(PromptBucket::Short, &[], 0.1, 3, &mut rng)
            .expect_err("empty candidates must error");
        assert!(format!("{error}").contains("at least one candidate"));
    }

    #[test]
    fn select_exploits_model_with_highest_success_rate_once_warm() {
        let store = StateStore::in_memory().expect("store");
        let board = store.scoreboard();
        for _ in 0..5 {
            board
                .record_outcome(PromptBucket::Short, "claude-haiku-4-5", true, 100, 10, 5)
                .unwrap();
            board
                .record_outcome(PromptBucket::Short, "claude-sonnet-4-6", false, 200, 10, 5)
                .unwrap();
            board
                .record_outcome(PromptBucket::Short, "claude-opus-4-6", false, 400, 10, 5)
                .unwrap();
        }
        let mut rng = SequenceRng::new(vec![0.99], vec![]);
        let pick = board
            .select(PromptBucket::Short, &candidates(), 0.0, 5, &mut rng)
            .expect("select");
        assert_eq!(pick.model, "claude-haiku-4-5");
        assert_eq!(pick.reason, SelectionReason::Exploit);
    }

    #[test]
    fn select_explores_when_rng_rolls_below_epsilon() {
        let store = StateStore::in_memory().expect("store");
        let board = store.scoreboard();
        for _ in 0..5 {
            for candidate in candidates() {
                board
                    .record_outcome(PromptBucket::Medium, &candidate, true, 100, 10, 5)
                    .unwrap();
            }
        }
        let mut rng = SequenceRng::new(vec![0.05], vec![2]);
        let pick = board
            .select(PromptBucket::Medium, &candidates(), 0.5, 5, &mut rng)
            .expect("select");
        assert_eq!(pick.reason, SelectionReason::Explore);
        assert_eq!(pick.model, candidates()[2]);
    }

    #[test]
    fn record_outcome_accumulates_counts_and_totals() {
        let store = StateStore::in_memory().expect("store");
        let board = store.scoreboard();
        board
            .record_outcome(PromptBucket::Long, "claude-opus-4-6", true, 800, 2000, 300)
            .unwrap();
        board
            .record_outcome(
                PromptBucket::Long,
                "claude-opus-4-6",
                false,
                1200,
                2500,
                100,
            )
            .unwrap();
        let stats = board
            .stats(PromptBucket::Long, "claude-opus-4-6")
            .expect("stats");
        assert_eq!(stats.successes, 1);
        assert_eq!(stats.failures, 1);
        assert_eq!(stats.total_latency_ms, 2000);
        assert_eq!(stats.total_input_tokens, 4500);
        assert_eq!(stats.total_output_tokens, 400);
        assert_eq!(stats.average_latency_ms(), 1000);
    }

    #[test]
    fn snapshot_returns_all_recorded_rows() {
        let store = StateStore::in_memory().expect("store");
        let board = store.scoreboard();
        board
            .record_outcome(PromptBucket::Short, "claude-haiku-4-5", true, 50, 10, 5)
            .unwrap();
        board
            .record_outcome(PromptBucket::Long, "claude-opus-4-6", false, 900, 100, 20)
            .unwrap();
        let rows = board.snapshot().expect("snapshot");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(
            |(bucket, model, _)| *bucket == PromptBucket::Short && model == "claude-haiku-4-5"
        ));
        assert!(rows
            .iter()
            .any(|(bucket, model, _)| *bucket == PromptBucket::Long && model == "claude-opus-4-6"));
    }

    #[test]
    fn model_stats_success_rate_is_laplace_smoothed() {
        let empty = ModelStats::default();
        assert!((empty.success_rate() - 0.5).abs() < 1e-9);

        let nine_of_ten = ModelStats {
            successes: 9,
            failures: 1,
            ..Default::default()
        };
        assert!((nine_of_ten.success_rate() - (10.0 / 12.0)).abs() < 1e-9);
    }
}
