//! In-process router driven by per-model success/failure statistics.
//!
//! Each turn, the runtime picks one of the configured candidate models
//! based on a scoreboard keyed by a coarse prompt bucket (short / medium /
//! long). After the turn completes the outcome (success or failure, plus
//! latency and token counts) is folded back into the scoreboard so the
//! next decision is better informed.
//!
//! Selection strategy:
//! 1. **Warm-start** — any candidate with fewer than `min_samples` recorded
//!    observations in the current bucket is picked so the scoreboard
//!    accumulates a baseline before the greedy phase kicks in.
//! 2. **Epsilon-greedy exploration** — with probability `epsilon` pick a
//!    uniformly random candidate so the scoreboard keeps adapting when the
//!    provider mix changes (new model versions, pricing shifts, etc.).
//! 3. **Greedy** — otherwise pick the candidate with the highest empirical
//!    success rate; ties broken by lower average latency.
//!
//! The scoreboard persists as JSON at a caller-supplied path. Writes use a
//! write-then-rename pattern so a crash mid-write can't corrupt the file.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

const SCOREBOARD_VERSION: u32 = 1;

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| {
            // Truncation is intentional — we want a stable u64 epoch-ms.
            #[allow(clippy::cast_possible_truncation)]
            let ms = duration.as_millis() as u64;
            ms
        })
        .unwrap_or(0)
}

/// Multiply `successes` / `failures` / `total_latency_ms` by
/// `0.5 ^ (elapsed_ms / half_life_ms)`. Rows without a prior
/// observation (loaded from older scoreboards or freshly inserted) are
/// skipped so decay doesn't punish data that predates the feature.
fn apply_decay(entry: &mut ModelStats, now_ms: u64, half_life_ms: u64) {
    if entry.last_observation_ms == 0 || half_life_ms == 0 || now_ms <= entry.last_observation_ms {
        return;
    }
    let elapsed_ms = now_ms.saturating_sub(entry.last_observation_ms);
    // `0.5 ^ k` = `exp(-k * ln 2)`. We never use more than `elapsed /
    // half_life` half-lives of decay; even across 100 years of elapsed
    // time with a 1-hour half-life the exponent is finite and safe in
    // f64 (factor ≈ 0 well before any overflow).
    #[allow(clippy::cast_precision_loss)]
    let ratio = (elapsed_ms as f64) / (half_life_ms as f64);
    let factor = (-ratio * std::f64::consts::LN_2).exp();
    entry.successes = decay_count(entry.successes, factor);
    entry.failures = decay_count(entry.failures, factor);
    entry.total_latency_ms = decay_count(entry.total_latency_ms, factor);
}

fn decay_count(value: u64, factor: f64) -> u64 {
    if factor <= 0.0 || value == 0 {
        return 0;
    }
    if factor >= 1.0 {
        return value;
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let decayed = ((value as f64) * factor).round() as u64;
    decayed
}

/// Coarse prompt bucket used as the scoreboard key. Keeping the alphabet
/// small means the scoreboard converges fast even on small session
/// corpora; finer-grained bucketing (topic, tool density, …) is a natural
/// follow-up once there's evidence the global success signal doesn't
/// discriminate enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptBucket {
    Short,
    Medium,
    Long,
}

impl PromptBucket {
    /// Classify a prompt by raw character length. Thresholds chosen to
    /// roughly match the Anthropic tokenizer's ~4-chars-per-token ratio:
    /// `short` ≈ sub-128 tokens, `medium` ≈ 128–1500 tokens, `long` ≳ 1500.
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
}

impl Display for PromptBucket {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Aggregated statistics for one candidate model inside a single bucket.
///
/// Counts decay toward zero over time when the scoreboard is recorded
/// against with a non-`None` `half_life_ms` — the exponent is
/// `0.5 ^ (elapsed_ms / half_life_ms)`, applied to `successes`,
/// `failures`, and `total_latency_ms` before the new sample is
/// added. Token totals are not decayed (they're only kept for cost
/// analysis, not the selection decision).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelStats {
    #[serde(default)]
    pub successes: u64,
    #[serde(default)]
    pub failures: u64,
    #[serde(default)]
    pub total_latency_ms: u64,
    #[serde(default)]
    pub total_input_tokens: u64,
    #[serde(default)]
    pub total_output_tokens: u64,
    /// Cumulative provider cost in micro-dollars (10⁻⁶ USD) for this
    /// (bucket, model) row. Storing an integer keeps the JSON schema
    /// free of floating-point values; a u64 of micro-dollars holds up to
    /// ~18 trillion dollars, which is enough headroom. Not decayed — we
    /// want a faithful lifetime cost figure for the `claw eval` report.
    #[serde(default)]
    pub total_cost_micros: u64,
    /// Unix-millis of the last observation merged into this row.
    /// Default 0 on rows loaded from pre-decay scoreboards; the first
    /// recorded outcome after load skips decay so we don't penalize
    /// historical data simply because it predates the feature.
    #[serde(default)]
    pub last_observation_ms: u64,
}

impl ModelStats {
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.successes + self.failures
    }

    /// Laplace-smoothed success rate so unsampled candidates don't tie at
    /// 0/0 = undefined. Returns a value in `[0, 1]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        // Cast precision: sample counts are bounded by the session
        // corpus size, safely inside f64's 53-bit mantissa.
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

    /// Cumulative cost in dollars for this (bucket, model) row. Derived
    /// from `total_cost_micros`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn total_cost_usd(&self) -> f64 {
        (self.total_cost_micros as f64) / 1_000_000.0
    }
}

/// Reason the selector returned a particular model. Surfaced to stderr for
/// debugging and to the scoreboard for post-hoc analysis.
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

/// One selection decision returned by [`RouterScoreboard::select`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub model: String,
    pub bucket: PromptBucket,
    pub reason: SelectionReason,
}

/// On-disk schema. Nested map: bucket → model → stats.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct ScoreboardState {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    buckets: BTreeMap<PromptBucket, BTreeMap<String, ModelStats>>,
}

const fn default_version() -> u32 {
    SCOREBOARD_VERSION
}

/// Errors raised while loading, updating, or persisting a scoreboard.
#[derive(Debug)]
pub enum ScoreboardError {
    Io(io::Error),
    Serde(serde_json::Error),
    Config(String),
}

impl Display for ScoreboardError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "scoreboard I/O error: {error}"),
            Self::Serde(error) => write!(f, "scoreboard serde error: {error}"),
            Self::Config(message) => write!(f, "scoreboard config error: {message}"),
        }
    }
}

impl std::error::Error for ScoreboardError {}

impl From<io::Error> for ScoreboardError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ScoreboardError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serde(error)
    }
}

/// Deterministic pseudo-random source for epsilon-greedy exploration. The
/// router does not need cryptographic quality and we would rather avoid
/// pulling `rand` into the dep graph for a single u64 pick per turn.
pub trait RandomSource {
    /// Returns a value uniformly in `[0.0, 1.0)`.
    fn next_unit_float(&mut self) -> f64;

    /// Returns a value uniformly in `[0, upper)`. `upper` is guaranteed to
    /// be non-zero by the caller.
    fn next_bounded(&mut self, upper: usize) -> usize;
}

/// Splitmix64-style PRNG seeded from the system clock. Good enough for
/// exploration decisions; not suitable for anything security-sensitive.
pub struct DefaultRng {
    state: u64,
}

impl DefaultRng {
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    #[must_use]
    pub fn from_system_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x9E37_79B9_7F4A_7C15, |duration| {
                // Truncation is intentional: we only need entropy bits,
                // not a faithful nanosecond count.
                #[allow(clippy::cast_possible_truncation)]
                let truncated = duration.as_nanos() as u64;
                truncated
            });
        Self::from_seed(nanos ^ u64::from(std::process::id()).rotate_left(32))
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl RandomSource for DefaultRng {
    fn next_unit_float(&mut self) -> f64 {
        // Take top 53 bits for a uniform float in [0, 1). The shift
        // guarantees the value fits in f64's mantissa losslessly.
        let value = self.next_u64() >> 11;
        #[allow(clippy::cast_precision_loss)]
        let as_float = value as f64 / ((1u64 << 53) as f64);
        as_float
    }

    fn next_bounded(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0, "upper must be non-zero");
        // On 32-bit targets the u64 -> usize cast narrows, but the
        // modulo by `upper` (a usize) bounds the final value correctly.
        #[allow(clippy::cast_possible_truncation)]
        let raw = self.next_u64() as usize;
        raw % upper
    }
}

/// File-backed router scoreboard. Loads and saves JSON at `path`; writes
/// are atomic (write-then-rename) so a crash during persistence can't
/// leave a half-written file.
#[derive(Debug, Clone)]
pub struct RouterScoreboard {
    path: Option<PathBuf>,
    state: ScoreboardState,
}

impl RouterScoreboard {
    /// Create an empty, in-memory-only scoreboard. Useful in tests.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            path: None,
            state: ScoreboardState {
                version: SCOREBOARD_VERSION,
                buckets: BTreeMap::new(),
            },
        }
    }

    /// Load an existing scoreboard file, or create an empty one bound to
    /// `path` if the file does not yet exist.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, ScoreboardError> {
        let path = path.into();
        let state = if path.exists() {
            let bytes = fs::read(&path)?;
            if bytes.is_empty() {
                ScoreboardState::default()
            } else {
                serde_json::from_slice(&bytes)?
            }
        } else {
            ScoreboardState::default()
        };
        Ok(Self {
            path: Some(path),
            state,
        })
    }

    pub fn save(&self) -> Result<(), ScoreboardError> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let serialized = serde_json::to_vec_pretty(&self.state)?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &serialized)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Current stats for `(bucket, model)`; returns the default (all zeros)
    /// when no samples have been recorded yet.
    #[must_use]
    pub fn stats(&self, bucket: PromptBucket, model: &str) -> ModelStats {
        self.state
            .buckets
            .get(&bucket)
            .and_then(|models| models.get(model))
            .cloned()
            .unwrap_or_default()
    }

    /// Iterate over every recorded `(bucket, model, stats)` triple.
    pub fn iter(&self) -> impl Iterator<Item = (PromptBucket, &str, &ModelStats)> {
        self.state.buckets.iter().flat_map(|(bucket, models)| {
            models
                .iter()
                .map(move |(model, stats)| (*bucket, model.as_str(), stats))
        })
    }

    /// Record a turn's outcome. `input_tokens` / `output_tokens` are purely
    /// informational (cost analysis); the routing decision itself only
    /// depends on `success`.
    ///
    /// Equivalent to [`record_outcome_with_decay`](Self::record_outcome_with_decay)
    /// called with `half_life_ms = None` and `cost_micros = 0`.
    pub fn record_outcome(
        &mut self,
        bucket: PromptBucket,
        model: &str,
        success: bool,
        latency_ms: u64,
        input_tokens: u32,
        output_tokens: u32,
    ) {
        self.record_outcome_with_decay(
            bucket,
            model,
            success,
            latency_ms,
            input_tokens,
            output_tokens,
            0,
            None,
        );
    }

    /// Record a turn's outcome, optionally decaying prior counts first
    /// and crediting cost against the row.
    ///
    /// - `cost_micros`: provider cost for this turn in micro-dollars.
    ///   Added to the row's lifetime cumulative cost; not decayed.
    /// - `half_life_ms`: if `Some(h)` and the row has a recorded
    ///   `last_observation_ms`, existing `successes`, `failures`, and
    ///   `total_latency_ms` are multiplied by `0.5 ^ (elapsed_ms / h)`
    ///   before the new sample lands. Passing `None` preserves the
    ///   pre-decay append-forever behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn record_outcome_with_decay(
        &mut self,
        bucket: PromptBucket,
        model: &str,
        success: bool,
        latency_ms: u64,
        input_tokens: u32,
        output_tokens: u32,
        cost_micros: u64,
        half_life_ms: Option<u64>,
    ) {
        let now_ms = now_unix_ms();
        let entry = self
            .state
            .buckets
            .entry(bucket)
            .or_default()
            .entry(model.to_string())
            .or_default();

        if let Some(half_life) = half_life_ms {
            apply_decay(entry, now_ms, half_life);
        }

        if success {
            entry.successes = entry.successes.saturating_add(1);
        } else {
            entry.failures = entry.failures.saturating_add(1);
        }
        entry.total_latency_ms = entry.total_latency_ms.saturating_add(latency_ms);
        entry.total_input_tokens = entry
            .total_input_tokens
            .saturating_add(u64::from(input_tokens));
        entry.total_output_tokens = entry
            .total_output_tokens
            .saturating_add(u64::from(output_tokens));
        entry.total_cost_micros = entry.total_cost_micros.saturating_add(cost_micros);
        entry.last_observation_ms = now_ms;
    }

    /// Pick the model for the next turn.
    ///
    /// # Errors
    /// Returns [`ScoreboardError::Config`] if `candidates` is empty, since
    /// the router has no choice to make.
    pub fn select(
        &self,
        bucket: PromptBucket,
        candidates: &[String],
        epsilon: f64,
        min_samples: u32,
        rng: &mut dyn RandomSource,
    ) -> Result<Selection, ScoreboardError> {
        if candidates.is_empty() {
            return Err(ScoreboardError::Config(
                "eval-driven router requires at least one candidate model".into(),
            ));
        }

        // Warm-start: any candidate that hasn't hit min_samples gets
        // picked so we build up a baseline before ranking.
        if let Some(needs_data) = candidates
            .iter()
            .find(|candidate| self.stats(bucket, candidate).samples() < u64::from(min_samples))
        {
            return Ok(Selection {
                model: needs_data.clone(),
                bucket,
                reason: SelectionReason::WarmStart,
            });
        }

        // Epsilon-greedy exploration.
        if epsilon > 0.0 && rng.next_unit_float() < epsilon {
            let index = rng.next_bounded(candidates.len());
            return Ok(Selection {
                model: candidates[index].clone(),
                bucket,
                reason: SelectionReason::Explore,
            });
        }

        // Greedy: highest success rate, break ties with lower avg latency.
        let best = candidates
            .iter()
            .map(|candidate| {
                let stats = self.stats(bucket, candidate);
                (candidate, stats)
            })
            .max_by(|(_, a), (_, b)| {
                let rate_a = a.success_rate();
                let rate_b = b.success_rate();
                rate_a
                    .partial_cmp(&rate_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.average_latency_ms().cmp(&a.average_latency_ms()))
            })
            .map_or_else(|| candidates[0].clone(), |(candidate, _)| candidate.clone());

        Ok(Selection {
            model: best,
            bucket,
            reason: SelectionReason::Exploit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            PromptBucket::from_prompt_chars(1_000_000),
            PromptBucket::Long
        );
    }

    #[test]
    fn model_stats_success_rate_is_laplace_smoothed() {
        let empty = ModelStats::default();
        assert!((empty.success_rate() - 0.5).abs() < 1e-9);

        let mostly_good = ModelStats {
            successes: 9,
            failures: 1,
            ..Default::default()
        };
        // (9 + 1) / (10 + 2) == 10/12
        assert!((mostly_good.success_rate() - (10.0 / 12.0)).abs() < 1e-9);
    }

    #[test]
    fn select_warm_starts_through_all_candidates() {
        let board = RouterScoreboard::in_memory();
        let mut rng = SequenceRng::new(vec![], vec![]);
        let pick = board
            .select(PromptBucket::Short, &candidates(), 0.1, 3, &mut rng)
            .expect("select");
        assert_eq!(pick.reason, SelectionReason::WarmStart);
        assert_eq!(pick.model, candidates()[0]);
    }

    #[test]
    fn select_errors_when_no_candidates() {
        let board = RouterScoreboard::in_memory();
        let mut rng = SequenceRng::new(vec![], vec![]);
        let error = board
            .select(PromptBucket::Short, &[], 0.1, 3, &mut rng)
            .expect_err("empty candidates must error");
        assert!(format!("{error}").contains("at least one candidate"));
    }

    #[test]
    fn select_exploits_model_with_highest_success_rate_once_warm() {
        let mut board = RouterScoreboard::in_memory();
        // Seed each candidate with 5 samples so warm-start is satisfied.
        for _ in 0..5 {
            board.record_outcome(PromptBucket::Short, "claude-haiku-4-5", true, 100, 10, 5);
            board.record_outcome(PromptBucket::Short, "claude-sonnet-4-6", false, 200, 10, 5);
            board.record_outcome(PromptBucket::Short, "claude-opus-4-6", false, 400, 10, 5);
        }
        // Epsilon 0 so we never explore.
        let mut rng = SequenceRng::new(vec![0.99], vec![]);
        let pick = board
            .select(PromptBucket::Short, &candidates(), 0.0, 5, &mut rng)
            .expect("select");
        assert_eq!(pick.model, "claude-haiku-4-5");
        assert_eq!(pick.reason, SelectionReason::Exploit);
    }

    #[test]
    fn select_explores_when_rng_rolls_below_epsilon() {
        let mut board = RouterScoreboard::in_memory();
        for _ in 0..5 {
            for candidate in candidates() {
                board.record_outcome(PromptBucket::Medium, &candidate, true, 100, 10, 5);
            }
        }
        // float < epsilon triggers exploration; bounded picks index 2.
        let mut rng = SequenceRng::new(vec![0.05], vec![2]);
        let pick = board
            .select(PromptBucket::Medium, &candidates(), 0.5, 5, &mut rng)
            .expect("select");
        assert_eq!(pick.reason, SelectionReason::Explore);
        assert_eq!(pick.model, candidates()[2]);
    }

    #[test]
    fn select_breaks_success_rate_ties_with_lower_latency() {
        let mut board = RouterScoreboard::in_memory();
        // Both have identical success rates but haiku is faster.
        for _ in 0..10 {
            board.record_outcome(PromptBucket::Short, "claude-haiku-4-5", true, 50, 1, 1);
            board.record_outcome(PromptBucket::Short, "claude-sonnet-4-6", true, 500, 1, 1);
        }
        let mut rng = SequenceRng::new(vec![0.99], vec![]);
        let pick = board
            .select(
                PromptBucket::Short,
                &[
                    "claude-haiku-4-5".to_string(),
                    "claude-sonnet-4-6".to_string(),
                ],
                0.0,
                5,
                &mut rng,
            )
            .expect("select");
        assert_eq!(pick.model, "claude-haiku-4-5");
    }

    #[test]
    fn record_outcome_accumulates_into_correct_bucket_and_model() {
        let mut board = RouterScoreboard::in_memory();
        board.record_outcome(PromptBucket::Long, "claude-opus-4-6", true, 800, 2000, 300);
        board.record_outcome(
            PromptBucket::Long,
            "claude-opus-4-6",
            false,
            1200,
            2500,
            100,
        );
        board.record_outcome(PromptBucket::Short, "claude-opus-4-6", true, 80, 120, 20);

        let long = board.stats(PromptBucket::Long, "claude-opus-4-6");
        assert_eq!(long.successes, 1);
        assert_eq!(long.failures, 1);
        assert_eq!(long.total_latency_ms, 2000);
        assert_eq!(long.total_input_tokens, 4500);
        assert_eq!(long.total_output_tokens, 400);
        assert_eq!(long.average_latency_ms(), 1000);

        let short = board.stats(PromptBucket::Short, "claude-opus-4-6");
        assert_eq!(short.successes, 1);
        assert_eq!(short.failures, 0);
    }

    #[test]
    fn decay_halves_counts_after_one_half_life() {
        // decay_count is the actual workhorse; verify the math directly
        // instead of spinning a real clock.
        assert_eq!(decay_count(100, 0.5), 50);
        assert_eq!(decay_count(10, 0.25), 3); // 2.5 rounds to 3
        assert_eq!(decay_count(0, 0.5), 0);
        assert_eq!(decay_count(1, 1.0), 1);
        assert_eq!(decay_count(5, 0.0), 0);
    }

    #[test]
    fn apply_decay_skips_rows_with_no_prior_observation() {
        let mut stats = ModelStats {
            successes: 10,
            failures: 5,
            total_latency_ms: 1_000,
            last_observation_ms: 0, // no prior observation → skip
            ..Default::default()
        };
        apply_decay(&mut stats, 1_000_000, 100_000);
        assert_eq!(stats.successes, 10);
        assert_eq!(stats.failures, 5);
        assert_eq!(stats.total_latency_ms, 1_000);
    }

    #[test]
    fn apply_decay_scales_counts_by_half_life_exponent() {
        let mut stats = ModelStats {
            successes: 100,
            failures: 40,
            total_latency_ms: 10_000,
            total_input_tokens: 7_777, // not decayed — cost data only
            total_output_tokens: 3_333,
            total_cost_micros: 5_000_000, // $5, not decayed
            last_observation_ms: 1_000,
        };
        // One half-life elapsed: factor should be ~0.5.
        apply_decay(&mut stats, 1_000 + 3_600_000, 3_600_000);
        assert!(
            (49..=51).contains(&stats.successes),
            "expected ~50, got {}",
            stats.successes
        );
        assert!(
            (19..=21).contains(&stats.failures),
            "expected ~20, got {}",
            stats.failures
        );
        assert!(
            (4_900..=5_100).contains(&stats.total_latency_ms),
            "expected ~5000, got {}",
            stats.total_latency_ms
        );
        // Tokens and cost untouched — decay applies only to counts.
        assert_eq!(stats.total_input_tokens, 7_777);
        assert_eq!(stats.total_output_tokens, 3_333);
        assert_eq!(stats.total_cost_micros, 5_000_000);
    }

    #[test]
    fn record_outcome_with_decay_forgets_old_samples() {
        let mut board = RouterScoreboard::in_memory();
        // Seed a stale observation from "a year ago" (last_observation_ms
        // well in the past). Set that directly to avoid waiting on
        // wall-clock time during the test.
        {
            let entry = board
                .state
                .buckets
                .entry(PromptBucket::Short)
                .or_default()
                .entry("claude-haiku-4-5".to_string())
                .or_default();
            entry.successes = 100;
            entry.failures = 0;
            // 30 half-lives in the past: decay factor 2^-30 ≈ 1e-9, so
            // everything rounds to zero.
            let thirty_half_lives_ago = now_unix_ms().saturating_sub(30 * 3_600_000);
            entry.last_observation_ms = thirty_half_lives_ago;
        }

        board.record_outcome_with_decay(
            PromptBucket::Short,
            "claude-haiku-4-5",
            true,
            100,
            10,
            5,
            0,
            Some(3_600_000), // 1-hour half-life
        );
        let stats = board.stats(PromptBucket::Short, "claude-haiku-4-5");
        // All 100 historical successes should have decayed to 0; only
        // the newly recorded success remains.
        assert_eq!(stats.successes, 1);
        assert_eq!(stats.failures, 0);
    }

    #[test]
    fn record_outcome_with_decay_accumulates_cost_without_decaying_it() {
        let mut board = RouterScoreboard::in_memory();
        board.record_outcome_with_decay(
            PromptBucket::Short,
            "claude-haiku-4-5",
            true,
            50,
            100,
            40,
            125_000, // $0.125 for this turn
            Some(3_600_000),
        );
        board.record_outcome_with_decay(
            PromptBucket::Short,
            "claude-haiku-4-5",
            true,
            75,
            200,
            80,
            250_000, // another $0.25
            Some(3_600_000),
        );
        let stats = board.stats(PromptBucket::Short, "claude-haiku-4-5");
        // Both costs should land even though the row saw decay on its
        // counts; cost is not among the decayed fields.
        assert_eq!(stats.total_cost_micros, 375_000);
        assert!((stats.total_cost_usd() - 0.375).abs() < 1e-9);
    }

    #[test]
    fn record_outcome_wrapper_does_not_decay() {
        let mut board = RouterScoreboard::in_memory();
        // Seed a huge count with an old timestamp.
        {
            let entry = board
                .state
                .buckets
                .entry(PromptBucket::Short)
                .or_default()
                .entry("claude-opus-4-6".to_string())
                .or_default();
            entry.successes = 50;
            entry.last_observation_ms = 1; // ancient
        }
        // The non-decay wrapper should preserve the count.
        board.record_outcome(PromptBucket::Short, "claude-opus-4-6", true, 100, 10, 5);
        let stats = board.stats(PromptBucket::Short, "claude-opus-4-6");
        assert_eq!(stats.successes, 51);
    }

    #[test]
    fn save_and_load_roundtrip_preserves_scores() {
        let dir = tempdir();
        let path = dir.join("scoreboard.json");

        {
            let mut board = RouterScoreboard::load(&path).expect("load empty");
            board.record_outcome(
                PromptBucket::Medium,
                "claude-sonnet-4-6",
                true,
                300,
                500,
                200,
            );
            board.record_outcome(
                PromptBucket::Medium,
                "claude-sonnet-4-6",
                false,
                900,
                800,
                50,
            );
            board.save().expect("save");
        }

        let reloaded = RouterScoreboard::load(&path).expect("reload");
        let stats = reloaded.stats(PromptBucket::Medium, "claude-sonnet-4-6");
        assert_eq!(stats.successes, 1);
        assert_eq!(stats.failures, 1);
        assert_eq!(stats.total_latency_ms, 1200);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn save_is_atomic_via_rename() {
        let dir = tempdir();
        let path = dir.join("scoreboard.json");

        let mut board = RouterScoreboard::load(&path).expect("load empty");
        board.record_outcome(PromptBucket::Short, "claude-haiku-4-5", true, 100, 10, 5);
        board.save().expect("save");

        // Temp sibling must be cleaned up (rename completed).
        let tmp = path.with_extension("json.tmp");
        assert!(
            !tmp.exists(),
            "temp file should be renamed away, got {tmp:?}"
        );
        assert!(path.exists());

        std::fs::remove_dir_all(dir).ok();
    }

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("eval-router-test-{nanos}-{unique}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn load_returns_empty_scoreboard_for_missing_file() {
        let dir = tempdir();
        let path = dir.join("does-not-exist.json");
        let board = RouterScoreboard::load(&path).expect("load non-existent");
        assert!(board.iter().next().is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn default_rng_produces_uniform_ish_floats() {
        let mut rng = DefaultRng::from_seed(42);
        for _ in 0..1000 {
            let value = rng.next_unit_float();
            assert!((0.0..1.0).contains(&value), "out of range: {value}");
        }
    }
}
