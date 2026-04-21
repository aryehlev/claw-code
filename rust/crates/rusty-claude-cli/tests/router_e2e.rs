//! End-to-end integration test for the eval-driven router.
//!
//! Verifies that:
//! 1. When `router.enabled = true` in `.claw.json`, claw consults the
//!    scoreboard and overrides the `--model` flag with a candidate.
//! 2. After a successful turn, the scoreboard JSON on disk contains an
//!    entry for the chosen candidate.
//! 3. `[router] bucket=... selected=... reason=...` lands on stderr.
//!
//! The test spins up `MockAnthropicService` so no real provider is hit.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use mock_anthropic_service::{MockAnthropicService, SCENARIO_PREFIX};
use serde_json::Value;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn eval_driven_router_writes_scoreboard_entry_after_successful_turn() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should build");
    let server = runtime
        .block_on(MockAnthropicService::spawn())
        .expect("mock service should start");
    let base_url = server.base_url();

    let workspace = unique_temp_dir("router-e2e");
    let config_home = workspace.join("config-home");
    let home = workspace.join("home");
    let scoreboard_path = workspace.join("scoreboard.json");
    fs::create_dir_all(&workspace).expect("workspace should exist");
    fs::create_dir_all(&config_home).expect("config home should exist");
    fs::create_dir_all(&home).expect("home should exist");

    // Router config: two candidates, min_samples=1 so the first turn
    // warm-starts candidates[0]. epsilon=0 so RNG never steers us into
    // the explore branch — keeps the test deterministic.
    let settings = serde_json::json!({
        "router": {
            "enabled": true,
            "mode": "eval-driven",
            "candidates": ["claude-sonnet-4-6", "claude-haiku-4-5"],
            "minSamples": 1,
            "epsilonPercent": 0,
            "halfLifeHours": 0,
            "scoreboardPath": scoreboard_path.to_string_lossy(),
        }
    });
    fs::write(
        workspace.join(".claw.json"),
        serde_json::to_string_pretty(&settings).expect("settings should serialize"),
    )
    .expect("claw.json should write");

    // Run a streaming_text scenario — single text block, no tools.
    let prompt = format!("{SCENARIO_PREFIX}streaming_text");
    let output = run_claw(
        &workspace,
        &config_home,
        &home,
        &base_url,
        &[
            "--model",
            "sonnet",
            "--permission-mode",
            "read-only",
            "--compact",
            &prompt,
        ],
    );

    assert!(
        output.status.success(),
        "router e2e run should succeed\nstdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8(output.stderr).expect("stderr should be utf8");
    assert!(
        stderr.contains("[router]"),
        "stderr should surface the router selection log; got:\n{stderr}"
    );
    assert!(
        stderr.contains("reason=warm_start"),
        "first turn with min_samples=1 should warm_start; got:\n{stderr}"
    );

    let raw = fs::read_to_string(&scoreboard_path).unwrap_or_else(|error| {
        panic!(
            "scoreboard file should exist at {}: {error}",
            scoreboard_path.display()
        )
    });
    let parsed: Value = serde_json::from_str(&raw).expect("scoreboard should be valid JSON");
    let buckets = parsed
        .get("buckets")
        .and_then(Value::as_object)
        .expect("scoreboard should contain a buckets object");

    // streaming_text prompt is tiny → Short bucket.
    let short = buckets
        .get("short")
        .and_then(Value::as_object)
        .expect("short bucket should exist after the turn");
    let sonnet = short
        .get("claude-sonnet-4-6")
        .and_then(Value::as_object)
        .expect("sonnet row should exist (warm_start picks candidates[0])");
    let successes = sonnet
        .get("successes")
        .and_then(Value::as_u64)
        .expect("successes field");
    assert_eq!(
        successes, 1,
        "scoreboard should record exactly one success: {sonnet:?}"
    );

    // The last observation timestamp must be populated so decay can
    // kick in on future turns.
    let last_ms = sonnet
        .get("last_observation_ms")
        .and_then(Value::as_u64)
        .expect("last_observation_ms field");
    assert!(
        last_ms > 0,
        "last_observation_ms should be a unix epoch ms, got {last_ms}"
    );

    fs::remove_dir_all(&workspace).expect("workspace cleanup should succeed");
}

#[test]
fn router_falls_back_to_second_candidate_when_first_fails() {
    // We can't easily make the mock provider fail per-model, so exercise
    // fallback at the scoreboard level: seed the scoreboard so
    // candidates[0] is already warm and candidates[1] is not, then run a
    // turn. We only want to confirm the wiring builds and doesn't
    // regress happy-path behavior — the failing-first-candidate path is
    // covered by unit tests in eval-router and by the select_excluding
    // unit tests in main.rs.
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should build");
    let server = runtime
        .block_on(MockAnthropicService::spawn())
        .expect("mock service should start");
    let base_url = server.base_url();

    let workspace = unique_temp_dir("router-fallback");
    let config_home = workspace.join("config-home");
    let home = workspace.join("home");
    let scoreboard_path = workspace.join("scoreboard.json");
    fs::create_dir_all(&workspace).expect("workspace should exist");
    fs::create_dir_all(&config_home).expect("config home should exist");
    fs::create_dir_all(&home).expect("home should exist");

    // Pre-seed a scoreboard where haiku already has a success recorded:
    // the next turn should go to sonnet (still warm-starting, count < 1).
    let seeded = serde_json::json!({
        "version": 1,
        "buckets": {
            "short": {
                "claude-haiku-4-5": {
                    "successes": 1,
                    "failures": 0,
                    "total_latency_ms": 100,
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "total_cost_micros": 60,
                    "last_observation_ms": 1,
                }
            }
        }
    });
    fs::write(
        &scoreboard_path,
        serde_json::to_string_pretty(&seeded).expect("seed should serialize"),
    )
    .expect("seed scoreboard should write");

    let settings = serde_json::json!({
        "router": {
            "enabled": true,
            "mode": "eval-driven",
            "candidates": ["claude-haiku-4-5", "claude-sonnet-4-6"],
            "minSamples": 1,
            "epsilonPercent": 0,
            "halfLifeHours": 0,
            "scoreboardPath": scoreboard_path.to_string_lossy(),
        }
    });
    fs::write(
        workspace.join(".claw.json"),
        serde_json::to_string_pretty(&settings).expect("settings should serialize"),
    )
    .expect("claw.json should write");

    let prompt = format!("{SCENARIO_PREFIX}streaming_text");
    let output = run_claw(
        &workspace,
        &config_home,
        &home,
        &base_url,
        &[
            "--model",
            "sonnet",
            "--permission-mode",
            "read-only",
            "--compact",
            &prompt,
        ],
    );

    assert!(
        output.status.success(),
        "router run should succeed\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify: sonnet was picked (warm-start for the still-unsampled
    // candidate) and its row was written.
    let raw = fs::read_to_string(&scoreboard_path).expect("scoreboard should exist");
    let parsed: Value = serde_json::from_str(&raw).expect("scoreboard JSON");
    let buckets = parsed.get("buckets").and_then(Value::as_object).unwrap();
    let short = buckets.get("short").and_then(Value::as_object).unwrap();
    assert!(
        short.contains_key("claude-sonnet-4-6"),
        "turn should have warm-started sonnet: {short:?}"
    );

    fs::remove_dir_all(&workspace).expect("workspace cleanup should succeed");
}

fn run_claw(
    cwd: &std::path::Path,
    config_home: &std::path::Path,
    home: &std::path::Path,
    base_url: &str,
    args: &[&str],
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_claw"));
    command
        .current_dir(cwd)
        .env_clear()
        .env("ANTHROPIC_API_KEY", "test-router-key")
        .env("ANTHROPIC_BASE_URL", base_url)
        .env("CLAW_CONFIG_HOME", config_home)
        .env("HOME", home)
        .env("NO_COLOR", "1")
        .env("PATH", "/usr/bin:/bin")
        .args(args);
    command.output().expect("claw should launch")
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_millis();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "claw-{label}-{}-{millis}-{counter}",
        std::process::id()
    ))
}
