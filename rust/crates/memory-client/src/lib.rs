//! External memory client for the `claw` runtime.
//!
//! Provides a `MemoryClient` trait with a synchronous surface that the
//! conversation runtime calls before and after each turn. The default
//! implementation targets [Zep](https://www.getzep.com/)'s REST API so
//! recall and ingestion run in a sidecar service rather than in-process.

use std::fmt::{Display, Formatter};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Role of a persisted message as understood by Zep. Mirrors the runtime's
/// [`MessageRole`](runtime::MessageRole) but is decoupled so this crate has no
/// runtime dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One message forwarded to the memory backend for fact extraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryMessage {
    pub role: MemoryRole,
    pub content: String,
}

impl MemoryMessage {
    #[must_use]
    pub fn new(role: MemoryRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Errors raised by [`MemoryClient`] implementations.
#[derive(Debug)]
pub enum MemoryError {
    Network(String),
    Protocol(String),
    Config(String),
}

impl Display for MemoryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(message) => write!(f, "memory network error: {message}"),
            Self::Protocol(message) => write!(f, "memory protocol error: {message}"),
            Self::Config(message) => write!(f, "memory config error: {message}"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Synchronous memory client surface. Implementations may block on an internal
/// async runtime, but callers (runtime `run_turn`) stay synchronous.
pub trait MemoryClient: Send {
    /// Retrieve up to `limit` memory snippets relevant to `query` for
    /// `session_id`. Returning an empty Vec is not an error.
    fn recall(
        &mut self,
        session_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<String>, MemoryError>;

    /// Forward new conversation messages to the backend so it can extract and
    /// store facts.
    fn ingest(&mut self, session_id: &str, messages: &[MemoryMessage]) -> Result<(), MemoryError>;
}

/// A memory client that does nothing. Used when memory is disabled in config.
pub struct NoopMemoryClient;

impl MemoryClient for NoopMemoryClient {
    fn recall(
        &mut self,
        _session_id: &str,
        _query: &str,
        _limit: usize,
    ) -> Result<Vec<String>, MemoryError> {
        Ok(Vec::new())
    }

    fn ingest(
        &mut self,
        _session_id: &str,
        _messages: &[MemoryMessage],
    ) -> Result<(), MemoryError> {
        Ok(())
    }
}

// ---- Zep REST client ----

/// Configuration for [`ZepMemoryClient`].
#[derive(Debug, Clone)]
pub struct ZepConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub user_id: String,
    pub timeout: Duration,
}

impl ZepConfig {
    #[must_use]
    pub fn new(base_url: impl Into<String>, user_id: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            user_id: user_id.into(),
            timeout: Duration::from_secs(5),
        }
    }

    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Memory client backed by Zep's REST API.
///
/// Endpoints used (Zep Cloud / self-hosted parity):
/// - `POST /api/v2/sessions/{session_id}/messages` for ingestion
/// - `POST /api/v2/sessions/{session_id}/search` for recall
pub struct ZepMemoryClient {
    config: ZepConfig,
    http: reqwest::blocking::Client,
}

impl ZepMemoryClient {
    pub fn new(config: ZepConfig) -> Result<Self, MemoryError> {
        if config.base_url.is_empty() {
            return Err(MemoryError::Config("base_url is empty".into()));
        }
        if config.user_id.is_empty() {
            return Err(MemoryError::Config("user_id is empty".into()));
        }
        let http = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|error| MemoryError::Config(error.to_string()))?;
        Ok(Self { config, http })
    }

    fn url(&self, path: &str) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}{path}")
    }

    fn apply_auth(
        &self,
        builder: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        if let Some(api_key) = &self.config.api_key {
            builder.header("Authorization", format!("Api-Key {api_key}"))
        } else {
            builder
        }
    }
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    text: &'a str,
    limit: usize,
    user_id: &'a str,
}

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    results: Vec<SearchResult>,
}

#[derive(Deserialize)]
struct SearchResult {
    #[serde(default)]
    fact: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

#[derive(Serialize)]
struct IngestRequest<'a> {
    messages: &'a [MemoryMessage],
    user_id: &'a str,
}

impl MemoryClient for ZepMemoryClient {
    fn recall(
        &mut self,
        session_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<String>, MemoryError> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let path = format!("/api/v2/sessions/{session_id}/search");
        let body = SearchRequest {
            text: query,
            limit,
            user_id: &self.config.user_id,
        };
        let request = self.apply_auth(self.http.post(self.url(&path)).json(&body));
        let response = request
            .send()
            .map_err(|error| MemoryError::Network(error.to_string()))?;
        if !response.status().is_success() {
            return Err(MemoryError::Protocol(format!(
                "search returned HTTP {}",
                response.status()
            )));
        }
        let payload: SearchResponse = response
            .json()
            .map_err(|error| MemoryError::Protocol(error.to_string()))?;
        Ok(payload
            .results
            .into_iter()
            .filter_map(|entry| entry.fact.or(entry.content))
            .filter(|text| !text.trim().is_empty())
            .collect())
    }

    fn ingest(&mut self, session_id: &str, messages: &[MemoryMessage]) -> Result<(), MemoryError> {
        if messages.is_empty() {
            return Ok(());
        }
        let path = format!("/api/v2/sessions/{session_id}/messages");
        let body = IngestRequest {
            messages,
            user_id: &self.config.user_id,
        };
        let request = self.apply_auth(self.http.post(self.url(&path)).json(&body));
        let response = request
            .send()
            .map_err(|error| MemoryError::Network(error.to_string()))?;
        if !response.status().is_success() {
            return Err(MemoryError::Protocol(format!(
                "ingest returned HTTP {}",
                response.status()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_returns_empty_recall() {
        let mut client = NoopMemoryClient;
        let recalled = client.recall("session", "hello", 5).expect("recall");
        assert!(recalled.is_empty());
    }

    #[test]
    fn noop_accepts_ingest() {
        let mut client = NoopMemoryClient;
        let message = MemoryMessage::new(MemoryRole::User, "hi");
        client.ingest("session", &[message]).expect("ingest");
    }

    #[test]
    fn zep_client_rejects_empty_base_url() {
        let config = ZepConfig::new("", "alice");
        match ZepMemoryClient::new(config) {
            Err(MemoryError::Config(_)) => {}
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("expected empty base_url to fail"),
        }
    }

    #[test]
    fn zep_client_rejects_empty_user_id() {
        let config = ZepConfig::new("https://zep.example", "");
        match ZepMemoryClient::new(config) {
            Err(MemoryError::Config(_)) => {}
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("expected empty user_id to fail"),
        }
    }

    #[test]
    fn zep_recall_short_circuits_on_empty_query() {
        let config = ZepConfig::new("https://zep.invalid", "alice");
        let mut client = ZepMemoryClient::new(config).expect("client");
        let recalled = client.recall("session", "   ", 5).expect("recall");
        assert!(recalled.is_empty());
    }

    #[test]
    fn zep_recall_short_circuits_on_zero_limit() {
        let config = ZepConfig::new("https://zep.invalid", "alice");
        let mut client = ZepMemoryClient::new(config).expect("client");
        let recalled = client.recall("session", "question", 0).expect("recall");
        assert!(recalled.is_empty());
    }

    #[test]
    fn zep_ingest_short_circuits_on_empty_messages() {
        let config = ZepConfig::new("https://zep.invalid", "alice");
        let mut client = ZepMemoryClient::new(config).expect("client");
        client.ingest("session", &[]).expect("ingest");
    }
}
