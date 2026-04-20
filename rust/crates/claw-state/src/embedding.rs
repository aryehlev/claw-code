//! Embedding provider trait.
//!
//! The default `NoopEmbeddingProvider` returns empty vectors so the rest
//! of the store works end-to-end without an embedding model. The candle-
//! backed implementation lives in the CLI to keep the ML deps out of
//! this crate's compile tree; it's plugged in via
//! [`StateStore::memory_with_embeddings`](crate::StateStore::memory_with_embeddings).

use crate::error::StateError;

/// Produces dense float vectors for text. Implementations are expected
/// to produce L2-normalized vectors so a dot product equals the cosine
/// similarity — semantic-recall code relies on that.
pub trait EmbeddingProvider: Send + Sync {
    fn dimension(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Vec<f32>, StateError>;
}

/// Provider that returns an empty vector. `dimension()` reports 0 so
/// downstream code can detect that semantic recall is disabled and skip
/// the similarity path.
pub struct NoopEmbeddingProvider;

impl EmbeddingProvider for NoopEmbeddingProvider {
    fn dimension(&self) -> usize {
        0
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, StateError> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_reports_zero_dimension_and_empty_embedding() {
        let provider = NoopEmbeddingProvider;
        assert_eq!(provider.dimension(), 0);
        assert!(provider.embed("anything").unwrap().is_empty());
    }
}
