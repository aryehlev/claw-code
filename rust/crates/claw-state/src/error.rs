use std::fmt::{Display, Formatter};

/// Errors raised by any operation on the state store.
#[derive(Debug)]
pub enum StateError {
    Sqlite(rusqlite::Error),
    Serde(serde_json::Error),
    Io(std::io::Error),
    InvalidInput(String),
}

impl Display for StateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sqlite(error) => write!(f, "state sqlite error: {error}"),
            Self::Serde(error) => write!(f, "state serde error: {error}"),
            Self::Io(error) => write!(f, "state io error: {error}"),
            Self::InvalidInput(message) => write!(f, "state invalid input: {message}"),
        }
    }
}

impl std::error::Error for StateError {}

impl From<rusqlite::Error> for StateError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sqlite(value)
    }
}

impl From<serde_json::Error> for StateError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value)
    }
}

impl From<std::io::Error> for StateError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
