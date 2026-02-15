use thiserror::Error;

/// Base error type shared across all provider SDKs.
#[derive(Error, Debug)]
pub enum SdkError {
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Provider error: {0}")]
    ProviderError(String),
    #[error("Unknown error: {0}")]
    Unknown(String),
}
