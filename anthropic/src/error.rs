use thiserror::Error;

/// Anthropic-specific error type that wraps the core SdkError.
#[derive(Error, Debug)]
pub enum AnthropicError {
    #[error(transparent)]
    Sdk(#[from] inference_sdk_core::SdkError),
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
    #[error("Unknown error: {0}")]
    Unknown(String),
}
impl From<AnthropicError> for inference_sdk_core::SdkError {
    fn from(err: AnthropicError) -> Self {
        match err {
            AnthropicError::Sdk(e) => e,
            AnthropicError::ApiError(msg) => inference_sdk_core::SdkError::ApiError(msg),
            AnthropicError::NetworkError(e) => inference_sdk_core::SdkError::NetworkError(e),
            AnthropicError::SerializationError(e) => inference_sdk_core::SdkError::SerializationError(e),
            AnthropicError::ConfigError(msg) => inference_sdk_core::SdkError::ConfigError(msg),
            AnthropicError::StreamError(msg) => inference_sdk_core::SdkError::StreamError(msg),
            AnthropicError::Unknown(msg) => inference_sdk_core::SdkError::Unknown(msg),
        }
    }
}
