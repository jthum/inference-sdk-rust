use crate::error::SdkError;
use reqwest::header::HeaderMap;
use std::time::Duration;

/// Per-request options that are provider-agnostic.
#[derive(Clone, Debug, Default)]
pub struct RequestOptions {
    pub timeout: Option<Duration>,
    pub headers: Option<HeaderMap>,
    pub max_retries: Option<u32>,
}

impl RequestOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_header(
        mut self,
        key: impl reqwest::header::IntoHeaderName,
        value: &str,
    ) -> Result<Self, SdkError> {
        let mut headers = self.headers.unwrap_or_default();
        let header_value = reqwest::header::HeaderValue::from_str(value)
            .map_err(|e| SdkError::ConfigError(format!("Invalid header value: {}", e)))?;
        headers.insert(key, header_value);
        self.headers = Some(headers);
        Ok(self)
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }
}
