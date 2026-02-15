use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use std::fmt;
use std::time::Duration;

use crate::SdkError;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct ClientConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
    pub max_retries: u32,
    pub headers: HeaderMap,
}

// Manually implement Debug to redact the API key
impl fmt::Debug for ClientConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientConfig")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl ClientConfig {
    pub fn new(api_key: String) -> Result<Self, SdkError> {
        let header_value = HeaderValue::from_str(&api_key)
            .map_err(|e| SdkError::ConfigError(format!("Invalid API key: {}", e)))?;

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", header_value);
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: DEFAULT_TIMEOUT,
            max_retries: 2,
            headers,
        })
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}
