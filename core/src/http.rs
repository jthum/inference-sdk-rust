use crate::error::SdkError;
use reqwest::Method;
use reqwest::header::HeaderMap;
use serde::Serialize;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub headers: HeaderMap,
    pub timeout: Option<std::time::Duration>,
    pub max_retries: Option<u32>,
}

impl RequestOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_header(
        mut self,
        key: &'static str,
        value: &str,
    ) -> Result<Self, crate::error::SdkError> {
        let val = reqwest::header::HeaderValue::from_str(value)
            .map_err(|e| crate::error::SdkError::ConfigError(e.to_string()))?;
        self.headers.insert(key, val);
        Ok(self)
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Alias for `with_retries`, kept for API consistency with client configs.
    pub fn with_max_retries(self, retries: u32) -> Self {
        self.with_retries(retries)
    }
}

/// Retry configuration extracted from a client's defaults and per-request options.
pub struct RetryConfig {
    pub base_url: String,
    pub endpoint: String,
    pub max_retries: u32,
}

const BASE_BACKOFF_MS: u64 = 250;
const MAX_BACKOFF_MS: u64 = 8_000;
const MAX_RETRIES_CAP: u32 = 10;
const JITTER_RANGE_MS: u64 = 200;

fn should_retry_status(status: reqwest::StatusCode) -> bool {
    status.as_u16() == 408 || status.as_u16() == 429 || status.is_server_error()
}

fn should_retry_network_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_request()
}

fn retry_delay(attempt: u32) -> Duration {
    let capped_attempt = attempt.min(10);
    let exp_multiplier = 2_u64.saturating_pow(capped_attempt.saturating_sub(1));
    let base_ms = BASE_BACKOFF_MS
        .saturating_mul(exp_multiplier)
        .min(MAX_BACKOFF_MS);

    let jitter_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jitter_ms = if JITTER_RANGE_MS == 0 {
        0
    } else {
        jitter_seed % JITTER_RANGE_MS
    };

    Duration::from_millis(base_ms.saturating_add(jitter_ms))
}

/// Send an HTTP POST request with exponential backoff retry.
///
/// This is the shared "Physics" layer: every provider SDK uses this
/// to send requests and handle transient failures identically.
pub async fn send_with_retry<T: Serialize>(
    http_client: &reqwest::Client,
    config: &RetryConfig,
    request_body: &T,
    options: &RequestOptions,
) -> Result<reqwest::Response, SdkError> {
    let url = format!("{}{}", config.base_url, config.endpoint);
    let max_retries = options
        .max_retries
        .unwrap_or(config.max_retries)
        .min(MAX_RETRIES_CAP);
    let mut retries = 0;

    loop {
        let mut request_builder = http_client.request(Method::POST, &url).json(request_body);

        if let Some(timeout) = options.timeout {
            request_builder = request_builder.timeout(timeout);
        }

        if !options.headers.is_empty() {
            request_builder = request_builder.headers(options.headers.clone());
        }

        let response_result = request_builder.send().await;

        match response_result {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }

                let status = response.status();
                if should_retry_status(status) && retries < max_retries {
                    retries += 1;
                    let wait = retry_delay(retries);
                    tokio::time::sleep(wait).await;
                    continue;
                }

                let error_text = response.text().await.unwrap_or_default();
                return Err(SdkError::ApiError(format!(
                    "API request failed (status {}): {}",
                    status, error_text
                )));
            }
            Err(e) => {
                if should_retry_network_error(&e) && retries < max_retries {
                    retries += 1;
                    let wait = retry_delay(retries);
                    tokio::time::sleep(wait).await;
                    continue;
                }
                return Err(SdkError::NetworkError(e));
            }
        }
    }
}
