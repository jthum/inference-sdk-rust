use crate::error::SdkError;
use reqwest::header::HeaderMap;
use reqwest::Method;
use serde::Serialize;

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

    pub fn with_header(mut self, key: &'static str, value: &str) -> Result<Self, crate::error::SdkError> {
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
}

/// Retry configuration extracted from a client's defaults and per-request options.
pub struct RetryConfig {
    pub base_url: String,
    pub endpoint: String,
    pub max_retries: u32,
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
    let max_retries = options.max_retries.unwrap_or(config.max_retries);
    let mut retries = 0;

    loop {
        let mut request_builder = http_client
            .request(Method::POST, &url)
            .json(request_body);

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
                if (status.is_server_error() || status.as_u16() == 429) && retries < max_retries {
                    retries += 1;
                    let wait = std::time::Duration::from_millis(500 * 2_u64.pow(retries - 1));
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
                if retries < max_retries {
                    retries += 1;
                    let wait = std::time::Duration::from_millis(500 * 2_u64.pow(retries - 1));
                    tokio::time::sleep(wait).await;
                    continue;
                }
                return Err(SdkError::NetworkError(e));
            }
        }
    }
}
