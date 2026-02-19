use crate::config::ClientConfig;
use crate::resources::messages::MessagesResource;
use inference_sdk_core::SdkError;
use reqwest::Client as HttpClient;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Client {
    pub(crate) http_client: HttpClient,
    pub(crate) config: Arc<ClientConfig>,
}

impl Client {
    pub fn new(api_key: impl Into<String>) -> Result<Self, SdkError> {
        let config = ClientConfig::new(api_key.into())?;
        Self::from_config(config)
    }

    pub fn from_config(config: ClientConfig) -> Result<Self, SdkError> {
        let http_client = HttpClient::builder()
            .timeout(config.timeout)
            .default_headers(config.headers.clone())
            .build()
            .map_err(|e| SdkError::ConfigError(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self {
            http_client,
            config: Arc::new(config),
        })
    }

    pub fn messages(&self) -> MessagesResource {
        MessagesResource::new(self.clone())
    }
}
