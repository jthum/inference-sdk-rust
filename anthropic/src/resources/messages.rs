use crate::client::Client;
use crate::error::AnthropicError;
use crate::types::message::{MessageRequest, MessageResponse, StreamEvent};
use eventsource_stream::Eventsource;
use futures::Stream;
use futures::StreamExt;
use inference_sdk_core::http::{send_with_retry, RetryConfig};
use inference_sdk_core::RequestOptions;
use std::pin::Pin;

#[derive(Clone, Debug)]
pub struct MessagesResource {
    pub(crate) client: Client,
}

impl MessagesResource {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Create a Message (non-streaming)
    ///
    /// POST /v1/messages
    pub async fn create(&self, request: MessageRequest) -> Result<MessageResponse, AnthropicError> {
        self.create_with_options(request, RequestOptions::default())
            .await
    }

    /// Create a Message with custom options
    pub async fn create_with_options(
        &self,
        request: MessageRequest,
        options: RequestOptions,
    ) -> Result<MessageResponse, AnthropicError> {
        let config = RetryConfig {
            base_url: self.client.config.base_url.clone(),
            endpoint: "/messages".to_string(),
            max_retries: self.client.config.max_retries,
        };
        let response = send_with_retry(&self.client.http_client, &config, &request, &options)
            .await
            .map_err(AnthropicError::from)?;
        response
            .json::<MessageResponse>()
            .await
            .map_err(AnthropicError::from)
    }

    /// Create a Message Stream
    ///
    /// POST /v1/messages (returning an SSE stream)
    pub async fn create_stream(
        &self,
        request: MessageRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamEvent, AnthropicError>> + Send + 'static>>,
        AnthropicError,
    > {
        self.create_stream_with_options(request, RequestOptions::default())
            .await
    }

    /// Create a Message Stream with custom options
    pub async fn create_stream_with_options(
        &self,
        mut request: MessageRequest,
        options: RequestOptions,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamEvent, AnthropicError>> + Send + 'static>>,
        AnthropicError,
    > {
        request.stream = Some(true);

        let config = RetryConfig {
            base_url: self.client.config.base_url.clone(),
            endpoint: "/messages".to_string(),
            max_retries: self.client.config.max_retries,
        };
        let response = send_with_retry(&self.client.http_client, &config, &request, &options)
            .await
            .map_err(AnthropicError::from)?;
        let stream = response.bytes_stream().eventsource();

        let mapped_stream = stream.map(|event_result| match event_result {
            Ok(event) => {
                if event.event == "ping" {
                    return Ok(StreamEvent::Ping);
                }
                serde_json::from_str::<StreamEvent>(&event.data)
                    .map_err(AnthropicError::SerializationError)
            }
            Err(e) => Err(AnthropicError::StreamError(e.to_string())),
        });

        Ok(Box::pin(mapped_stream))
    }
}
