pub mod client;
pub mod config;
pub mod resources;
pub mod types;
pub mod normalization;

pub use client::Client;
pub use config::ClientConfig;
pub use normalization::AnthropicRequestExt;

// Re-export core types
pub use inference_sdk_core::{SdkError, InferenceProvider, InferenceRequest, InferenceEvent, InferenceResult, InferenceStream, InferenceMessage, RequestOptions, InferenceContent, InferenceRole, StopReason, Usage};
use futures::{StreamExt, future::BoxFuture};

impl InferenceProvider for Client {
    // Default complete() implementation from trait is used.

    fn stream<'a>(&'a self, request: InferenceRequest, options: Option<RequestOptions>) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let anthropic_req = normalization::to_anthropic_request(request)?;
            
            // Handle beta headers (e.g. for thinking)
            let mut opts = options.unwrap_or_default();
            if anthropic_req.thinking.is_some() {
                 opts = opts.beta("output-128k-2025-02-19")?;
            }

            let stream = self.messages().create_stream_with_options(anthropic_req, opts).await?;
            
            // Stateful adapter
            let mut adapter = normalization::AnthropicStreamAdapter::new();
            
            let mapped_stream = stream.map(move |event_res: Result<types::message::StreamEvent, SdkError>| {
                match event_res {
                    Ok(event) => adapter.process_event(event),
                    Err(e) => vec![Ok(InferenceEvent::Error { message: e.to_string() })],
                }
            });
            
            let flat_stream = mapped_stream.flat_map(futures::stream::iter);

            Ok(Box::pin(flat_stream) as InferenceStream)
        })
    }
}

