pub mod client;
pub mod resources;
pub mod types;
pub mod normalization;

pub use client::{Client, ClientConfig};
pub use types::embedding::EmbeddingRequest;
pub use inference_sdk_core::{RequestOptions, SdkError, InferenceProvider, InferenceRequest, InferenceEvent, InferenceResult, InferenceStream, InferenceMessage, InferenceRole, InferenceContent, StopReason, Usage};
use futures::{StreamExt, future::BoxFuture};

impl InferenceProvider for Client {
    // Default complete() implementation from trait is used.

    fn stream<'a>(&'a self, request: InferenceRequest, options: Option<RequestOptions>) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let openai_req = normalization::to_openai_request(request)?;
            let stream = self.chat().create_stream_with_options(openai_req, options.unwrap_or_default()).await?;
            
            // Stateful adapter
            let mut adapter = normalization::OpenAiStreamAdapter::new();
            
            let mapped_stream = stream.map(move |chunk_res: Result<types::chat::ChatCompletionChunk, SdkError>| {
                match chunk_res {
                    Ok(chunk) => adapter.process_chunk(chunk),
                    Err(e) => vec![Ok(InferenceEvent::Error { message: e.to_string() })],
                }
            });

            // Flatten Vec<Result> to Stream
            let flat_stream = mapped_stream.flat_map(futures::stream::iter);

            Ok(Box::pin(flat_stream) as InferenceStream)
        })
    }
}
