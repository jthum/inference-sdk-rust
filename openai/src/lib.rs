pub mod client;
pub mod normalization;
pub mod resources;
pub mod types;

pub use client::{Client, ClientConfig};
use futures_util::{StreamExt, future::BoxFuture};
pub use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceMessage, InferenceProvider, InferenceRequest,
    InferenceResult, InferenceRole, InferenceStream, RequestOptions, RetryNetworkRule, RetryPolicy,
    RetryStatusRule, SdkError, StopReason, TimeoutPolicy, Usage,
};
pub use types::embedding::EmbeddingRequest;

impl InferenceProvider for Client {
    // Default complete() implementation from trait is used.

    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let openai_req = normalization::to_openai_request(request)?;
            let stream = self
                .chat()
                .create_stream_with_options(openai_req, options.unwrap_or_default())
                .await?;

            // Stateful adapter
            let mut adapter = normalization::OpenAiStreamAdapter::new();

            let mapped_stream = stream.map(
                move |chunk_res: Result<types::chat::ChatCompletionChunk, SdkError>| match chunk_res
                {
                    Ok(chunk) => adapter.process_chunk(chunk),
                    Err(e) => vec![Err(e)],
                },
            );

            // Flatten Vec<Result> to Stream
            let flat_stream = mapped_stream.flat_map(futures_util::stream::iter);

            Ok(Box::pin(flat_stream) as InferenceStream)
        })
    }
}
