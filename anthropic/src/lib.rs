pub mod client;
pub mod config;
pub mod error;
pub mod resources;
pub mod types;

pub use client::Client;
pub use config::ClientConfig;
pub use error::AnthropicError;

// Re-export core types
pub use inference_sdk_core::{SdkError, InferenceProvider, InferenceRequest, InferenceEvent, InferenceResult, InferenceStream};
use inference_sdk_core::futures::{StreamExt, future::BoxFuture};

// Re-export core types for convenience
pub use inference_sdk_core::RequestOptions;


impl InferenceProvider for Client {
    fn complete<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceResult, SdkError>> {
        Box::pin(async move {
                // TODO: Implement blocking completion re-using stream aggregation or direct call
        // For now, let's use the blocking `.create()` on messages resource
        let anthropic_req = to_anthropic_request(request)?;
        let response = self.messages().create(anthropic_req).await?;
        
        let content = response.content.iter()
            .filter_map(|c| match c {
                types::message::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
            
        Ok(InferenceResult {
            content,
            usage: inference_sdk_core::Usage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
            }
            })
        })
    }

    fn stream<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let anthropic_req = to_anthropic_request(request)?;
            
            // Handle beta headers (e.g. for thinking)
            let mut opts = crate::RequestOptions::default();
            if anthropic_req.thinking.is_some() {
                 opts = opts.beta("output-128k-2025-02-19")?;
            }

            let stream = self.messages().create_stream_with_options(anthropic_req, opts).await?;
            
            let mapped_stream = stream.map(|event_res| {
                match event_res {
                    Ok(event) => normalize_anthropic_event(event),
                    Err(e) => Ok(Some(InferenceEvent::Error { message: e.to_string() })),
                }
            });
            
            // Flatten Option<InferenceEvent> (filter None) and handle Result
            let filtered = mapped_stream.filter_map(|evt| async move {
                match evt {
                    Ok(Some(e)) => Some(Ok(e)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

            Ok(Box::pin(filtered) as InferenceStream)
        })
    }
}

fn to_anthropic_request(req: InferenceRequest) -> Result<crate::types::message::MessageRequest, SdkError> {
    let mut messages = Vec::new();
    
    for msg in req.messages {
        let role = match msg.role {
            inference_sdk_core::InferenceRole::User => types::message::Role::User,
            inference_sdk_core::InferenceRole::Assistant => types::message::Role::Assistant,
        };
        
        let mut content_blocks = Vec::new();
        for content in msg.content {
            match content {
                inference_sdk_core::InferenceContent::Text { text } => {
                    content_blocks.push(types::message::ContentBlock::Text { text });
                }
                inference_sdk_core::InferenceContent::ToolUse { id, name, input } => {
                     content_blocks.push(types::message::ContentBlock::ToolUse { id, name, input });
                }
                inference_sdk_core::InferenceContent::ToolResult { tool_use_id, content, is_error } => {
                    content_blocks.push(types::message::ContentBlock::ToolResult {
                        tool_use_id,
                        content: Some(vec![types::message::ContentBlock::Text { text: content }]),
                        is_error: Some(is_error),
                    });
                }
            }
        }
        
        messages.push(types::message::Message { role, content: types::message::Content::Blocks(content_blocks) });
    }

    let tools: Option<Vec<types::message::Tool>> = req.tools.map(|ts| {
        ts.into_iter().map(|t| types::message::Tool {
            name: t.name,
            description: t.description,
            input_schema: t.input_schema,
        }).collect()
    });

    // Handle Thinking
    let thinking = req.thinking_budget.map(|budget| types::message::ThinkingConfig {
        thinking_type: "enabled".to_string(),
        budget_tokens: budget,
    });

    Ok(types::message::MessageRequest::builder()
        .model(req.model)
        .messages(messages)
        .maybe_system(req.system)
        .max_tokens(req.max_tokens.unwrap_or(8192))
        .maybe_temperature(req.temperature)
        .maybe_tools(tools)
        .maybe_thinking(thinking)
        .build())
}

fn normalize_anthropic_event(event: crate::types::message::StreamEvent) -> Result<Option<InferenceEvent>, SdkError> {
    match event {
        types::message::StreamEvent::MessageStart { message } => Ok(Some(InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: message.model,
            provider_id: "anthropic".to_string(),
        })),
        types::message::StreamEvent::ContentBlockDelta { delta, .. } => match delta {
            types::message::ContentBlockDelta::TextDelta { text } => Ok(Some(InferenceEvent::MessageDelta { content: text })),
            types::message::ContentBlockDelta::ThinkingDelta { thinking } => Ok(Some(InferenceEvent::ThinkingDelta { content: thinking })),
            _ => Ok(None),
        },
        types::message::StreamEvent::ContentBlockStart { content_block, .. } => match content_block {
            types::message::ContentBlock::ToolUse { id, name, input } => Ok(Some(InferenceEvent::ToolCall { id, name, args: input })),
            _ => Ok(None),
        },
        types::message::StreamEvent::MessageDelta { usage, .. } => Ok(Some(InferenceEvent::MessageEnd {
            input_tokens: 0, // Not provided in delta
            output_tokens: usage.output_tokens,
        })),
        types::message::StreamEvent::Error { error } => Ok(Some(InferenceEvent::Error { message: error.message })),
        _ => Ok(None),
    }
}

/// Anthropic-specific extensions for `RequestOptions`.
pub trait AnthropicRequestExt {
    /// Add the `anthropic-beta` header to the request options.
    fn beta(self, version: &str) -> Result<RequestOptions, AnthropicError>;
}

impl AnthropicRequestExt for RequestOptions {
    fn beta(self, version: &str) -> Result<RequestOptions, AnthropicError> {
        self.with_header("anthropic-beta", version)
            .map_err(AnthropicError::from)
    }
}

