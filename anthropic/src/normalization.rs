use inference_sdk_core::{SdkError, InferenceRequest, InferenceRole, InferenceContent, InferenceEvent, RequestOptions};
use crate::types;

pub fn to_anthropic_request(req: InferenceRequest) -> Result<types::message::MessageRequest, SdkError> {
    let mut messages = Vec::new();
    
    for msg in req.messages {
        match msg.role {
            InferenceRole::User => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    if let InferenceContent::Text { text } = content {
                        content_blocks.push(types::message::ContentBlock::Text { text });
                    }
                }
                if !content_blocks.is_empty() {
                    messages.push(types::message::Message { 
                        role: types::message::Role::User, 
                        content: types::message::Content::Blocks(content_blocks) 
                    });
                }
            }
            InferenceRole::Assistant => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    match content {
                        InferenceContent::Text { text } => {
                            content_blocks.push(types::message::ContentBlock::Text { text });
                        }
                        InferenceContent::ToolUse { id, name, input } => {
                             content_blocks.push(types::message::ContentBlock::ToolUse { id, name, input });
                        }
                        _ => {}
                    }
                }
                messages.push(types::message::Message { 
                    role: types::message::Role::Assistant, 
                    content: types::message::Content::Blocks(content_blocks) 
                });
            }
            InferenceRole::Tool => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    match content {
                        InferenceContent::ToolResult { tool_use_id, content, is_error } => {
                            content_blocks.push(types::message::ContentBlock::ToolResult {
                                tool_use_id,
                                content: Some(vec![types::message::ContentBlock::Text { text: content }]),
                                is_error: Some(is_error),
                            });
                        }
                        _ => {}
                    }
                }
                if !content_blocks.is_empty() {
                    messages.push(types::message::Message { 
                        role: types::message::Role::User, 
                        content: types::message::Content::Blocks(content_blocks) 
                    });
                }
            }
        }
    }

    let tools: Option<Vec<types::message::Tool>> = req.tools.map(|ts| {
        ts.into_iter().map(|t| types::message::Tool {
            name: t.name,
            description: Some(t.description),
            input_schema: t.input_schema,
        }).collect()
    });

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

pub struct AnthropicStreamAdapter {
    input_tokens: u32,
}

impl AnthropicStreamAdapter {
    pub fn new() -> Self {
        Self { input_tokens: 0 }
    }

    pub fn process_event(&mut self, event: types::message::StreamEvent) -> Vec<Result<InferenceEvent, SdkError>> {
        match event {
            types::message::StreamEvent::MessageStart { message } => {
                self.input_tokens = message.usage.input_tokens;
                
                vec![Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: message.model,
                    provider_id: "anthropic".to_string(),
                })]
            },
            types::message::StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                types::message::ContentBlockDelta::TextDelta { text } => vec![Ok(InferenceEvent::MessageDelta { content: text })],
                types::message::ContentBlockDelta::ThinkingDelta { thinking } => vec![Ok(InferenceEvent::ThinkingDelta { content: thinking })],
                _ => vec![],
            },
            types::message::StreamEvent::ContentBlockStart { content_block, .. } => match content_block {
                types::message::ContentBlock::ToolUse { id, name, input } => vec![Ok(InferenceEvent::ToolCall { id, name, args: input })],
                _ => vec![],
            },
            types::message::StreamEvent::MessageDelta { usage, .. } => {
                vec![Ok(InferenceEvent::MessageEnd {
                    input_tokens: self.input_tokens,
                    output_tokens: usage.output_tokens,
                })]
            },
            types::message::StreamEvent::Error { error } => vec![Ok(InferenceEvent::Error { message: error.message })],
            _ => vec![],
        }
    }
}

/// Anthropic-specific extensions for `RequestOptions`.
pub trait AnthropicRequestExt {
    /// Add the `anthropic-beta` header to the request options.
    fn beta(self, version: &str) -> Result<RequestOptions, SdkError>;
}

impl AnthropicRequestExt for RequestOptions {
    fn beta(self, version: &str) -> Result<RequestOptions, SdkError> {
        self.with_header("anthropic-beta", version)
            .map_err(|e| SdkError::ConfigError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{StreamEvent, MessageResponse, Usage as AnthropicUsage, MessageDeltaUsage};

    #[test]
    fn test_anthropic_adapter_captures_usage() {
        let mut adapter = AnthropicStreamAdapter::new();

        let start_event = StreamEvent::MessageStart {
            message: MessageResponse {
                id: "msg_123".to_string(),
                response_type: "message".to_string(),
                role: crate::types::message::Role::Assistant,
                content: vec![],
                model: "claude-3-5-sonnet".to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: 10,
                    output_tokens: 1, 
                },
            }
        };
        
        let events = adapter.process_event(start_event);
        assert_eq!(events.len(), 1);
        if let Ok(InferenceEvent::MessageStart { provider_id, .. }) = &events[0] {
            assert_eq!(provider_id, "anthropic");
        } else {
            panic!("Expected MessageStart");
        }
        assert_eq!(adapter.input_tokens, 10);

        let delta_event = StreamEvent::MessageDelta {
            delta: crate::types::message::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
            },
            usage: MessageDeltaUsage {
                output_tokens: 20,
            }
        };

        let events = adapter.process_event(delta_event);
        assert_eq!(events.len(), 1);
        if let Ok(InferenceEvent::MessageEnd { input_tokens, output_tokens }) = events[0] {
            assert_eq!(input_tokens, 10);
            assert_eq!(output_tokens, 20);
        } else {
            panic!("Expected MessageEnd");
        }
    }
}
