use crate::types;
use base64::Engine;
use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceRequest, InferenceRole, RequestOptions, SdkError,
    StopReason,
};
use std::fs;
use std::path::Path;

pub fn to_anthropic_request(
    req: InferenceRequest,
) -> Result<types::message::MessageRequest, SdkError> {
    if req.response_format.is_some() {
        return Err(SdkError::ConfigError(
            "structured response_format is not supported by the anthropic provider driver"
                .to_string(),
        ));
    }

    let mut messages = Vec::new();

    for msg in req.messages {
        match msg.role {
            InferenceRole::User => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    match content {
                        InferenceContent::Text { text } => {
                            content_blocks.push(types::message::ContentBlock::Text { text });
                        }
                        InferenceContent::Image {
                            content_type,
                            url,
                            local_path,
                            ..
                        } => {
                            content_blocks.push(types::message::ContentBlock::Image {
                                source: encode_anthropic_image(
                                    content_type.as_deref(),
                                    local_path.as_deref(),
                                    url.as_deref(),
                                )?,
                            });
                        }
                        InferenceContent::File {
                            name,
                            content_type,
                            url,
                            local_path,
                        } => {
                            content_blocks.push(types::message::ContentBlock::Text {
                                text: render_file_fallback_text(
                                    name.as_deref(),
                                    content_type.as_deref(),
                                    url.as_deref(),
                                    local_path.as_deref(),
                                ),
                            });
                        }
                        _ => {}
                    }
                }

                if !content_blocks.is_empty() {
                    messages.push(types::message::Message {
                        role: types::message::Role::User,
                        content: types::message::Content::Blocks(content_blocks),
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
                            content_blocks.push(types::message::ContentBlock::ToolUse {
                                id,
                                name,
                                input,
                            });
                        }
                        InferenceContent::Thinking { content, signature } => {
                            content_blocks.push(types::message::ContentBlock::Thinking {
                                thinking: content,
                                signature,
                            });
                        }
                        _ => {}
                    }
                }

                if !content_blocks.is_empty() {
                    messages.push(types::message::Message {
                        role: types::message::Role::Assistant,
                        content: types::message::Content::Blocks(content_blocks),
                    });
                }
            }
            InferenceRole::Tool => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    if let InferenceContent::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = content
                    {
                        content_blocks.push(types::message::ContentBlock::ToolResult {
                            tool_use_id,
                            content: Some(types::message::ToolResultContent::Text(content)),
                            is_error: is_error.then_some(true),
                        });
                    }
                }

                if !content_blocks.is_empty() {
                    // Anthropic expects tool results to be sent as a user role message.
                    messages.push(types::message::Message {
                        role: types::message::Role::User,
                        content: types::message::Content::Blocks(content_blocks),
                    });
                }
            }
        }
    }

    let tools: Option<Vec<types::message::Tool>> = req.tools.map(|ts| {
        ts.into_iter()
            .map(|t| types::message::Tool {
                name: t.name,
                description: Some(t.description),
                input_schema: t.input_schema,
            })
            .collect()
    });

    let thinking = req
        .thinking_budget
        .map(|budget| types::message::ThinkingConfig {
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

fn encode_anthropic_image(
    content_type: Option<&str>,
    local_path: Option<&str>,
    url: Option<&str>,
) -> Result<types::message::ImageSource, SdkError> {
    let local_path = local_path.ok_or_else(|| {
        let suffix = url
            .map(|value| {
                format!("; url-only image refs are not supported by the anthropic driver: {value}")
            })
            .unwrap_or_default();
        SdkError::ConfigError(format!(
            "anthropic image content must include local_path{suffix}"
        ))
    })?;

    let bytes = fs::read(local_path)?;
    let media_type = infer_image_media_type(content_type, local_path)?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);

    Ok(types::message::ImageSource {
        source_type: "base64".to_string(),
        media_type,
        data,
    })
}

fn infer_image_media_type(
    content_type: Option<&str>,
    local_path: &str,
) -> Result<String, SdkError> {
    if let Some(content_type) = content_type {
        let content_type = content_type.trim().to_ascii_lowercase();
        return match content_type.as_str() {
            "image/png" => Ok("image/png".to_string()),
            "image/jpeg" | "image/jpg" => Ok("image/jpeg".to_string()),
            "image/gif" => Ok("image/gif".to_string()),
            "image/webp" => Ok("image/webp".to_string()),
            other if other.starts_with("image/") => Ok(other.to_string()),
            other => Err(SdkError::ConfigError(format!(
                "unsupported image content_type '{other}'"
            ))),
        };
    }

    let extension = Path::new(local_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("png") => Ok("image/png".to_string()),
        Some("jpg" | "jpeg") => Ok("image/jpeg".to_string()),
        Some("gif") => Ok("image/gif".to_string()),
        Some("webp") => Ok("image/webp".to_string()),
        _ => Err(SdkError::ConfigError(format!(
            "could not infer image media type for '{local_path}'"
        ))),
    }
}

fn render_file_fallback_text(
    name: Option<&str>,
    content_type: Option<&str>,
    url: Option<&str>,
    local_path: Option<&str>,
) -> String {
    let mut details = Vec::new();

    if let Some(name) = name.filter(|value| !value.trim().is_empty()) {
        details.push(format!("name={name}"));
    }
    if let Some(content_type) = content_type.filter(|value| !value.trim().is_empty()) {
        details.push(format!("type={content_type}"));
    }
    if let Some(local_path) = local_path.filter(|value| !value.trim().is_empty()) {
        details.push(format!("local_path={local_path}"));
    } else if let Some(url) = url.filter(|value| !value.trim().is_empty()) {
        details.push(format!("url={url}"));
    }

    if details.is_empty() {
        "[file attachment]".to_string()
    } else {
        format!("[file attachment: {}]", details.join(", "))
    }
}

#[derive(Default)]
pub struct AnthropicStreamAdapter {
    input_tokens: u32,
    cache_read_input_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
}

impl AnthropicStreamAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_event(
        &mut self,
        event: types::message::StreamEvent,
    ) -> Vec<Result<InferenceEvent, SdkError>> {
        match event {
            types::message::StreamEvent::MessageStart { message } => {
                self.input_tokens = message.usage.input_tokens;
                self.cache_read_input_tokens = message.usage.cache_read_input_tokens;
                self.cache_creation_input_tokens = message.usage.cache_creation_input_tokens;

                vec![Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: message.model,
                    provider_id: "anthropic".to_string(),
                })]
            }
            types::message::StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                types::message::ContentBlockDelta::TextDelta { text } => {
                    vec![Ok(InferenceEvent::MessageDelta { content: text })]
                }
                types::message::ContentBlockDelta::ThinkingDelta { thinking } => {
                    vec![Ok(InferenceEvent::ThinkingDelta { content: thinking })]
                }
                types::message::ContentBlockDelta::SignatureDelta { signature } => {
                    vec![Ok(InferenceEvent::ThinkingSignatureDelta { signature })]
                }
                types::message::ContentBlockDelta::InputJsonDelta { partial_json } => {
                    vec![Ok(InferenceEvent::ToolCallDelta {
                        delta: partial_json,
                    })]
                }
            },
            types::message::StreamEvent::ContentBlockStart {
                content_block: types::message::ContentBlock::ToolUse { id, name, .. },
                ..
            } => vec![Ok(InferenceEvent::ToolCallStart { id, name })],
            types::message::StreamEvent::MessageDelta { delta, usage } => {
                if let Some(input_tokens) = usage.input_tokens {
                    self.input_tokens = input_tokens;
                }
                if usage.cache_read_input_tokens.is_some() {
                    self.cache_read_input_tokens = usage.cache_read_input_tokens;
                }
                if usage.cache_creation_input_tokens.is_some() {
                    self.cache_creation_input_tokens = usage.cache_creation_input_tokens;
                }
                let stop_reason = delta.stop_reason.map(|s| match s.as_str() {
                    "end_turn" => StopReason::EndTurn,
                    "max_tokens" => StopReason::MaxTokens,
                    "tool_use" => StopReason::ToolUse,
                    "stop_sequence" => StopReason::StopSequence,
                    _ => StopReason::Unknown,
                });

                vec![Ok(InferenceEvent::MessageEnd {
                    input_tokens: self
                        .input_tokens
                        .saturating_add(self.cache_read_input_tokens.unwrap_or(0))
                        .saturating_add(self.cache_creation_input_tokens.unwrap_or(0)),
                    output_tokens: usage.output_tokens,
                    cache_read_input_tokens: self.cache_read_input_tokens,
                    cache_creation_input_tokens: self.cache_creation_input_tokens,
                    stop_reason,
                })]
            }
            types::message::StreamEvent::Error { error } => {
                vec![Err(SdkError::ProviderError(error.message))]
            }
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
    use crate::types::message::{
        MessageDeltaUsage, MessageResponse, StreamEvent, Usage as AnthropicUsage,
    };

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
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
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
                input_tokens: None,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let events = adapter.process_event(delta_event);
        assert_eq!(events.len(), 1);
        if let Ok(InferenceEvent::MessageEnd {
            input_tokens,
            output_tokens,
            stop_reason,
            ..
        }) = &events[0]
        {
            assert_eq!(*input_tokens, 10);
            assert_eq!(*output_tokens, 20);
            assert_eq!(*stop_reason, Some(StopReason::EndTurn));
        } else {
            panic!("Expected MessageEnd");
        }
    }

    #[test]
    fn test_anthropic_adapter_normalizes_cache_usage_and_total_input() {
        let mut adapter = AnthropicStreamAdapter::new();
        adapter.process_event(StreamEvent::MessageStart {
            message: MessageResponse {
                id: "msg_cached".to_string(),
                response_type: "message".to_string(),
                role: crate::types::message::Role::Assistant,
                content: vec![],
                model: "claude-sonnet".to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: 12,
                    output_tokens: 1,
                    cache_creation_input_tokens: Some(30),
                    cache_read_input_tokens: Some(80),
                },
            },
        });

        let events = adapter.process_event(StreamEvent::MessageDelta {
            delta: crate::types::message::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
            },
            usage: MessageDeltaUsage {
                output_tokens: 20,
                input_tokens: None,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        });

        assert!(matches!(
            events[0],
            Ok(InferenceEvent::MessageEnd {
                input_tokens: 122,
                output_tokens: 20,
                cache_read_input_tokens: Some(80),
                cache_creation_input_tokens: Some(30),
                ..
            })
        ));
    }

    #[test]
    fn test_anthropic_adapter_emits_tool_argument_deltas() {
        let mut adapter = AnthropicStreamAdapter::new();
        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: types::message::ContentBlockDelta::InputJsonDelta {
                partial_json: "{\"city\":\"S".to_string(),
            },
        };
        let events = adapter.process_event(event);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::ToolCallDelta { ref delta }) if delta == "{\"city\":\"S"
        ));
    }

    #[test]
    fn test_anthropic_adapter_emits_thinking_signature_deltas() {
        let mut adapter = AnthropicStreamAdapter::new();
        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: types::message::ContentBlockDelta::SignatureDelta {
                signature: "sig_abc".to_string(),
            },
        };
        let events = adapter.process_event(event);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::ThinkingSignatureDelta { ref signature }) if signature == "sig_abc"
        ));
    }
}

#[cfg(test)]
mod request_normalization_tests {
    use super::to_anthropic_request;
    use inference_sdk_core::{
        InferenceContent, InferenceMessage, InferenceRequest, InferenceResponseFormat,
        InferenceRole,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_fixture_path(extension: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("inference_sdk_anthropic_{nonce}.{extension}"))
    }

    #[test]
    fn preserves_assistant_thinking_blocks_in_request_history() {
        let req = InferenceRequest::builder()
            .model("test-model")
            .messages(vec![InferenceMessage {
                role: InferenceRole::Assistant,
                content: vec![
                    InferenceContent::Thinking {
                        content: "deliberation".to_string(),
                        signature: Some("sig-123".to_string()),
                    },
                    InferenceContent::ToolUse {
                        id: "toolu_1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({ "path": "nonce.txt" }),
                    },
                ],
                tool_call_id: None,
            }])
            .max_tokens(128)
            .build();

        let out = to_anthropic_request(req).expect("request should normalize");
        assert_eq!(out.messages.len(), 1);

        match &out.messages[0].content {
            crate::types::message::Content::Blocks(blocks) => {
                assert!(matches!(
                    &blocks[0],
                    crate::types::message::ContentBlock::Thinking {
                        thinking,
                        signature: Some(signature),
                    } if thinking == "deliberation" && signature == "sig-123"
                ));
                assert!(matches!(
                    &blocks[1],
                    crate::types::message::ContentBlock::ToolUse { id, name, .. }
                    if id == "toolu_1" && name == "read_file"
                ));
            }
            other => panic!("unexpected content form: {other:?}"),
        }
    }

    #[test]
    fn rejects_structured_response_format_hints() {
        let req = InferenceRequest::builder()
            .model("test-model")
            .messages(vec![InferenceMessage {
                role: InferenceRole::User,
                content: vec![InferenceContent::Text {
                    text: "hello".to_string(),
                }],
                tool_call_id: None,
            }])
            .max_tokens(128)
            .response_format(InferenceResponseFormat::JsonObject)
            .build();

        let err = to_anthropic_request(req).expect_err("request should reject response_format");
        assert!(matches!(err, inference_sdk_core::SdkError::ConfigError(_)));
        assert!(
            err.to_string().contains("structured response_format"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn maps_user_images_and_files() {
        let image_path = temp_fixture_path("png");
        fs::write(&image_path, [1_u8, 2, 3, 4]).expect("fixture image");

        let req = InferenceRequest::builder()
            .model("test-model")
            .messages(vec![InferenceMessage {
                role: InferenceRole::User,
                content: vec![
                    InferenceContent::Text {
                        text: "inspect these".to_string(),
                    },
                    InferenceContent::Image {
                        name: Some("diagram.png".to_string()),
                        content_type: Some("image/png".to_string()),
                        url: None,
                        local_path: Some(image_path.display().to_string()),
                        detail: None,
                    },
                    InferenceContent::File {
                        name: Some("spec.pdf".to_string()),
                        content_type: Some("application/pdf".to_string()),
                        url: Some("https://example.test/spec.pdf".to_string()),
                        local_path: None,
                    },
                ],
                tool_call_id: None,
            }])
            .max_tokens(128)
            .build();

        let out = to_anthropic_request(req).expect("request should normalize");
        match &out.messages[0].content {
            crate::types::message::Content::Blocks(blocks) => {
                assert!(matches!(
                    &blocks[0],
                    crate::types::message::ContentBlock::Text { text } if text == "inspect these"
                ));
                assert!(matches!(
                    &blocks[1],
                    crate::types::message::ContentBlock::Image { source }
                    if source.source_type == "base64"
                        && source.media_type == "image/png"
                        && !source.data.is_empty()
                ));
                assert!(matches!(
                    &blocks[2],
                    crate::types::message::ContentBlock::Text { text }
                    if text.contains("spec.pdf") && text.contains("application/pdf")
                ));
            }
            other => panic!("unexpected content form: {other:?}"),
        }

        let _ = fs::remove_file(image_path);
    }
}

#[cfg(test)]
mod tool_result_request_shape_tests {
    use super::to_anthropic_request;
    use inference_sdk_core::{InferenceContent, InferenceMessage, InferenceRequest, InferenceRole};

    #[test]
    fn tool_results_serialize_as_string_content_and_omit_false_is_error() {
        let req = InferenceRequest::builder()
            .model("test-model")
            .messages(vec![InferenceMessage {
                role: InferenceRole::Tool,
                content: vec![InferenceContent::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: "ok".to_string(),
                    is_error: false,
                }],
                tool_call_id: Some("toolu_1".to_string()),
            }])
            .max_tokens(128)
            .build();

        let out = to_anthropic_request(req).expect("request should normalize");
        let json = serde_json::to_value(out).expect("request should serialize");
        let block = &json["messages"][0]["content"][0];

        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["tool_use_id"], "toolu_1");
        assert_eq!(block["content"], "ok");
        assert!(
            block.get("is_error").is_none(),
            "is_error=false should be omitted"
        );
    }
}
