use crate::types;
use base64::Engine;
use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceRequest, InferenceResponseFormat, InferenceRole,
    SdkError, StopReason,
};
use std::fs;
use std::path::Path;

pub fn to_openai_request(
    req: InferenceRequest,
) -> Result<types::chat::ChatCompletionRequest, SdkError> {
    let mut messages = Vec::new();

    if let Some(system) = req.system {
        messages.push(types::chat::ChatMessage {
            role: types::chat::ChatRole::System,
            content: Some(types::chat::ChatContent::Text(system)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for msg in req.messages {
        match msg.role {
            InferenceRole::User => {
                if let Some(content) = normalize_user_content(msg.content)? {
                    messages.push(types::chat::ChatMessage {
                        role: types::chat::ChatRole::User,
                        content: Some(content),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            InferenceRole::Assistant => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<types::chat::ToolCall> = Vec::new();

                for content in msg.content {
                    match content {
                        InferenceContent::Text { text } => text_parts.push(text),
                        InferenceContent::ToolUse { id, name, input } => {
                            let arguments = serde_json::to_string(&input)
                                .map_err(SdkError::SerializationError)?;
                            tool_calls.push(types::chat::ToolCall {
                                id,
                                call_type: "function".to_string(),
                                function: types::chat::FunctionCall { name, arguments },
                            });
                        }
                        _ => {}
                    }
                }

                if !text_parts.is_empty() || !tool_calls.is_empty() {
                    messages.push(types::chat::ChatMessage {
                        role: types::chat::ChatRole::Assistant,
                        content: if text_parts.is_empty() {
                            None
                        } else {
                            Some(types::chat::ChatContent::Text(text_parts.join("\n")))
                        },
                        name: None,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                        tool_call_id: None,
                    });
                }
            }
            InferenceRole::Tool => {
                for content in msg.content {
                    if let InferenceContent::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = content
                    {
                        messages.push(types::chat::ChatMessage {
                            role: types::chat::ChatRole::Tool,
                            content: Some(types::chat::ChatContent::Text(content)),
                            name: None,
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id),
                        });
                    }
                }
            }
        }
    }

    let tools: Option<Vec<types::chat::Tool>> = req.tools.map(|ts| {
        ts.into_iter()
            .map(|t| types::chat::Tool {
                tool_type: "function".to_string(),
                function: types::chat::FunctionDefinition {
                    name: t.name,
                    description: Some(t.description),
                    parameters: t.input_schema,
                    strict: None,
                },
            })
            .collect()
    });

    let tool_choice = if tools.as_ref().is_some_and(|ts| !ts.is_empty()) {
        Some(types::chat::ToolChoice::Mode("auto".to_string()))
    } else {
        None
    };

    let response_format = req
        .response_format
        .map(normalize_response_format)
        .transpose()?;

    Ok(types::chat::ChatCompletionRequest::builder()
        .model(req.model)
        .messages(messages)
        .maybe_temperature(req.temperature)
        .maybe_max_tokens(req.max_tokens)
        .maybe_tools(tools)
        .maybe_tool_choice(tool_choice)
        .maybe_response_format(response_format)
        .build())
}

fn normalize_user_content(
    content: Vec<InferenceContent>,
) -> Result<Option<types::chat::ChatContent>, SdkError> {
    let mut text_parts = Vec::new();
    let mut mixed_parts = Vec::new();
    let mut saw_non_text = false;

    for item in content {
        match item {
            InferenceContent::Text { text } => {
                if saw_non_text {
                    mixed_parts.push(types::chat::ContentPart::Text { text });
                } else {
                    text_parts.push(text);
                }
            }
            InferenceContent::Image {
                content_type,
                url,
                local_path,
                detail,
                ..
            } => {
                saw_non_text = true;
                if !text_parts.is_empty() {
                    mixed_parts.extend(
                        text_parts
                            .drain(..)
                            .map(|text| types::chat::ContentPart::Text { text }),
                    );
                }

                mixed_parts.push(types::chat::ContentPart::ImageUrl {
                    image_url: types::chat::ImageUrl {
                        url: normalize_openai_image_url(
                            content_type.as_deref(),
                            local_path.as_deref(),
                            url.as_deref(),
                        )?,
                        detail,
                    },
                });
            }
            InferenceContent::File {
                name,
                content_type,
                url,
                local_path,
            } => {
                saw_non_text = true;
                if !text_parts.is_empty() {
                    mixed_parts.extend(
                        text_parts
                            .drain(..)
                            .map(|text| types::chat::ContentPart::Text { text }),
                    );
                }

                mixed_parts.push(types::chat::ContentPart::Text {
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

    if saw_non_text {
        if mixed_parts.is_empty() {
            Ok(None)
        } else {
            Ok(Some(types::chat::ChatContent::Parts(mixed_parts)))
        }
    } else if text_parts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(types::chat::ChatContent::Text(text_parts.join("\n"))))
    }
}

fn normalize_openai_image_url(
    content_type: Option<&str>,
    local_path: Option<&str>,
    url: Option<&str>,
) -> Result<String, SdkError> {
    if let Some(local_path) = local_path {
        let bytes = fs::read(local_path)?;
        let media_type = infer_image_media_type(content_type, local_path)?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        return Ok(format!("data:{media_type};base64,{encoded}"));
    }

    url.map(str::to_string).ok_or_else(|| {
        SdkError::ConfigError("image content must include either local_path or url".to_string())
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
            "image/bmp" => Ok("image/bmp".to_string()),
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
        Some("bmp") => Ok("image/bmp".to_string()),
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

fn normalize_response_format(
    response_format: InferenceResponseFormat,
) -> Result<types::chat::ResponseFormat, SdkError> {
    Ok(match response_format {
        InferenceResponseFormat::Text => types::chat::ResponseFormat::Text,
        InferenceResponseFormat::JsonObject => types::chat::ResponseFormat::JsonObject,
        InferenceResponseFormat::JsonSchema { json_schema } => {
            types::chat::ResponseFormat::JsonSchema {
                json_schema: types::chat::JsonSchemaConfig {
                    name: json_schema.name,
                    description: json_schema.description,
                    schema: json_schema.schema,
                    strict: json_schema.strict,
                },
            }
        }
    })
}

#[derive(Default)]
pub struct OpenAiStreamAdapter {
    stop_reason: Option<StopReason>,
    message_started: bool,
}

impl OpenAiStreamAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_chunk(
        &mut self,
        chunk: types::chat::ChatCompletionChunk,
    ) -> Vec<Result<InferenceEvent, SdkError>> {
        let mut events = Vec::new();

        if chunk.choices.is_empty() {
            if let Some(usage) = chunk.usage {
                events.push(Ok(InferenceEvent::MessageEnd {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    cache_read_input_tokens: usage
                        .prompt_tokens_details
                        .and_then(|details| details.cached_tokens),
                    cache_creation_input_tokens: None,
                    stop_reason: self.stop_reason.clone(),
                }));
            }
            return events;
        }

        let choice = &chunk.choices[0];
        let model_name = chunk.model.clone();

        if let Some(types::chat::ChatRole::Assistant) = &choice.delta.role
            && !self.message_started
        {
            self.message_started = true;
            events.push(Ok(InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: model_name,
                provider_id: "openai".to_string(),
            }));
        }

        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            events.push(Ok(InferenceEvent::MessageDelta {
                content: content.clone(),
            }));
        }

        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                if let Some(func) = &tc.function {
                    if let (Some(id), Some(name)) = (&tc.id, &func.name) {
                        events.push(Ok(InferenceEvent::ToolCallStart {
                            id: id.clone(),
                            name: name.clone(),
                        }));
                    }

                    if let Some(arguments) = &func.arguments
                        && !arguments.is_empty()
                    {
                        events.push(Ok(InferenceEvent::ToolCallDelta {
                            delta: arguments.clone(),
                        }));
                    }
                }
            }
        }

        if let Some(finish_reason) = &choice.finish_reason {
            self.stop_reason = Some(match finish_reason.as_str() {
                "stop" => StopReason::EndTurn,
                "length" => StopReason::MaxTokens,
                "tool_calls" => StopReason::ToolUse,
                "content_filter" => StopReason::Unknown,
                _ => StopReason::Unknown,
            });
        }

        // Some OpenAI-compatible providers (e.g. MiniMax) emit the final usage chunk
        // with a non-empty `choices` array containing only an empty delta (often with
        // repeated assistant role) instead of the OpenAI-style empty-choices usage chunk.
        if let Some(usage) = chunk.usage {
            let empty_content = choice.delta.content.as_deref().is_none_or(str::is_empty);
            let no_tool_calls = choice.delta.tool_calls.as_ref().is_none_or(Vec::is_empty);
            if empty_content && no_tool_calls {
                events.push(Ok(InferenceEvent::MessageEnd {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    cache_read_input_tokens: usage
                        .prompt_tokens_details
                        .and_then(|details| details.cached_tokens),
                    cache_creation_input_tokens: None,
                    stop_reason: self.stop_reason.clone(),
                }));
            }
        }

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{
        ChatCompletionChunk, ChunkChoice, ChunkDelta, ChunkFunctionCall, ChunkToolCall, Usage,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_fixture_path(extension: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("inference_sdk_openai_{nonce}.{extension}"))
    }

    fn make_choice_chunk(
        tool_calls: Option<Vec<ChunkToolCall>>,
        finish_reason: Option<String>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chk_123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    tool_calls,
                },
                finish_reason,
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
        }
    }

    fn make_usage_chunk(prompt_tokens: u32, completion_tokens: u32) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chk_usage".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567891,
            model: "gpt-4o".to_string(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
            }),
            system_fingerprint: None,
        }
    }

    fn make_mixed_usage_chunk(
        content: Option<&str>,
        role: Option<types::chat::ChatRole>,
        tool_calls: Option<Vec<ChunkToolCall>>,
        finish_reason: Option<String>,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chk_usage_mixed".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567892,
            model: "gpt-4o".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role,
                    content: content.map(str::to_string),
                    tool_calls,
                },
                finish_reason,
                logprobs: None,
            }],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
            }),
            system_fingerprint: None,
        }
    }

    #[test]
    fn test_openai_adapter_emits_tool_start_and_deltas() {
        let mut adapter = OpenAiStreamAdapter::new();

        let chunk1 = make_choice_chunk(
            Some(vec![ChunkToolCall {
                index: 0,
                id: Some("call_123".to_string()),
                function: Some(ChunkFunctionCall {
                    name: Some("weather".to_string()),
                    arguments: Some("{\"loc".to_string()),
                }),
                call_type: Some("function".to_string()),
            }]),
            None,
        );
        let events = adapter.process_chunk(chunk1);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::ToolCallStart { ref id, ref name }) if id == "call_123" && name == "weather"
        ));
        assert!(matches!(
            events[1],
            Ok(InferenceEvent::ToolCallDelta { ref delta }) if delta == "{\"loc"
        ));

        let chunk2 = make_choice_chunk(
            Some(vec![ChunkToolCall {
                index: 0,
                id: None,
                function: Some(ChunkFunctionCall {
                    name: None,
                    arguments: Some("ation\": \"SF\"}".to_string()),
                }),
                call_type: None,
            }]),
            None,
        );
        let events = adapter.process_chunk(chunk2);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::ToolCallDelta { ref delta }) if delta == "ation\": \"SF\"}"
        ));
    }

    #[test]
    fn test_openai_adapter_emits_message_end_from_usage_chunk() {
        let mut adapter = OpenAiStreamAdapter::new();

        let finish_chunk = make_choice_chunk(None, Some("stop".to_string()));
        assert!(adapter.process_chunk(finish_chunk).is_empty());

        let usage_chunk = make_usage_chunk(12, 34);
        let events = adapter.process_chunk(usage_chunk);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::MessageEnd {
                input_tokens: 12,
                output_tokens: 34,
                stop_reason: Some(StopReason::EndTurn),
                ..
            })
        ));
    }

    #[test]
    fn test_openai_adapter_captures_cached_prompt_tokens() {
        let mut adapter = OpenAiStreamAdapter::new();
        let mut usage_chunk = make_usage_chunk(120, 34);
        usage_chunk
            .usage
            .as_mut()
            .expect("usage chunk")
            .prompt_tokens_details = Some(types::chat::PromptTokensDetails {
            cached_tokens: Some(96),
        });

        let events = adapter.process_chunk(usage_chunk);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::MessageEnd {
                input_tokens: 120,
                cache_read_input_tokens: Some(96),
                cache_creation_input_tokens: None,
                ..
            })
        ));
    }

    #[test]
    fn test_openai_adapter_emits_message_end_from_mixed_usage_chunk() {
        let mut adapter = OpenAiStreamAdapter::new();
        // Provider repeats assistant role on content chunks; adapter should only emit one MessageStart.
        let start_chunk = ChatCompletionChunk {
            id: "chk_start".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some(types::chat::ChatRole::Assistant),
                    content: Some("hi".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
        };
        let ev1 = adapter.process_chunk(start_chunk);
        assert_eq!(ev1.len(), 2);
        assert!(matches!(ev1[0], Ok(InferenceEvent::MessageStart { .. })));
        assert!(matches!(ev1[1], Ok(InferenceEvent::MessageDelta { .. })));

        let usage_chunk = make_mixed_usage_chunk(
            Some(""),
            Some(types::chat::ChatRole::Assistant),
            None,
            None,
            10,
            20,
        );
        let ev2 = adapter.process_chunk(usage_chunk);
        assert_eq!(ev2.len(), 1);
        assert!(matches!(
            ev2[0],
            Ok(InferenceEvent::MessageEnd {
                input_tokens: 10,
                output_tokens: 20,
                ..
            })
        ));
    }

    #[test]
    fn test_openai_adapter_avoids_duplicate_message_start_when_role_repeats() {
        let mut adapter = OpenAiStreamAdapter::new();
        let mk = |content: &str| ChatCompletionChunk {
            id: "chk_rep".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some(types::chat::ChatRole::Assistant),
                    content: Some(content.to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
        };
        let ev1 = adapter.process_chunk(mk("a"));
        let ev2 = adapter.process_chunk(mk("b"));
        assert!(matches!(ev1[0], Ok(InferenceEvent::MessageStart { .. })));
        assert_eq!(
            ev1.iter()
                .filter(|e| matches!(e, Ok(InferenceEvent::MessageStart { .. })))
                .count(),
            1
        );
        assert_eq!(
            ev2.iter()
                .filter(|e| matches!(e, Ok(InferenceEvent::MessageStart { .. })))
                .count(),
            0
        );
    }

    #[test]
    fn test_to_openai_request_sets_tool_choice_auto_when_tools_present() {
        let req = InferenceRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![inference_sdk_core::InferenceMessage {
                role: InferenceRole::User,
                content: vec![InferenceContent::Text {
                    text: "hello".to_string(),
                }],
                tool_call_id: None,
            }],
            system: None,
            tools: Some(vec![inference_sdk_core::Tool {
                name: "read_file".to_string(),
                description: "Read file".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            }]),
            temperature: None,
            max_tokens: None,
            thinking_budget: None,
            response_format: None,
        };

        let out = to_openai_request(req).expect("request normalization");
        assert!(matches!(
            out.tool_choice,
            Some(types::chat::ToolChoice::Mode(ref mode)) if mode == "auto"
        ));
    }

    #[test]
    fn test_to_openai_request_omits_tool_choice_without_tools() {
        let req = InferenceRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![inference_sdk_core::InferenceMessage {
                role: InferenceRole::User,
                content: vec![InferenceContent::Text {
                    text: "hello".to_string(),
                }],
                tool_call_id: None,
            }],
            system: None,
            tools: None,
            temperature: None,
            max_tokens: None,
            thinking_budget: None,
            response_format: None,
        };

        let out = to_openai_request(req).expect("request normalization");
        assert!(out.tool_choice.is_none());
    }

    #[test]
    fn test_to_openai_request_maps_json_schema_response_format() {
        let req = InferenceRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![inference_sdk_core::InferenceMessage {
                role: InferenceRole::User,
                content: vec![InferenceContent::Text {
                    text: "hello".to_string(),
                }],
                tool_call_id: None,
            }],
            system: None,
            tools: None,
            temperature: None,
            max_tokens: None,
            thinking_budget: None,
            response_format: Some(InferenceResponseFormat::JsonSchema {
                json_schema: inference_sdk_core::InferenceJsonSchemaConfig {
                    name: "review_result".to_string(),
                    description: Some("Structured review output".to_string()),
                    schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "approved": { "type": "boolean" }
                        },
                        "required": ["approved"]
                    }),
                    strict: Some(true),
                },
            }),
        };

        let out = to_openai_request(req).expect("request normalization");
        match out.response_format {
            Some(types::chat::ResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, "review_result");
                assert_eq!(json_schema.strict, Some(true));
                assert_eq!(
                    json_schema.schema["required"],
                    serde_json::json!(["approved"])
                );
            }
            other => panic!("expected json_schema response_format, got {other:?}"),
        }
    }

    #[test]
    fn test_to_openai_request_maps_images_and_files_in_user_content() {
        let image_path = temp_fixture_path("png");
        fs::write(&image_path, [1_u8, 2, 3, 4]).expect("fixture image");

        let req = InferenceRequest {
            model: "gpt-4o".to_string(),
            messages: vec![inference_sdk_core::InferenceMessage {
                role: InferenceRole::User,
                content: vec![
                    InferenceContent::Text {
                        text: "please inspect".to_string(),
                    },
                    InferenceContent::Image {
                        name: Some("diagram.png".to_string()),
                        content_type: Some("image/png".to_string()),
                        url: None,
                        local_path: Some(image_path.display().to_string()),
                        detail: Some("high".to_string()),
                    },
                    InferenceContent::File {
                        name: Some("spec.pdf".to_string()),
                        content_type: Some("application/pdf".to_string()),
                        url: Some("https://example.test/spec.pdf".to_string()),
                        local_path: None,
                    },
                ],
                tool_call_id: None,
            }],
            system: None,
            tools: None,
            temperature: None,
            max_tokens: None,
            thinking_budget: None,
            response_format: None,
        };

        let out = to_openai_request(req).expect("request normalization");
        let user_message = out
            .messages
            .into_iter()
            .find(|message| message.role == types::chat::ChatRole::User)
            .expect("user message");

        match user_message.content.expect("user content") {
            types::chat::ChatContent::Parts(parts) => {
                assert!(matches!(
                    &parts[0],
                    types::chat::ContentPart::Text { text } if text == "please inspect"
                ));
                assert!(matches!(
                    &parts[1],
                    types::chat::ContentPart::ImageUrl { image_url }
                    if image_url.url.starts_with("data:image/png;base64,")
                        && image_url.detail.as_deref() == Some("high")
                ));
                assert!(matches!(
                    &parts[2],
                    types::chat::ContentPart::Text { text }
                    if text.contains("spec.pdf") && text.contains("application/pdf")
                ));
            }
            other => panic!("expected content parts, got {other:?}"),
        }

        let _ = fs::remove_file(image_path);
    }
}
