use crate::types;
use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceRequest, InferenceRole, SdkError, StopReason,
};

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
                let mut text_parts: Vec<String> = Vec::new();
                for content in msg.content {
                    if let InferenceContent::Text { text } = content {
                        text_parts.push(text);
                    }
                }

                if !text_parts.is_empty() {
                    messages.push(types::chat::ChatMessage {
                        role: types::chat::ChatRole::User,
                        content: Some(types::chat::ChatContent::Text(text_parts.join("\n"))),
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

    Ok(types::chat::ChatCompletionRequest::builder()
        .model(req.model)
        .messages(messages)
        .maybe_temperature(req.temperature)
        .maybe_max_tokens(req.max_tokens)
        .maybe_tools(tools)
        .build())
}

#[derive(Default)]
pub struct OpenAiStreamAdapter {
    stop_reason: Option<StopReason>,
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
                    stop_reason: self.stop_reason.clone(),
                }));
            }
            return events;
        }

        let choice = &chunk.choices[0];

        if let Some(types::chat::ChatRole::Assistant) = &choice.delta.role {
            events.push(Ok(InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: chunk.model,
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

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{
        ChatCompletionChunk, ChunkChoice, ChunkDelta, ChunkFunctionCall, ChunkToolCall, Usage,
    };

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
                stop_reason: Some(StopReason::EndTurn)
            })
        ));
    }
}
