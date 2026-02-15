use inference_sdk_core::{SdkError, InferenceRequest, InferenceRole, InferenceContent, InferenceEvent};
use crate::types;
use std::collections::HashMap;

pub fn to_openai_request(req: InferenceRequest) -> Result<types::chat::ChatCompletionRequest, SdkError> {
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
                let text_parts: Vec<&str> = msg.content.iter().filter_map(|c| match c {
                    InferenceContent::Text { text } => Some(text.as_str()),
                    _ => None,
                }).collect();
                
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
                let text_parts: Vec<&str> = msg.content.iter().filter_map(|c| match c {
                    InferenceContent::Text { text } => Some(text.as_str()),
                    _ => None,
                }).collect();
                
                let tool_calls: Vec<types::chat::ToolCall> = msg.content.iter().filter_map(|c| match c {
                    InferenceContent::ToolUse { id, name, input } => Some(types::chat::ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: types::chat::FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        }
                    }),
                    _ => None,
                }).collect();

                messages.push(types::chat::ChatMessage {
                    role: types::chat::ChatRole::Assistant,
                    content: if text_parts.is_empty() { None } else { Some(types::chat::ChatContent::Text(text_parts.join("\n"))) },
                    name: None,
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                    tool_call_id: None,
                });
            }
            InferenceRole::Tool => {
                for content in msg.content {
                    match content {
                        InferenceContent::ToolResult { tool_use_id, content, .. } => {
                             messages.push(types::chat::ChatMessage {
                                role: types::chat::ChatRole::Tool,
                                content: Some(types::chat::ChatContent::Text(content)),
                                name: None,
                                tool_calls: None,
                                tool_call_id: Some(tool_use_id),
                            });
                        }
                        _ => {} 
                    }
                }
            }
        }
    }

    let tools: Option<Vec<types::chat::Tool>> = req.tools.map(|ts| {
        ts.into_iter().map(|t| types::chat::Tool {
            tool_type: "function".to_string(),
            function: types::chat::FunctionDefinition {
                name: t.name,
                description: Some(t.description),
                parameters: t.input_schema,
                strict: None,
            }
        }).collect()
    });

    Ok(types::chat::ChatCompletionRequest::builder()
        .model(req.model)
        .messages(messages)
        .maybe_temperature(req.temperature)
        .maybe_max_tokens(req.max_tokens)
        .maybe_tools(tools)
        .build())
}

pub struct OpenAiStreamAdapter {
    tool_buffers: HashMap<u32, ToolCallBuffer>,
}

struct ToolCallBuffer {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OpenAiStreamAdapter {
    pub fn new() -> Self {
        Self { tool_buffers: HashMap::new() }
    }

    pub fn process_chunk(&mut self, chunk: types::chat::ChatCompletionChunk) -> Vec<Result<InferenceEvent, SdkError>> {
        let mut events = Vec::new();
        
        if chunk.choices.is_empty() {
             if let Some(usage) = chunk.usage {
                events.push(Ok(InferenceEvent::MessageEnd {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                }));
            }
            return events;
        }

        let choice = &chunk.choices[0];
        
        if let Some(role) = &choice.delta.role {
            match role {
                types::chat::ChatRole::Assistant => events.push(Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: chunk.model.clone(),
                    provider_id: "openai".to_string(),
                })),
                _ => {},
            }
        }

        if let Some(content) = &choice.delta.content {
            if !content.is_empty() {
                events.push(Ok(InferenceEvent::MessageDelta { content: content.clone() }));
            }
        }

        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                let index = tc.index;
                let buffer = self.tool_buffers.entry(index).or_insert(ToolCallBuffer {
                    id: None,
                    name: None,
                    arguments: String::new(),
                });

                if let Some(id) = &tc.id {
                    buffer.id = Some(id.clone());
                }
                if let Some(func) = &tc.function {
                    if let Some(name) = &func.name {
                        buffer.name = Some(name.clone());
                    }
                    if let Some(args) = &func.arguments {
                        buffer.arguments.push_str(args);
                    }
                }
            }
        }

        if let Some(finish_reason) = &choice.finish_reason {
            if finish_reason == "tool_calls" {
                let mut tools: Vec<_> = self.tool_buffers.drain().collect();
                tools.sort_by_key(|(k, _)| *k);
                
                for (_, buffer) in tools {
                    if let (Some(id), Some(name)) = (buffer.id, buffer.name) {
                        let args = serde_json::from_str(&buffer.arguments).unwrap_or(serde_json::json!({}));
                        events.push(Ok(InferenceEvent::ToolCall {
                            id,
                            name,
                            args,
                        }));
                    }
                }
            }
        }

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{ChatCompletionChunk, ChunkChoice, ChunkDelta, ChunkToolCall, ChunkFunctionCall, Usage as OpenAiUsage};

    fn make_chunk(tool_calls: Option<Vec<ChunkToolCall>>, finish_reason: Option<String>) -> ChatCompletionChunk {
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

    #[test]
    fn test_openai_adapter_accumulates_tool_calls() {
        let mut adapter = OpenAiStreamAdapter::new();

        let chunk1 = make_chunk(
            Some(vec![ChunkToolCall {
                index: 0,
                id: Some("call_123".to_string()),
                function: Some(ChunkFunctionCall {
                    name: Some("weather".to_string()),
                    arguments: Some("{\"loc".to_string()),
                }),
                call_type: Some("function".to_string()),
            }]),
            None
        );
        let events = adapter.process_chunk(chunk1);
        assert!(events.is_empty());

        let chunk2 = make_chunk(
            Some(vec![ChunkToolCall {
                index: 0,
                id: None,
                function: Some(ChunkFunctionCall {
                    name: None,
                    arguments: Some("ation\": \"SF\"}".to_string()),
                }),
                call_type: None,
            }]),
            None
        );
        let events = adapter.process_chunk(chunk2);
        assert!(events.is_empty());

        let chunk3 = make_chunk(None, Some("tool_calls".to_string()));
        let events = adapter.process_chunk(chunk3);
        
        assert_eq!(events.len(), 1);
        match &events[0] {
            Ok(InferenceEvent::ToolCall { id, name, args }) => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "weather");
                assert_eq!(args["location"], "SF");
            },
            _ => panic!("Expected ToolCall event, got {:?}", events[0]),
        }
    }
}
