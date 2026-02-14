pub mod client;
pub mod resources;
pub mod types;

pub use client::{Client, ClientConfig};
pub use types::embedding::EmbeddingRequest;
pub use inference_sdk_core::{RequestOptions, SdkError, InferenceProvider, InferenceRequest, InferenceEvent, InferenceResult, InferenceStream};
use inference_sdk_core::futures::{StreamExt, future::BoxFuture};


impl InferenceProvider for Client {
    fn complete<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceResult, SdkError>> {
        Box::pin(async move {
        let openai_req = to_openai_request(request)?;
        let response = self.chat().create(openai_req).await?;
        
        let content = response.choices.first()
            .and_then(|c| c.message.content.as_ref())
            .map(|c| match c {
                types::chat::ChatContent::Text(t) => t.as_str(),
                _ => "",
            })
            .unwrap_or("")
            .to_string();
            
        Ok(InferenceResult {
            content,
            usage: inference_sdk_core::Usage {
                input_tokens: response.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                output_tokens: response.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
            }
            })
        })
    }

    fn stream<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let openai_req = to_openai_request(request)?;
            let stream = self.chat().create_stream(openai_req).await?;
            
            // This mapping logic needs to handle chunk accumulation or just emit raw events.
            // For simplicity in this iteration, we map what we can see in single chunks.
            // Full tool call re-assembly might be needed if chunks are fragmented, 
            // but Bedrock was handling that. We should ideally handle it here or verify OpenAI SDK handles it.
            // The OpenAI SDK `ChatCompletionChunk` returns generic deltas.
            
            let mapped_stream = stream.map(|chunk_res: Result<types::chat::ChatCompletionChunk, SdkError>| {
                match chunk_res {
                    Ok(chunk) => normalize_openai_chunk(chunk),
                    Err(e) => Ok(Some(InferenceEvent::Error { message: e.to_string() })),
                }
            });

            // Filter None and flatten
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

fn to_openai_request(req: InferenceRequest) -> Result<types::chat::ChatCompletionRequest, SdkError> {
    let mut messages = Vec::new();

    // Handle System Prompt as first message
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
            inference_sdk_core::InferenceRole::User => {
                let text: String = msg.content.iter().filter_map(|c| match c {
                    inference_sdk_core::InferenceContent::Text { text } => Some(text.as_str()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                
                messages.push(types::chat::ChatMessage {
                    role: types::chat::ChatRole::User,
                    content: Some(types::chat::ChatContent::Text(text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
                
                // Tool results
                for content in msg.content {
                    if let inference_sdk_core::InferenceContent::ToolResult { tool_use_id, content, .. } = content {
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
            inference_sdk_core::InferenceRole::Assistant => {
                let text_parts: Vec<&str> = msg.content.iter().filter_map(|c| match c {
                    inference_sdk_core::InferenceContent::Text { text } => Some(text.as_str()),
                    _ => None,
                }).collect();
                
                let tool_calls: Vec<types::chat::ToolCall> = msg.content.iter().filter_map(|c| match c {
                    inference_sdk_core::InferenceContent::ToolUse { id, name, input } => Some(types::chat::ToolCall {
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
        }
    }

    let tools: Option<Vec<types::chat::Tool>> = req.tools.map(|ts| {
        ts.into_iter().map(|t| types::chat::Tool {
            tool_type: "function".to_string(),
            function: types::chat::FunctionDefinition {
                name: t.name,
                description: t.description,
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

fn normalize_openai_chunk(chunk: types::chat::ChatCompletionChunk) -> Result<Option<InferenceEvent>, SdkError> {
    if chunk.choices.is_empty() {
        if let Some(usage) = chunk.usage {
            return Ok(Some(InferenceEvent::MessageEnd {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            }));
        }
        return Ok(None);
    }
    
    let choice = &chunk.choices[0];
    
    // Message Start (implicit in first chunk usually, or explicit logic needed? logic: if role is present)
    if let Some(role) = choice.delta.role.as_ref() {
         match role {
            types::chat::ChatRole::Assistant => return Ok(Some(InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: chunk.model.clone(),
                provider_id: "openai".to_string(),
            })),
            _ => {},
         }
    }

    // Text Content
    if let Some(content) = choice.delta.content.as_ref() {
        if !content.is_empty() {
            return Ok(Some(InferenceEvent::MessageDelta { content: content.clone() }));
        }
    }
    
    // Tool Calls
    if let Some(tool_calls) = choice.delta.tool_calls.as_ref() {
        for tc in tool_calls {
            // Only emit if it's the start of a tool call (has ID)
            if let (Some(id), Some(func)) = (&tc.id, &tc.function) {
                if let Some(name) = &func.name {
                    let args = serde_json::from_str(func.arguments.as_deref().unwrap_or("{}")).unwrap_or(serde_json::json!({}));
                    return Ok(Some(InferenceEvent::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        args,
                    }));
                }
            }
        }
    }

    Ok(None)
}
