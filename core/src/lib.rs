use serde::{Deserialize, Serialize};

use std::pin::Pin;
use futures::Stream;
use futures::future::BoxFuture;

pub mod error;
pub mod http;

pub use error::SdkError;
pub use http::RequestOptions;


/// A provider that can fulfill inference requests.
pub trait InferenceProvider: Send + Sync {
    fn complete<'a>(&'a self, request: InferenceRequest, options: Option<RequestOptions>) -> BoxFuture<'a, Result<InferenceResult, SdkError>> {
        Box::pin(async move {
            let stream = self.stream(request, options).await?;
            InferenceResult::from_stream(stream).await
        })
    }
    
    fn stream<'a>(&'a self, request: InferenceRequest, options: Option<RequestOptions>) -> BoxFuture<'a, Result<InferenceStream, SdkError>>;
}

pub type InferenceStream = Pin<Box<dyn Stream<Item = Result<InferenceEvent, SdkError>> + Send + 'static>>;

/// A standardized request for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// The model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
    pub model: String,

    /// The conversation history.
    pub messages: Vec<InferenceMessage>,

    /// Optional system prompt (if not provided in messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Available tools for the model to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Sampling temperature (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Optional "thinking" budget for reasoning models (e.g. Claude 3.7, o1).
    /// Providers that support it will use this; others will ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
}

// Bon builder
#[bon::bon]
impl InferenceRequest {
    #[builder]
    pub fn new(
        #[builder(into)]
        model: String,
        messages: Vec<InferenceMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        #[builder(into)]
        system: Option<String>,
        tools: Option<Vec<Tool>>,
        thinking_budget: Option<u32>,
    ) -> Self {
        Self {
            model,
            messages,
            temperature,
            max_tokens,
            system,
            tools,
            thinking_budget,
        }
    }
}

/// A normalized message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMessage {
    pub role: InferenceRole,
    pub content: Vec<InferenceContent>,
    // Optional field to link a tool result to a tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceRole {
    User,
    Assistant,
    // System is handled via the separate `system` field or normalization logic
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceContent {
    Text { text: String },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    Thinking {
        content: String,
    },
}

/// Normalized definition of a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub content: Vec<InferenceContent>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub usage: Usage,
}

impl InferenceResult {
    /// Helper to extract all text content combined.
    pub fn text(&self) -> String {
        self.content.iter().filter_map(|c| match c {
            InferenceContent::Text { text } => Some(text.as_str()),
            _ => None,
        }).collect::<Vec<_>>().join("")
    }

    /// Collects a stream into a single result.
    pub async fn from_stream(mut stream: InferenceStream) -> Result<Self, SdkError> {
        let mut content_parts = Vec::new();
        let mut model = String::new();
        let mut stop_reason = None;
        let mut usage = Usage { input_tokens: 0, output_tokens: 0 };
        
// Tool call accumulation state
        let mut current_tool_id: Option<String> = None;
        let mut current_tool_name: Option<String> = None;
        let mut current_tool_json: String = String::new();

        while let Some(event_res) = futures::StreamExt::next(&mut stream).await {
            match event_res {
                Ok(event) => match event {
                    InferenceEvent::MessageStart { model: m, .. } => {
                        model = m;
                    }
                    InferenceEvent::MessageDelta { content } => {
                         if let Some(InferenceContent::Text { text }) = content_parts.last_mut() {
                             text.push_str(&content);
                         } else {
                             content_parts.push(InferenceContent::Text { text: content });
                         }
                    }
                    InferenceEvent::ThinkingDelta { content } => {
                         if let Some(InferenceContent::Thinking { content: text }) = content_parts.last_mut() {
                             text.push_str(&content);
                         } else {
                             content_parts.push(InferenceContent::Thinking { content });
                         }
                    }
                    InferenceEvent::ToolCallStart { id, name } => {
                        // If we were building a previous tool, we should probably finalize it (though APIs usually don't interleave this way)
                        // For now, simplicity: start new buffer
                        current_tool_id = Some(id);
                        current_tool_name = Some(name);
                        current_tool_json.clear();
                    }
                    InferenceEvent::ToolCallDelta { delta } => {
                        current_tool_json.push_str(&delta);
                    }
                    InferenceEvent::MessageEnd { input_tokens, output_tokens, stop_reason: sr } => {
                        // Finalize any pending tool call
                        if let (Some(id), Some(name)) = (current_tool_id.take(), current_tool_name.take()) {
                            let input = serde_json::from_str(&current_tool_json).unwrap_or(serde_json::Value::Null); 
                            content_parts.push(InferenceContent::ToolUse { id, name, input });
                            current_tool_json.clear();
                        }
                        
                        usage = Usage { input_tokens, output_tokens };
                        stop_reason = sr;
                    }
                },
                Err(e) => return Err(e),
            }
        }
        
        Ok(InferenceResult {
            content: content_parts,
            model,
            stop_reason,
            usage,
        })
    }
}

/// Events emitted during a streaming inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum InferenceEvent {
    /// The start of a message response.
    MessageStart {
        role: String,
        model: String, // from API
        /// The provider attempting to fulfill this request.
        provider_id: String, 
    },
    /// A text delta for the message content.
    MessageDelta {
        content: String,
    },
    /// A thought process delta (for reasoning models).
    ThinkingDelta {
        content: String,
    },
    /// A tool call started.
    ToolCallStart {
        id: String,
        name: String,
    },
    /// A delta for a tool call argument (JSON fragment).
    ToolCallDelta {
        delta: String,
    },
    /// The end of a message response, including usage statistics.
    MessageEnd {
        input_tokens: u32,
        output_tokens: u32,
        stop_reason: Option<StopReason>,
    },
}

