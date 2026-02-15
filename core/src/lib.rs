use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use futures::future::BoxFuture;

pub mod error;
pub mod http;

pub use error::SdkError;
pub use http::RequestOptions;

/// A provider that can fulfill inference requests.
pub trait InferenceProvider: Send + Sync {
    fn complete<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceResult, SdkError>>;
    
    // We'll update the trait signature in Phase 3 or later if needed, 
    // for now we just keep the simple signature but the types change.
    fn stream<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceStream, SdkError>>;
}

pub type InferenceStream = futures::stream::BoxStream<'static, Result<InferenceEvent, SdkError>>;

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
#[derive(bon::Builder)]
impl InferenceRequest {
    #[builder]
    pub fn new(
        model: String,
        messages: Vec<InferenceMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
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
    /// A tool call detected in the stream.
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    /// The end of a message response, including usage statistics.
    MessageEnd {
        input_tokens: u32,
        output_tokens: u32,
    },
    /// An error occurred during the stream.
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub content: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub type InferenceStream = Pin<Box<dyn Stream<Item = Result<InferenceEvent, SdkError>> + Send + 'static>>;

/// The core trait that all providers must implement.
pub trait InferenceProvider: Send + Sync {
    /// Generate a unified completion (non-streaming).
    fn complete<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceResult, SdkError>>;

    /// Generate a streaming response.
    fn stream<'a>(&'a self, request: InferenceRequest) -> BoxFuture<'a, Result<InferenceStream, SdkError>>;
}
