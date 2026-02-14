use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use futures::future::BoxFuture;
pub use futures;

pub mod config;
pub mod error;
pub mod http;

pub use config::*;
pub use error::*;

/// A standardized request for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
pub struct InferenceRequest {
    /// The model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
    #[builder(into)]
    pub model: String,

    /// The conversation history.
    pub messages: Vec<InferenceMessage>,

    /// Optional system prompt (if not provided in messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub system: Option<String>,

    /// Available tools for the model to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,

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

/// A normalized message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMessage {
    pub role: InferenceRole,
    pub content: Vec<InferenceContent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceRole {
    User,
    Assistant,
    // System is handled via the separate `system` field or normalization logic
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
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Events emitted during a streaming inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    /// The start of a message response.
    MessageStart {
        role: String,
        model: String,
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
