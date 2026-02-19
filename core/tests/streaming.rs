use futures_util::stream;
use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceResult, SdkError, StopReason,
    StreamInvariantViolation,
};

#[tokio::test]
async fn test_from_stream_accumulates_tool_calls() {
    let tool_id = "call_123";
    let tool_name = "weather";

    let events = vec![
        Ok(InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        }),
        Ok(InferenceEvent::MessageDelta {
            content: "Thinking...".to_string(),
        }),
        Ok(InferenceEvent::ThinkingDelta {
            content: "I should check the weather.".to_string(),
        }),
        Ok(InferenceEvent::ToolCallStart {
            id: tool_id.to_string(),
            name: tool_name.to_string(),
        }),
        Ok(InferenceEvent::ToolCallDelta {
            delta: "{\"loc".to_string(),
        }),
        Ok(InferenceEvent::ToolCallDelta {
            delta: "ation\": \"SF\"}".to_string(),
        }),
        Ok(InferenceEvent::MessageEnd {
            input_tokens: 10,
            output_tokens: 20,
            stop_reason: Some(StopReason::ToolUse),
        }),
    ];

    let stream = Box::pin(stream::iter(events));
    let result = InferenceResult::from_stream(stream)
        .await
        .expect("Stream failed");

    assert_eq!(result.model, "test-model");
    assert_eq!(result.stop_reason, Some(StopReason::ToolUse));
    // Text + Thinking + ToolUse
    assert_eq!(result.content.len(), 3);

    match &result.content[0] {
        InferenceContent::Text { text } => assert_eq!(text, "Thinking..."),
        _ => panic!("Expected Text"),
    }
    match &result.content[1] {
        InferenceContent::Thinking { content } => {
            assert_eq!(content, "I should check the weather.")
        }
        _ => panic!("Expected Thinking"),
    }
    match &result.content[2] {
        InferenceContent::ToolUse { id, name, input } => {
            assert_eq!(id, tool_id);
            assert_eq!(name, tool_name);
            assert_eq!(input["location"], "SF");
        }
        _ => panic!("Expected ToolUse"),
    }
}

#[tokio::test]
async fn test_from_stream_returns_error_for_invalid_tool_json() {
    let events = vec![
        Ok(InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        }),
        Ok(InferenceEvent::ToolCallStart {
            id: "call_123".to_string(),
            name: "weather".to_string(),
        }),
        Ok(InferenceEvent::ToolCallDelta {
            delta: "{\"city\":".to_string(),
        }),
        Ok(InferenceEvent::MessageEnd {
            input_tokens: 1,
            output_tokens: 2,
            stop_reason: Some(StopReason::ToolUse),
        }),
    ];

    let stream = Box::pin(stream::iter(events));
    let result = InferenceResult::from_stream(stream).await;
    assert!(result.is_err(), "invalid tool JSON should return an error");
}

#[tokio::test]
async fn test_from_stream_returns_error_when_stream_missing_message_end() {
    let events = vec![
        Ok(InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        }),
        Ok(InferenceEvent::MessageDelta {
            content: "partial".to_string(),
        }),
    ];

    let stream = Box::pin(stream::iter(events));
    let result = InferenceResult::from_stream(stream).await;

    assert!(matches!(
        result,
        Err(SdkError::StreamInvariantViolation(
            StreamInvariantViolation::MissingMessageEnd
        ))
    ));
}

#[tokio::test]
async fn test_from_stream_returns_error_when_delta_precedes_message_start() {
    let events = vec![
        Ok(InferenceEvent::MessageDelta {
            content: "oops".to_string(),
        }),
        Ok(InferenceEvent::MessageEnd {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: Some(StopReason::EndTurn),
        }),
    ];

    let stream = Box::pin(stream::iter(events));
    let result = InferenceResult::from_stream(stream).await;

    assert!(matches!(
        result,
        Err(SdkError::StreamInvariantViolation(
            StreamInvariantViolation::MessageNotStarted
        ))
    ));
}
