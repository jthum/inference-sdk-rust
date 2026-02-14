# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-14

### Added
- **Capability Normalization**:
  - Introduced `InferenceProvider` trait in `inference-sdk-core` for unified provider access.
  - Standardized `InferenceRequest` and `InferenceEvent` types across all SDKs.
  - Added support for standardized streaming events (`MessageStart`, `MessageDelta`, `ToolCall`, etc.).
  - Both `anthropic-sdk` and `openai-sdk` now implement the common `InferenceProvider` interface.
  - `ProviderKind` now implements `FromStr` for better configuration parsing.

---

## [0.2.0] - 2026-02-14

### Added
- **Adaptive Thinking Support**:
  - `anthropic-sdk`: Added `ThinkingConfig` and `ThinkingDelta` events for Claude 3.7 Sonnet / Opus 4.6.
  - Support for `beta` headers in `MessageRequest`.
- **Embeddings Support**:
  - `openai-sdk`: Implementation of the Embeddings API.
  - Added `EmbeddingRequest`, `EmbeddingResponse`, and `EmbeddingData` models.
- **Improved Request Handling**:
  - Standardized `RequestOptions` to support per-request configuration (e.g., custom headers, proxy settings).

## [0.1.0] - 2026-02-10

### Added
- **Initial Release**: Established the modular Rust workspace for LLM inference.
- **`inference-sdk-core`**:
  - Unified `RequestOptions` for per-request overrides (headers, timeouts).
  - Shared `SdkError` enum for consistent cross-provider error handling.
  - Built-in retry logic with exponential backoff on transient failures.
- **`anthropic-sdk`**:
  - Implementation of the Anthropic Messages API.
  - Support for streaming responses using Server-Sent Events (SSE).
  - Type-safe request builders integrated with the `bon` crate.
  - `AnthropicRequestExt` trait for easy beta header injection.
- **`openai-sdk`**:
  - Implementation of the OpenAI Chat Completions API.
  - Support for streaming responses.
  - Type-safe models for multiple message roles (system, user, assistant, tool).
  - Shared core logic for Bearer token authentication.
