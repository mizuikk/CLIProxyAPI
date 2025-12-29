// Package claude provides response translation functionality for OpenAI to Anthropic API.
// This package handles the conversion of OpenAI Chat Completions API responses into Anthropic API-compatible
// JSON format, transforming streaming events and non-streaming responses into the format
// expected by Anthropic API clients. It supports both streaming and non-streaming modes,
// handling text content, tool calls, and usage metadata appropriately.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var (
	dataTag = []byte("data:")
)

// ConvertOpenAIResponseToAnthropicParams holds parameters for response conversion
type ConvertOpenAIResponseToAnthropicParams struct {
	MessageID string
	Model     string
	CreatedAt int64
	// Content accumulator for streaming
	ContentAccumulator strings.Builder
	// Tool calls accumulator for streaming
	ToolCallsAccumulator map[int]*ToolCallAccumulator
	// Track if text content block has been started
	TextContentBlockStarted bool
	// Track if thinking content block has been started
	ThinkingContentBlockStarted bool
	// Track finish reason for later use
	FinishReason string
	// Track if content blocks have been stopped
	ContentBlocksStopped bool
	// Track if message_delta has been sent
	MessageDeltaSent bool
	// Track if message_start has been sent
	MessageStarted bool
	// Track if message_stop has been sent
	MessageStopSent bool
	// Tool call content block index mapping
	ToolCallBlockIndexes map[int]int
	// Index assigned to text content block
	TextContentBlockIndex int
	// Index assigned to thinking content block
	ThinkingContentBlockIndex int
	// Next available content block index
	NextContentBlockIndex int

	// ThinkTagOpen tracks whether we are currently inside a "<think>...</think>" section while
	// translating providers that embed reasoning directly into the "content" stream.
	ThinkTagOpen bool
	// ThinkTagCarry stores a partial "<think>" or "</think>" tag prefix when the tag is split across
	// streaming chunks. This enables correct reconstruction without leaking tag fragments to clients.
	ThinkTagCarry string
}

// ToolCallAccumulator holds the state for accumulating tool call data
type ToolCallAccumulator struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

const (
	thinkOpenTag  = "<think>"
	thinkCloseTag = "</think>"
)

type contentSegment struct {
	Thinking bool
	Text     string
}

// shouldParseThinkTagsForModel reports whether this translator should reinterpret "<think>...</think>" tags
// as Anthropic thinking blocks for the given model name.
//
// Root cause: Some providers may legitimately output literal "<think>" text (e.g., discussing XML/HTML or
// showing code samples). Blindly parsing tags for every model would silently rewrite user-visible content.
// Solution: Gate this compatibility behaviour to MiniMax models only (MiniMax-*), where we have observed
// the upstream embedding reasoning in delta.content rather than "reasoning_content".
func shouldParseThinkTagsForModel(model string) bool {
	model = strings.TrimSpace(model)
	if model == "" {
		return false
	}
	return strings.HasPrefix(strings.ToLower(model), "minimax-")
}

// parseThinkTaggedText splits a string into ordered segments of normal text and "<think>...</think>" reasoning.
//
// Root cause: Some OpenAI-compatible providers (observed with MiniMax-M2.1) do not emit OpenAI-style
// "reasoning_content" fields. Instead, they stream reasoning inside delta.content using "<think>...</think>" tags.
// The existing OpenAI->Anthropic translator only maps "reasoning_content" to Anthropic "thinking" content blocks,
// so downstream Anthropic clients see no thinking (and may hide the tagged text).
//
// Solution: Detect and parse "<think>...</think>" sections and convert them into Anthropic "thinking" blocks,
// while emitting the remaining text as regular Anthropic "text" blocks, preserving the original stream order.
func parseThinkTaggedText(text string, initialThinkOpen bool) (segments []contentSegment, thinkOpen bool) {
	thinkOpen = initialThinkOpen
	remaining := text

	appendSegment := func(thinking bool, value string) {
		if strings.TrimSpace(value) == "" {
			return
		}
		segments = append(segments, contentSegment{Thinking: thinking, Text: value})
	}

	for remaining != "" {
		if !thinkOpen {
			openIdx := strings.Index(remaining, thinkOpenTag)
			closeIdx := strings.Index(remaining, thinkCloseTag)

			// Handle stray close tags by dropping them while preserving surrounding text.
			if closeIdx != -1 && (openIdx == -1 || closeIdx < openIdx) {
				appendSegment(false, remaining[:closeIdx])
				remaining = remaining[closeIdx+len(thinkCloseTag):]
				continue
			}

			if openIdx == -1 {
				appendSegment(false, remaining)
				return segments, thinkOpen
			}

			appendSegment(false, remaining[:openIdx])
			remaining = remaining[openIdx+len(thinkOpenTag):]
			thinkOpen = true
			continue
		}

		// thinkOpen == true
		closeIdx := strings.Index(remaining, thinkCloseTag)
		if closeIdx == -1 {
			appendSegment(true, remaining)
			return segments, thinkOpen
		}
		appendSegment(true, remaining[:closeIdx])
		remaining = remaining[closeIdx+len(thinkCloseTag):]
		thinkOpen = false
	}

	return segments, thinkOpen
}

// splitTrailingThinkTagPrefix removes and returns a partial "<think>" or "</think>" prefix at the end of s.
// This supports cases where the provider splits a tag across streaming chunks.
func splitTrailingThinkTagPrefix(s string) (processed string, carry string) {
	processed = s
	carry = ""
	if s == "" {
		return processed, carry
	}

	maxPrefix := len(thinkCloseTag) - 1
	if openMax := len(thinkOpenTag) - 1; openMax > maxPrefix {
		maxPrefix = openMax
	}
	if maxPrefix <= 0 {
		return processed, carry
	}
	if len(s) < maxPrefix {
		maxPrefix = len(s)
	}

	for i := maxPrefix; i >= 1; i-- {
		openPrefix := thinkOpenTag[:i]
		closePrefix := thinkCloseTag[:i]
		if strings.HasSuffix(s, openPrefix) || strings.HasSuffix(s, closePrefix) {
			return s[:len(s)-i], s[len(s)-i:]
		}
	}
	return processed, carry
}

// parseThinkTaggedStreamDelta parses a streaming delta.content string into ordered thinking/text segments,
// carrying incomplete "<think>" or "</think>" prefixes across chunks.
func parseThinkTaggedStreamDelta(param *ConvertOpenAIResponseToAnthropicParams, delta string) []contentSegment {
	if param == nil {
		segments, _ := parseThinkTaggedText(delta, false)
		return segments
	}

	combined := param.ThinkTagCarry + delta
	param.ThinkTagCarry = ""

	processed, carry := splitTrailingThinkTagPrefix(combined)
	param.ThinkTagCarry = carry

	segments, thinkOpen := parseThinkTaggedText(processed, param.ThinkTagOpen)
	param.ThinkTagOpen = thinkOpen
	return segments
}

func emitThinkingDelta(param *ConvertOpenAIResponseToAnthropicParams, results *[]string, text string) {
	if param == nil || results == nil {
		return
	}
	if strings.TrimSpace(text) == "" {
		return
	}
	stopTextContentBlock(param, results)
	if !param.ThinkingContentBlockStarted {
		if param.ThinkingContentBlockIndex == -1 {
			param.ThinkingContentBlockIndex = param.NextContentBlockIndex
			param.NextContentBlockIndex++
		}
		contentBlockStartJSON := `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`
		contentBlockStartJSON, _ = sjson.Set(contentBlockStartJSON, "index", param.ThinkingContentBlockIndex)
		*results = append(*results, "event: content_block_start\ndata: "+contentBlockStartJSON+"\n\n")
		param.ThinkingContentBlockStarted = true
	}

	thinkingDeltaJSON := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}`
	thinkingDeltaJSON, _ = sjson.Set(thinkingDeltaJSON, "index", param.ThinkingContentBlockIndex)
	thinkingDeltaJSON, _ = sjson.Set(thinkingDeltaJSON, "delta.thinking", text)
	*results = append(*results, "event: content_block_delta\ndata: "+thinkingDeltaJSON+"\n\n")
}

func emitTextDelta(param *ConvertOpenAIResponseToAnthropicParams, results *[]string, text string) {
	if param == nil || results == nil {
		return
	}
	if strings.TrimSpace(text) == "" {
		return
	}
	if !param.TextContentBlockStarted {
		stopThinkingContentBlock(param, results)
		if param.TextContentBlockIndex == -1 {
			param.TextContentBlockIndex = param.NextContentBlockIndex
			param.NextContentBlockIndex++
		}
		contentBlockStartJSON := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
		contentBlockStartJSON, _ = sjson.Set(contentBlockStartJSON, "index", param.TextContentBlockIndex)
		*results = append(*results, "event: content_block_start\ndata: "+contentBlockStartJSON+"\n\n")
		param.TextContentBlockStarted = true
	}

	contentDeltaJSON := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}`
	contentDeltaJSON, _ = sjson.Set(contentDeltaJSON, "index", param.TextContentBlockIndex)
	contentDeltaJSON, _ = sjson.Set(contentDeltaJSON, "delta.text", text)
	*results = append(*results, "event: content_block_delta\ndata: "+contentDeltaJSON+"\n\n")
	param.ContentAccumulator.WriteString(text)
}

// ConvertOpenAIResponseToClaude converts OpenAI streaming response format to Anthropic API format.
// This function processes OpenAI streaming chunks and transforms them into Anthropic-compatible JSON responses.
// It handles text content, tool calls, and usage metadata, outputting responses that match the Anthropic API format.
//
// Parameters:
//   - ctx: The context for the request.
//   - modelName: The name of the model.
//   - rawJSON: The raw JSON response from the OpenAI API.
//   - param: A pointer to a parameter object for the conversion.
//
// Returns:
//   - []string: A slice of strings, each containing an Anthropic-compatible JSON response.
func ConvertOpenAIResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &ConvertOpenAIResponseToAnthropicParams{
			MessageID:                   "",
			Model:                       "",
			CreatedAt:                   0,
			ContentAccumulator:          strings.Builder{},
			ToolCallsAccumulator:        nil,
			TextContentBlockStarted:     false,
			ThinkingContentBlockStarted: false,
			FinishReason:                "",
			ContentBlocksStopped:        false,
			MessageDeltaSent:            false,
			ToolCallBlockIndexes:        make(map[int]int),
			TextContentBlockIndex:       -1,
			ThinkingContentBlockIndex:   -1,
			NextContentBlockIndex:       0,
		}
	}

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	// Check if this is the [DONE] marker
	rawStr := strings.TrimSpace(string(rawJSON))
	if rawStr == "[DONE]" {
		return convertOpenAIDoneToAnthropic((*param).(*ConvertOpenAIResponseToAnthropicParams))
	}

	streamResult := gjson.GetBytes(originalRequestRawJSON, "stream")
	if !streamResult.Exists() || (streamResult.Exists() && streamResult.Type == gjson.False) {
		return convertOpenAINonStreamingToAnthropic(rawJSON)
	} else {
		return convertOpenAIStreamingChunkToAnthropic(rawJSON, (*param).(*ConvertOpenAIResponseToAnthropicParams))
	}
}

// convertOpenAIStreamingChunkToAnthropic converts OpenAI streaming chunk to Anthropic streaming events
func convertOpenAIStreamingChunkToAnthropic(rawJSON []byte, param *ConvertOpenAIResponseToAnthropicParams) []string {
	root := gjson.ParseBytes(rawJSON)
	var results []string

	// Initialize parameters if needed
	if param.MessageID == "" {
		param.MessageID = root.Get("id").String()
	}
	if param.Model == "" {
		param.Model = root.Get("model").String()
	}
	if param.CreatedAt == 0 {
		param.CreatedAt = root.Get("created").Int()
	}

	// Emit message_start on the very first chunk, regardless of whether it has a role field.
	// Some providers (like Copilot) may send tool_calls in the first chunk without a role field.
	if delta := root.Get("choices.0.delta"); delta.Exists() {
		hasReasoningDelta := false
		if !param.MessageStarted {
			// Send message_start event
			messageStartJSON := `{"type":"message_start","message":{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}`
			messageStartJSON, _ = sjson.Set(messageStartJSON, "message.id", param.MessageID)
			messageStartJSON, _ = sjson.Set(messageStartJSON, "message.model", param.Model)
			results = append(results, "event: message_start\ndata: "+messageStartJSON+"\n\n")
			param.MessageStarted = true

			// Don't send content_block_start for text here - wait for actual content
		}

		// Handle reasoning content delta
		if reasoning := delta.Get("reasoning_content"); reasoning.Exists() {
			hasReasoningDelta = true
			for _, reasoningText := range collectOpenAIReasoningTexts(reasoning) {
				if reasoningText == "" {
					continue
				}
				stopTextContentBlock(param, &results)
				if !param.ThinkingContentBlockStarted {
					if param.ThinkingContentBlockIndex == -1 {
						param.ThinkingContentBlockIndex = param.NextContentBlockIndex
						param.NextContentBlockIndex++
					}
					contentBlockStartJSON := `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`
					contentBlockStartJSON, _ = sjson.Set(contentBlockStartJSON, "index", param.ThinkingContentBlockIndex)
					results = append(results, "event: content_block_start\ndata: "+contentBlockStartJSON+"\n\n")
					param.ThinkingContentBlockStarted = true
				}

				thinkingDeltaJSON := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}`
				thinkingDeltaJSON, _ = sjson.Set(thinkingDeltaJSON, "index", param.ThinkingContentBlockIndex)
				thinkingDeltaJSON, _ = sjson.Set(thinkingDeltaJSON, "delta.thinking", reasoningText)
				results = append(results, "event: content_block_delta\ndata: "+thinkingDeltaJSON+"\n\n")
			}
		}

		// Handle content delta
		if content := delta.Get("content"); content.Exists() && content.String() != "" {
			// Only apply "<think>...</think>" parsing when OpenAI-style reasoning_content isn't present
			// to avoid duplicating thinking output in the Anthropic stream.
			if !hasReasoningDelta && shouldParseThinkTagsForModel(param.Model) {
				segments := parseThinkTaggedStreamDelta(param, content.String())
				for _, seg := range segments {
					if seg.Thinking {
						emitThinkingDelta(param, &results, seg.Text)
					} else {
						emitTextDelta(param, &results, seg.Text)
					}
				}
			} else {
				emitTextDelta(param, &results, content.String())
			}
		}

		// Handle tool calls
		if toolCalls := delta.Get("tool_calls"); toolCalls.Exists() && toolCalls.IsArray() {
			if param.ToolCallsAccumulator == nil {
				param.ToolCallsAccumulator = make(map[int]*ToolCallAccumulator)
			}

			toolCalls.ForEach(func(_, toolCall gjson.Result) bool {
				index := int(toolCall.Get("index").Int())
				blockIndex := param.toolContentBlockIndex(index)

				// Initialize accumulator if needed
				if _, exists := param.ToolCallsAccumulator[index]; !exists {
					param.ToolCallsAccumulator[index] = &ToolCallAccumulator{}
				}

				accumulator := param.ToolCallsAccumulator[index]

				// Handle tool call ID
				if id := toolCall.Get("id"); id.Exists() {
					accumulator.ID = id.String()
				}

				// Handle function name
				if function := toolCall.Get("function"); function.Exists() {
					if name := function.Get("name"); name.Exists() {
						accumulator.Name = name.String()

						stopThinkingContentBlock(param, &results)

						stopTextContentBlock(param, &results)

						// Send content_block_start for tool_use
						contentBlockStartJSON := `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"","name":"","input":{}}}`
						contentBlockStartJSON, _ = sjson.Set(contentBlockStartJSON, "index", blockIndex)
						contentBlockStartJSON, _ = sjson.Set(contentBlockStartJSON, "content_block.id", accumulator.ID)
						contentBlockStartJSON, _ = sjson.Set(contentBlockStartJSON, "content_block.name", accumulator.Name)
						results = append(results, "event: content_block_start\ndata: "+contentBlockStartJSON+"\n\n")
					}

					// Handle function arguments
					if args := function.Get("arguments"); args.Exists() {
						argsText := args.String()
						if argsText != "" {
							accumulator.Arguments.WriteString(argsText)
						}
					}
				}

				return true
			})
		}
	}

	// Handle finish_reason (but don't send message_delta/message_stop yet)
	if finishReason := root.Get("choices.0.finish_reason"); finishReason.Exists() && finishReason.String() != "" {
		reason := finishReason.String()
		param.FinishReason = reason

		// Send content_block_stop for thinking content if needed
		if param.ThinkingContentBlockStarted {
			contentBlockStopJSON := `{"type":"content_block_stop","index":0}`
			contentBlockStopJSON, _ = sjson.Set(contentBlockStopJSON, "index", param.ThinkingContentBlockIndex)
			results = append(results, "event: content_block_stop\ndata: "+contentBlockStopJSON+"\n\n")
			param.ThinkingContentBlockStarted = false
			param.ThinkingContentBlockIndex = -1
		}

		// Send content_block_stop for text if text content block was started
		stopTextContentBlock(param, &results)

		// Send content_block_stop for any tool calls
		if !param.ContentBlocksStopped {
			for index := range param.ToolCallsAccumulator {
				accumulator := param.ToolCallsAccumulator[index]
				blockIndex := param.toolContentBlockIndex(index)

				// Send complete input_json_delta with all accumulated arguments
				if accumulator.Arguments.Len() > 0 {
					inputDeltaJSON := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`
					inputDeltaJSON, _ = sjson.Set(inputDeltaJSON, "index", blockIndex)
					inputDeltaJSON, _ = sjson.Set(inputDeltaJSON, "delta.partial_json", util.FixJSON(accumulator.Arguments.String()))
					results = append(results, "event: content_block_delta\ndata: "+inputDeltaJSON+"\n\n")
				}

				contentBlockStopJSON := `{"type":"content_block_stop","index":0}`
				contentBlockStopJSON, _ = sjson.Set(contentBlockStopJSON, "index", blockIndex)
				results = append(results, "event: content_block_stop\ndata: "+contentBlockStopJSON+"\n\n")
				delete(param.ToolCallBlockIndexes, index)
			}
			param.ContentBlocksStopped = true
		}

		// Don't send message_delta here - wait for usage info or [DONE]
	}

	// Handle usage information separately (this comes in a later chunk)
	// Only process if usage has actual values (not null)
	if param.FinishReason != "" {
		usage := root.Get("usage")
		var inputTokens, outputTokens int64
		if usage.Exists() && usage.Type != gjson.Null {
			// Check if usage has actual token counts
			promptTokens := usage.Get("prompt_tokens")
			completionTokens := usage.Get("completion_tokens")

			if promptTokens.Exists() && completionTokens.Exists() {
				inputTokens = promptTokens.Int()
				outputTokens = completionTokens.Int()
			}
		}
		// Send message_delta with usage
		messageDeltaJSON := `{"type":"message_delta","delta":{"stop_reason":"","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`
		messageDeltaJSON, _ = sjson.Set(messageDeltaJSON, "delta.stop_reason", mapOpenAIFinishReasonToAnthropic(param.FinishReason))
		messageDeltaJSON, _ = sjson.Set(messageDeltaJSON, "usage.input_tokens", inputTokens)
		messageDeltaJSON, _ = sjson.Set(messageDeltaJSON, "usage.output_tokens", outputTokens)
		results = append(results, "event: message_delta\ndata: "+messageDeltaJSON+"\n\n")
		param.MessageDeltaSent = true

		emitMessageStopIfNeeded(param, &results)

	}

	return results
}

// convertOpenAIDoneToAnthropic handles the [DONE] marker and sends final events
func convertOpenAIDoneToAnthropic(param *ConvertOpenAIResponseToAnthropicParams) []string {
	var results []string

	// Ensure all content blocks are stopped before final events
	if param.ThinkingContentBlockStarted {
		contentBlockStopJSON := `{"type":"content_block_stop","index":0}`
		contentBlockStopJSON, _ = sjson.Set(contentBlockStopJSON, "index", param.ThinkingContentBlockIndex)
		results = append(results, "event: content_block_stop\ndata: "+contentBlockStopJSON+"\n\n")
		param.ThinkingContentBlockStarted = false
		param.ThinkingContentBlockIndex = -1
	}

	stopTextContentBlock(param, &results)

	if !param.ContentBlocksStopped {
		for index := range param.ToolCallsAccumulator {
			accumulator := param.ToolCallsAccumulator[index]
			blockIndex := param.toolContentBlockIndex(index)

			if accumulator.Arguments.Len() > 0 {
				inputDeltaJSON := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`
				inputDeltaJSON, _ = sjson.Set(inputDeltaJSON, "index", blockIndex)
				inputDeltaJSON, _ = sjson.Set(inputDeltaJSON, "delta.partial_json", util.FixJSON(accumulator.Arguments.String()))
				results = append(results, "event: content_block_delta\ndata: "+inputDeltaJSON+"\n\n")
			}

			contentBlockStopJSON := `{"type":"content_block_stop","index":0}`
			contentBlockStopJSON, _ = sjson.Set(contentBlockStopJSON, "index", blockIndex)
			results = append(results, "event: content_block_stop\ndata: "+contentBlockStopJSON+"\n\n")
			delete(param.ToolCallBlockIndexes, index)
		}
		param.ContentBlocksStopped = true
	}

	// If we haven't sent message_delta yet (no usage info was received), send it now
	if param.FinishReason != "" && !param.MessageDeltaSent {
		messageDeltaJSON := `{"type":"message_delta","delta":{"stop_reason":"","stop_sequence":null}}`
		messageDeltaJSON, _ = sjson.Set(messageDeltaJSON, "delta.stop_reason", mapOpenAIFinishReasonToAnthropic(param.FinishReason))
		results = append(results, "event: message_delta\ndata: "+messageDeltaJSON+"\n\n")
		param.MessageDeltaSent = true
	}

	emitMessageStopIfNeeded(param, &results)

	return results
}

// convertOpenAINonStreamingToAnthropic converts OpenAI non-streaming response to Anthropic format
func convertOpenAINonStreamingToAnthropic(rawJSON []byte) []string {
	root := gjson.ParseBytes(rawJSON)

	out := `{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}`
	out, _ = sjson.Set(out, "id", root.Get("id").String())
	out, _ = sjson.Set(out, "model", root.Get("model").String())

	// Process message content and tool calls
	if choices := root.Get("choices"); choices.Exists() && choices.IsArray() && len(choices.Array()) > 0 {
		choice := choices.Array()[0] // Take first choice

		reasoningNode := choice.Get("message.reasoning_content")
		reasoningTexts := collectOpenAIReasoningTexts(reasoningNode)
		for _, reasoningText := range reasoningTexts {
			if reasoningText == "" {
				continue
			}
			block := `{"type":"thinking","thinking":""}`
			block, _ = sjson.Set(block, "thinking", reasoningText)
			out, _ = sjson.SetRaw(out, "content.-1", block)
		}

		// Handle text content
		if content := choice.Get("message.content"); content.Exists() && content.String() != "" {
			// Only attempt "<think>...</think>" parsing when "reasoning_content" is absent/empty.
			if len(reasoningTexts) == 0 && shouldParseThinkTagsForModel(root.Get("model").String()) {
				segments, _ := parseThinkTaggedText(content.String(), false)
				for _, seg := range segments {
					if seg.Thinking {
						block := `{"type":"thinking","thinking":""}`
						block, _ = sjson.Set(block, "thinking", seg.Text)
						out, _ = sjson.SetRaw(out, "content.-1", block)
						continue
					}
					block := `{"type":"text","text":""}`
					block, _ = sjson.Set(block, "text", seg.Text)
					out, _ = sjson.SetRaw(out, "content.-1", block)
				}
			} else {
				block := `{"type":"text","text":""}`
				block, _ = sjson.Set(block, "text", content.String())
				out, _ = sjson.SetRaw(out, "content.-1", block)
			}
		}

		// Handle tool calls
		if toolCalls := choice.Get("message.tool_calls"); toolCalls.Exists() && toolCalls.IsArray() {
			toolCalls.ForEach(func(_, toolCall gjson.Result) bool {
				toolUseBlock := `{"type":"tool_use","id":"","name":"","input":{}}`
				toolUseBlock, _ = sjson.Set(toolUseBlock, "id", toolCall.Get("id").String())
				toolUseBlock, _ = sjson.Set(toolUseBlock, "name", toolCall.Get("function.name").String())

				argsStr := util.FixJSON(toolCall.Get("function.arguments").String())
				if argsStr != "" && gjson.Valid(argsStr) {
					argsJSON := gjson.Parse(argsStr)
					if argsJSON.IsObject() {
						toolUseBlock, _ = sjson.SetRaw(toolUseBlock, "input", argsJSON.Raw)
					} else {
						toolUseBlock, _ = sjson.SetRaw(toolUseBlock, "input", "{}")
					}
				} else {
					toolUseBlock, _ = sjson.SetRaw(toolUseBlock, "input", "{}")
				}

				out, _ = sjson.SetRaw(out, "content.-1", toolUseBlock)
				return true
			})
		}

		// Set stop reason
		if finishReason := choice.Get("finish_reason"); finishReason.Exists() {
			out, _ = sjson.Set(out, "stop_reason", mapOpenAIFinishReasonToAnthropic(finishReason.String()))
		}
	}

	// Set usage information
	if usage := root.Get("usage"); usage.Exists() {
		out, _ = sjson.Set(out, "usage.input_tokens", usage.Get("prompt_tokens").Int())
		out, _ = sjson.Set(out, "usage.output_tokens", usage.Get("completion_tokens").Int())
		reasoningTokens := int64(0)
		if v := usage.Get("completion_tokens_details.reasoning_tokens"); v.Exists() {
			reasoningTokens = v.Int()
		}
		out, _ = sjson.Set(out, "usage.reasoning_tokens", reasoningTokens)
	}

	return []string{out}
}

// mapOpenAIFinishReasonToAnthropic maps OpenAI finish reasons to Anthropic equivalents
func mapOpenAIFinishReasonToAnthropic(openAIReason string) string {
	switch openAIReason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "end_turn" // Anthropic doesn't have direct equivalent
	case "function_call": // Legacy OpenAI
		return "tool_use"
	default:
		return "end_turn"
	}
}

func (p *ConvertOpenAIResponseToAnthropicParams) toolContentBlockIndex(openAIToolIndex int) int {
	if idx, ok := p.ToolCallBlockIndexes[openAIToolIndex]; ok {
		return idx
	}
	idx := p.NextContentBlockIndex
	p.NextContentBlockIndex++
	p.ToolCallBlockIndexes[openAIToolIndex] = idx
	return idx
}

func collectOpenAIReasoningTexts(node gjson.Result) []string {
	var texts []string
	if !node.Exists() {
		return texts
	}

	if node.IsArray() {
		node.ForEach(func(_, value gjson.Result) bool {
			texts = append(texts, collectOpenAIReasoningTexts(value)...)
			return true
		})
		return texts
	}

	switch node.Type {
	case gjson.String:
		if text := strings.TrimSpace(node.String()); text != "" {
			texts = append(texts, text)
		}
	case gjson.JSON:
		if text := node.Get("text"); text.Exists() {
			if trimmed := strings.TrimSpace(text.String()); trimmed != "" {
				texts = append(texts, trimmed)
			}
		} else if raw := strings.TrimSpace(node.Raw); raw != "" && !strings.HasPrefix(raw, "{") && !strings.HasPrefix(raw, "[") {
			texts = append(texts, raw)
		}
	}

	return texts
}

func stopThinkingContentBlock(param *ConvertOpenAIResponseToAnthropicParams, results *[]string) {
	if !param.ThinkingContentBlockStarted {
		return
	}
	contentBlockStopJSON := `{"type":"content_block_stop","index":0}`
	contentBlockStopJSON, _ = sjson.Set(contentBlockStopJSON, "index", param.ThinkingContentBlockIndex)
	*results = append(*results, "event: content_block_stop\ndata: "+contentBlockStopJSON+"\n\n")
	param.ThinkingContentBlockStarted = false
	param.ThinkingContentBlockIndex = -1
}

func emitMessageStopIfNeeded(param *ConvertOpenAIResponseToAnthropicParams, results *[]string) {
	if param.MessageStopSent {
		return
	}
	*results = append(*results, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	param.MessageStopSent = true
}

func stopTextContentBlock(param *ConvertOpenAIResponseToAnthropicParams, results *[]string) {
	if !param.TextContentBlockStarted {
		return
	}
	contentBlockStopJSON := `{"type":"content_block_stop","index":0}`
	contentBlockStopJSON, _ = sjson.Set(contentBlockStopJSON, "index", param.TextContentBlockIndex)
	*results = append(*results, "event: content_block_stop\ndata: "+contentBlockStopJSON+"\n\n")
	param.TextContentBlockStarted = false
	param.TextContentBlockIndex = -1
}

// ConvertOpenAIResponseToClaudeNonStream converts a non-streaming OpenAI response to a non-streaming Anthropic response.
//
// Parameters:
//   - ctx: The context for the request.
//   - modelName: The name of the model.
//   - rawJSON: The raw JSON response from the OpenAI API.
//   - param: A pointer to a parameter object for the conversion.
//
// Returns:
//   - string: An Anthropic-compatible JSON response.
func ConvertOpenAIResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	root := gjson.ParseBytes(rawJSON)
	out := `{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}`
	out, _ = sjson.Set(out, "id", root.Get("id").String())
	out, _ = sjson.Set(out, "model", root.Get("model").String())

	hasToolCall := false
	stopReasonSet := false

	if choices := root.Get("choices"); choices.Exists() && choices.IsArray() && len(choices.Array()) > 0 {
		choice := choices.Array()[0]

		if finishReason := choice.Get("finish_reason"); finishReason.Exists() {
			out, _ = sjson.Set(out, "stop_reason", mapOpenAIFinishReasonToAnthropic(finishReason.String()))
			stopReasonSet = true
		}

		if message := choice.Get("message"); message.Exists() {
			if contentResult := message.Get("content"); contentResult.Exists() {
				if contentResult.IsArray() {
					var textBuilder strings.Builder
					var thinkingBuilder strings.Builder

					flushText := func() {
						if textBuilder.Len() == 0 {
							return
						}
						block := `{"type":"text","text":""}`
						block, _ = sjson.Set(block, "text", textBuilder.String())
						out, _ = sjson.SetRaw(out, "content.-1", block)
						textBuilder.Reset()
					}

					flushThinking := func() {
						if thinkingBuilder.Len() == 0 {
							return
						}
						block := `{"type":"thinking","thinking":""}`
						block, _ = sjson.Set(block, "thinking", thinkingBuilder.String())
						out, _ = sjson.SetRaw(out, "content.-1", block)
						thinkingBuilder.Reset()
					}

					for _, item := range contentResult.Array() {
						switch item.Get("type").String() {
						case "text":
							flushThinking()
							textBuilder.WriteString(item.Get("text").String())
						case "tool_calls":
							flushThinking()
							flushText()
							toolCalls := item.Get("tool_calls")
							if toolCalls.IsArray() {
								toolCalls.ForEach(func(_, tc gjson.Result) bool {
									hasToolCall = true
									toolUse := `{"type":"tool_use","id":"","name":"","input":{}}`
									toolUse, _ = sjson.Set(toolUse, "id", tc.Get("id").String())
									toolUse, _ = sjson.Set(toolUse, "name", tc.Get("function.name").String())

									argsStr := util.FixJSON(tc.Get("function.arguments").String())
									if argsStr != "" && gjson.Valid(argsStr) {
										argsJSON := gjson.Parse(argsStr)
										if argsJSON.IsObject() {
											toolUse, _ = sjson.SetRaw(toolUse, "input", argsJSON.Raw)
										} else {
											toolUse, _ = sjson.SetRaw(toolUse, "input", "{}")
										}
									} else {
										toolUse, _ = sjson.SetRaw(toolUse, "input", "{}")
									}

									out, _ = sjson.SetRaw(out, "content.-1", toolUse)
									return true
								})
							}
						case "reasoning":
							flushText()
							if thinking := item.Get("text"); thinking.Exists() {
								thinkingBuilder.WriteString(thinking.String())
							}
						default:
							flushThinking()
							flushText()
						}
					}

					flushThinking()
					flushText()
				} else if contentResult.Type == gjson.String {
					textContent := contentResult.String()
					if textContent != "" {
						block := `{"type":"text","text":""}`
						block, _ = sjson.Set(block, "text", textContent)
						out, _ = sjson.SetRaw(out, "content.-1", block)
					}
				}
			}

			if reasoning := message.Get("reasoning_content"); reasoning.Exists() {
				for _, reasoningText := range collectOpenAIReasoningTexts(reasoning) {
					if reasoningText == "" {
						continue
					}
					block := `{"type":"thinking","thinking":""}`
					block, _ = sjson.Set(block, "thinking", reasoningText)
					out, _ = sjson.SetRaw(out, "content.-1", block)
				}
			}

			if toolCalls := message.Get("tool_calls"); toolCalls.Exists() && toolCalls.IsArray() {
				toolCalls.ForEach(func(_, toolCall gjson.Result) bool {
					hasToolCall = true
					toolUseBlock := `{"type":"tool_use","id":"","name":"","input":{}}`
					toolUseBlock, _ = sjson.Set(toolUseBlock, "id", toolCall.Get("id").String())
					toolUseBlock, _ = sjson.Set(toolUseBlock, "name", toolCall.Get("function.name").String())

					argsStr := util.FixJSON(toolCall.Get("function.arguments").String())
					if argsStr != "" && gjson.Valid(argsStr) {
						argsJSON := gjson.Parse(argsStr)
						if argsJSON.IsObject() {
							toolUseBlock, _ = sjson.SetRaw(toolUseBlock, "input", argsJSON.Raw)
						} else {
							toolUseBlock, _ = sjson.SetRaw(toolUseBlock, "input", "{}")
						}
					} else {
						toolUseBlock, _ = sjson.SetRaw(toolUseBlock, "input", "{}")
					}

					out, _ = sjson.SetRaw(out, "content.-1", toolUseBlock)
					return true
				})
			}
		}
	}

	if respUsage := root.Get("usage"); respUsage.Exists() {
		out, _ = sjson.Set(out, "usage.input_tokens", respUsage.Get("prompt_tokens").Int())
		out, _ = sjson.Set(out, "usage.output_tokens", respUsage.Get("completion_tokens").Int())
	}

	if !stopReasonSet {
		if hasToolCall {
			out, _ = sjson.Set(out, "stop_reason", "tool_use")
		} else {
			out, _ = sjson.Set(out, "stop_reason", "end_turn")
		}
	}

	return out
}

func ClaudeTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"input_tokens":%d}`, count)
}
