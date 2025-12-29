package claude

import (
	"context"
	"strings"
	"testing"
)

// TestConvertOpenAIResponseToClaude_ThinkTagsBecomeThinkingBlocks verifies that providers which embed
// reasoning in delta.content via "<think>...</think>" tags are converted into Anthropic thinking blocks.
func TestConvertOpenAIResponseToClaude_ThinkTagsBecomeThinkingBlocks(t *testing.T) {
	originalReq := []byte(`{"stream":true}`)
	rawChunk := []byte(`data: {"id":"id1","model":"MiniMax-M2.1","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"<think>reason</think>answer"}}]}`)

	var param any
	events := ConvertOpenAIResponseToClaude(context.Background(), "MiniMax-M2.1", originalReq, nil, rawChunk, &param)
	joined := strings.Join(events, "")

	if strings.Contains(joined, thinkOpenTag) || strings.Contains(joined, thinkCloseTag) {
		t.Fatalf("expected think tags to be removed from Anthropic stream, got: %s", joined)
	}
	if !strings.Contains(joined, `"type":"thinking_delta"`) || !strings.Contains(joined, `"thinking":"reason"`) {
		t.Fatalf("expected thinking delta to be emitted, got: %s", joined)
	}
	if !strings.Contains(joined, `"type":"text_delta"`) || !strings.Contains(joined, `"text":"answer"`) {
		t.Fatalf("expected text delta to be emitted, got: %s", joined)
	}
}

// TestConvertOpenAIResponseToClaude_ThinkTagsAcrossChunks verifies tag parsing when "<think>...</think>"
// spans multiple streaming chunks.
func TestConvertOpenAIResponseToClaude_ThinkTagsAcrossChunks(t *testing.T) {
	originalReq := []byte(`{"stream":true}`)

	chunk1 := []byte(`data: {"id":"id2","model":"MiniMax-M2.1","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"<think>rea"}}]}`)
	chunk2 := []byte(`data: {"id":"id2","model":"MiniMax-M2.1","created":1,"choices":[{"index":0,"delta":{"content":"son</think>ans"}}]}`)

	var param any
	events1 := ConvertOpenAIResponseToClaude(context.Background(), "MiniMax-M2.1", originalReq, nil, chunk1, &param)
	events2 := ConvertOpenAIResponseToClaude(context.Background(), "MiniMax-M2.1", originalReq, nil, chunk2, &param)
	joined := strings.Join(append(events1, events2...), "")

	if strings.Contains(joined, thinkOpenTag) || strings.Contains(joined, thinkCloseTag) {
		t.Fatalf("expected think tags to be removed from Anthropic stream, got: %s", joined)
	}
	if !strings.Contains(joined, `"thinking":"rea"`) || !strings.Contains(joined, `"thinking":"son"`) {
		t.Fatalf("expected thinking deltas split across chunks, got: %s", joined)
	}
	if !strings.Contains(joined, `"text":"ans"`) {
		t.Fatalf("expected final text delta to be emitted, got: %s", joined)
	}
}

// TestConvertOpenAIResponseToClaude_ThinkTagsIgnoredForNonMiniMax verifies we do not reinterpret literal
// "<think>...</think>" text for non-MiniMax models to avoid rewriting user-visible content.
func TestConvertOpenAIResponseToClaude_ThinkTagsIgnoredForNonMiniMax(t *testing.T) {
	originalReq := []byte(`{"stream":true}`)
	rawChunk := []byte(`data: {"id":"id3","model":"gpt-4o","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"<think>literal</think>text"}}]}`)

	var param any
	events := ConvertOpenAIResponseToClaude(context.Background(), "gpt-4o", originalReq, nil, rawChunk, &param)
	joined := strings.Join(events, "")

	if strings.Contains(joined, `"type":"thinking_delta"`) {
		t.Fatalf("expected no thinking delta for non-MiniMax models, got: %s", joined)
	}
	if !strings.Contains(joined, `<think>literal</think>text`) {
		t.Fatalf("expected literal think tags to be preserved as text, got: %s", joined)
	}
}
