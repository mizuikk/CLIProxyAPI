package claude

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertClaudeRequestToOpenAI_InjectsClaudeCodeContinuation_ForMiniMax(t *testing.T) {
	in := `{
	  "model": "MiniMax-M2.1",
	  "system": [{"type":"text","text":"` + claudeCodeSystemMarker + `"}],
	  "messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`

	out := ConvertClaudeRequestToOpenAI("MiniMax-M2.1", []byte(in), true)
	root := gjson.ParseBytes(out)

	found := false
	root.Get("messages.0.content").ForEach(func(_, value gjson.Result) bool {
		if strings.Contains(value.Get("text").String(), claudeCodeAgentContinuationInstruction) {
			found = true
			return false
		}
		return true
	})
	if !found {
		t.Fatalf("expected continuation instruction to be injected for MiniMax Claude Code session, got: %s", root.Get("messages.0.content").Raw)
	}
}

func TestConvertClaudeRequestToOpenAI_DoesNotInject_ForNonMiniMax(t *testing.T) {
	in := `{
	  "model": "gpt-4o",
	  "system": [{"type":"text","text":"` + claudeCodeSystemMarker + `"}],
	  "messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`

	out := ConvertClaudeRequestToOpenAI("gpt-4o", []byte(in), true)
	root := gjson.ParseBytes(out)

	root.Get("messages.0.content").ForEach(func(_, value gjson.Result) bool {
		if strings.Contains(value.Get("text").String(), claudeCodeAgentContinuationInstruction) {
			t.Fatalf("did not expect continuation instruction for non-MiniMax model, got: %s", root.Get("messages.0.content").Raw)
		}
		return true
	})
}

func TestConvertClaudeRequestToOpenAI_DoesNotInject_WhenNotClaudeCode(t *testing.T) {
	in := `{
	  "model": "MiniMax-M2.1",
	  "system": [{"type":"text","text":"You are a helpful assistant."}],
	  "messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`

	out := ConvertClaudeRequestToOpenAI("MiniMax-M2.1", []byte(in), true)
	root := gjson.ParseBytes(out)

	root.Get("messages.0.content").ForEach(func(_, value gjson.Result) bool {
		if strings.Contains(value.Get("text").String(), claudeCodeAgentContinuationInstruction) {
			t.Fatalf("did not expect continuation instruction when request is not Claude Code, got: %s", root.Get("messages.0.content").Raw)
		}
		return true
	})
}

func TestConvertClaudeRequestToOpenAI_ForcesToolChoiceRequired_ForMiniMaxClaudeCodeWithTools(t *testing.T) {
	in := `{
	  "model": "MiniMax-M2.1",
	  "system": [{"type":"text","text":"` + claudeCodeSystemMarker + `"}],
	  "messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
	  "tools": [{"name":"Bash","description":"Run a command","input_schema":{"type":"object"}}]
	}`

	out := ConvertClaudeRequestToOpenAI("MiniMax-M2.1", []byte(in), true)
	root := gjson.ParseBytes(out)

	if root.Get("tool_choice").String() != "required" {
		t.Fatalf("expected tool_choice=required for MiniMax Claude Code session with tools, got: %s", root.Get("tool_choice").Raw)
	}
}

func TestConvertClaudeRequestToOpenAI_DoesNotOverrideExplicitToolChoice_ForMiniMaxClaudeCode(t *testing.T) {
	in := `{
	  "model": "MiniMax-M2.1",
	  "system": [{"type":"text","text":"` + claudeCodeSystemMarker + `"}],
	  "messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
	  "tools": [{"name":"Bash","description":"Run a command","input_schema":{"type":"object"}}],
	  "tool_choice": {"type":"auto"}
	}`

	out := ConvertClaudeRequestToOpenAI("MiniMax-M2.1", []byte(in), true)
	root := gjson.ParseBytes(out)

	if root.Get("tool_choice").String() != "auto" {
		t.Fatalf("expected explicit tool_choice to be preserved, got: %s", root.Get("tool_choice").Raw)
	}
}
