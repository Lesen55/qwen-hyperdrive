// Qwen3.x chat template (ChatML + <think>), equivalent to the template shipped in the GGUF for
// system/user/assistant/tool text messages.
#pragma once
#include <string>
#include <vector>

#include "json.h"

struct ChatMessage {
  std::string role, content, reasoning;
  Json tool_calls;  // assistant: OpenAI tool_calls array (or null)
  std::string name, tool_call_id;  // tool results: function name / id of the call answered
};

// Prompt format of the loaded model: Qwen ChatML (<think>, XML tool calls) or Inkling 2
// (<|message_*|> / <|content_*|> markers, JSON tool calls, a "Thinking effort level" system line)
enum class ChatFormat { Qwen, Inkling };
void chat_set_format(ChatFormat f);
ChatFormat chat_format();
// text that closes a reasoning section early and opens the visible answer
std::string chat_close_reasoning();

std::vector<ChatMessage> chat_messages_from_json(const Json& messages);
// tools: OpenAI "tools" array (or null); reasoning_effort: "xhigh" (default), "medium", "low"
std::string chat_render(const std::vector<ChatMessage>& msgs, bool add_generation_prompt, bool enable_thinking,
                        const Json& tools = Json(), const std::string& reasoning_effort = "");

// Streaming filter for the visible answer: text before the first <tool_call> passes through, the
// tool-call blocks are held back and parsed at the end.
struct ToolCallFilter {
  bool in_call = false;
  std::string pending, raw;
  std::string feed(const std::string& text, bool final);
};
// Parse "<tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>" blocks into an
// OpenAI tool_calls array; parameter values are typed with the tool's JSON schema when available.
Json parse_tool_calls(const std::string& raw, const Json& tools, int id_base);

// Incremental splitter of generated text into reasoning (inside <think>...</think>) and content.
struct ThinkSplitter {
  bool in_think;
  bool content_started = false;
  std::string pending;  // text that may be the start of a tag
  // Inkling: the part being read (0 none / header, 1 thinking, 2 text, 3 tool call) and the header so far
  int part = 0;
  std::string header;
  explicit ThinkSplitter(bool starts_in_think) : in_think(chat_format() == ChatFormat::Qwen && starts_in_think) {}
  // feed new text; appends to reasoning/content outputs
  void feed(const std::string& text, std::string& reasoning, std::string& content, bool final = false);
};
