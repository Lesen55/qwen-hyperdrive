// Qwen3.x chat template (ChatML + <think>), equivalent to the template shipped in the GGUF for
// system/user/assistant/tool text messages.
#pragma once
#include <string>
#include <vector>

#include "json.h"

struct ChatMessage {
  std::string role, content, reasoning;
};

std::vector<ChatMessage> chat_messages_from_json(const Json& messages);
std::string chat_render(const std::vector<ChatMessage>& msgs, bool add_generation_prompt, bool enable_thinking);

// Incremental splitter of generated text into reasoning (inside <think>...</think>) and content.
struct ThinkSplitter {
  bool in_think;
  bool content_started = false;
  std::string pending;  // text that may be the start of a tag
  explicit ThinkSplitter(bool starts_in_think) : in_think(starts_in_think) {}
  // feed new text; appends to reasoning/content outputs
  void feed(const std::string& text, std::string& reasoning, std::string& content, bool final = false);
};
