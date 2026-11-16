#include "chat.h"

static std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\t' || s[a] == '\r')) a++;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\t' || s[b - 1] == '\r')) b--;
  return s.substr(a, b - a);
}

static std::string content_text(const Json& c) {
  if (c.is_string()) return c.str();
  std::string out;
  if (c.is_array())
    for (auto& part : c.arr())
      if (part.get_str("type") == "text") out += part.get_str("text");
  return out;
}

std::vector<ChatMessage> chat_messages_from_json(const Json& messages) {
  std::vector<ChatMessage> v;
  for (auto& m : messages.arr()) {
    ChatMessage cm;
    cm.role = m.get_str("role");
    cm.content = content_text(m["content"]);
    cm.reasoning = m.get_str("reasoning_content");
    v.push_back(std::move(cm));
  }
  return v;
}

std::string chat_render(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking) {
  std::string o;
  size_t i0 = 0;
  if (!msgs.empty() && msgs[0].role == "system") {
    o += "<|im_start|>system\n" + trim(msgs[0].content) + "<|im_end|>\n";
    i0 = 1;
  }
  for (size_t i = i0; i < msgs.size(); i++) {
    const auto& m = msgs[i];
    const std::string content = trim(m.content);
    if (m.role == "user") {
      o += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      o += "<|im_start|>assistant\n<think>\n" + trim(m.reasoning) + "\n</think>\n\n" + content + "<|im_end|>\n";
    } else if (m.role == "tool") {
      if (i == 0 || msgs[i - 1].role != "tool") o += "<|im_start|>user";
      o += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") o += "<|im_end|>\n";
    } else if (m.role == "system") {
      o += "<|im_start|>system\n" + content + "<|im_end|>\n";
    }
  }
  if (add_gen) {
    o += "<|im_start|>assistant\n";
    o += enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return o;
}

void ThinkSplitter::feed(const std::string& text, std::string& reasoning, std::string& content, bool final) {
  pending += text;
  for (;;) {
    const std::string tag = in_think ? "</think>" : "<think>";
    const size_t p = pending.find(tag);
    if (p != std::string::npos) {
      (in_think ? reasoning : content) += pending.substr(0, p);
      pending.erase(0, p + tag.size());
      in_think = !in_think;
      if (!in_think) {  // drop the blank lines the template puts after </think>
        while (!pending.empty() && pending[0] == '\n') pending.erase(0, 1);
      }
      continue;
    }
    // keep a possible partial tag at the end
    size_t keep = 0;
    if (!final)
      for (size_t k = std::min(tag.size() - 1, pending.size()); k > 0; k--)
        if (pending.compare(pending.size() - k, k, tag, 0, k) == 0) { keep = k; break; }
    std::string emit = pending.substr(0, pending.size() - keep);
    if (!in_think && !content_started && !emit.empty()) {
      // strip leading newlines of the visible answer
      size_t a = 0;
      while (a < emit.size() && emit[a] == '\n') a++;
      emit = emit.substr(a);
      if (!emit.empty()) content_started = true;
    }
    (in_think ? reasoning : content) += emit;
    pending.erase(0, pending.size() - keep);
    return;
  }
}
