#include "chat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

static ChatFormat g_format = ChatFormat::Qwen;
void chat_set_format(ChatFormat f) { g_format = f; }
ChatFormat chat_format() { return g_format; }
std::string chat_close_reasoning() {
  if (g_format == ChatFormat::Inkling) return "<|end_message|><|message_model|><|content_text|>";
  return "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";
}

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
    if (cm.role == "developer") cm.role = "system";  // OpenAI's newer name for the system role
    cm.content = content_text(m["content"]);
    cm.reasoning = m.get_str("reasoning_content");
    cm.name = m.get_str("name");
    cm.tool_call_id = m.get_str("tool_call_id");
    if (m["tool_calls"].is_array()) cm.tool_calls = m["tool_calls"];
    v.push_back(std::move(cm));
  }
  return v;
}

static std::string render_tool_calls(const Json& calls, const std::string& content) {
  std::string o;
  bool first = true;
  for (const Json& tc0 : calls.arr()) {
    const Json& tc = tc0["function"].is_object() ? tc0["function"] : tc0;
    const std::string name = tc.get_str("name");
    if (first) o += trim(content).empty() ? "<tool_call>\n<function=" + name + ">\n" : "\n\n<tool_call>\n<function=" + name + ">\n";
    else o += "\n<tool_call>\n<function=" + name + ">\n";
    first = false;
    Json args = tc["arguments"];
    if (args.is_string()) {  // OpenAI clients send arguments as a JSON string
      try { args = Json::parse(args.str()); } catch (...) { args = Json(); }
    }
    if (args.is_object())
      for (auto& [k, v] : args.items()) o += "<parameter=" + k + ">\n" + (v.is_string() ? v.str() : v.dump_py()) + "\n</parameter>\n";
    o += "</function>\n</tool_call>";
  }
  return o;
}

// keys sorted at every level (Jinja tojson(sort_keys=true))
static Json sorted_json(const Json& j) {
  if (j.is_array()) {
    Json a = Json::array();
    for (const Json& x : j.arr()) a.push_back(sorted_json(x));
    return a;
  }
  if (!j.is_object()) return j;
  std::vector<std::pair<std::string, Json>> it(j.items().begin(), j.items().end());
  std::stable_sort(it.begin(), it.end(), [](auto& a, auto& b) { return a.first < b.first; });
  Json o = Json::object();
  for (auto& [k, v] : it) o[k] = sorted_json(v);
  return o;
}

// Inkling 2 chat template (as shipped with the model): tool declarations, the thinking-effort system line before
// the first non-system message, <|message_ROLE|><|content_KIND|>...<|end_message|> parts, JSON tool calls, and
// <|content_model_end_sampling|> after each model turn.  There is no generation prompt: the model opens its turn.
static std::string inkling_render(const std::vector<ChatMessage>& msgs, bool enable_thinking, const Json& tools,
                                  const std::string& effort) {
  std::string o;
  if (tools.is_array() && !tools.arr().empty()) {
    Json specs = Json::array();
    for (const Json& t : tools.arr()) {
      const Json& fn = t["function"].is_object() ? t["function"] : t;
      Json spec = Json::object();
      spec["description"] = Json(fn.get_str("description"));
      spec["name"] = Json(fn.get_str("name"));
      spec["parameters"] = fn["parameters"].is_object() ? fn["parameters"] : Json::object();
      spec["type"] = Json(t.get_str("type").empty() ? std::string("function") : t.get_str("type"));
      specs.push_back(spec);
    }
    o += "<|message_system|>tool_declare<|content_xml|>" + sorted_json(specs).dump() + "<|end_message|>";
  }
  std::string level = "0.9";
  if (!enable_thinking || effort == "none") level = "0";
  else if (effort == "minimal") level = "0.1";
  else if (effort == "low") level = "0.2";
  else if (effort == "medium") level = "0.7";
  else if (effort == "xhigh" || effort == "max") level = "0.99";
  const std::string eff = "<|message_system|><|content_text|>Thinking effort level: " + level + "<|end_message|>";
  bool effort_done = false;
  for (size_t i = 0; i < msgs.size(); i++) {
    const ChatMessage& m = msgs[i];
    if (!effort_done && m.role != "system") {
      o += eff;
      effort_done = true;
    }
    std::string r = m.role == "user" ? "<|message_user|>" : m.role == "assistant" ? "<|message_model|>"
                  : m.role == "tool" ? "<|message_tool|>" : "<|message_system|>";
    if (m.role == "tool") {
      std::string name = m.name;
      if (name.empty() && !m.tool_call_id.empty())
        for (const ChatMessage& p : msgs)
          if (p.role == "assistant" && p.tool_calls.is_array())
            for (const Json& tc : p.tool_calls.arr())
              if (tc.get_str("id") == m.tool_call_id) name = tc["function"].get_str("name");
      r += name;
    }
    const bool has_calls = m.role == "assistant" && m.tool_calls.is_array() && !m.tool_calls.arr().empty();
    if (m.role == "assistant" && !m.reasoning.empty()) o += r + "<|content_thinking|>" + m.reasoning + "<|end_message|>";
    if (!m.content.empty() || !(m.role == "assistant" && (has_calls || !m.reasoning.empty())))
      o += r + "<|content_text|>" + m.content + "<|end_message|>";
    if (has_calls) {
      for (const Json& tc0 : m.tool_calls.arr()) {
        const Json& fn = tc0["function"].is_object() ? tc0["function"] : tc0;
        const std::string name = fn.get_str("name");
        Json args = fn["arguments"];
        if (args.is_string()) {
          try { args = Json::parse(args.str()); } catch (...) { args = Json::object(); }
        }
        if (!args.is_object()) args = Json::object();
        std::vector<std::pair<std::string, Json>> kv(args.items().begin(), args.items().end());
        std::stable_sort(kv.begin(), kv.end(), [](auto& a, auto& b) { return a.first < b.first; });
        o += "<|message_model|>" + name + "<|content_invoke_tool_json|>{\"name\":" + Json(name).dump() + ",\"args\":{";
        for (size_t k = 0; k < kv.size(); k++) o += (k ? "," : "") + Json(kv[k].first).dump() + ":" + kv[k].second.dump();
        o += "}}<|end_message|>";
      }
    }
    if (m.role == "assistant") o += "<|content_model_end_sampling|>";
  }
  if (!effort_done) o += eff;
  return o;
}

// Mirrors the chat template shipped in the GGUF (tools block, reasoning-effort instructions, thinking
// preserved in the history, tool calls / tool responses).
std::string chat_render(const std::vector<ChatMessage>& msgs, bool add_gen, bool enable_thinking, const Json& tools,
                        const std::string& reasoning_effort) {
  if (g_format == ChatFormat::Inkling) return inkling_render(msgs, enable_thinking, tools, reasoning_effort);
  std::string o, ri;
  if (enable_thinking) {
    const std::string e = reasoning_effort.empty() ? "xhigh" : reasoning_effort;
    if (e == "xhigh")
      ri = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
           "alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    else if (e == "low")
      ri = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary "
           "elaboration.";
  }
  const bool has_sys = !msgs.empty() && msgs[0].role == "system";
  const std::string sys = has_sys ? trim(msgs[0].content) : "";
  if (tools.is_array() && !tools.arr().empty()) {
    o += "<|im_start|>system\n";
    if (!ri.empty()) o += ri + "\n\n";
    o += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const Json& t : tools.arr()) o += "\n" + t.dump_py();
    o += "\n</tools>";
    o += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
         "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
         "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
         "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
         "parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function "
         "call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do "
         "not tell the user about function calls\n</IMPORTANT>";
    if (!sys.empty()) o += "\n\n" + sys;
    o += "<|im_end|>\n";
  } else if (!sys.empty()) {
    o += "<|im_start|>system\n" + (ri.empty() ? "" : ri + "\n\n") + sys + "<|im_end|>\n";
  } else if (!ri.empty()) {
    o += "<|im_start|>system\n" + ri + "<|im_end|>\n";
  }
  for (size_t i = has_sys ? 1 : 0; i < msgs.size(); i++) {
    const auto& m = msgs[i];
    const std::string content = trim(m.content);
    if (m.role == "user") {
      o += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      o += "<|im_start|>assistant\n<think>\n" + trim(m.reasoning) + "\n</think>\n\n" + content;
      if (m.tool_calls.is_array()) o += render_tool_calls(m.tool_calls, content);
      o += "<|im_end|>\n";
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

std::string ToolCallFilter::feed(const std::string& text, bool final) {
  static const std::string tag = "<tool_call>";
  if (in_call) { raw += text; return ""; }
  pending += text;
  const size_t p = pending.find(tag);
  if (p != std::string::npos) {
    std::string out = pending.substr(0, p);
    raw = pending.substr(p);
    pending.clear();
    in_call = true;
    return out;
  }
  size_t keep = 0;
  if (!final)
    for (size_t k = std::min(tag.size() - 1, pending.size()); k > 0; k--)
      if (pending.compare(pending.size() - k, k, tag, 0, k) == 0) { keep = k; break; }
  std::string out = pending.substr(0, pending.size() - keep);
  pending.erase(0, pending.size() - keep);
  return out;
}

static std::string strip_nl(std::string v) {
  if (!v.empty() && v.front() == '\n') v.erase(0, 1);
  if (!v.empty() && v.back() == '\n') v.pop_back();
  return v;
}

Json parse_tool_calls(const std::string& raw, const Json& tools, int id_base) {
  Json calls = Json::array();
  size_t p = 0;
  if (raw.find("<function=") == std::string::npos) {
    // JSON bodies: <tool_call>{"name": .., "args"|"arguments": {..}}</tool_call>
    for (;;) {
      const size_t a = raw.find("<tool_call>", p);
      if (a == std::string::npos) break;
      size_t e = raw.find("</tool_call>", a);
      if (e == std::string::npos) e = raw.size();
      Json j;
      try { j = Json::parse(trim(raw.substr(a + 11, e - a - 11))); } catch (...) { j = Json(); }
      p = e;
      if (!j.is_object() || j.get_str("name").empty()) continue;
      Json args = j["args"].is_object() ? j["args"] : j["arguments"].is_object() ? j["arguments"] : Json::object();
      Json fn = Json::object();
      fn["name"] = Json(j.get_str("name"));
      fn["arguments"] = Json(args.dump());
      Json tc = Json::object();
      tc["id"] = Json("call_" + std::to_string(id_base) + "_" + std::to_string(calls.arr().size()));
      tc["type"] = Json(std::string("function"));
      tc["function"] = fn;
      calls.push_back(tc);
    }
    (void)tools;
    return calls;
  }
  for (;;) {
    const size_t a = raw.find("<function=", p);
    if (a == std::string::npos) break;
    const size_t ne = raw.find('>', a);
    if (ne == std::string::npos) break;
    const std::string name = trim(raw.substr(a + 10, ne - a - 10));
    size_t fe = raw.find("</function>", ne);
    if (fe == std::string::npos) fe = raw.size();
    const Json* props = nullptr;
    if (tools.is_array())
      for (const Json& t : tools.arr()) {
        const Json& f = t["function"].is_object() ? t["function"] : t;
        if (f.get_str("name") == name) { props = &f["parameters"]["properties"]; break; }
      }
    Json args = Json::object();
    size_t q = ne + 1;
    for (;;) {
      const size_t pa = raw.find("<parameter=", q);
      if (pa == std::string::npos || pa >= fe) break;
      const size_t pe = raw.find('>', pa);
      if (pe == std::string::npos) break;
      const std::string key = trim(raw.substr(pa + 11, pe - pa - 11));
      size_t ve = raw.find("</parameter>", pe);
      const size_t nx = raw.find("<parameter=", pe);
      if (ve == std::string::npos || (nx != std::string::npos && nx < ve)) ve = std::min(nx == std::string::npos ? fe : nx, fe);
      const std::string val = strip_nl(raw.substr(pe + 1, ve - pe - 1));
      std::string type;
      if (props && props->is_object() && (*props)[key].is_object()) type = (*props)[key].get_str("type");
      Json v(val);
      if (type != "string") {
        try {
          Json j = Json::parse(val);
          if (!type.empty() || !j.is_string()) v = j;
        } catch (...) {}
      }
      args[key] = v;
      q = ve;
    }
    Json fn = Json::object();
    fn["name"] = name;
    fn["arguments"] = args.dump();
    Json tc = Json::object();
    tc["id"] = "call_" + std::to_string(id_base) + "_" + std::to_string(calls.arr().size());
    tc["type"] = "function";
    tc["function"] = fn;
    calls.push_back(tc);
    p = fe;
  }
  return calls;
}

// Inkling output: <|message_model|>[NAME]<|content_thinking|>...<|end_message|><|message_model|><|content_text|>...
// <|end_message|>[<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>].  Tool calls
// are passed on to the content as "<tool_call>{json}</tool_call>" (ToolCallFilter / parse_tool_calls take them).
static void inkling_feed(ThinkSplitter& sp, const std::string& text, std::string& reasoning, std::string& content, bool final) {
  static const char* kMarks[] = {"<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                                 "<|content_invoke_tool_json|>", "<|end_message|>"};
  // text of the current part
  auto put = [&](std::string t) {
    if (t.empty()) return;
    if (sp.part == 0) { sp.header += t; return; }  // between <|message_model|> and the content marker
    if (sp.part == 1) { reasoning += t; return; }
    if (sp.part == 2 && !sp.content_started) {
      size_t a = 0;
      while (a < t.size() && t[a] == '\n') a++;
      t.erase(0, a);
      if (t.empty()) return;
      sp.content_started = true;
    }
    content += t;
  };
  sp.pending += text;
  for (;;) {
    size_t best = std::string::npos;
    int mi = -1;
    for (int i = 0; i < 5; i++) {
      const size_t p = sp.pending.find(kMarks[i]);
      if (p < best) { best = p; mi = i; }
    }
    if (mi < 0) break;
    put(sp.pending.substr(0, best));
    sp.pending.erase(0, best + strlen(kMarks[mi]));
    if (mi >= 1 && mi <= 3) sp.header.clear();  // the part's name (a tool call repeats it in its JSON)
    switch (mi) {
      case 0: sp.part = 0; sp.header.clear(); sp.in_think = false; break;
      case 1: sp.part = 1; sp.in_think = true; break;
      case 2: sp.part = 2; sp.in_think = false; break;
      case 3: sp.part = 3; sp.in_think = false; content += "<tool_call>"; break;
      case 4:
        if (sp.part == 3) content += "</tool_call>";
        if (sp.part == 0 && !sp.header.empty()) { content += sp.header; sp.header.clear(); }
        sp.part = 0;
        sp.in_think = false;
        break;
    }
  }
  // hold back a possible partial marker
  size_t keep = 0;
  if (!final) {
    const size_t lt = sp.pending.rfind('<');
    if (lt != std::string::npos && sp.pending.find('>', lt) == std::string::npos && sp.pending.size() - lt < 32)
      keep = sp.pending.size() - lt;
  }
  put(sp.pending.substr(0, sp.pending.size() - keep));
  sp.pending.erase(0, sp.pending.size() - keep);
  if (final && sp.part == 0 && !sp.header.empty()) {  // stray text outside any part
    content += sp.header;
    sp.header.clear();
  }
}

void ThinkSplitter::feed(const std::string& text, std::string& reasoning, std::string& content, bool final) {
  if (chat_format() == ChatFormat::Inkling) {
    inkling_feed(*this, text, reasoning, content, final);
    return;
  }
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
