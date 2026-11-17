#pragma once
// Web search for the chat UI: DuckDuckGo HTML results + readable text of the top pages.
// Pages are fetched with curl (spawned without a shell); only http(s) URLs taken from the search
// results are fetched, and private / loopback hosts are refused.
#include <string>
#include <vector>

struct WebResult {
  std::string title, url, snippet, text;
};

std::vector<WebResult> web_search(const std::string& query, int n_results, int n_fetch, size_t max_chars);
std::string url_decode(const std::string& s);
