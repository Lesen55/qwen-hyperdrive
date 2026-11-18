// Backend factory: the CUDA build links backend_cuda.cpp, the Vulkan build backend_vk.cpp.
#pragma once
#include <memory>

#include "imodel.h"

std::unique_ptr<IModel> create_model(const EngineOptions& opt);
// small draft model (same tokenizer) for speculative decoding of a target without MTP: loaded with
// just the memory it needs (weights + its KV at the target's context), before the target
std::unique_ptr<IModel> create_draft_model(const std::string& path, const EngineOptions& target);
const char* backend_name();
