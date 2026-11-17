#include "backend.h"
#include "vk_model.h"

std::unique_ptr<IModel> create_model(const EngineOptions& opt) { return std::make_unique<VkModel>(opt); }
const char* backend_name() { return "vulkan"; }
