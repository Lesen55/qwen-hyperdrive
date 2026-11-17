// Backend factory: the CUDA build links backend_cuda.cpp, the Vulkan build backend_vk.cpp.
#pragma once
#include <memory>

#include "imodel.h"

std::unique_ptr<IModel> create_model(const EngineOptions& opt);
const char* backend_name();
