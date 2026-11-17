#include "backend.h"
#include "model.h"

std::unique_ptr<IModel> create_model(const EngineOptions& opt) { return std::make_unique<Model>(opt); }
const char* backend_name() { return "cuda"; }
