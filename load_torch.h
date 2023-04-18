#ifndef LOADTORCH_H
#define LOADTORCH_H
#include "macro_ABI.h"
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
#include <iostream>

extern torch::jit::script::Module rexnet, net;

bool load_module(const std::string &module_path, torch::jit::script::Module &module, bool toCuda);


#endif
