#include "load_torch.h"

bool load_module(const std::string &module_path, torch::jit::script::Module &module, bool toCUDA)
{
	try{
		module = torch::jit::load(module_path);
        if(toCUDA)
            module.to(torch::kCUDA);
	}
	catch(...)
	{
		std::cout<<"load failed, chek the path integrity."<<std::endl;
		return false;
	}
	return true;
}
