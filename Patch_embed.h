#include "Tensor.h"
#include "weight.h"

#ifndef PATCH_H
#define PATCH_H

Tensor* Patch_embedding(Tensor* input_tensor, W_Tensor* weight_tensor);

#endif