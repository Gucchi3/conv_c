#include "Tensor.h"
#include "weight.h"
#ifndef UTILS_H
#define UTILS_H


Tensor* make_Tensor(int H, int W, int C);
Tensor* make_Tensor_from_array(int H, int W, int C, const float* init_data);
void free_Tensor(Tensor* Tensor);
void print_Tensor(Tensor* Tensor, int show_tensor_contents);
void print_W_Tensor(W_Tensor* W_Tensor, int show_tensor_contents);
#endif