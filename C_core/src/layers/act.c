#include "../../include/core/Tensor.h"
#include <stdint.h>
#include <math.h>


Tensor* RELU(Tensor* input_tensor){
  for(int32_t i=0; i < input_tensor->H * input_tensor->W * input_tensor->C; i++){
    input_tensor->data[i] = (input_tensor->data[i] < 0) ?  0 : input_tensor->data[i]  ;
  }
  return input_tensor;
}


Tensor* RELU6(Tensor* input_tensor){
  for(int32_t i=0; i < input_tensor->H * input_tensor->W * input_tensor->C; i++){
    input_tensor->data[i] = (input_tensor->data[i] < 0) ?  0 : ((input_tensor->data[i] > 6) ? 6 : input_tensor->data[i]);
    
  }
  return input_tensor;
}


Tensor* SiLU(Tensor* input_tensor){
    for(int32_t i=0; i < input_tensor->H * input_tensor->W * input_tensor->C; i++){
    input_tensor->data[i] =  input_tensor->data[i] / (1 + expf(- input_tensor->data[i])); 
  }
  return input_tensor;
}
