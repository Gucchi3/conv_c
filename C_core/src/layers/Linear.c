#include "../../include/core/Tensor.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "../../include/core/utils.h"






Tensor* Linear(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor){ 
//　引数例外処理
if (!input_tensor || !weight_tensor){
  return NULL;
}

// 出力Tensor形状計算
int32_t output_tensor_H = 1;
int32_t output_tensor_W = 1;
int32_t output_tensor_C = weight_tensor->OC;

// 出力Tensor形状例外処理
const int32_t input_total_size = input_tensor->H * input_tensor->W * input_tensor->C;
const int32_t expected_total_size = weight_tensor->H * weight_tensor->W * weight_tensor->INC;
if (output_tensor_C <= 0 || (input_total_size != expected_total_size)){
  return NULL;
}

// 出力Tensor作成・メモリ作成 -> NULL確認
Tensor* output_tensor = make_Tensor(output_tensor_H, output_tensor_W, output_tensor_C);
if (!output_tensor){
  return NULL;
}

// Linear
  const float* weight_point = weight_tensor->data;
  float* output_point = output_tensor->data;

  for(int32_t oc=0; oc < output_tensor->C; oc++){
    // 画像データ先頭ポインタ格納
    const float* input_point = input_tensor->data;
    // 重み初期化(biasがある場合は事前に加算しておく)
    float sum = (bias_tensor)? bias_tensor->data[oc] : 0.0f;

    for(int32_t i=0; i < input_total_size; i++){
      sum += *input_point++ * *weight_point++;
    }
    // 格納
    *output_point++ = sum;
  }

  // return
  return output_tensor;
}