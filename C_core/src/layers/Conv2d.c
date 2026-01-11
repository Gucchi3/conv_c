#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/core/Tensor.h"
#include "../../include/core/utils.h"


Tensor* Conv2d(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, int32_t stride, int32_t padding){
  // 代入
  int32_t kernel_size = weight_tensor->H; int32_t output_channel = weight_tensor->OC;
  // 引数例外処理
  if (!input_tensor || kernel_size <= 0 || output_channel <= 0 || stride <= 0 || padding < 0 || (bias_tensor && bias_tensor->OC != output_channel)) {
    return NULL;
  }
  // 出力Tensor形状計算
  int32_t output_tensor_H = ((input_tensor->H + 2 * padding - kernel_size) / stride) + 1;
  int32_t output_tensor_W = ((input_tensor->W + 2 * padding - kernel_size) / stride) + 1;
  int32_t output_tensor_C = output_channel;
  // 計算結果が有効かチェック
  if (output_tensor_H <= 0 || output_tensor_W <= 0) {
    return NULL;
  }
  // 出力Tensor作成・メモリ確保 -> NULL確認
  Tensor* output_tensor = make_Tensor(output_tensor_H, output_tensor_W, output_tensor_C);
  if (!output_tensor){
    return NULL;
  }

  //Conv2d
  const float* weight_head = weight_tensor->data;
  float* output_point = output_tensor->data;


  for(int32_t h=0; h < output_tensor->H; h++){
    for(int32_t w=0; w < output_tensor->W; w++){
      // 重みの先頭アドレス格納
      const float* weight_point = weight_head;
// カーネル左上の画像内位置計算
      int32_t in_lt_h = (h * stride) - padding;
      int32_t in_lt_w = (w * stride) - padding; 
      for(int32_t oc=0; oc < output_tensor->C; oc++){
        // 重み初期化
        float sum = (bias_tensor) ? bias_tensor->data[oc] : 0.0f;
        for(int32_t kh=0; kh < kernel_size; kh++){
          for(int32_t kw=0; kw < kernel_size; kw++){
            // 計算座標計算
            int32_t in_h = in_lt_h + kh;
            int32_t in_w = in_lt_w + kw;
            // paddingか画像か判定 -> paddingならアドレス移動してkcループをスキップ
            if(in_h < 0 || in_h >= input_tensor->H || in_w < 0 || in_w >= input_tensor->W){
              weight_point += input_tensor->C;
              continue;
            }
            // 対象画像座標があるアドレス計算
            int32_t in_idx = ((in_h * input_tensor->W) + in_w) * input_tensor->C;
            const float* in_point = &input_tensor->data[in_idx];
            for(int32_t kc=0; kc < input_tensor->C; kc++){
              // 乗算 -> 加算　
              sum += *in_point++ * *weight_point++;
            }
          }
        }
        *output_point++ = sum;
      }
    }
  }

  // return
  return output_tensor;
}


Tensor* Conv2d_BN_ACT(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, int32_t stride, int32_t padding, const char* act){
// Now building...
  // 代入
  int32_t kernel_size = weight_tensor->H; int32_t output_channel = weight_tensor->OC;
  // 引数例外処理
  if (!input_tensor || kernel_size <= 0 || output_channel <= 0 || stride <= 0 || padding < 0 || (bias_tensor && bias_tensor->OC != output_channel)) {
    return NULL;
  }
  // 使用活性化関数選択
  int32_t act_type = 0; // 0:None, 1:RELU, 2:RELU6, 3:SiLU
  if (act) { // NULLチェック
    if (strcmp(act, "RELU") == 0)       act_type = 1;
    else if (strcmp(act, "RELU6") == 0) act_type = 2;
    else if (strcmp(act, "SiLU") == 0)  act_type = 3;
}
  // 出力Tensor形状計算
  int32_t output_tensor_H = ((input_tensor->H + 2 * padding - kernel_size) / stride) + 1;
  int32_t output_tensor_W = ((input_tensor->W + 2 * padding - kernel_size) / stride) + 1;
  int32_t output_tensor_C = output_channel;
  // 計算結果が有効かチェック
  if (output_tensor_H <= 0 || output_tensor_W <= 0) {
    return NULL;
  }
  // 出力Tensor作成・メモリ確保 -> NULL確認
  Tensor* output_tensor = make_Tensor(output_tensor_H, output_tensor_W, output_tensor_C);
  if (!output_tensor){
    return NULL;
  }

  //Conv2d
  const float* weight_head = weight_tensor->data;
  float* output_point = output_tensor->data;


  for(int32_t h=0; h < output_tensor->H; h++){
    for(int32_t w=0; w < output_tensor->W; w++){
      // 重みの先頭アドレス格納
      const float* weight_point = weight_head;
// カーネル左上の画像内位置計算
      int32_t in_lt_h = (h * stride) - padding;
      int32_t in_lt_w = (w * stride) - padding; 
      for(int32_t oc=0; oc < output_tensor->C; oc++){
        // 重み初期化
        float sum = (bias_tensor) ? bias_tensor->data[oc] : 0.0f;
        for(int32_t kh=0; kh < kernel_size; kh++){
          for(int32_t kw=0; kw < kernel_size; kw++){
            // 計算座標計算
            int32_t in_h = in_lt_h + kh;
            int32_t in_w = in_lt_w + kw;
            // paddingか画像か判定 -> paddingならアドレス移動してkcループをスキップ
            if(in_h < 0 || in_h >= input_tensor->H || in_w < 0 || in_w >= input_tensor->W){
              weight_point += input_tensor->C;
              continue;
            }
            // 対象画像座標があるアドレス計算
            int32_t in_idx = ((in_h * input_tensor->W) + in_w) * input_tensor->C;
            const float* in_point = &input_tensor->data[in_idx];
            for(int32_t kc=0; kc < input_tensor->C; kc++){
              // 乗算 -> 加算　
              sum += *in_point++ * *weight_point++;
            }
          }
        }
        if(act_type == 1){
          *output_point++ = (sum < 0) ? 0 : sum;
        }else if(act_type == 2){
          *output_point++ = (sum < 0) ? 0 : ((sum > 6) ? 6: sum);
        }else if(act_type == 3){
          *output_point++ = sum / (1 + expf(- sum));
        }else if (act_type == 0){
          *output_point++ = sum;
        }
      }
    }
  }

  // return
  return output_tensor ;
}


Tensor* PConv2d(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor){
  // comming soon...
}


Tensor* PConv2d_BN_ACT(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, const char* act){
  // comming soon...
}


Tensor* DConv2d(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, int32_t stride, int32_t padding){
  // comming soon...
}

Tensor* DConv2d_BN_ACT(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, int32_t stride, int32_t padding, const char* act){
// comming soon...
}
