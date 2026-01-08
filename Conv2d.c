//*---------------------------------note-----------------------------------------
//*
//*                           next --> add "bias"
//*
//*------------------------------------------------------------------------------



#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Tensor.h"
#include "utils.h"


Tensor* Conv2d(Tensor* input_tensor, W_Tensor* weight_tensor, int stride, int padding){
  // 代入
  int kernel_size = weight_tensor->H; int output_channel = weight_tensor->OC;
  // 引数例外処理
  if (!input_tensor || kernel_size <= 0 || output_channel <= 0 || stride <= 0 || padding < 0) {
    printf("Error：----- 引数が足りないか、Tensorが空です。 -----\n");
    //free_Tensor(input_tensor);
    return NULL;
  }
  // 出力Tensor形状計算
  int output_tensor_H = ((input_tensor->H + 2 * padding - kernel_size) / stride) + 1;
  int output_tensor_W = ((input_tensor->W + 2 * padding - kernel_size) / stride) + 1;
  int output_tensor_C = output_channel;
  // 計算結果が有効かチェック
  if (output_tensor_H <= 0 || output_tensor_W <= 0) {
    printf("Error：----- カーネルサイズが入力サイズを超えています。 -----\n");
    return NULL;
  }
  // 出力Tensor作成・メモリ確保 -> NULL確認
  Tensor* output_tensor = make_Tensor(output_tensor_H, output_tensor_W, output_tensor_C);
  if (!output_tensor){
    printf("Error：----- 出力Tensorのメモリ確保に失敗しました。 -----\n");
    //free_Tensor(input_tensor);
    return NULL;
  }

  //Conv2d
  const float* weight_head = weight_tensor->data;
  float* output_point = output_tensor->data;


  for(int h=0; h < output_tensor->H; h++){
    for(int w=0; w < output_tensor->W; w++){
      // 重みの先頭アドレス格納
      const float* weight_point = weight_head;
// カーネル左上の画像内位置計算
      int in_lt_h = (h * stride) - padding;
      int in_lt_w = (w * stride) - padding; 
      for(int oc=0; oc < output_tensor->C; oc++){
        // 重み初期化
        float sum = 0.0f;
        for(int kh=0; kh < kernel_size; kh++){
          for(int kw=0; kw < kernel_size; kw++){
            // 計算座標計算
            int in_h = in_lt_h + kh;
            int in_w = in_lt_w + kw;
            // paddingか画像か判定 -> paddingならアドレス移動してkcループをスキップ
            if(in_h < 0 || in_h >= input_tensor->H || in_w < 0 || in_w >= input_tensor->W){
              weight_point += input_tensor->C;
              continue;
            }
            // 対象画像座標があるアドレス計算
            int in_idx = ((in_h * input_tensor->W) + in_w) * input_tensor->C;
            const float* in_point = &input_tensor->data[in_idx];
            for(int kc=0; kc < input_tensor->C; kc++){
              sum += *in_point++ * *weight_point++;
            }
          }
        }
        *output_point++ = sum;
      }
    }
  }



  // return
  return output_tensor ;
}

