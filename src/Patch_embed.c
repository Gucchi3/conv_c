//! -----------------------------------------------------------------------------------------------------------------//!
//!                                                                                                                  //!
//!    この「Patch_embedding」関数は、「Conv2d」へと全面的に置き換えられました。（使用不可・保守終了）  Thank you for using...    //!
//!                                                                                                                  //!
//! -----------------------------------------------------------------------------------------------------------------//!



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Tensor.h"
#include "utils.h"


//@note Patch_embedding
Tensor* Patch_embedding(Tensor* input_tensor, W_Tensor* weight_tensor){
  // 代入
  int kernel_size = weight_tensor->H;int output_channel = weight_tensor->OC;
  // 引数例外処理
  if (!input_tensor || kernel_size <= 0 || output_channel <= 0) {
    printf("Error：----- 引数が足りないか、Tensorが空です。 -----\n");
    free_Tensor(input_tensor);
    return NULL;
  }
  // 出力Tensor形状計算
  int output_tensor_H = input_tensor->H / kernel_size;
  int output_tensor_W = input_tensor->W / kernel_size;
  int output_tensor_C = output_channel;
  
  // 出力Tensor作成・メモリ確保 -> NULL確認
  Tensor* output_tensor = make_Tensor(output_tensor_H, output_tensor_W, output_tensor_C);
  if (!output_tensor){
    printf("Error：----- 出力Tensorのメモリ確保に失敗しました。 -----\n");
    free_Tensor(input_tensor);
    return NULL;
  }
  // -------------------------------------------------------------------------------------------------------------
  //@audit paddingを考慮するなら、メモリ番地的にcal_pointを計算するんじゃなくて、H,Wで画像座標ベースのほうが良かったかもね。
  // -------------------------------------------------------------------------------------------------------------
  // Patch_embedding
  int cal_point_lt = 0; int cal_point_c = 0; int cal_point = 0; 
  int w_point_hwc = 0; int w_point_c = 0; int w_point = 0; 
  float sum = 0;
  int output_index = 0;
  
  for(int h=0; h < input_tensor->H/kernel_size; h++){
    for(int w=0; w < input_tensor->W/kernel_size; w++){
      //カーネル内左上に来る画像内の座標位置特定
      cal_point_lt = ((h * kernel_size * input_tensor->W)+(w * kernel_size))*input_tensor->C;
      for(int oc=0; oc< output_tensor->C; oc++ ){
        // sumを0.0にリセット
        sum = 0.0f;
        // 重みのI位置を特定
        w_point_hwc = kernel_size * kernel_size * input_tensor->C * oc;
        for(int kh=0; kh < kernel_size; kh++){
          for(int kw=0; kw < kernel_size; kw++){
            // カーネル内の計算位置の画像内における座標特定
            cal_point_c = cal_point_lt + ((kh * input_tensor->W) + kw) * input_tensor->C;
            // 重みのIHWを特定
            w_point_c = w_point_hwc + ((kernel_size * kh + kw) * input_tensor->C);
            for(int kc=0; kc < input_tensor->C; kc++){
              // CH方向も含めた最終的な計算対象座標計算
              cal_point = cal_point_c + kc;
              // 重みの最終的な座標を計算
              w_point = w_point_c + kc;
              // 乗算 -> 加算
              sum += (input_tensor->data[cal_point] * weight_tensor->data[w_point]);
            }
          }
        }
        // 格納先index計算
        output_index = (((h * output_tensor->W)+w)*output_tensor->C)+oc;
        // 格納
        output_tensor->data[output_index] = sum;
      }
    }
  }

  // return
  return output_tensor;
}

