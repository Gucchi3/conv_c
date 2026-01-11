// Tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

/*-------------------------------------------------------
Tensor構造体：入力画像、特徴マップのための構造体

//* explanation
H：高さ
W：幅
C：チャンネル数
**data：データが入っている初めの番地

基本的にはutils.cで定義されている、make_Tensor 関数にて生成、メモリ確保を行う。
///!dataには「HWC」型の高さ優先で格納される画像もしくは特徴マップの最初の番地が格納される。
また、同じくutils.cのfree_Tensor関数にて、指定したTensorのメモリ開放を行うことができる。

///!画像は必ず「HWC」順にしてください！ 
-----------------------------------------------------------*/
//@note Tensor型の定義
typedef struct {
  int32_t H, W, C;
  float *data;
} Tensor;


/*-----------------------------------------------------------
W_Tensor構造体：重みのための構造体

//*explanation
H：高さ
W：幅
INC：入力CH
OC：出力CH
*data：データが入っている番地
-------------------------------------------------------------*/
typedef struct{
  int32_t OC, INC, H, W;
  const float *data;
} W_Tensor;

/*-----------------------------------------------------------
B_Tensor構造体：バイアスのための構造体

//*explanation
OC：要素数
*data：データが入っている番地
-------------------------------------------------------------*/
typedef struct{
  int32_t OC;
  const float* data;
} B_Tensor;

#endif
