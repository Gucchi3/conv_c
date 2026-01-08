// Tensor.h
#ifndef TENSOR_H
#define TENSOR_H

//@note Tensor型の定義
typedef struct {
  int H, W, C;
  float *data;
} Tensor;

#endif



/*
//* explanation
H：高さ
W：幅
C：チャンネル数
**data：データが入っている初めの番地

基本的にはutils.cで定義されている、make_Tensor 関数にて生成、メモリ確保を行う。
**dataには「HWC」型の高さ優先で格納される画像もしくは特徴マップの最初の番地が格納される。

また、同じくutils.cのfree_Tensor関数にて、指定したTensorのメモリ開放を行うことができる。
*/

