#include "Tensor.h"
#ifndef CONV2D_H
#define CONV2D_H


/* -------------------------------------------------
Conv2d：畳み込み演算を提供します。
msut：input_tensor, weight_tensor, stride, padding

explanation：
　一般的な畳み込み層を提供します。
　任意のstride, padding, kernel_sizeに対応可能です。
　kernel_sizeはweight_tensorから自動的に取得されます。
　Patch_embedding, pointwise conv にも使用可能です。(pointwise convは別関数制作予定)

///!! 注意 !!//
　//! カーネルは正方形しか保守されません！長方形は保守不可！無理！
　//! 無理なカーネルやストライド等の値を入力すると、正常に動作しない可能性があります！保守不可！
-----------------------------------------------------*/
Tensor* Conv2d(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor,  int stride, int padding);


/* -------------------------------------------------
Conv2d_BN_RELU：畳み込み演算(BN吸収済み)+RELUを提供します。
msut：input_tensor, weight_tensor, stride, padding

explanation：
　一般的な畳み込み層(BN層吸収済み)+RELUを提供します。
　任意のstride, padding, kernel_sizeに対応可能です。
　kernel_sizeはweight_tensorから自動的に取得されます。
　Patch_embedding, pointwise conv にも使用可能です。(pointwise convは別関数制作予定)

///!! 注意 !!//
　//! カーネルは正方形しか保守されません！長方形は保守不可！無理！
　//! 無理なカーネルやストライド等の値を入力すると、正常に動作しない可能性があります！保守不可！
　
　//! この関数はBN層をConb2dの重みに吸収してある前提の関数です。
-----------------------------------------------------*/
Tensor* Conv2d_BN_RELU(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, int stride, int padding);

#endif 