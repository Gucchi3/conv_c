#ifndef CONV2D_H
#define CONV2D_H

#include "../core/Tensor.h"


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
Tensor* Conv2d(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor,  int32_t stride, int32_t padding);


/* -------------------------------------------------
Conv2d_BN_ACT：畳み込み演算(BN吸収済み)+活性化関数を提供します。
msut：input_tensor, weight_tensor, stride, padding, act

explanation：
　一般的な畳み込み層(BN層吸収済み)+活性化関数を提供します。
　任意のstride, padding, kernel_sizeに対応可能です。
　kernel_sizeはweight_tensorから自動的に取得されます。
　Patch_embedding, pointwise conv にも使用可能です。(pointwise convは別関数制作予定)
　actには使用する活性化関数を指定できます。(act.h参照)

///!! 注意 !!//
　//! 使用できる活性化関数は現状 RELU, RELU6, SiLU のみです。
　//! カーネルは正方形しか保守されません！長方形は保守不可！無理！
　//! 無理なカーネルやストライド等の値を入力すると、正常に動作しない可能性があります！保守不可！
　
　//! この関数はBN層をConb2dの重みに吸収してある前提の関数です。
　//! バッチ正規化の処理は記述されていません。数学的に統合してください！
-----------------------------------------------------*/
Tensor* Conv2d_BN_ACT(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, int32_t stride, int32_t padding, char act);




/*---------------------------------------------------
//! --- 未検証 ---
PConv2d_BN_ACT：Pointwise_Conv+BNを提供します。
must：input_tensor, weight_tensor

explanation：
　Pointwise Convを"HWC"順で提供する。
　Activationはact.hに作成されている物が対応可能。
-----------------------------------------------------*/
Tensor* PConv2d_BN_ACT(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, const char* act);



/*---------------------------------------------------
//! --- 未検証 ---
DConv2d_BN_ACT： Depthwise_Conv+BNを提供します。
must：input_tensor, weight_tensor, stride, padding


explanation：
　Depthwise Convを"HWC"順で提供します。
　任意のstride, paddingに対応可能です。
　Activationはact.hに作成されているものが使用可能です。(act.h参照)

//! 注意 !//
　//! 画像と重みの並びは、HWC, NHWCです。 CHWではありません！
　//! 正規化処理は記述されていません。数学的に統合してください。
　
-----------------------------------------------------*/
Tensor* DConv2d_BN_ACT(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor, int32_t stride, int32_t padding, const char* act);

#endif 
