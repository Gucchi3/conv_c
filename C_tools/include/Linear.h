#include "../../Tensor.h"

/*-----------------------------------------
Linear：ごく一般的な線形層を提供します。
must：input_tensor, weight_tensor, bias_tensor

explanation：
　ごく一般的な線形層です。　
　出力形状はweight_tensorから自動的に計算されます。

//!! 注意 !!//
　//! 入力データ配置順(HWCかCHW)が、重みの配置順と一致しているかを確認してください。
　//! 本リポジトリでは、基本的にすべてHWCで統一してあります。
　//! pytorchでは基本CHWであることに注意してください。
　//! pytorch -> weight.hに展開時には、permute(CHW_to_HWC)関数を提供する予定です。


-------------------------------------------*/
Tensor* Linear(Tensor* input_tensor, W_Tensor* weight_tensor, B_Tensor* bias_tensor);