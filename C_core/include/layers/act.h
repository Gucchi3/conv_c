#ifndef ACT_H
#define ACT_H

#include "../core/Tensor.h"



/*----------------------------------------------------
RELU：RELU関数
must：input_tensor

explanation：
　Tensor->dataの0未満のものを0にクリップする。
------------------------------------------------------*/
Tensor* RELU(Tensor* input_tensor);


/*----------------------------------------------------
RELU6：RELU6関数
must：input_tensor

explanation：
　Tensor->dataの0未満のものを0に、6以上のものを6にクリップする。
------------------------------------------------------*/
Tensor* RELU6(Tensor* input_tensor);



/*----------------------------------------------------
SiLU：SiLU関数
must：input_tensor

explanation：
x = x / (1 + e^(-x)) を提供します。
------------------------------------------------------*/
Tensor* SiLU(Tensor* input_tensor);

#endif