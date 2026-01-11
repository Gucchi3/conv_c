#ifndef UTILS_H
#define UTILS_H

#include "Tensor.h"

/* -------------------------------------------------
make_Tensor：Tensor構造体を作成します。
msut：H, W, C

explanation：
　引数のH,W,CをもとにTensor構造体を作成します。
　Tensor構造体の詳細については、Tensor.hを確認してください。
　*data内のデータはこの関数では作成されません。
　ただし、メモリ確保だけは行われます。
　H,W,Cのみ作成され、かつメモリ確保が行われます。
-----------------------------------------------------*/
Tensor* make_Tensor(int32_t H, int32_t W, int32_t C);

/*---------------------------------------------------
make_Tensor_from_array：Tensor構造体の作成及び、*dataに事前に定義してある配列をコピーします。
must：H, W, C, init_data

explanation：
　引数をもとにTensor構造体を作成すると共に、*data領域に事前に定義された配列をコピーして格納します。
　init_dataには、事前に定義してある配列のアドレスを渡す必要があります。
　また、init_dataのサイズは、H*W*Cと一致している必要があります。
　Tensor構造体については、Tensor.hを確認してください。
-----------------------------------------------------*/
Tensor* make_Tensor_from_array(int32_t H, int32_t W, int32_t C, const float* init_data);

/*---------------------------------------------------
free_Tensor：引数のTensor構造体のメモリを開放します。
must：Tensor

explanation：
　引数のTensorは、重みのW_TensorではなくてTensor型である必要があります。
　Tensor->dataの中身と、Tensor構造体自体の両方を開放します。
-----------------------------------------------------*/
void free_Tensor(Tensor* Tensor);

/*---------------------------------------------------
print_Tensor：引数のTensorの形状、アドレス、中身を表示します。
must：Tensor, show_tensor_contents

explanation：
　渡されたTensorのH,W,Cの中身、Tensor構造体の番地、*dataの番地を表示します。
　show_tensor_contentsに0以外の整数を渡すことで、*dataの中身も表示することができます。
　重みのW_Tensorはprint_W_Tensorが使用できます。
-----------------------------------------------------*/
void print_Tensor(Tensor* Tensor, int32_t show_tensor_contents);


/*---------------------------------------------------
print_W_Tensor：引数のW_Tensorの形状、アドレス、中身を表示します。
must：W_Tensor, show_tensor_contents

explanation：
　渡されたTensorのH,W,INC, OCの中身、W_Tensor構造体の番地、*dataの番地を表示します。
　show_tensor_contentsに0以外を渡すことで、*dataの中身を表示できます。
　通常のTensorはprint_Tensorが使用できます。
-----------------------------------------------------*/
void print_W_Tensor(W_Tensor* W_Tensor, int32_t show_tensor_contents);


void print_B_Tensor(B_Tensor* bt, int32_t show_tensor_contents);





#endif
