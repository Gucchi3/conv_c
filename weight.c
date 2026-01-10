#include "Tensor.h"
#include "weight.h"

/*----------------------------------------------
weight.c：レイヤーの重みをW_Tensor型に配置する場所

explanation：
　ここで、weight.hで定義された重み配列を、.data = 「配列名」の形で定義したW_Tensorを作成してください。
　これは、main.cにて、「Patch_embedding(feat1, &patch_weight);」　このように引数で渡されます。

作成例：
  W_Tensor patch_weight = {
    .OC = 3,
    .INC = 3,
    .H = 4,
    .W = 4,
    .data = patch_embed_weight　(<-weight.hで定義されている配列名)
  };

------------------------------------------------*/





W_Tensor patch_weight = {
  .OC = 3,
  .INC = 3,
  .H = 3,
  .W = 3,
  .data = patch_embed_weight
};


B_Tensor conv_bias = {
    .OC = 3,
    .data = conv1_bias_data
};

/*----------------------------------------------
Linear Debug Tensors
------------------------------------------------*/
W_Tensor linear_weight = {
  .OC = 2,
  .INC = 3,   // 入力のチャンネル数
  .H = 8,     // 入力の高さ
  .W = 8,     // 入力の幅
  .data = linear_debug_weight_data
};

B_Tensor linear_bias = {
  .OC = 2,
  .data = linear_debug_bias_data
};