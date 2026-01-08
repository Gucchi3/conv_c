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
  .H = 4,
  .W = 4,
  .data = patch_embed_weight
};