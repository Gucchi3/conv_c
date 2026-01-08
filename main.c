#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "input.h"
#include "Patch_embed.h"
#include "weight.h"

//@note main 関数
int main(void){

  //　Tensorを作成してみる。
  Tensor* feat1 = make_Tensor_from_array(8,8,3,input_data);
  // Tensor情報を表示
  print_Tensor(feat1,1);
  print_W_Tensor(&patch_weight, 1);
  //patch_embed
  Tensor* feat2 = Patch_embedding(feat1, &patch_weight);
  // feat1の開放
  free_Tensor(feat1);
  //feat2をprintf
  print_Tensor(feat2, 1);



  // 解放
  free_Tensor(feat2);
  return 0;
}






















































