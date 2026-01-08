#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Tensor.h"
#include "weight.h"



//@note make_Tensor 関数
Tensor* make_Tensor(int H,int W,int C){
  // 特徴マップの構造体を定義
  Tensor *feat = (Tensor*)malloc(sizeof(Tensor)); 
  // 確保したTensorがNULLでないか確認　
  if (!feat) return NULL;
  // Tensorに中身を付与
  feat->H = H; feat->W = W; feat->C = C; 
  // H*W*Cサイズのメモリ確保かつ、それをdataに格納、callocで0に初期化
  feat->data = (float*)calloc(H*W*C, sizeof(float));
  // callocのNULL確認、Trueの場合、先にfeatを解放してからreturn
  if (!feat->data){free(feat); return NULL;};
  printf("メモリ確保に成功...\n");
  // return
  return feat;
}


//@note make_Tensor_from_array 関数
Tensor* make_Tensor_from_array(int H, int W, int C, const float* init_data){
  // 入力画像データの構造体を定義
  Tensor* image = (Tensor*)malloc(sizeof(Tensor));
  // 確保したTensorがNULLでないか確認
  if (!image) return NULL;
  // Tensorに中身(H, W, C)を付与
  image->H = H; image->W = W; image->C = C;
  // 画像分のメモリを確保
  image->data = (float*)malloc(H*W*C*sizeof(float));
  // NULLかどうか確認
  if (!image->data){free(image); return NULL;}
  printf("メモリ確保に成功...\n");
  // init_dataに格納されている画像データを、image->dataにコピー
  if (init_data){
    memcpy(image->data, init_data, H*W*C*sizeof(float));
  }else{
    printf("Error：----- init_dataがNULLです。 -----");
  }
  //return
  return image;
}

//@note Tensor_free 関数
void free_Tensor(Tensor* Tensor){
  // メモリ開放
if(!Tensor){printf("----- Tensorが空です。 -----"); return;}
free(Tensor->data);
free(Tensor);
printf("メモリ開放完了...\n");
// return
return;
}


//@note print_Tensor 関数
void print_Tensor(Tensor* Tensor, int show_tensor_contents){
  if (Tensor){
    printf("Tensor structure ->>\n");
    printf("H=%d, W=%d, C=%d\n", Tensor->H, Tensor->W, Tensor->C);
    printf("Tensor：%p\n",(void*)Tensor);
    printf("Tensor->data：%p\n", (void*)Tensor->data);
    if(show_tensor_contents){
      int total = Tensor->H * Tensor->W * Tensor->C;
      for(int i = 0; i < total; ++i){
        printf("%6.2f", Tensor->data[i]);
        // 区切り: チャンネル末尾でスペース、行末で改行
        if ((i + 1) % Tensor->C == 0) printf(" ");
        if ((i + 1) % (Tensor->W * Tensor->C) == 0) printf("\n");
      }
      printf("\n");
    }
  }else{
    printf("Error：----- Tensor引数が空です。 -----");
  }
  // return
  return;
}


//@note print_WTensor 関数
void print_W_Tensor(W_Tensor* Tensor, int show_tensor_contents){
  if (Tensor){
    printf("W_Tensor structure ->>\n");
    printf("OC=%d, INC=%d, H=%d, W=%d\n", Tensor->OC, Tensor->INC, Tensor->H, Tensor->W);
    printf("W_Tensor：%p\n",(void*)Tensor);
    printf("W_Tensor->data：%p\n", (void*)Tensor->data);
    if(show_tensor_contents){
      // 総要素数計算
      int total = Tensor->OC * Tensor->INC * Tensor->H * Tensor->W;   
      // データの並び順: [OC][H][W][INC] を想定
      for(int i = 0; i < total; ++i){
        printf("%6.2f", Tensor->data[i]);     
        // 区切り1: 入力チャンネル(INC)末尾でスペース
        if ((i + 1) % Tensor->INC == 0) printf(" ");
        // 区切り2: カーネルの1行(W * INC)末尾で改行
        if ((i + 1) % (Tensor->W * Tensor->INC) == 0) printf("\n");
        // 区切り3: 1つのフィルタ(OC)が終わるごとに空行を追加して見やすくする
        if ((i + 1) % (Tensor->H * Tensor->W * Tensor->INC) == 0 && (i + 1) < total) {
            printf("----------------\n"); 
        }
      }
      printf("\n");
    }
  }else{
    printf("Error：----- W_Tensor引数が空です。 -----\n");
  }
  // return
  return;
}