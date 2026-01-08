#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Tensor.h"




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
  printf("メモリ確保成功...\n");
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
  printf("メモリ確保成功...\n");
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
void print_Tensor(Tensor* t, int show_tensor_contents){
  if (!t){
    printf("Error: Tensor is NULL.\n");
    return;
  }

  printf("=== Tensor Structure ===\n");
  printf("  Address : %p\n", (void*)t);
  printf("  Data    : %p\n", (void*)t->data);
  printf("  Shape   : H=%d, W=%d, C=%d\n", t->H, t->W, t->C);

  if(show_tensor_contents){
    printf("--- Data Content ---\n");
    int total_pixels = t->H * t->W;
    
    // データが多すぎる場合の安全策（必要ならコメントアウト解除）
    // if (total_pixels > 200) { printf(" (Too large to print all...)\n"); return; }

    for(int h = 0; h < t->H; h++){
      printf("Row %2d: ", h); // 行番号表示
      
      for(int w = 0; w < t->W; w++){
        printf("[");
        for(int c = 0; c < t->C; c++){
          // index計算
          int idx = ((h * t->W) + w) * t->C + c;
          
          // 値表示 (7.2f で桁を揃える)
          printf("%7.2f", t->data[idx]);
          
          // 最後のチャンネル以外はカンマを入れる
          if(c < t->C - 1) printf(",");
        }
        printf("] ");
      }
      printf("\n"); // W方向が終わったら改行
    }
    printf("========================\n\n");
  }
}

//@note print_WTensor 関数
void print_W_Tensor(W_Tensor* wt, int show_tensor_contents){
  if (!wt){
    printf("Error: W_Tensor is NULL.\n");
    return;
  }

  printf("=== W_Tensor Structure ===\n");
  printf("  Address : %p\n", (void*)wt);
  printf("  Data    : %p\n", (void*)wt->data);
  printf("  Shape   : OC=%d, INC=%d, H=%d, W=%d\n", wt->OC, wt->INC, wt->H, wt->W);

  if(show_tensor_contents){
    printf("--- Data Content (Format: [INC0, INC1...]) ---\n");
    
    // データ順序: [OC][H][W][INC]
    
    for(int oc = 0; oc < wt->OC; oc++){
      printf("Filter %2d (OC=%d):\n", oc, oc); // フィルタごとの見出し
      
      for(int h = 0; h < wt->H; h++){
        printf("  Row %2d: ", h); // カーネルの行番号
        
        for(int w = 0; w < wt->W; w++){
          printf("[");
          for(int inc = 0; inc < wt->INC; inc++){
            // 4次元インデックス計算
            // index = oc*(H*W*INC) + h*(W*INC) + w*(INC) + inc
            int idx = (oc * wt->H * wt->W * wt->INC) + 
                      (h  * wt->W * wt->INC) + 
                      (w  * wt->INC) + inc;

            printf("%6.2f", wt->data[idx]);

            // カンマ区切り（最後以外）
            if(inc < wt->INC - 1) printf(",");
          }
          printf("] ");
        }
        printf("\n"); 
      }
      printf("--------------------------------\n"); // フィルタ区切り線
    }
  }
}

//@note print_B_Tensor 関数
void print_B_Tensor(B_Tensor* bt, int show_tensor_contents){
  if (!bt){
    printf("B_Tensor is NULL (No Bias)\n");
    return;
  }
  printf("=== B_Tensor Structure ===\n");
  printf("  Shape : OC=%d\n", bt->OC);
  if(show_tensor_contents){
    printf("  Data  : [ ");
    for(int i=0; i < bt->OC; i++){
        printf("%6.2f", bt->data[i]);
        if(i < bt->OC - 1) printf(", ");
    }
  }
  printf(" ]\n\n");
}