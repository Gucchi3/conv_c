#ifndef WEIGHT_H
#define WEIGHT_H

// ... (patch_embed_weightなどは変更なし) ...

static const float patch_embed_weight[81] = {
    // ... (既存のデータそのまま) ...
    // 行0 (Top)
    0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,
    // 行1 (Center) -> 真ん中のInput Ch0だけ 1.0
    0.0f, 0.0f, 0.0f,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,
    // 行2 (Bottom)
    0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,

    // OC=1
    0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,   1.0f, 1.0f, 1.0f,   0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,

    // OC=2
    1.0f, 1.0f, 1.0f,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.0f,   100.0f, 100.0f, 100.0f
};

extern W_Tensor patch_weight;

static const float conv1_bias_data[3] = { 0.1f, -0.5f, 0.01f };
extern B_Tensor conv_bias;


// =================================================================
// Linear Debug Weights (Complex Test)
// =================================================================
// 入力想定: 8x8x3 = 192要素 (値は 0.0, 1.0, ..., 191.0)
// 出力: 2要素
// =================================================================

static const float linear_debug_weight_data[384] = {
    // --- OC 0 : ゼロ和・負数・小数テスト ---
    // Target: Input[10]=10, Input[11]=11, Input[20]=20
    // Calc: (10*0.5) + (11*-0.5) + (20*0.25) 
    //     = 5.0 - 5.5 + 5.0 = 4.5
    [10] = 0.5f,
    [11] = -0.5f,
    [20] = 0.25f,

    // --- OC 1 : 桁の異なる加算テスト ---
    // Offset 192 start
    // Target: Input[100]=100, Input[101]=101
    // Calc: (100 * 0.1) + (101 * 10.0)
    //     = 10.0 + 1010.0 = 1020.0
    [192 + 100] = 0.1f,
    [192 + 101] = 10.0f
};

static const float linear_debug_bias_data[2] = { 
    -4.5f,  // OC0: 4.5 + (-4.5) = 0.0
    5.55f   // OC1: 1020.0 + 5.55 = 1025.55
};

extern W_Tensor linear_weight;
extern B_Tensor linear_bias;

#endif