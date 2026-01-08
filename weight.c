#include "weight.h"




W_Tensor patch_weight = {
  .OC = 3,
  .INC = 3,
  .H = 4,
  .W = 4,
  .data = patch_embed_weight
};