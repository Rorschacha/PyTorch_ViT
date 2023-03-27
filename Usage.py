
import torch
from models import vit




v = vit.ViT(
    image_size = 256, # 其他尺寸：384
    patch_size = 32, # 切出来小片的边长，必须被image_size 整除
    num_classes = 1000, # 分类数量
    dim = 1024, # encoder前 线性层 输出的 矢量的长度
    depth = 6,  # transformer block的数量
    heads = 16, # Number of heads in Multi-head Attention layer
    mlp_dim = 2048,# MLP层缩放参数
    dropout = 0.1,
    emb_dropout = 0.1
)




img = torch.randn(1, 3, 256, 256)



preds = v(img) # (1, 1000)
print(preds)