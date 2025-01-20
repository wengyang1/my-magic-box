import torch
import torchvision.models as models
from thop import profile

# 加载预训练的VGG16模型
model = models.vgg16(pretrained=True)

# 计算FLOPs
# thop.profile函数会同时返回模型的FLOPs和参数量
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
flops = flops / 1e9  # GFlops Giga=10亿
print("FLOPs:", flops)
print("Parameters:", params)
