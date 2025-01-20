import torch
import torchvision.models as models

# 加载预训练的 VGG16 模型
# 缓存在 /Users/wengyang/.cache/torch/hub/checkpoints/vgg16-397923af.pth
vgg16 = models.vgg16(pretrained=True)


# model = models.vgg16(pretrained=False)

# 打印每一层参数的名称和形状，并计算总参数量
def print_model_parameters(model):
    total_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            print(f"Layer: {name}")
            print(f"  Weight shape: {module.weight.shape}")
            total_params += module.weight.numel()
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"  Bias shape: {module.bias.shape}")
            total_params += module.bias.numel()
    return total_params


total_params = print_model_parameters(vgg16)
print(f"Total trainable parameters: {total_params}")


# 计算模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(vgg16)
print(f"VGG16 total params: {num_params}")