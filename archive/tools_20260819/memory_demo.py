import torch
import torch.nn as nn
import gc

def get_memory_usage():
    """获取当前GPU显存使用量（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 2000)
        self.layer2 = nn.Linear(2000, 2000)
        self.layer3 = nn.Linear(2000, 2000)
        self.layer4 = nn.Linear(2000, 1000)
        self.layer5 = nn.Linear(1000, 10)
        
    def forward(self, x):
        print(f"Input: {get_memory_usage():.2f} MB")
        
        x = torch.relu(self.layer1(x))
        print(f"After layer1: {get_memory_usage():.2f} MB")
        
        x = torch.relu(self.layer2(x))
        print(f"After layer2: {get_memory_usage():.2f} MB")
        
        x = torch.relu(self.layer3(x))
        print(f"After layer3: {get_memory_usage():.2f} MB")
        
        x = torch.relu(self.layer4(x))
        print(f"After layer4: {get_memory_usage():.2f} MB")
        
        x = self.layer5(x)
        print(f"After layer5: {get_memory_usage():.2f} MB")
        
        return x

def demonstrate_backward_memory():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和数据
    model = SimpleModel().to(device)
    batch_size = 32
    input_data = torch.randn(batch_size, 1000, device=device, requires_grad=True)
    target = torch.randint(0, 10, (batch_size,), device=device)
    
    print("=== 前向传播 ===")
    torch.cuda.empty_cache()
    print(f"开始前向传播: {get_memory_usage():.2f} MB")
    
    output = model(input_data)
    print(f"前向传播完成: {get_memory_usage():.2f} MB")
    
    loss = nn.CrossEntropyLoss()(output, target)
    print(f"计算loss后: {get_memory_usage():.2f} MB")
    
    print("\n=== 反向传播 ===")
    print(f"开始反向传播: {get_memory_usage():.2f} MB")
    
    # 使用hook来监控每层梯度计算完成后的显存变化
    def backward_hook(module, grad_input, grad_output):
        print(f"梯度计算完成 - {module.__class__.__name__}: {get_memory_usage():.2f} MB")
    
    # 注册hook
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_backward_hook(backward_hook)
            hooks.append(hook)
    
    loss.backward()
    print(f"反向传播完成: {get_memory_usage():.2f} MB")
    
    # 清理hooks
    for hook in hooks:
        hook.remove()
    
    print(f"清理计算图后: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    demonstrate_backward_memory() 