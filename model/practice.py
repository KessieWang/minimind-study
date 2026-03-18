import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):  # dim输入维度、eps极小值
        super().__init__()  
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  
        # nn.Parameter告诉pytorch这是一个可学习参数，训练时请自动更新

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
    def forward(self, x):
        return self.weight *self._norm(x.float()).type_as(x)
    
    # 输入x形状[2, 3, 4] 批次=2，序列长度=3，特征维度=4


