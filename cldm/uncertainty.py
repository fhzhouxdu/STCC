import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

# class UncertaintyLoss(nn.Module):
#     def __init__(self, num=2):
#         super().__init__()
#         sigma = torch.zeros(num, requires_grad=True)
#         self.sigma = torch.nn.Parameter(sigma)
        
#     def forward(self, x):
#         loss_sum = 0.
#         for i, [y_true, y_pred] in enumerate(x):
#             loss = F.mse_loss(y_true, y_pred)
#             pre = torch.exp(-self.sigma[i])
#             loss_sum += pre*loss+self.sigma[i]
#         return loss_sum

def UncertaintyLoss(y):
    loss_sum = 0.
    for i, [y_true, y_pred, theta] in enumerate(y):
        loss_sum += torch.mean((y_true - y_pred) ** 2 / (2 * torch.exp(theta)) + theta)
    return loss_sum
        
class ReconstModel(nn.Module):
    def __init__(self, num_blocks_en=2, num_blocks_de=2):
        super().__init__()
        self.encoder = nn.Sequential(*[
            Block(
                dim=1024,
                num_heads=8,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(num_blocks_en)]).to('cuda')
        
        self.decoder = nn.Sequential(*[
            Block(
                dim=1024,
                num_heads=8,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(num_blocks_de)]).to('cuda')
        
        self.mu = nn.Sequential(*[
            Block(
                dim=1024,
                num_heads=8,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )]).to('cuda')
        
        self.theta = nn.Sequential(*[
            Block(
                dim=1024,
                num_heads=8,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            ),
            nn.ReLU()]).to('cuda')
            
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.decoder(x)
        mu = self.mu(x)
        theta = self.theta(x)
        return mu, -theta
        
class UncertaintyWeigh(nn.Module):
    def __init__(self, num=2):
        super().__init__()
        self.reconst_model = nn.Sequential(*[
            ReconstModel() 
            for i in range(num)]).to('cuda')
        self.norm = nn.AdaptiveAvgPool2d((1, 1024))
        self.linear = nn.Sequential(*[
            nn.Linear(1024,1)
            for i in range(num)]).to('cuda')
        
    def forward(self, *x):
        y = []
        w = []
        for i, y_true in enumerate(x):
            y_pred, theta = self.reconst_model[i](y_true)
            y.append([y_true, y_pred, theta])
            w.append(self.linear[i](self.norm(torch.sqrt(torch.exp(theta))).squeeze(1)))
        loss = UncertaintyLoss(y)
        w = torch.cat(w, dim=1)
        w = F.softmax(w, dim=1)
        w = w.unsqueeze(2).unsqueeze(3)
        return loss, w
        
    