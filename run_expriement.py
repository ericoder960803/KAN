import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
from training_module import train_with_val_safe
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import timm



class TaylorLinear(nn.Module):
    def __init__(self, in_f, out_f, order=3):
        super().__init__()
        self.order = order
        self.coeffs = nn.Parameter(torch.randn(out_f, in_f, order + 1) * 0.02)
    def forward(self, x):
        powers = torch.arange(self.order + 1, device=x.device).float()
        x_pow = torch.pow(x.unsqueeze(-1), powers)
        return torch.einsum("...ik,oik->...o", x_pow, self.coeffs)

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        grid = torch.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.02)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def compute_bsplines(self, x):
        x_uns = x.unsqueeze(-1)
        value = (x_uns >= self.grid[:-1]) & (x_uns < self.grid[1:])
        value = value.float()
        for k in range(1, self.spline_order + 1):
            left = (x_uns - self.grid[: -(k + 1)]) / (self.grid[k:-1] - self.grid[: -(k + 1)])
            right = (self.grid[k + 1 :] - x_uns) / (self.grid[k + 1 :] - self.grid[1:-k])
            value = left * value[..., :-1] + right * value[..., 1:]
        return value

    def forward(self, x):
        base_output = torch.einsum("...i,oi->...o", F.silu(x), self.base_weight)
        bases = self.compute_bsplines(x)
        spline_output = torch.einsum("...ik,oik->...o", bases, self.spline_weight)
        return base_output + spline_output

class AdaptiveHybridKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.taylor = TaylorLinear(in_dim, out_dim)
        self.spline = KANLinear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, out_dim), nn.Sigmoid())
    def forward(self, x):
        t, s, a = self.taylor(x), self.spline(x), self.gate(x)
        return a * t + (1 - a) * s



class KANGlobalPyramidInteraction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.interaction_kan = AdaptiveHybridKANLayer(d_model * 2, d_model)
        self.scale_fusion = nn.Sequential(nn.Linear(21 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B, L, D = x.shape
        H = W = int(L**0.5)
        x_2d = x.transpose(1, 2).reshape(B, D, H, W)
        p1 = F.adaptive_avg_pool2d(x_2d, (1, 1)).reshape(B, -1)
        p2 = F.adaptive_avg_pool2d(x_2d, (2, 2)).reshape(B, -1)
        p3 = F.adaptive_avg_pool2d(x_2d, (4, 4)).reshape(B, -1)
        soul = self.scale_fusion(torch.cat([p1, p2, p3], dim=1)).unsqueeze(1)
        return self.norm(x + self.interaction_kan(torch.cat([x, soul.expand(-1, L, -1)], dim=-1)))

class Native_KAN_CV_Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.local = AdaptiveHybridKANLayer(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.pyramid = KANGlobalPyramidInteraction(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.local(self.ln1(x)))
        x = x + self.dropout(self.pyramid(self.ln2(x)))
        return x

class Native_KAN_CV(nn.Module):
    def __init__(self, d_model=128, depth=4, num_classes=100):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList([Native_KAN_CV_Block(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for block in self.blocks: x = block(x)
        return self.head(self.norm(x.mean(1)))



def get_dataloaders(batch_size=128):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=tf_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=tf_test)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size)

def plot_final_results(histories):
    plt.figure(figsize=(10, 6))
    for name, h in histories.items():
        plt.plot(h['test_acc'], label=f'{name} (Best: {max(h["test_acc"]):.4f})')
    plt.axhline(y=0.445, color='gray', linestyle=':', label='ViT Baseline')
    plt.title("CIFAR-100: KAN vs ViT Performance")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.show()



train_loader, test_loader = get_dataloaders()


contestants = {
    "ViT_Tiny": timm.create_model('vit_tiny_patch4_32', num_classes=100, embed_dim=128, depth=4, num_heads=4),
    "Native_KAN_Gold(128)": Native_KAN_CV(d_model=128, depth=4),
    "Native_KAN_Middle(63)":Native_KAN_CV(d_model=64,depth=4),
    "Native_KAN_Slim(48)": Native_KAN_CV(d_model=48,depth=4),
    "Native_KAN_Tiny(32)": Native_KAN_CV(d_model=24,depth=4),
}


results = train_with_val_safe(contestants, train_loader, test_loader, epochs=30)

# 4. 生成圖表
plot_final_results(results)