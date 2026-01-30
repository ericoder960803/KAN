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