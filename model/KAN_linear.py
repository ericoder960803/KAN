import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 1. 網格 (Grid)：釘子的位置，使用 register_buffer 確保它不計算梯度但跟隨模型移動
        grid = torch.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)

        # 2.(Spline Weight)：對應權重 c，形狀為 (out, in, j)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.1
        )

        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # 4. 剪枝遮罩 (Mask)：1 代表連線存在，0 代表已剪枝
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def compute_efficient_bsplines(self, x):
        """
        核心『磨圓機器』：將輸入 x 轉換為 B-spline 亮度矩陣 (Bases)
        """
        x = x.unsqueeze(-1) # (batch, in, 1)
        grid = self.grid

        # k=0: 方塊山
        value = (x >= grid[:-1]) & (x < grid[1:])
        value = value.float()

        # k=1~spline_order: 遞迴升級 (1-abs 三角形到平滑山丘)
        for k in range(1, self.spline_order + 1):
            left_slope = (x - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)])
            right_slope = (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:-k])
            value = left_slope * value[..., :-1] + right_slope * value[..., 1:]

        return value # 返回 (batch, in, j)

    def forward(self, x):
        # A. 基礎線性路徑 (Base Path)
        # 套用 mask 確保剪枝後的連線不發揮作用
        base_output = F.linear(F.silu(x), self.base_weight * self.mask)

        # B. 曲線路徑 (Spline Path)
        bases = self.compute_efficient_bsplines(x) # (b, i, j)

        # 使用 einsum 進行『大熔爐』加總：bij, oij -> bo
        # 同時考慮 mask
        active_weight = self.spline_weight * self.mask.unsqueeze(-1)
        #spline_output = torch.einsum("bij,oij->bo", bases, active_weight)
        spline_output = torch.einsum("...ij,oij->...o", bases, active_weight)

        return base_output + spline_output

    @torch.no_grad()
    def get_influence_matrix(self, x):
        """
        計算論文中的樣本平均強度 mu (phi_ij 的絕對值平均)
        """
        bases = self.compute_efficient_bsplines(x)
        # 這裡用 boi 標籤，不對 i 加總，為了看每條連線的表現
        # 這邊在一些狀況可能有 bug
        phi_val = torch.einsum("bij,oij->boi", bases, self.spline_weight)
        # 對 batch 維度取平均，對應公式中的 1/N * sum(|phi(x)|)
        influence = phi_val.abs().mean(dim=0)
        return influence

    def prune(self, x, threshold=0.01):
        """
        執行剪枝：強度低於門檻的連線將被設為 0
        """
        influence = self.get_influence_matrix(x)
        self.mask = (influence > threshold).float()
        print(f"Pruning done! Active connections: {int(self.mask.sum())}/{self.mask.numel()}")

    def plot_influence(self, x):
        """
        視覺化連線強度熱力圖
        """
        influence = self.get_influence_matrix(x).cpu().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(influence, annot=True, cmap="YlGnBu", fmt=".4f")
        plt.title("Connection Influence (mu)")
        plt.xlabel("Input Features")
        plt.ylabel("Output Nodes")
        plt.show()