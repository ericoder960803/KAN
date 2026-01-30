class TaylorLinear(nn.Module):
    def __init__(self, in_features, out_features, order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        # 形狀 (out, in, order + 1)
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, order + 1) * 0.1)

    def forward(self, x):
        device = x.device
        # 1. 產生多項式基底，保持跟 x 一樣的前面維度
        # x shape: (..., in_features)
        powers = torch.arange(self.order + 1, device=device).float()
        
        # 使用 unsqueeze(-1) 讓 x 變成 (..., in, 1)，然後 pow 變成 (..., in, order+1)
        x_pow = torch.pow(x.unsqueeze(-1), powers) 

        # 2. einsum 核心運算：使用 '...' 自動適應前面的所有維度 (Batch, Seq, Head 等)
        # ... : 代表前面所有的維度
        # i : input_features, k : order+1, o : out_features
        
        out = torch.einsum("...ik,oik->...o", x_pow, self.coeffs)
        
        return out