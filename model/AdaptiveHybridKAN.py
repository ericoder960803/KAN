class AdaptiveHybridKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.taylor = TaylorLinear(in_dim, out_dim)
        self.spline = KANLinear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, out_dim), nn.Sigmoid())
    def forward(self, x):
        t, s, a = self.taylor(x), self.spline(x), self.gate(x)
        return a * t + (1 - a) * s