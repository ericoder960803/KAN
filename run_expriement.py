import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import gc
class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # 標準 Transformer 層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, l = x.shape
        x = self.embedding(x) + self.pos_encoding[:, :l, :]
        x = self.transformer(x)
        return self.head(x)

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


class MemoryEfficientHybridBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # multiheadattention -> adaptivehybridkanlayer
        # A: Attention (處理全域搬運) ---
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # B: Adaptive KAN (處理局部深度變換
        self.norm2 = nn.LayerNorm(d_model)
       
        self.kan_ffn = AdaptiveHybridKANLayer(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
       
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + self.dropout(attn_out)
        
       
        res = x
        x_norm = self.norm2(x)
        
        if self.use_checkpoint and self.training:
            # 關鍵：這裡會釋放 Spline 展開的巨大基函數矩陣
            # 需要在反向傳播時重新計算，但能省下極大 VRAM
            kan_out = checkpoint(self.kan_ffn, x_norm, use_reentrant=False)
        else:
            kan_out = self.kan_ffn(x_norm)
            
        x = res + self.dropout(kan_out)
        return x

class MemoryoptimizedHybridKANLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, depth=4, nhead=8, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # 堆疊混合區塊
        self.blocks = nn.ModuleList([
            MemoryEfficientHybridBlock(d_model, nhead) for _ in range(depth)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, l = x.shape
        x = self.embedding(x) + self.pos_embed[:, :l, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

def get_enwiki_data(batch_size=32, seq_len=128):
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format("torch")
    
    train_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
    return train_loader, len(tokenizer)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters : {total_params / 1e6:.2f} M")
    return total_params

def train_enwiki_final_v3(models_dict, train_loader, tokenizer, epochs=35):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_history = {}
    
    # 設定 Sequence Length (The expriement use 128 )
    print("Start training (d_model=128)")

    for name, model in models_dict.items():
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n" + "="*50)
        print(f"📦 正在優化訓練: {name}")
        print(f"="*50)
        
        model.to(device)
        
        # 加強權重衰減，約束 B-Spline 曲線不要太極端 
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        hist = {'loss': [], 'ppl': [], 'grad_norm': []}
        best_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_grad_norm = 0
            
            for i, batch in enumerate(train_loader):
                inputs = batch['input_ids'].to(device)
                targets = inputs.clone()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                logits = outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1))
                labels = targets[:, 1:].contiguous().view(-1)
                
                loss = criterion(logits, labels)
                loss.backward()
                
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                total_grad_norm += grad_norm.item()
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            avg_grad_norm = total_grad_norm / len(train_loader)
            ppl = math.exp(avg_loss) if avg_loss < 10 else 999.9
            
            scheduler.step()
            curr_lr = optimizer.param_groups[0]['lr']
            
            hist['loss'].append(avg_loss)
            hist['ppl'].append(ppl)
            hist['grad_norm'].append(avg_grad_norm)
            
            print(f"Epoch {epoch+1:02d} | PPL: {ppl:.2f} | Grad: {avg_grad_norm:.4f} | LR: {curr_lr:.2e}")

            model.eval()
            test_prompt = "The science of artificial intelligence is"
            input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated = input_ids
                for _ in range(30): # 預測長一點
                    outputs = model(generated)
                    logits = outputs[:, -1, :] / 0.8 # Temperature
                    
                    # lower the same patteron 
                    for token_id in set(generated[0].tolist()):
                        logits[0, token_id] /= 1.2
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)
                
                print(f"🔮 [隨機採樣預測] -> {tokenizer.decode(generated[0])}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"v6_best_{name}.pth")
        
        all_history[name] = hist
        model.to('cpu')
    
    return all_history

import matplotlib.pyplot as plt

def plot_enwiki_full_results(all_history):
    # 設定圖表風格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Loss 曲線
    for name, h in all_history.items():
        axes[0].plot(h['loss'], label=name, marker='o', markersize=4)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    
    # 2. PPL 曲線 (用對數座標看更精準)
    for name, h in all_history.items():
        axes[1].plot(h['ppl'], label=name, marker='s', markersize=4)
    axes[1].set_title("Perplexity (PPL)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_yscale('log')
    axes[1].legend()
    
    
    for name, h in all_history.items():
        if 'grad_norm' in h:
            axes[2].plot(h['grad_norm'], label=name, marker='^', markersize=4)
            axes[2].set_title("Average Gradient Norm")
        else:
            
            key = 'mem' if 'mem' in h else 'loss'
            axes[2].plot(h[key], label=f"{name} ({key})")
            axes[2].set_title(f"Training {key}")
            
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("KAN_vs_Transformer_Analysis.png")
    plt.show()

#main
train_loader, vocab_size = get_enwiki_data()
enwiki_models = {
    "Transformer_Baseline": BaselineTransformer(vocab_size, d_model=128),
    
    "MemoryoptimizedHybridKANLanguageModel":MemoryoptimizedHybridKANLanguageModel(vocab_size,d_model=128)
}
tokenizer = AutoTokenizer.from_pretrained("gpt2")

enwiki_results = train_enwiki_final_v3(enwiki_models,train_loader,tokenizer, epochs=35)

plot_enwiki_full_results(enwiki_results)
