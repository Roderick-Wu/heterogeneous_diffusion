import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# mnist is 28*28
# patch image into 4*4 patches -> 7*7 patches

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# adaptive layer norm
class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ada_lin = nn.Sequential(
            nn.SiLU(), # dont ened linear, remember timestep embedding already went through a small mlp
            nn.Linear(hidden_dim, 4 * hidden_dim)
        )

    def forward(self, x, t_emb):
        # t_emb shape: (B, hidden_dim)
        shift1, scale1, shift2, scale2 = self.ada_lin(t_emb).chunk(4, dim=-1)
        
        # Expand condition to sequence length: (B, 1, hidden_dim)
        shift1, scale1 = shift1.unsqueeze(1), scale1.unsqueeze(1)
        shift2, scale2 = shift2.unsqueeze(1), scale2.unsqueeze(1)

        # Attention block with AdaLN
        normed_x = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + attn_out

        # MLP block with AdaLN
        normed_x = self.norm2(x) * (1 + scale2) + shift2
        mlp_out = self.mlp(normed_x)
        x = x + mlp_out
        
        return x

class DiT(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, img_size=28, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # learned position embedding -- this would be (1, 49, hidden_dim) for 7*7 patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim))
        
        # timestep embedding -- gemini recommended this actually (traditional sinusoid through a small mlp)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Output unpatchify head
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, in_channels * patch_size * patch_size)

    def forward(self, x, t):
        B, C, H, W = x.shape
        
        # Patchify and Embed: (B, C, H, W) -> (B, hidden_dim, H/P, W/P) -> (B, L, hidden_dim)
        # For 28x28 with patch size 4, this results in 7x7=49 patches, so L=49
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Transformer
        for block in self.blocks:
            x = block(x, t_emb)
            
        # Output head
        x = self.norm_out(x)
        x = self.head(x) # (B, L, patch_size^2 * C)
        
        # Unpatchify back to image shape
        P = self.patch_size
        Grid = H // P
        x = x.view(B, Grid, Grid, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        
        return x


