import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import math

# -----------------------
# 1. Sinusoidal timestep embedding
# -----------------------


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

# -----------------------
# 2. Residual block with timestep conditioning
# -----------------------


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.SiLU()
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv2d(
            in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        tb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + tb
        h = self.act(self.conv2(h))
        return h + self.res_conv(x)

# -----------------------
# 3. Downsample / Upsample (width-only pooling)
# -----------------------


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        # only upsample width dimension
        x = F.interpolate(x, scale_factor=(1, 2), mode='nearest')
        return self.conv(x)

# -----------------------
# 4. U-Net model for [1×5×72] data with width-downsampling only
# -----------------------


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128):
        super().__init__()
        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        # down blocks
        self.down1 = ResBlock(in_ch, base_ch, time_emb_dim)
        self.down2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down3 = ResBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        # mid blocks
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        # up blocks
        self.up1 = Upsample(base_ch * 4)
        self.res3 = ResBlock(base_ch * 8, base_ch * 2, time_emb_dim)
        self.up2 = Upsample(base_ch * 2)
        self.res2 = ResBlock(base_ch * 4, base_ch, time_emb_dim)
        self.up3 = Upsample(base_ch)
        self.res1 = ResBlock(base_ch * 2, base_ch, time_emb_dim)
        # final projection
        self.final = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        # downsample only width dimension
        d1 = self.down1(x, t_emb)
        d2 = self.down2(F.avg_pool2d(d1, (1, 2)), t_emb)
        d3 = self.down3(F.avg_pool2d(d2, (1, 2)), t_emb)
        # mid (width-pool once more)
        m = self.mid2(self.mid1(F.avg_pool2d(d3, (1, 2)), t_emb), t_emb)
        # upsample only width
        u1 = self.up1(m)
        u1 = self.res3(torch.cat([u1, d3], dim=1), t_emb)
        u2 = self.up2(u1)
        u2 = self.res2(torch.cat([u2, d2], dim=1), t_emb)
        u3 = self.up3(u2)
        u3 = self.res1(torch.cat([u3, d1], dim=1), t_emb)
        return self.final(u3)

# -----------------------
# 5. Cosine noise schedule
# -----------------------


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# -----------------------
# 6. Prepare schedules
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000
betas = cosine_beta_schedule(T).to(device)
alphas = 1 - betas
alpha_prod = torch.cumprod(alphas, dim=0)
sqrt_alpha_prod = torch.sqrt(alpha_prod)
sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_prod)

# -----------------------
# 7. Forward noising q_sample for 4D tensors [B,C,H,W]
# -----------------------


@torch.no_grad()
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    coeff_x = sqrt_alpha_prod[t].view(-1, 1, 1, 1)
    coeff_n = sqrt_one_minus_alpha_prod[t].view(-1, 1, 1, 1)
    return coeff_x * x0 + coeff_n * noise, noise

# rest unchanged ...
