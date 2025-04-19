import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import math
import pandas as pd
import argparse
from scripts.clf import train_clf, evaluate_clf
from scripts.gan import train_gan, train_labeledgan, gen_synthetic, gen_synthetic_labeledgan, save_gan
from scripts.data_loader import get_dataloader, get_windows
from scripts.eval_dist import calculate_metrics
from sklearn.preprocessing import MinMaxScaler


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
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


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




class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        # only upsample width dimension
        x = F.interpolate(x, scale_factor=(1, 2), mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=128):
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
        # x = torch.unsqueeze(x, dim=1)
        t_emb = self.time_mlp(t)
        # downsample only width dimension
        d1 = self.down1(x, t_emb)
        d2 = self.down2(F.avg_pool2d(d1, (1, 2)), t_emb)
        d3 = self.down3(F.avg_pool2d(d2, (1, 2)), t_emb)
        # mid (width-pool once more)
        m = self.mid2(self.mid1(F.avg_pool2d(d3, (1, 2)), t_emb), t_emb)
        # upsample only width
        u1 = self.up1(m)
        # import pdb;pdb.set_trace()
        u1 = self.res3(torch.cat([u1, d3], dim=1), t_emb)
        u2 = self.up2(u1)
        u2 = self.res2(torch.cat([u2, d2], dim=1), t_emb)
        u3 = self.up3(u2)
        u3 = self.res1(torch.cat([u3, d1], dim=1), t_emb)
        return self.final(u3)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


device = "cuda" if torch.cuda.is_available() else "cpu"

T = 1000
betas = cosine_beta_schedule(T)
alphas = 1 - betas
alpha_prod = torch.cumprod(alphas, dim=0)
sqrt_alpha_prod = torch.sqrt(alpha_prod)
sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_prod)

# after computing betas, alphas, etc.
betas = betas.to(device)
alphas = alphas.to(device)
alpha_prod = alpha_prod.to(device)
sqrt_alpha_prod = sqrt_alpha_prod.to(device)
sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(device)


@torch.no_grad()
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    return sqrt_alpha_prod[t, None, None, None] * x0 + sqrt_one_minus_alpha_prod[t, None, None, None] * noise, noise

# @torch.no_grad()
# def q_sample(x0, t, noise=None):
#     if noise is None:
#         noise = torch.randn_like(x0)
#     # sqrt_alpha_prod[t] has shape [B]
#     # [None, None] → broadcast over H and W
#     coef_x = sqrt_alpha_prod[t, None, None]
#     coef_n = sqrt_one_minus_alpha_prod[t, None, None]
#     x_t = coef_x * x0 + coef_n * noise
#     return x_t, noise


@torch.no_grad()
def p_sample_loop(model, shape, device):
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_tensor)
        alpha_t = alphas[t]
        alpha_prod_t = alpha_prod[t]
        beta_t = betas[t]

        coef1 = 1 / torch.sqrt(alphas[t])
        coef2 = (1 - alphas[t]) / sqrt_one_minus_alpha_prod[t]
        mean = coef1 * (x - coef2 * eps_pred)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean
    return x


def get_dataset(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    label = df['Label'].values
    df.drop(columns='Label', inplace=True)

    return df, label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--window', type=int, default=5)

    parser.add_argument('--epoch-num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--lr-clf', type=float, default=0.001)

    parser.add_argument('--epoch-num-gan', type=int, default=100)
    parser.add_argument('--batch-size-gan', type=int, default=64)
    parser.add_argument('--block-size-gan', type=int, default=10)
    parser.add_argument('--lr-g', type=float, default=0.002)
    parser.add_argument('--lr-d', type=float, default=0.002)

    return parser.parse_args()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    scaler = MinMaxScaler()
    A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
    A[A.columns] = scaler.fit_transform(A)

    A, A_label = get_windows(A, A_label, args.window)
    A_train_loader, A_test_loader = get_dataloader(
        A, A_label, device=device, batch_size=args.batch_size, train_test_split=.2)

    model = UNet(in_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    for epoch in range(epochs):
        for imgs, _ in A_train_loader:
            # for imgs,_ in loader:
            imgs = imgs.to(device)
            imgs = torch.unsqueeze(imgs, dim=1)
            t = torch.randint(0, T, (imgs.size(0),), device=device)
            x_t, noise = q_sample(imgs, t)
            # import pdb;pdb.set_trace()
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")

    # sample
    samples = p_sample_loop(model, (1, 1, 5, 72), device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, "cifar10_diffusion_samples.png", nrow=4)
    print("Saved samples → cifar10_diffusion_samples.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
