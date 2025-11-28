import torch
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import os
import matplotlib.pyplot as plt
import utils
import dataset


class NoiseScheduler():
    def __init__(self, num_timesteps=50, type="linear", device='cpu', beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.type = type
        self.device = device
        if type == "linear":
            self.init_linear_schedule(beta_start, beta_end)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")

    def init_linear_schedule(self, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32, device=self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def __len__(self):
        return self.num_timesteps


class SimpleUNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, max_timesteps=1000):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t = t[:, None, None, None].float() / self.max_timesteps
        t = t.expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([x, t], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


class DDIM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        super().__init__()
        self.model = SimpleUNet(n_channels=n_dim+1, n_classes=n_dim, max_timesteps=n_steps)
        self.n_steps = n_steps

    def forward(self, x, t):
        return self.model(x, t)


def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    model.train()
    losses = []
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            x0, _ = batch
            x0 = x0.to(device)
            batch_size = x0.size(0)
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x0)
            sqrt_alphas_cumprod_t = noise_scheduler.sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = noise_scheduler.sqrt_one_minus_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
            xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
            noise_pred = model(xt, t)
            loss = mse(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            os.makedirs(run_name, exist_ok=True)
            model_path = os.path.join(run_name, f"ddpm_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            loss_plot_path = os.path.join(run_name, 'loss_curve.png')
            plt.savefig(loss_plot_path)
            plt.close()
    torch.save(model.state_dict(), os.path.join(run_name, "model.pth"))
    return losses


@torch.no_grad()
def sample_ddim(model, n_samples, image_shape, noise_scheduler, eta=0.0, return_intermediate=False):
    device = next(model.parameters()).device
    x = torch.randn((n_samples, *image_shape), device=device)
    intermediates = []
    alphas = noise_scheduler.alphas
    alphas_cumprod = noise_scheduler.alphas_cumprod
    for i in reversed(range(1, noise_scheduler.num_timesteps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        t_prev = torch.full((n_samples,), i - 1, device=device, dtype=torch.long)
        alpha_cum_t = alphas_cumprod[i]
        alpha_cum_t_prev = alphas_cumprod[i - 1]
        beta_t = noise_scheduler.betas[i]
        eps = model(x, t)
        x0 = (x - (1 - alpha_cum_t).sqrt() * eps) / alpha_cum_t.sqrt()
        sigma_t = eta * ((1 - alpha_cum_t_prev) / (1 - alpha_cum_t) * beta_t).sqrt()
        mean_pred = alpha_cum_t_prev.sqrt() * x0 + (1 - alpha_cum_t_prev - sigma_t ** 2).sqrt() * eps
        noise = torch.randn_like(x) if eta > 0 else 0
        x = mean_pred + sigma_t * noise if eta > 0 else mean_pred
        if return_intermediate:
            intermediates.append(x.detach().cpu())
    return intermediates if return_intermediate else x.detach().cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--lbeta", type=float, default=1e-4)
    parser.add_argument("--ubeta", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=[28, 28])
    args = parser.parse_args()

    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.dataset}'
    os.makedirs(run_name, exist_ok=True)

    image_shape = (args.n_dim, *args.image_size)
    model = DDIM(n_dim=args.n_dim, n_steps=args.n_steps).to(device)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta, device=device)

    if args.mode == 'train':
        data_X, data_y = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, noise_scheduler, dataloader, optimizer, args.epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample_ddim(model, args.n_samples, image_shape, noise_scheduler, eta=args.eta)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
