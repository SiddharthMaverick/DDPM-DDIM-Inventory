import torch
import torch.utils
import numpy as np
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
from advanced_unet import Unet  # Assuming advanced UNet is saved as advanced_unet.py



def get_run_dirs(latent_path, n_dim, n_steps, lbeta, ubeta, seed):
    # latent_path = "HX/latents64x64.npz"
             # "HX"
    latent_name = os.path.splitext(os.path.basename(latent_path))[0]  # "latents64x64"

    run_base = os.path.join("runs",f"{latent_name}_ddpm_{n_dim}_{n_steps}_{lbeta}_{ubeta}")

    train_dir = os.path.join(run_base, "train")
    gen_dir = os.path.join(run_base, "generated", f"seed_{seed}")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    return train_dir, gen_dir


def sample_in_batches(model,total_samples,noise_scheduler,batch_size=64):
    device=next(model.parameters()).device
    all_samples=[]

    for start in range(0,total_samples,batch_size):
        cur_batch=min(batch_size,total_samples-start)
        x=sample(model,cur_batch,noise_scheduler)
        all_samples.append(x.cpu())
    return torch.cat(all_samples,dim=0)




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


class DDPM(nn.Module):
    def __init__(self, n_dim=1, n_steps=200):
        super().__init__()
        self.n_steps = n_steps
        model_config = {
            'im_channels': n_dim,
            'down_channels': [32, 64, 128, 256],
            'mid_channels': [256, 256, 128],
            'time_emb_dim': 128,
            'down_sample': [True, True, False],
            'num_down_layers': 1,
            'num_mid_layers': 1,
            'num_up_layers': 1
        }
        self.model = Unet(model_config)

    def forward(self, x, t):
        return self.model(x, t)


def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    device = next(model.parameters()).device
    mse = nn.MSELoss()
    model.train()
    losses = []
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
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
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            os.makedirs(run_name, exist_ok=True)
            model_path = os.path.join(run_name, f"ddpm_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint at {model_path}")

            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_name, 'loss_curve.png'))
            plt.close()

    torch.save(model.state_dict(), os.path.join(run_name, "model.pth"))
    return losses


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False):
    device = next(model.parameters()).device
    x = torch.randn((n_samples, 1, 64, 64), device=device)
    intermediates = []

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        noise_pred = model(x, t_batch)

        beta_t = noise_scheduler.betas[t]
        alpha_t = noise_scheduler.alphas[t]
        alpha_cumprod_t = noise_scheduler.alphas_cumprod[t]
        sqrt_recip_alpha_t = noise_scheduler.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = noise_scheduler.sqrt_one_minus_alphas_cumprod[t]

        mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)
        if t > 0:
            var = noise_scheduler.posterior_variance[t]
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

        if return_intermediate:
            intermediates.append(x.detach().cpu())

    return intermediates if return_intermediate else x.detach().cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--latent", type=str, required=True, help="Path to latent .npz file")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--lbeta", type=float, default=1e-4)
    parser.add_argument("--ubeta", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=1)

    args = parser.parse_args()
    utils.seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # run folders
    # inside __main__
    train_dir, gen_dir = get_run_dirs(args.latent, args.n_dim, args.n_steps, args.lbeta, args.ubeta, args.seed)


    # model + scheduler
    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps).to(device)
    noise_scheduler = NoiseScheduler(
        num_timesteps=args.n_steps,
        beta_start=args.lbeta,
        beta_end=args.ubeta,
        device=device
    )

    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_latent(args.latent)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_X, data_y),
            batch_size=args.batch_size,
            shuffle=True
        )
        train(model, noise_scheduler, dataloader, optimizer, args.epochs, train_dir)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(os.path.join(train_dir, "model.pth")))
        samples = sample_in_batches(model,total_samples=1280,noise_scheduler=noise_scheduler,batch_size=64)

        samples_np=samples.cpu().numpy()

        save_path=os.path.join(gen_dir,f"samples_{args.seed}_{args.n_samples}.npz")
        np.savez_compressed(save_path,samples=samples_np)

        print(f"Saved samples at {save_path} with shape {samples_np.shape}")
