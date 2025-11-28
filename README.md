# DDPM for Latent Diffusion (PyTorch Implementation)

This repository contains a PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** designed to operate directly on **latent representations** rather than raw pixel images. The model uses an **advanced UNet architecture** (`advanced_unet.py`) and supports training and sampling from latent `.npz` files.

---

## 🚀 Features

* Custom Noise Scheduler with linear beta schedule
* Advanced UNet-based denoiser with learnable time embeddings
* Two operating modes: **train** and **sample**
* Automatic output directory creation for logs & generated samples
* Periodic model checkpointing
* Saves training loss curve
* Batch sampling and `.npz` sample export

---

## 📦 Repository Structure

```
.
├── advanced_unet.py     # Custom UNet architecture
├── AdvDDPM.py           # Main DDPM training & sampling script
├── utils.py             # Utility helpers (seed setup, paths)
├── DDIM.py              # Main DDIM training & sampling script
└── README.md
```

---

## 🧠 How it Works

This implementation trains a DDPM to denoise progressively corrupted latent representations.
The objective is MSE noise prediction:

$$
L = \mathbb{E}*{t, x_0, \epsilon} \left| \epsilon - \epsilon*\theta(x_t,t) \right|^2
$$

Sampling reverses the diffusion process step-by-step using the model’s noise prediction.

---

## 🔧 Installation

```bash
git clone https://github.com/your-repo/ddpm-latent.git
cd ddpm-latent
pip install -r requirements.txt
```

### Dependencies

* PyTorch
* NumPy
* tqdm
* Matplotlib

---

## 📁 Latent Data Format

The model expects a `.npz` latent file containing arrays `X` and `y`:

```python
data_X, data_y = dataset.load_latent("latents64x64.npz")
```

Expected shape:

```
(N, C, 64, 64)
```

---

## 🏋️ Training

To train the model:

```bash
python ddpm_latent.py \
  --mode train \
  --latent path/to/latents.npz \
  --n_steps 50 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4
```

Outputs saved under:

```
runs/<latent_name>_ddpm_<dim>_<steps>_<lbeta>_<ubeta>/train/
```

---

## ✨ Sampling

```bash
python ddpm_latent.py \
  --mode sample \
  --latent path/to/latents.npz \
  --n_steps 50 \
  --seed 42
```

Generated latent samples stored in:

```
runs/.../generated/seed_<seed>/samples_<seed>_<n_samples>.npz
```

---

## 📊 Output Files

| File              | Description                |
| ----------------- | -------------------------- |
| `model.pth`       | Final model                |
| `ddpm_epoch_*.pt` | Saved checkpoints          |
| `loss_curve.png`  | Loss history visualization |
| `samples_*.npz`   | Generated latent samples   |

---

## 💡 Customizing Model Config

Modify UNet configuration inside `DDPM()`:

```python
model_config = {
    'im_channels': n_dim,
    'down_channels': [32, 64, 128, 256],
    'mid_channels': [256, 256, 128],
    'time_emb_dim': 128
}
```

---

## 🙏 Acknowledgements

Based on:

* Ho et al. (2020), *Denoising Diffusion Probabilistic Models*
* Open-source diffusion model community

---

## 📜 License

MIT License
