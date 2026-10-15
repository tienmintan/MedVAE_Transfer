"""
diffusion_model.py: Cobb-conditioned latent diffusion model (U-Net, Gaussian diffusion, EMA).

Conditional latent diffusion model for spine X-ray synthesis in MedVAE's
latent space, conditioned on three Cobb angles via FiLM modulation.

Architecture:
  - 2D U-Net with 4 resolution levels (256x96 -> 32x12 at bottleneck)
  - Each ResNet block applies FiLM modulation from (time + Cobb) embedding
  - Self-attention at the bottleneck level only
  - Cosine noise schedule (Nichol & Dhariwal 2021), T=1000 steps
  - Classifier-free guidance via learned null Cobb embedding

Designed for the 384x1024 whole-image latents at MedVAE 4x/1ch compression,
i.e. inputs of shape (B, 1, 256, 96). Other latent shapes should also work
as long as both spatial dims are divisible by 8 (3 down/upsampling stages).
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Embeddings ---

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional embedding for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP."""
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.dim))


class CobbConditioning(nn.Module):
    """Embed standardised Cobb angles via a small MLP.

    Supports classifier-free guidance through a learned null embedding that
    replaces the conditional embedding when ``drop_mask`` is True.
    """
    def __init__(self, embed_dim: int = 256, n_angles: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_angles, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Learned null embedding (initialised to zero, learned through CFG dropout)
        self.null_embed = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, cobb: torch.Tensor,
                drop_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            cobb: (B, n_angles) standardised Cobb angles
            drop_mask: (B,) bool tensor; True positions use null embedding
        Returns:
            (B, embed_dim) conditioning vectors
        """
        emb = self.mlp(cobb)
        if drop_mask is not None:
            emb = torch.where(
                drop_mask[:, None],
                self.null_embed[None, :].expand_as(emb),
                emb,
            )
        return emb


# --- Building blocks ---

class ResBlock(nn.Module):
    """ResNet block with FiLM conditioning.

    The (time + Cobb) embedding gets projected to per-channel (scale, shift)
    via a linear layer, applied after the second GroupNorm.
    """
    def __init__(self, in_ch: int, out_ch: int, embed_dim: int,
                 groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.film = nn.Linear(embed_dim, 2 * out_ch)

        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # FiLM modulation on the post-norm features
        scale, shift = self.film(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention block, applied at the bottleneck only.

    For the bottleneck latent at 32x12, attention is over 384 positions,
    well within memory budget for a small batch.
    """
    def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, num_heads, head_dim, H*W)
        d = C // self.num_heads
        q = q.reshape(B, self.num_heads, d, H * W)
        k = k.reshape(B, self.num_heads, d, H * W)
        v = v.reshape(B, self.num_heads, d, H * W)

        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) / math.sqrt(d)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        h = h.reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# --- Main U-Net ---

class UNet(nn.Module):
    """Conditional U-Net for diffusion in MedVAE's 1-channel latent space.

    Default channel widths (64, 128, 256, 256) and embed_dim=256 give about
    30-40M parameters, appropriate for ~500 training samples.
    """
    def __init__(self,
                 in_channels: int = 1,
                 channels: Tuple[int, ...] = (64, 128, 256, 256),
                 embed_dim: int = 256,
                 n_angles: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.embed_dim = embed_dim

        self.time_embed = TimestepEmbedding(embed_dim)
        self.cobb_embed = CobbConditioning(embed_dim, n_angles=n_angles)

        # Stem
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder: at each level, 2 ResBlocks + downsample (except last)
        self.down_blocks = nn.ModuleList()
        for i, c_out in enumerate(channels):
            c_in = channels[i - 1] if i > 0 else channels[0]
            is_last = (i == len(channels) - 1)
            self.down_blocks.append(nn.ModuleList([
                ResBlock(c_in, c_out, embed_dim),
                ResBlock(c_out, c_out, embed_dim),
                Downsample(c_out) if not is_last else nn.Identity(),
            ]))

        # Middle: ResBlock + Attn + ResBlock at the deepest level
        c_mid = channels[-1]
        self.mid_block1 = ResBlock(c_mid, c_mid, embed_dim)
        self.mid_attn = SelfAttention(c_mid)
        self.mid_block2 = ResBlock(c_mid, c_mid, embed_dim)

        # Decoder: mirror of encoder, with skip-connection concatenation
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels))):
            # Channels coming in from the previous decoder level (or from
            # the middle block, for the first/deepest iteration).
            if i == len(channels) - 1:
                c_in_prev = c_mid
            else:
                c_in_prev = channels[i + 1]
            c_skip = channels[i]
            c_out = channels[i]
            self.up_blocks.append(nn.ModuleList([
                ResBlock(c_in_prev + c_skip, c_out, embed_dim),
                ResBlock(c_out, c_out, embed_dim),
                Upsample(c_out) if i > 0 else nn.Identity(),
            ]))

        # Head
        self.norm_out = nn.GroupNorm(8, channels[0])
        self.conv_out = nn.Conv2d(channels[0], in_channels, 3, padding=1)

        # Zero-init the final conv (helps diffusion training stability)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cobb: torch.Tensor,
                cond_drop_mask: torch.Tensor = None) -> torch.Tensor:
        # Combined time + cobb embedding
        emb = self.time_embed(t) + self.cobb_embed(cobb, cond_drop_mask)

        h = self.conv_in(x)
        skips = []
        for res1, res2, down in self.down_blocks:
            h = res1(h, emb)
            h = res2(h, emb)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for res1, res2, up in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = res1(h, emb)
            h = res2(h, emb)
            h = up(h)

        return self.conv_out(F.silu(self.norm_out(h)))


# --- Gaussian diffusion process ---

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021 (arXiv:2102.09672)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


class GaussianDiffusion:
    """Wrapper around the forward and reverse processes.

    All buffers are precomputed at construction. The class is stateless with
    respect to the model; any model that implements forward(x, t, cobb,
    drop_mask) can be passed.

    Supports two parameterizations:
      - "eps" (default): model predicts the noise ε. Standard DDPM.
      - "v": model predicts the v-target from Salimans & Ho 2022. More
        numerically stable across timesteps because the target magnitude
        stays consistent: at high noise v ≈ ε (unit-normal), at low noise
        v ≈ -x_0 (bounded by data scale). Recommended for heterogeneous
        datasets and small-data regimes where ε-prediction is fragile at
        low-α timesteps.

    Buffers are kept on CPU until moved with ``.to(device)``.
    """
    def __init__(self, T: int = 1000, device: torch.device = None,
                 parameterization: str = "eps"):
        if parameterization not in ("eps", "v"):
            raise ValueError(f"parameterization must be 'eps' or 'v', "
                             f"got {parameterization!r}")
        self.parameterization = parameterization
        self.T = T
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        if device is not None:
            self.to(device)

    def to(self, device: torch.device):
        for name in ["betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
                     "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "sqrt_recip_alphas", "posterior_variance"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _extract(self, buf: torch.Tensor, t: torch.Tensor,
                 shape: Tuple[int, ...]) -> torch.Tensor:
        out = buf.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sa = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        soma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sa * x_0 + soma * noise

    def _get_v(self, x_0: torch.Tensor, noise: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
        """Compute the v-target from Salimans & Ho 2022:
            v = sqrt(alpha_t) * noise - sqrt(1 - alpha_t) * x_0
        """
        sa = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        soma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sa * noise - soma * x_0

    def _model_pred_to_x0_and_eps(self, model_output: torch.Tensor,
                                    x_t: torch.Tensor,
                                    t: torch.Tensor):
        """Convert raw model output to (x_0_pred, eps_pred), regardless of
        parameterization. This lets the samplers share code between eps
        and v parameterizations.

        For 'eps' parameterization: model_output is eps_pred.
            x_0_pred = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)

        For 'v' parameterization: model_output is v_pred.
            x_0_pred = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v
            eps_pred = sqrt(alpha_t) * v + sqrt(1 - alpha_t) * x_t
        """
        sa = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        soma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        if self.parameterization == "eps":
            eps_pred = model_output
            x_0_pred = (x_t - soma * eps_pred) / sa
        else:   # 'v'
            v_pred = model_output
            x_0_pred = sa * x_t - soma * v_pred
            eps_pred = sa * v_pred + soma * x_t
        return x_0_pred, eps_pred

    def training_loss(self, model: nn.Module, x_0: torch.Tensor,
                      cobb: torch.Tensor,
                      cond_drop_prob: float = 0.15) -> torch.Tensor:
        """DDPM-style training loss. Target depends on parameterization:
        ε for 'eps' mode, v for 'v' mode.

        Randomly drops Cobb conditioning to support classifier-free guidance.
        """
        B = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Classifier-free guidance dropout
        drop_mask = (torch.rand(B, device=device) < cond_drop_prob)

        pred = model(x_t, t, cobb, cond_drop_mask=drop_mask)
        if self.parameterization == "eps":
            target = noise
        else:   # 'v'
            target = self._get_v(x_0, noise, t)
        return F.mse_loss(pred, target)

    # --- Sampling ---
    @torch.no_grad()
    def _predict_raw_with_cfg(self, model: nn.Module, x_t: torch.Tensor,
                              t: torch.Tensor, cobb: torch.Tensor,
                              cfg_scale: float) -> torch.Tensor:
        """Run the model with classifier-free guidance, returning the raw
        output (eps if parameterization='eps', v if 'v').
        """
        if cfg_scale == 1.0:
            return model(x_t, t, cobb)
        pred_cond = model(x_t, t, cobb)
        null = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
        pred_uncond = model(x_t, t, cobb, cond_drop_mask=null)
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

    @torch.no_grad()
    def _predict_clamped_x0_and_consistent_eps(
            self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
            cobb: torch.Tensor, cfg_scale: float,
            clamp_value: float = 3.0):
        """Predict (x_0_pred, eps_pred) with x_0 clamped to standardised
        data range and eps re-derived to remain consistent.

        Returns (x_0_clamped, eps_consistent).

        Why re-derive eps: an unmodified DDIM update uses both x_0_pred and
        the direction from eps. If the model's eps is miscalibrated, the
        direction term diverges even after x_0_pred is clamped. Re-deriving
        eps from the clamped x_0 propagates the clamp into all subsequent
        computations.

        At numerical-boundary timesteps where sqrt(1-alpha_t) ≈ 0, the
        re-derivation is unstable. In that regime the unmodified eps is
        returned (any miscalibration there is harmless since the direction
        coefficient sqrt(1-alpha) is also near zero).
        """
        sa = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        soma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        pred = self._predict_raw_with_cfg(model, x_t, t, cobb, cfg_scale)
        x_0_pred, eps_pred = self._model_pred_to_x0_and_eps(pred, x_t, t)

        x_0_clamped = x_0_pred.clamp(-clamp_value, clamp_value)

        # Re-derive eps from the clamped x_0, but only where soma is large
        # enough for stable division: ε_consistent = (x_t - √α * x_0_clamp) / √(1-α).
        safe = (soma > 1e-3)
        eps_consistent = torch.where(
            safe,
            (x_t - sa * x_0_clamped) / soma.clamp_min(1e-3),
            eps_pred,
        )
        return x_0_clamped, eps_consistent

    @torch.no_grad()
    def sample_ddpm(self, model: nn.Module, shape: Tuple[int, ...],
                    cobb: torch.Tensor, cfg_scale: float = 3.0,
                    device: torch.device = None) -> torch.Tensor:
        """Full-step DDPM sampler. T=1000 reverse steps."""
        device = device or cobb.device
        x = torch.randn(shape, device=device)
        for t_int in reversed(range(self.T)):
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            _, eps = self._predict_clamped_x0_and_consistent_eps(
                model, x, t, cobb, cfg_scale)

            sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, shape)
            beta = self._extract(self.betas, t, shape)
            soma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, shape)

            mean = sqrt_recip_alpha * (x - beta * eps / soma)
            if t_int == 0:
                x = mean
            else:
                var = self._extract(self.posterior_variance, t, shape)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample_ddim(self, model: nn.Module, shape: Tuple[int, ...],
                    cobb: torch.Tensor, cfg_scale: float = 3.0,
                    n_steps: int = 50, eta: float = 0.0,
                    device: torch.device = None) -> torch.Tensor:
        """DDIM sampler (Song et al. 2021). eta=0 is deterministic."""
        device = device or cobb.device
        step_indices = torch.linspace(0, self.T - 1, n_steps + 1,
                                      device=device).long()
        x = torch.randn(shape, device=device)

        for i in reversed(range(n_steps)):
            t_cur = step_indices[i + 1].expand(shape[0])
            t_prev = step_indices[i].expand(shape[0])

            x_0_pred, eps = self._predict_clamped_x0_and_consistent_eps(
                model, x, t_cur, cobb, cfg_scale)

            alpha_prev = self._extract(self.alphas_cumprod, t_prev, shape)
            alpha_cur = self._extract(self.alphas_cumprod, t_cur, shape)

            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_cur) * (1 - alpha_cur / alpha_prev)
            )
            direction = torch.sqrt(1.0 - alpha_prev - sigma ** 2) * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_prev) * x_0_pred + direction + noise

        return x


# --- Exponential moving average over model weights (standard for diffusion) ---

class EMA:
    """Maintain an EMA copy of model weights for sampling."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(),
                                                     alpha=1 - self.decay)

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        device = next(iter(self.shadow.values())).device
        for k, v in sd.items():
            self.shadow[k] = v.to(device)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite model parameters with the EMA copy (use for sampling)."""
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def store(self, model: nn.Module) -> dict:
        """Back up current model weights (so they can be restored after sampling)."""
        return {n: p.detach().clone() for n, p in model.named_parameters()
                if p.requires_grad}

    def restore(self, model: nn.Module, backup: dict) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in backup:
                p.data.copy_(backup[n])
