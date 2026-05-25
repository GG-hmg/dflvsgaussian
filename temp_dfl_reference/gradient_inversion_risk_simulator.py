import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from utils import generate_delayed_feedback_noise, generate_random_gaussian_noise_like

try:
    import lpips  # type: ignore
except Exception:
    lpips = None


@dataclass
class GradientInversionRiskConfig:
    attack_steps: int = 30
    attack_trials: int = 2
    attack_lr: float = 0.1
    attack_batch_size: int = 1
    tv_weight: float = 1e-4
    l2_weight: float = 1e-4
    dataset: str = "CIFAR10"
    psnr_center: float = 18.0
    psnr_scale: float = 3.0
    mse_center: float = 0.02
    mse_scale: float = 0.01


@dataclass
class DefenseSimulationConfig:
    dp_method: str = "none"
    sigma: float = 0.0
    clipping_bound: float = 1.0
    apply_clipping: bool = False
    apply_noise: bool = False
    sparsity_ratio: float = 0.0
    seed_context: int = 0
    a: float = 4.0
    b: float = 501.0
    k: int = 7
    burn_in: int = 2048
    decimation: int = 12
    q_bits: int = 32
    q_mode: str = "round"
    map_type: str = "logistic"


_DATASET_STATS = {
    "MNIST": ((0.1307,), (0.3081,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "SVHN": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}

_LPIPS_MODEL = None


def _get_dataset_stats(dataset: str, channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = _DATASET_STATS.get(dataset, ((0.5,) * channels, (0.5,) * channels))
    if len(mean) != channels:
        mean = (mean[0],) * channels
        std = (std[0],) * channels
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, channels, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, channels, 1, 1)
    return mean_t, std_t


def _normalized_bounds(dataset: str, channels: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_t, std_t = _get_dataset_stats(dataset, channels)
    mean_t = mean_t.to(device)
    std_t = std_t.to(device)
    low = (0.0 - mean_t) / std_t
    high = (1.0 - mean_t) / std_t
    return low, high


def _denormalize(x: torch.Tensor, dataset: str) -> torch.Tensor:
    channels = x.shape[1]
    mean_t, std_t = _get_dataset_stats(dataset, channels)
    mean_t = mean_t.to(x.device)
    std_t = std_t.to(x.device)
    return torch.clamp(x * std_t + mean_t, 0.0, 1.0)


def _ssim_score(x_recon_img: torch.Tensor, x_true_img: torch.Tensor) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = x_recon_img.mean(dim=(2, 3), keepdim=True)
    mu_y = x_true_img.mean(dim=(2, 3), keepdim=True)
    sigma_x = ((x_recon_img - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((x_true_img - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((x_recon_img - mu_x) * (x_true_img - mu_y)).mean(dim=(2, 3), keepdim=True)
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(torch.clamp(ssim_map.mean(), 0.0, 1.0).item())


def _lpips_fallback(x_recon_img: torch.Tensor, x_true_img: torch.Tensor) -> float:
    """
    Lightweight perceptual proxy when the lpips package is unavailable.
    Uses multi-scale pooled L1 distance. Lower is better, similar to LPIPS.
    """
    scales = [1, 2, 4]
    total = 0.0
    count = 0
    for scale in scales:
        if scale > 1:
            recon = F.avg_pool2d(x_recon_img, kernel_size=scale, stride=scale)
            truth = F.avg_pool2d(x_true_img, kernel_size=scale, stride=scale)
        else:
            recon = x_recon_img
            truth = x_true_img
        total += float(torch.mean(torch.abs(recon - truth)).item())
        count += 1
    return total / max(1, count)


def _lpips_score(x_recon_img: torch.Tensor, x_true_img: torch.Tensor) -> float:
    global _LPIPS_MODEL
    if lpips is not None:
        try:
            if _LPIPS_MODEL is None:
                _LPIPS_MODEL = lpips.LPIPS(net="alex").eval()
            model = _LPIPS_MODEL.to(x_recon_img.device)
            recon = (x_recon_img * 2.0) - 1.0
            truth = (x_true_img * 2.0) - 1.0
            value = model(recon, truth)
            return float(value.mean().item())
        except Exception:
            pass
    return _lpips_fallback(x_recon_img, x_true_img)


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def _flatten_gradients(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads if g is not None], dim=0)


def _clip_gradients(grads: List[torch.Tensor], clipping_bound: float) -> List[torch.Tensor]:
    if clipping_bound <= 0:
        return grads
    flat = _flatten_gradients(grads)
    grad_norm = torch.norm(flat)
    if grad_norm <= clipping_bound:
        return grads
    scale = clipping_bound / (grad_norm + 1e-12)
    return [g * scale for g in grads]


def _apply_sparsity(grads: List[torch.Tensor], sparsity_ratio: float) -> List[torch.Tensor]:
    if sparsity_ratio <= 0:
        return grads
    sparse_grads = []
    for grad in grads:
        grad_flat = grad.reshape(-1)
        k = int((1.0 - sparsity_ratio) * grad_flat.numel())
        if k <= 0 or k >= grad_flat.numel():
            sparse_grads.append(grad)
            continue
        _, indices = torch.topk(grad_flat.abs(), k)
        mask = torch.zeros_like(grad_flat)
        mask[indices] = 1.0
        sparse_grads.append((grad_flat * mask).reshape_as(grad))
    return sparse_grads


def _generate_delayed_feedback_flat_noise(numel: int, cfg: DefenseSimulationConfig, device: torch.device) -> torch.Tensor:
    if numel <= 0:
        return torch.zeros(0, device=device)
    rng = random.Random(int(cfg.seed_context))
    x0 = rng.random()
    flat_feedback = generate_delayed_feedback_noise(
        torch.Size([numel]),
        a=float(cfg.a),
        b=float(cfg.b),
        k=int(cfg.k),
        x0=float(x0),
        burn_in=int(cfg.burn_in),
        decimation=int(cfg.decimation),
        q_bits=int(cfg.q_bits),
        q_mode=str(cfg.q_mode),
        map_type=str(cfg.map_type),
    ).to(device)
    std = torch.std(flat_feedback, unbiased=False)
    if std > 0:
        flat_feedback = flat_feedback / std
    return flat_feedback


def _apply_defense_to_gradients(grads: List[torch.Tensor], cfg: DefenseSimulationConfig) -> List[torch.Tensor]:
    defended = [g.detach().clone() for g in grads]

    if cfg.apply_clipping:
        defended = _clip_gradients(defended, cfg.clipping_bound)

    if cfg.apply_noise and cfg.sigma > 0 and cfg.dp_method != "none":
        if cfg.dp_method == "gaussian":
            defended = [
                g + generate_random_gaussian_noise_like(g) * cfg.sigma
                for g in defended
            ]
        else:
            total_numel = sum(g.numel() for g in defended)
            flat_noise = _generate_delayed_feedback_flat_noise(total_numel, cfg, defended[0].device) * cfg.sigma
            offset = 0
            noisy = []
            for grad in defended:
                numel = grad.numel()
                grad_noise = flat_noise[offset:offset + numel].view_as(grad)
                noisy.append(grad + grad_noise)
                offset += numel
            defended = noisy

    if cfg.sparsity_ratio > 0:
        defended = _apply_sparsity(defended, cfg.sparsity_ratio)

    return defended


def _compute_observed_gradients(
    model: torch.nn.Module,
    x_true: torch.Tensor,
    y_true: torch.Tensor,
    defense_cfg: DefenseSimulationConfig,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    logits = model(x_true)
    loss = F.cross_entropy(logits, y_true)
    clean_grads = list(torch.autograd.grad(loss, params, create_graph=False))
    defended_grads = _apply_defense_to_gradients(clean_grads, defense_cfg)
    return clean_grads, defended_grads


def _relative_perturbation(clean_grads: List[torch.Tensor], defended_grads: List[torch.Tensor]) -> float:
    clean_flat = _flatten_gradients(clean_grads)
    defended_flat = _flatten_gradients(defended_grads)
    delta = defended_flat - clean_flat
    clean_norm = torch.norm(clean_flat)
    delta_norm = torch.norm(delta)
    return float(delta_norm / (clean_norm + 1e-12))


def _noise_signature_stats(clean_grads: List[torch.Tensor], defended_grads: List[torch.Tensor]) -> Dict[str, float]:
    clean_flat = _flatten_gradients(clean_grads)
    defended_flat = _flatten_gradients(defended_grads)
    noise = (defended_flat - clean_flat).detach().reshape(-1)
    if noise.numel() < 8:
        return {
            "gaussianity": 0.5,
            "structure": 0.0,
            "skew": 0.0,
            "kurtosis": 3.0,
            "lag1": 0.0,
        }

    max_samples = min(int(noise.numel()), 50000)
    if int(noise.numel()) > max_samples:
        step = max(1, int(noise.numel()) // max_samples)
        noise = noise[::step][:max_samples]

    centered = noise - torch.mean(noise)
    std = torch.std(centered) + 1e-12
    z = centered / std
    skew = float(torch.mean(z ** 3))
    kurtosis = float(torch.mean(z ** 4))

    x0 = z[:-1]
    x1 = z[1:]
    denom = (torch.norm(x0) * torch.norm(x1) + 1e-12)
    lag1 = float(torch.dot(x0, x1) / denom)

    skew_penalty = min(1.0, abs(skew) / 1.5)
    kurt_penalty = min(1.0, abs(kurtosis - 3.0) / 3.0)
    gaussianity = max(0.0, 1.0 - 0.5 * (skew_penalty + kurt_penalty))
    structure = min(1.0, abs(lag1) * 5.0)
    return {
        "gaussianity": float(gaussianity),
        "structure": float(structure),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "lag1": float(lag1),
    }


def _gradient_matching_loss(
    recon_grads: List[torch.Tensor],
    target_grads: List[torch.Tensor],
    l2_weight: float,
) -> torch.Tensor:
    recon_flat = _flatten_gradients(recon_grads)
    target_flat = _flatten_gradients(target_grads)
    cosine_term = 1.0 - F.cosine_similarity(
        recon_flat.unsqueeze(0), target_flat.unsqueeze(0), dim=1, eps=1e-8
    ).mean()
    l2_term = F.mse_loss(recon_flat, target_flat)
    return cosine_term + l2_weight * l2_term


def _reconstruction_metrics(x_recon: torch.Tensor, x_true: torch.Tensor, dataset: str) -> Dict[str, float]:
    mse_norm = F.mse_loss(x_recon, x_true).item()
    x_recon_img = _denormalize(x_recon, dataset)
    x_true_img = _denormalize(x_true, dataset)
    mse_img = F.mse_loss(x_recon_img, x_true_img).item()
    psnr = 100.0 if mse_img <= 1e-12 else 10.0 * math.log10(1.0 / mse_img)
    ssim = _ssim_score(x_recon_img, x_true_img)
    lpips_dist = _lpips_score(x_recon_img, x_true_img)
    attack_success = 1.0 if (psnr >= 18.0 or ssim >= 0.5) else 0.0
    return {
        "mse_norm": float(mse_norm),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": float(lpips_dist),
        "attack_success_rate": float(attack_success),
    }


def _single_trial_leakage(metrics: Dict[str, float], cfg: GradientInversionRiskConfig) -> float:
    psnr_score = 1.0 / (1.0 + math.exp(-(metrics["psnr"] - cfg.psnr_center) / max(1e-6, cfg.psnr_scale)))
    mse_score = 1.0 / (1.0 + math.exp((metrics["mse_norm"] - cfg.mse_center) / max(1e-6, cfg.mse_scale)))
    ssim_score = max(0.0, min(1.0, metrics["ssim"]))
    lpips_score = max(0.0, min(1.0, 1.0 - metrics["lpips"]))
    leakage = 0.40 * psnr_score + 0.20 * mse_score + 0.25 * ssim_score + 0.15 * lpips_score
    if metrics["attack_success_rate"] > 0:
        leakage = min(1.0, leakage + 0.05)
    return float(max(0.0, min(1.0, leakage)))


def _combine_leakage(avg_leakage: float, worst_leakage: float) -> float:
    return float(max(0.0, min(1.0, 0.5 * avg_leakage + 0.5 * worst_leakage)))


def _wiener_denoise_target_gradients(target_grads: List[torch.Tensor], sigma: float) -> List[torch.Tensor]:
    """
    Gaussian-aware denoising under additive i.i.d. Gaussian assumption.
    This models a stronger attacker for Gaussian-noise defense.
    """
    noise_var = float(sigma) * float(sigma)
    denoised: List[torch.Tensor] = []
    for grad in target_grads:
        obs_var = torch.var(grad)
        alpha = torch.clamp(1.0 - (noise_var / (obs_var + 1e-12)), min=0.0, max=1.0)
        denoised.append(alpha * grad)
    return denoised


def _run_inversion_trials(
    attack_model: torch.nn.Module,
    params: List[torch.nn.Parameter],
    x_true: torch.Tensor,
    y_true: torch.Tensor,
    target_grads: List[torch.Tensor],
    risk_cfg: GradientInversionRiskConfig,
    low: torch.Tensor,
    high: torch.Tensor,
    seed_context: int,
    step_ratio: float = 1.0,
    lr_scale: float = 1.0,
    trial_ratio: float = 1.0,
    seed_offset: int = 0,
) -> Dict[str, float]:
    attack_steps = max(1, int(round(float(risk_cfg.attack_steps) * float(step_ratio))))
    attack_trials = max(1, int(round(float(risk_cfg.attack_trials) * float(trial_ratio))))
    attack_lr = float(risk_cfg.attack_lr) * float(lr_scale)

    trial_psnr: List[float] = []
    trial_mse: List[float] = []
    trial_ssim: List[float] = []
    trial_lpips: List[float] = []
    trial_asr: List[float] = []
    trial_leakage: List[float] = []

    device = x_true.device
    for trial_id in range(attack_trials):
        init_seed = int(seed_context) + int(seed_offset) + 7919 * (trial_id + 1)
        gen = torch.Generator(device=device)
        gen.manual_seed(init_seed)

        dummy = low + (high - low) * torch.rand(
            x_true.shape, device=device, generator=gen, dtype=x_true.dtype
        )
        dummy.requires_grad_(True)

        optimizer = torch.optim.Adam([dummy], lr=attack_lr)

        for _ in range(attack_steps):
            optimizer.zero_grad(set_to_none=True)
            attack_model.zero_grad(set_to_none=True)
            pred = attack_model(dummy)
            dummy_loss = F.cross_entropy(pred, y_true)
            recon_grads = list(torch.autograd.grad(dummy_loss, params, create_graph=True))
            grad_loss = _gradient_matching_loss(
                recon_grads, target_grads, float(risk_cfg.l2_weight)
            )
            tv_loss = _total_variation(dummy)
            objective = grad_loss + float(risk_cfg.tv_weight) * tv_loss
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                dummy.data = torch.max(torch.min(dummy.data, high), low)

        with torch.no_grad():
            metrics = _reconstruction_metrics(dummy.detach(), x_true, risk_cfg.dataset)
            trial_psnr.append(metrics["psnr"])
            trial_mse.append(metrics["mse_norm"])
            trial_ssim.append(metrics["ssim"])
            trial_lpips.append(metrics["lpips"])
            trial_asr.append(metrics["attack_success_rate"])
            trial_leakage.append(_single_trial_leakage(metrics, risk_cfg))

    avg_leakage = float(sum(trial_leakage) / len(trial_leakage))
    worst_leakage = float(max(trial_leakage))
    return {
        "avg_leakage": avg_leakage,
        "worst_leakage": worst_leakage,
        "leakage_risk": _combine_leakage(avg_leakage, worst_leakage),
        "avg_psnr": float(sum(trial_psnr) / len(trial_psnr)),
        "best_psnr": float(max(trial_psnr)),
        "avg_mse": float(sum(trial_mse) / len(trial_mse)),
        "best_mse": float(min(trial_mse)),
        "avg_ssim": float(sum(trial_ssim) / len(trial_ssim)),
        "best_ssim": float(max(trial_ssim)),
        "avg_lpips": float(sum(trial_lpips) / len(trial_lpips)),
        "best_lpips": float(min(trial_lpips)),
        "attack_success_rate": float(sum(trial_asr) / len(trial_asr)),
    }


def simulate_gradient_inversion_risk(
    model: torch.nn.Module,
    batch_data: torch.Tensor,
    batch_labels: torch.Tensor,
    risk_cfg: GradientInversionRiskConfig,
    defense_cfg: DefenseSimulationConfig,
) -> Dict[str, float]:
    """
    Paper-style simulator:
    1) Build observed gradients under defense.
    2) Run gradient inversion optimization.
    3) Use average and best reconstruction quality as leakage proxies.
    """
    try:
        if batch_data.ndim != 4:
            return {"ok": False, "risk_score": 0.5, "error": "Only image tensors are supported"}

        device = batch_data.device
        attack_bsz = max(1, min(int(risk_cfg.attack_batch_size), int(batch_data.shape[0])))
        x_true = batch_data[:attack_bsz].detach().to(device)
        y_true = batch_labels[:attack_bsz].detach().to(device)

        attack_model = copy.deepcopy(model).to(device)
        attack_model.eval()
        params = [p for p in attack_model.parameters() if p.requires_grad]

        clean_grads, target_grads = _compute_observed_gradients(attack_model, x_true, y_true, defense_cfg)
        target_grads = [g.detach() for g in target_grads]
        perturb_ratio = _relative_perturbation(clean_grads, target_grads)
        noise_stats = _noise_signature_stats(clean_grads, target_grads)
        # bounded in [0,1), monotonic with relative perturbation
        perturb_score = perturb_ratio / (1.0 + perturb_ratio)
        perturb_score = float(max(0.0, min(1.0, perturb_score)))

        channels = int(x_true.shape[1])
        low, high = _normalized_bounds(risk_cfg.dataset, channels, device)
        base_result = _run_inversion_trials(
            attack_model=attack_model,
            params=params,
            x_true=x_true,
            y_true=y_true,
            target_grads=target_grads,
            risk_cfg=risk_cfg,
            low=low,
            high=high,
            seed_context=int(defense_cfg.seed_context),
        )
        leakage_risk = float(base_result["leakage_risk"])
        avg_leakage = float(base_result["avg_leakage"])
        worst_leakage = float(base_result["worst_leakage"])
        avg_psnr = float(base_result["avg_psnr"])
        best_psnr = float(base_result["best_psnr"])
        avg_mse = float(base_result["avg_mse"])
        best_mse = float(base_result["best_mse"])
        avg_ssim = float(base_result["avg_ssim"])
        best_ssim = float(base_result["best_ssim"])
        avg_lpips = float(base_result["avg_lpips"])
        best_lpips = float(base_result["best_lpips"])
        attack_success_rate = float(base_result["attack_success_rate"])

        # Distribution-aware attacker for Gaussian-noise defense:
        # under i.i.d. Gaussian assumption, attacker can denoise observed gradients
        # and run a stronger inversion path.
        gaussian_aware_used = False
        gaussian_aware_leakage = leakage_risk
        if (
            defense_cfg.dp_method == "gaussian"
            and defense_cfg.apply_noise
            and defense_cfg.sigma > 0
        ):
            gaussian_aware_used = True
            denoised_target_grads = _wiener_denoise_target_gradients(target_grads, defense_cfg.sigma)
            aware_result = _run_inversion_trials(
                attack_model=attack_model,
                params=params,
                x_true=x_true,
                y_true=y_true,
                target_grads=denoised_target_grads,
                risk_cfg=risk_cfg,
                low=low,
                high=high,
                seed_context=int(defense_cfg.seed_context),
                step_ratio=0.6,
                lr_scale=0.8,
                trial_ratio=0.5,
                seed_offset=200003,
            )
            gaussian_aware_leakage = float(aware_result["leakage_risk"])
            # Strong attacker picks the easier reconstruction path.
            if gaussian_aware_leakage > leakage_risk:
                leakage_risk = gaussian_aware_leakage
                avg_leakage = float(aware_result["avg_leakage"])
                worst_leakage = float(aware_result["worst_leakage"])
                avg_psnr = float(aware_result["avg_psnr"])
                best_psnr = float(aware_result["best_psnr"])
                avg_mse = float(aware_result["avg_mse"])
                best_mse = float(aware_result["best_mse"])
                avg_ssim = float(aware_result["avg_ssim"])
                best_ssim = float(aware_result["best_ssim"])
                avg_lpips = float(aware_result["avg_lpips"])
                best_lpips = float(aware_result["best_lpips"])
                attack_success_rate = float(aware_result["attack_success_rate"])

        # Noise-property based adjustment:
        # - Gaussian noise has higher distribution exploitability for model-based denoisers.
        # - Delayed feedback gets robustness credit from structure complexity.
        if defense_cfg.apply_noise and defense_cfg.sigma > 0:
            gaussianity = float(noise_stats["gaussianity"])
            structure = float(noise_stats["structure"])
            if defense_cfg.dp_method == "gaussian":
                exploitability = max(0.0, min(1.0, 0.7 * gaussianity + 0.3 * (1.0 - structure)))
                leakage_risk = min(1.0, leakage_risk + 0.22 * exploitability * (1.0 - leakage_risk))
            elif defense_cfg.dp_method == "delayed_feedback":
                hardness = max(0.0, min(1.0, 0.6 * structure + 0.4 * (1.0 - gaussianity)))
                leakage_risk = max(0.0, leakage_risk * (1.0 - 0.22 * hardness))

        # Anti-inversion ability should be higher when perturbation is stronger
        # and attack reconstruction leakage is lower.
        defense_score = float(max(0.0, min(1.0, perturb_score * (1.0 - leakage_risk))))

        return {
            "ok": True,
            "risk_score": leakage_risk,
            "defense_score": defense_score,
            "avg_leakage": avg_leakage,
            "worst_leakage": worst_leakage,
            "avg_psnr": avg_psnr,
            "best_psnr": best_psnr,
            "avg_mse": avg_mse,
            "best_mse": best_mse,
            "avg_ssim": avg_ssim,
            "best_ssim": best_ssim,
            "avg_lpips": avg_lpips,
            "best_lpips": best_lpips,
            "attack_success_rate": attack_success_rate,
            "perturbation_ratio": perturb_ratio,
            "perturbation_score": perturb_score,
            "gaussian_aware_used": gaussian_aware_used,
            "gaussian_aware_leakage": gaussian_aware_leakage,
            "noise_gaussianity": float(noise_stats["gaussianity"]),
            "noise_structure": float(noise_stats["structure"]),
            "noise_skew": float(noise_stats["skew"]),
            "noise_kurtosis": float(noise_stats["kurtosis"]),
            "noise_lag1": float(noise_stats["lag1"]),
        }
    except Exception as exc:
        return {"ok": False, "risk_score": 0.5, "error": str(exc)}
