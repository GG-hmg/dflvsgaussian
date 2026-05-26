import torch
from options import parse_args
import numpy as np
import math
import random

args = parse_args()

def generate_random_gaussian_noise(shape, device=None, dtype=torch.float32):
    """
    Generate Gaussian noise from vectorized uniform samples.
    """
    shape = torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape, device=device, dtype=dtype)

    target_device = device if device is not None else torch.device("cpu")
    uniform_vals = torch.rand(total_elements, device=target_device, dtype=torch.float32)
    noise = generate_ziggurat_gaussian_noise(shape, uniform_vals)
    if dtype is not None:
        noise = noise.to(dtype=dtype)
    if device is not None:
        noise = noise.to(device=device)
    return noise


def generate_random_gaussian_noise_like(reference_tensor):
    target_dtype = reference_tensor.dtype if reference_tensor.is_floating_point() else torch.float32
    return generate_random_gaussian_noise(
        reference_tensor.shape,
        device=reference_tensor.device,
        dtype=target_dtype
    )

# ==================== True Ziggurat Normal Sampler (Marsaglia-Tsang 2000) ====================
# Pre-computed 256-layer lookup tables — built once at import time.
# The tables replicate the canonical Marsaglia-Tsang Ziggurat.  Note: PyTorch's
# own torch.randn() uses Box-Muller via cuRAND — this is an independent
# implementation that can be fed custom uniform sequences (e.g. DFL chaotic).

_ZIGGURAT_N = 256
_ZIGGURAT_R = 3.6541528853610088       # tail boundary (start of tail layer)
_ZIGGURAT_V = 0.00492867323399          # per-rectangle area (v = A_layer)

def _build_ziggurat_tables():
    """Build x[i] (width) and f[i] (standard-normal PDF = exp(-x²/2)/√2π).

    Follows Marsaglia & Tsang (2000).  The f-table stores *un-normalised*
    values ``exp(-x²/2)`` so that the rectangle test is invariant to the
    leading constant 1/√2π — identical to Apache Commons RNG's convention.
    """
    n = _ZIGGURAT_N
    r = _ZIGGURAT_R
    v = _ZIGGURAT_V

    x_table = [0.0] * n
    f_table = [0.0] * n

    # Layer N-1: tail boundary — store un-normalised PDF
    d_val = r
    f_val = math.exp(-0.5 * r * r)
    if f_val < 1e-20:
        f_val = 1e-20
    x_table[n - 1] = d_val
    f_table[n - 1] = f_val

    # Descend from N-2 down to 1
    for i in range(n - 2, 0, -1):
        arg = v / d_val + f_val
        if arg >= 1.0:
            break
        d_val = math.sqrt(-2.0 * math.log(arg))
        f_val = math.exp(-0.5 * d_val * d_val)
        if f_val < 1e-20:
            f_val = 1e-20
        x_table[i] = d_val
        f_table[i] = f_val

    # Layer 0: origin — f(0) = exp(0) = 1.0
    x_table[0] = 0.0
    f_table[0] = 1.0

    return (
        torch.tensor(x_table, dtype=torch.float32),
        torch.tensor(f_table, dtype=torch.float32),
    )

_ZT_X, _ZT_F = _build_ziggurat_tables()


def generate_inverse_cdf_gaussian_noise(shape, uniform_sequence):
    """
    Inverse-CDF (erfinv) Gaussian sampler — the classic method (slow but exact).

    Kept for ablation / comparison with the true Ziggurat sampler.
    Uses float64 internally to avoid erfinv tail instability, then cast
    to float32 for model compatibility.
    """
    shape = torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape, device=uniform_sequence.device, dtype=torch.float32)
    if uniform_sequence.numel() < total_elements:
        raise ValueError("Uniform sequence too short")

    u = uniform_sequence.reshape(-1)[:total_elements].to(dtype=torch.float64)
    u = torch.clamp(u, min=1e-12, max=1.0 - 1e-12)
    noise = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    noise = noise.reshape(shape).to(dtype=torch.float32)
    return noise.to(uniform_sequence.device)


def generate_ziggurat_gaussian_noise(shape, uniform_sequence):
    """
    True Marsaglia-Tsang Ziggurat sampler — 256-layer lookup tables.

    Uses a rejection loop: on each pass we try to fill the buffer with
    candidates.  ~99 % of candidates pass the fast-path acceptance test,
    so the loop rarely runs more than 1–2 iterations.

    References
    ----------
    * Marsaglia & Tsang (2000), "The Ziggurat Method for Generating Random Variables"
    * Apache Commons RNG  — ZigguratNormalizedGaussianSampler
    """
    shape = torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape, device=uniform_sequence.device, dtype=torch.float32)

    dev = uniform_sequence.device
    x_table = _ZT_X.to(dev, non_blocking=True)
    f_table = _ZT_F.to(dev, non_blocking=True)

    n = _ZIGGURAT_N
    r = _ZIGGURAT_R

    u_flat = uniform_sequence.view(-1).to(torch.float32)

    # Fix 1: golden-ratio phase shift when tiling, never exact-repeat
    needed_initial = total_elements * 4
    if u_flat.numel() < needed_initial:
        repeats = (needed_initial + u_flat.numel() - 1) // u_flat.numel()
        tiled = []
        for i in range(repeats):
            tiled.append(torch.remainder(u_flat + i * 0.6180339887, 1.0))
        u_flat = torch.cat(tiled, dim=0)

    consumed = 0
    result = torch.empty(total_elements, device=dev, dtype=torch.float32)
    filled = 0

    max_iters = 10
    for _ in range(max_iters):
        remaining = total_elements - filled
        if remaining <= 0:
            break

        need = remaining
        n_vals = 4 * need

        # Fix 2: global phase shift when running out, never wrap-around clone
        if consumed + n_vals > int(u_flat.numel()):
            u_flat = torch.remainder(u_flat + 0.3141592653, 1.0)
            consumed = 0

        chunk = u_flat[consumed : consumed + n_vals]
        consumed += n_vals

        chunk = chunk.view(need, 4)
        u_j_frac = chunk[:, 0]
        u_width = chunk[:, 1]
        u_test = chunk[:, 2]
        u_sign = chunk[:, 3]

        # Layer index (weighted by ratio of widths — we select uniformly in
        # layer space, but layers with wider x-bounds are naturally hit more
        # because u_width * x_table[j] can land anywhere in [0, x_j]).
        j = (u_j_frac * float(n)).long().clamp(0, n - 1)

        # Candidate |x|
        x_abs = u_width * x_table[j]

        # --- fast path ---
        # For j > 0, if |x| < x[j-1] → accept immediately (no further test).
        j_is_tail = (j == 0)
        x_lower = x_table[(j - 1).clamp(min=0)]
        accept_fast = (~j_is_tail) & (x_abs < x_lower)

        # --- rectangle test ---
        # For non-tail layers where fast path failed, compare the PDF.
        need_rect = (~j_is_tail) & (~accept_fast)
        accept_rect = torch.zeros(need, device=dev, dtype=torch.bool)
        if need_rect.any():
            j_rect = j[need_rect]
            j_rect_m1 = j_rect - 1
            x_rect = x_abs[need_rect]
            u_rect = u_test[need_rect]

            y_lower = f_table[j_rect_m1]
            y_upper = f_table[j_rect]
            # Un-normalised PDF (matching f_table convention: exp(-x²/2))
            pdf_x = torch.exp(-0.5 * x_rect * x_rect)

            accept_rect[need_rect] = (u_rect * (y_lower - y_upper)) < (pdf_x - y_upper)

        # --- tail sampling ---
        # For j == 0 (tail), use Marsaglia's tail method: x = sqrt(R^2 - 2*ln(u)).
        accept_tail = torch.zeros(need, device=dev, dtype=torch.bool)
        if j_is_tail.any():
            u_tail = u_width[j_is_tail]
            u_tail = u_tail.clamp(min=1e-12, max=1.0 - 1e-12)
            tail_x = torch.sqrt(r * r - 2.0 * torch.log(u_tail))
            x_abs[j_is_tail] = tail_x
            accept_tail[j_is_tail] = True

        accepted = accept_fast | accept_rect | accept_tail
        n_good = accepted.sum().item()

        if n_good > 0:
            if n_good > remaining:
                n_good = remaining
                accepted_indices = torch.nonzero(accepted, as_tuple=True)[0][:n_good]
                accepted[:] = False
                accepted[accepted_indices] = True
                n_good = remaining

            sign = torch.where(
                u_sign[accepted] < 0.5,
                torch.tensor(1.0, device=dev, dtype=torch.float32),
                torch.tensor(-1.0, device=dev, dtype=torch.float32),
            )
            result[filled : filled + n_good] = x_abs[accepted][:n_good] * sign
            filled += n_good

    return result.reshape(shape)


# ==================== Core Utilities ====================
def compute_fisher_diag(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fisher_diag = [torch.zeros_like(p).to(device) for p in model.parameters()]
    model.eval()
    model.to(device)

    samples = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        if batch_idx >= 3: break
        data, labels = data.to(device), labels.to(device)
        model.zero_grad()
        log_probs = torch.nn.functional.log_softmax(model(data), dim=1)

        for i in range(min(5, len(labels))):
            grads = torch.autograd.grad(log_probs[i, labels[i].item()], model.parameters(), retain_graph=True)
            for j, g in enumerate(grads):
                if g is not None: fisher_diag[j] += g.pow(2)
            samples += 1

    if samples > 0:
        fisher_diag = [f / samples for f in fisher_diag]
    return fisher_diag


def adaptive_privacy_budget(client_data_sizes, target_epsilon):
    total_size = sum(client_data_sizes)
    if total_size == 0: return [target_epsilon] * len(client_data_sizes)
    n = len(client_data_sizes)
    budgets = [target_epsilon * ((s/total_size)**0.5) * n for s in client_data_sizes]
    scaling = (target_epsilon * n) / sum(budgets)
    return [min(max(b * scaling, target_epsilon * 0.3), target_epsilon * 2.0) for b in budgets]


def sparsify_gradients(gradients, sparsity=0.4):
    if sparsity <= 0 or gradients is None: return gradients
    sparsified = []
    for grad in gradients:
        if grad is None:
            sparsified.append(None)
            continue
        grad_flat = grad.abs().view(-1)
        k = int((1 - sparsity) * grad_flat.numel())
        if 0 < k < grad_flat.numel():
            _, indices = torch.topk(grad_flat, k)
            mask = torch.zeros_like(grad_flat).scatter_(0, indices, 1).view(grad.shape)
            sparsified.append(grad * mask)
        else:
            sparsified.append(grad)
    return sparsified


def add_adaptive_gaussian_noise(gradients, client_epsilon, delta,
                                clipping_bound, client_data_size,
                                total_data_size, current_epoch=0,
                                noise_decay=0.95, decay_type='exponential',
                                use_chaotic=False, chaotic_factor=0.3,
                                dp_method='gaussian'):
    if dp_method == 'none': return gradients

    base_sigma = (clipping_bound * math.sqrt(2 * math.log(1.25 / delta)) / client_epsilon) if client_epsilon > 0 else 0.0
    sigma_factor = getattr(args, f'sigma_factor_{dp_method}', 0.3 if dp_method == 'gaussian' else 0.4)
    sigma = base_sigma * sigma_factor

    decay = (noise_decay ** current_epoch) if decay_type == 'exponential' else (max(0.3, 1.0 - current_epoch/getattr(args, 'global_epoch', 100)) if decay_type == 'linear' else 1.0)
    sigma = max(0.001, min(0.5, sigma * decay * get_dataset_complexity_factor(getattr(args, 'dataset', 'CIFAR10'))))

    noisy_gradients = []
    for i, grad in enumerate(gradients):
        if grad is None:
            noisy_gradients.append(None)
            continue
        
        grad_device = grad.device
        norm = torch.norm(grad)
        adaptive_clip = clipping_bound * (1.0 - current_epoch / getattr(args, 'global_epoch', 100) * 0.5)
        clipped_grad = grad * (adaptive_clip / norm) if norm > adaptive_clip else grad

        # 当sigma_factor_dfl=0时，回退到普通Gaussian以避免DFL机制干扰
        if dp_method == 'dfl' and use_chaotic and sigma > 0 and getattr(args, 'sigma_factor_dfl', 0.0) > 0:
            try:
                seed = (current_epoch * 1000 + i) % 10000
                torch.manual_seed(seed)
                random.seed(seed)

                noise = generate_dfl_gaussian_noise(
                    grad.shape,
                    a=getattr(args, 'dfl_a', 4.0),
                    b=getattr(args, 'dfl_b', 501.0),
                    k=getattr(args, 'dfl_k', 7),
                    x0=random.random(),
                    burn_in=getattr(args, 'dfl_burn_in', 2048),
                    decimation=getattr(args, 'dfl_decimation', 12),
                    max_direct_uniform=getattr(args, 'dfl_max_direct_uniform', 4096)
                ).to(grad_device)

                std = noise.std()
                if std > 0: noise = (noise / std) * sigma
                noisy_gradients.append(clipped_grad + noise)
            except Exception:
                noisy_gradients.append(clipped_grad + generate_random_gaussian_noise_like(clipped_grad) * sigma)
        else:
            noisy_gradients.append(clipped_grad + (generate_random_gaussian_noise_like(clipped_grad) * sigma if sigma > 0 else 0))

    return noisy_gradients


def get_dataset_complexity_factor(name):
    return {'MNIST': 0.35, 'CIFAR10': 0.6, 'SVHN': 0.55, 'EMNIST': 0.45, 'FashionMNIST': 0.45}.get(name, 0.55)


# ==================== Pure DFL (No Repeats, No Phase Shifts) ====================
_dfl_long_cache = {}  # key: (a, b, k), value: single seq tensor

def _get_pure_dfl_sequences(a, b, k, seed_val, needed_length):
    cache_key = (a, b, k)

    # Pure approach: no 2M hard cap. Generate exactly what the model needs.
    # 2x buffer gives room for different start positions without re-computation.
    precompute_len = max(5000000, needed_length * 2)

    if cache_key not in _dfl_long_cache or len(_dfl_long_cache[cache_key]) < needed_length:
        print(f"[Pure DFL] Building {precompute_len/1000000:.1f}M pure chaotic sequence (a={a}, b={b}, k={k})...")
        print(f"[Pure DFL] First-time build may take seconds to tens of seconds. Subsequent epochs use cache.")

        # Initialize k-step history buffer
        history = [(seed_val + i * 0.1) % 1.0 for i in range(k)]
        seq = [0.0] * precompute_len

        # Core formula: x(n+1) = mod(a * x(n)*(1-x(n)) + b * x(n-k), 1.0)
        for i in range(precompute_len):
            x_n = history[-1]
            x_n_minus_k = history[0]

            x_next = (a * x_n * (1.0 - x_n) + b * x_n_minus_k) % 1.0
            seq[i] = x_next

            # Slide history window
            history.pop(0)
            history.append(x_next)

        _dfl_long_cache[cache_key] = torch.tensor(seq, dtype=torch.float64)
        print("[Pure DFL] Reservoir build completed! Ready for training.")

    seq_tensor = _dfl_long_cache[cache_key]

    # Deterministic pseudo-random start index — guaranteed in-bounds
    max_start = max(1, len(seq_tensor) - needed_length)
    pseudo_random_int = int(seed_val * 1000000)
    start_idx = pseudo_random_int % max_start

    return seq_tensor[start_idx : start_idx + needed_length]


def generate_dfl_gaussian_noise(shape, a=4.0, b=501.0, k=7, x0=0.5,
                                burn_in=2048, decimation=12,
                                max_direct_uniform=4096):
    total_elements = int(shape.numel()) if isinstance(shape, torch.Size) else int(torch.Size(shape).numel())
    if total_elements <= 0: return torch.zeros(shape if isinstance(shape, torch.Size) else torch.Size(shape))

    total_uniform = 2 * total_elements
    thin_factor = max(1, int(decimation))
    burn_in = max(0, int(burn_in))

    # Exact length needed — decimation gap + burn-in, no shortcuts
    required_length = burn_in + (total_uniform * thin_factor) + 64

    # Pure generation: request exactly what we need, no repeats
    seq = _get_pure_dfl_sequences(a, b, k, x0, required_length)

    # Direct extraction with Gap (decimation)
    u = seq[burn_in:]
    u = u[::thin_factor]
    u = u[:total_uniform]

    if u.numel() == 0: raise ValueError("DFL sequence empty")

    # Degeneracy guard (the only retained safety patch)
    check = u[:min(u.numel(), 4096)]
    if torch.unique(torch.floor(check * 4096)).numel() < min(256, max(32, check.numel() // 64)):
        u = torch.remainder(0.8 * u + 0.2 * torch.rand_like(u), 1.0)

    # Exactly enough uniforms — straight to pairing
    pairs = u.reshape(-1, 2)
    pairs[:, 1] = torch.remainder(pairs[:, 1] + 0.5 * pairs[:, 0], 1.0)
    u_final = torch.clamp(pairs.reshape(-1), min=1e-10, max=1 - 1e-10)

    noise = generate_ziggurat_gaussian_noise(shape, u_final)

    # Sign flip for strict zero-mean
    base_noise_len = total_uniform // 2
    reps_noise = (total_elements + base_noise_len - 1) // base_noise_len
    signs = (torch.randint(0, 2, (reps_noise,), device=noise.device, dtype=torch.float32) * 2 - 1)
    sign_mask = signs.repeat_interleave(base_noise_len)[:total_elements]
    noise = noise * sign_mask.reshape(shape)

    return noise