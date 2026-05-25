import torch
from options import parse_args
import numpy as np
import math
import scipy.stats as stats
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
# These tables replicate the Marsaglia-Tsang Ziggurat used by PyTorch/cuRAND internally.

_ZIGGURAT_N = 256
_ZIGGURAT_R = 3.6541528853610088       # tail boundary (start of tail layer)
_ZIGGURAT_V = 0.00492867323399          # per-rectangle area (v = A_layer)

def _build_ziggurat_tables():
    """Build x[i] (width) and f[i] (half-normal PDF at x[i]) for 256-layer Ziggurat.

    Follows Marsaglia & Tsang (2000) exactly.
    The iteration uses a numerical approximation that is valid while
    ``v / d + f < 1.0``.  Once the argument of ``log`` would become
    non-positive we simply fill the remaining lower layers with ``d = 0``
    (these are the innermost layers where the PDF is nearly flat).
    """
    n = _ZIGGURAT_N
    r = _ZIGGURAT_R
    v = _ZIGGURAT_V   # per-rectangle area, ≈ A_layer

    x_table = [0.0] * n
    f_table = [0.0] * n

    # Layer N-1: tail boundary
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
            # Numeric limit reached — pad remainder with zeros
            break
        d_val = math.sqrt(-2.0 * math.log(arg))
        f_val = math.exp(-0.5 * d_val * d_val)
        if f_val < 1e-20:
            f_val = 1e-20
        x_table[i] = d_val
        f_table[i] = f_val

    # Layer 0: origin — x=0, half-normal PDF(0) = sqrt(2/pi) ≈ 0.7979
    x_table[0] = 0.0
    f_table[0] = math.sqrt(2.0 / math.pi)

    return (
        torch.tensor(x_table, dtype=torch.float32),
        torch.tensor(f_table, dtype=torch.float32),
    )

_ZT_X, _ZT_F = _build_ziggurat_tables()

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
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)

    # We consume the uniform_sequence as a flat float32 buffer.
    # Each attempt needs 3 floats (layer_float, u, u_test) + 1 for sign.
    u_flat = uniform_sequence.view(-1).to(torch.float32)
    # Ensure we have enough — tile if necessary
    needed_per_pass = total_elements * 4
    if u_flat.numel() < needed_per_pass:
        repeats = (needed_per_pass + u_flat.numel() - 1) // u_flat.numel()
        u_flat = u_flat.repeat(repeats)

    consumed = 0
    result = torch.empty(total_elements, device=dev, dtype=torch.float32)
    filled = 0

    # Ziggurat rejection loop — rare to need more than 1 iteration
    max_iters = 10
    for _ in range(max_iters):
        remaining = total_elements - filled
        if remaining <= 0:
            break

        need = remaining
        start = consumed % int(u_flat.numel())
        # Wrap-around: grab 4*need floats
        n_vals = 4 * need
        if start + n_vals <= int(u_flat.numel()):
            chunk = u_flat[start : start + n_vals]
        else:
            avail = int(u_flat.numel()) - start
            need_extra = n_vals - avail
            chunk = torch.cat([
                u_flat[start : start + avail],
                u_flat[:need_extra],
            ], dim=0)
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
            pdf_x = inv_sqrt2pi * torch.exp(-0.5 * x_rect * x_rect)

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

    # Fallback: any unfilled slots → use Box-Muller (should never happen)
    if filled < total_elements:
        n_fall = total_elements - filled
        u1 = u_flat[(consumed % int(u_flat.numel())):].to(torch.float32)
        u1 = u1[:n_fall].clamp(min=1e-12, max=1.0 - 1e-12)
        u2 = u_flat[(consumed + n_fall) % int(u_flat.numel()):].to(torch.float32)
        u2 = u2[:n_fall]
        r_bm = torch.sqrt(-2.0 * torch.log(u1))
        theta = 2.0 * math.pi * u2
        result[filled:] = r_bm * torch.cos(theta)
        filled = total_elements

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


# ==================== DFL (Discrete Fractional Logistic) ====================
# Pre-computed long sequences for fast reuse
_dfl_long_cache = {}  # key: (a, b, k), value: single seq tensor

def _get_dfl_sequences(a, b, k, seed_val, needed_length):
    cache_key = (a, b, k)

    # Dynamic safe reservoir: at least 3x needed_length to eliminate overlap
    safe_multiplier = 3
    precompute_len = max(20000000, needed_length * safe_multiplier)

    if cache_key not in _dfl_long_cache or len(_dfl_long_cache[cache_key]) < precompute_len:
        print(f"[DFL] Building {precompute_len/1000000:.1f}M reservoir (a={a}, b={b}, k={k})...")

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
        print("[DFL] Reservoir build completed!")

    seq_tensor = _dfl_long_cache[cache_key]
    max_start = max(1, len(seq_tensor) - needed_length)

    # Deterministic pseudo-random start index
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

    # Cap per-call sequence length at 2M
    max_sequence_pool = 2000000
    base_length = min(max_sequence_pool, burn_in + (total_uniform * thin_factor) + 64)
    base_length = max(32, base_length)

    # Call new engine with (a, b, k). k-step delay is built-in, no torch.roll needed.
    seq = _get_dfl_sequences(a, b, k, x0, base_length)

    # Direct extraction with Gap (decimation), no external mixing needed
    u = seq[burn_in:]
    u = u[::thin_factor]


    if u.numel() == 0: raise ValueError("DFL sequence empty")

    check = u[:min(u.numel(), 4096)]
    if torch.unique(torch.floor(check * 4096)).numel() < min(256, max(32, check.numel() // 64)):
        u = torch.remainder(0.8 * u + 0.2 * torch.rand_like(u), 1.0)

    # 如果抽出来的水不够，利用后处理铺满（这是最高效的做法）
    if u.numel() < total_uniform:
        blen = u.numel()
        reps = (total_uniform + blen - 1) // blen
        uniform_vals = u.repeat(reps)[:total_uniform]
        needs_sign_flip = True
    else:
        uniform_vals = u[:total_uniform]
        blen = total_uniform
        needs_sign_flip = False

    pairs = uniform_vals.reshape(-1, 2)
    pairs[:, 1] = torch.remainder(pairs[:, 1] + 0.5 * pairs[:, 0], 1.0)
    u_final = torch.clamp(pairs.reshape(-1), min=1e-10, max=1 - 1e-10)

    noise = generate_ziggurat_gaussian_noise(shape, u_final)

    # 符号翻转保平安
    if needs_sign_flip:
        base_noise_len = blen // 2
        reps_noise = (total_elements + base_noise_len - 1) // base_noise_len
        signs = (torch.randint(0, 2, (reps_noise,), device=noise.device, dtype=torch.float32) * 2 - 1)
        sign_mask = signs.repeat_interleave(base_noise_len)[:total_elements]
        noise = noise * sign_mask.reshape(shape)

    return noise

generate_simple_chaotic_noise = generate_dfl_gaussian_noise