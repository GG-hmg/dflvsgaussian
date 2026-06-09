import torch
import numpy as np
import math

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
    noise = generate_inverse_cdf_gaussian_noise(shape, uniform_vals)
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
        torch.tensor(x_table, dtype=torch.float64),
        torch.tensor(f_table, dtype=torch.float64),
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
    x_table = _ZT_X.to(dev, dtype=torch.float64, non_blocking=True)
    f_table = _ZT_F.to(dev, dtype=torch.float64, non_blocking=True)

    n = _ZIGGURAT_N
    r = _ZIGGURAT_R

    u_flat = uniform_sequence.view(-1).to(torch.float64)

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

            sign = (u_sign[accepted] < 0.5).to(torch.float64) * 2.0 - 1.0
            result[filled : filled + n_good] = (x_abs[accepted][:n_good] * sign).to(torch.float32)
            filled += n_good

    return result.reshape(shape)


def adaptive_privacy_budget(client_data_sizes, target_epsilon):
    total_size = sum(client_data_sizes)
    if total_size == 0: return [target_epsilon] * len(client_data_sizes)
    n = len(client_data_sizes)
    budgets = [target_epsilon * ((s/total_size)**0.5) * n for s in client_data_sizes]
    scaling = (target_epsilon * n) / sum(budgets)
    return [min(max(b * scaling, target_epsilon * 0.3), target_epsilon * 2.0) for b in budgets]


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

    # seed_val ∈ [0, 1) is already a well-spread hash from the caller
    # (see ours.py: (client_id, epoch, batch) → Knuth-multiply → x0).
    # Direct mapping; no second avalanche needed.
    max_start = max(1, len(seq_tensor) - needed_length)
    start_idx = int(seed_val * max_start) % max_start

    return seq_tensor[start_idx : start_idx + needed_length]


def _extract_dfl_uniform_sequence(total_elements, a, b, k, x0, burn_in, decimation):
    """
    Run the DFL chaotic map and produce `total_elements` samples in [0, 1)
    after burn-in + decimation + degeneracy guard + adjacent-pair coupling.

    Shared preprocessing for both dfl_uniform and dfl_gaussian — the two
    noise kinds differ only in the final step (linear rescale vs inverse-CDF).
    """
    if total_elements <= 0:
        return torch.zeros(0, dtype=torch.float32)

    # Even count for reshape(-1, 2) used by the pair-coupling step.
    total_uniform = total_elements + (total_elements % 2)

    thin_factor = max(1, int(decimation))
    burn_in = max(0, int(burn_in))
    required_length = burn_in + (total_uniform * thin_factor) + 64

    seq = _get_pure_dfl_sequences(a, b, k, x0, required_length)
    u = seq[burn_in:]
    u = u[::thin_factor]
    # .clone() severs the view from _dfl_long_cache — the pair-coupling
    # assignment below must not mutate the cache.
    u = u[:total_uniform].clone()

    if u.numel() == 0:
        raise ValueError("DFL sequence empty")

    # Degeneracy guard
    check = u[:min(u.numel(), 4096)]
    if torch.unique(torch.floor(check * 4096)).numel() < min(256, max(32, check.numel() // 64)):
        u = torch.remainder(0.8 * u + 0.2 * torch.rand_like(u), 1.0)

    # Adjacent-pair coupling for anti-pattern defense
    pairs = u.reshape(-1, 2)
    pairs[:, 1] = torch.remainder(pairs[:, 1] + 0.5 * pairs[:, 0], 1.0)
    u_final = torch.clamp(pairs.reshape(-1)[:total_elements], min=1e-10, max=1 - 1e-10)
    return u_final


def generate_dfl_gaussian_noise(shape, a=4.0, b=501.0, k=7, x0=0.5,
                                burn_in=2048, decimation=12, device=None):
    """
    Standard-normal noise sampled via inverse-CDF, with chaotic DFL random source.

    Marginal distribution = N(0, 1). Sample correlations follow the DFL map.
    Used by noise_kind in {dfl_gaussian, mix_dgauss_gauss, mix_dgauss_dchaos}.
    """
    shape = shape if isinstance(shape, torch.Size) else torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape)

    # 5x oversampling so inverse-CDF has plenty of input even after pair coupling drop
    u_final = _extract_dfl_uniform_sequence(5 * total_elements, a, b, k, x0, burn_in, decimation)
    if device is not None:
        u_final = u_final.to(device, non_blocking=True)
    return generate_inverse_cdf_gaussian_noise(shape, u_final)


def generate_dfl_uniform_noise(shape, a=4.0, b=501.0, k=7, x0=0.5,
                               burn_in=2048, decimation=12, device=None):
    """
    Mean-0 std-1 noise direct from the DFL chaotic map — NOT passed through
    a Gaussian sampler.

    Pipeline: raw DFL ∈ [0,1) → (u-0.5) * sqrt(12). Result has:
        - bounded support [-sqrt(3), +sqrt(3)) ≈ [-1.73, 1.73]
        - mean 0, variance 1 (exactly, by linearity)
        - uniform-shape marginal distribution
        - chaotic sample-to-sample correlation
    Used by noise_kind in {dfl_uniform, mix_dgauss_dchaos}.
    """
    shape = shape if isinstance(shape, torch.Size) else torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape)

    u_final = _extract_dfl_uniform_sequence(total_elements, a, b, k, x0, burn_in, decimation)
    noise = (u_final - 0.5) * math.sqrt(12.0)
    noise = noise.reshape(shape).to(dtype=torch.float32)
    if device is not None:
        noise = noise.to(device, non_blocking=True)
    return noise