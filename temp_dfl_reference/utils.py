import torch
from options import parse_args
import numpy as np
import math
import hashlib

args = parse_args()

_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MUL2 = 0x94D049BB133111EB

def generate_random_gaussian_noise(shape, device=None, dtype=torch.float32):
    """
    Generate Gaussian noise from vectorized uniform samples.
    This avoids Python-list allocations that can trigger MemoryError in long runs.
    """
    shape = torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape, device=device, dtype=dtype)

    # Keep generation vectorized to minimize temporary Python objects.
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

# ==================== Inverse-CDF Gaussian Sampler ====================
def generate_inverse_cdf_gaussian_noise(shape, uniform_sequence):
    """
    Generate N(0,1) samples from uniforms via inverse-CDF (erfinv).
    This is fully vectorized in torch and avoids large NumPy float64 allocations.
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


def _stable_u64_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return seed or 1


def _splitmix64_uniform_block(seed: int, count: int) -> np.ndarray:
    """Generate a block of [0, 1) uniforms using SplitMix64."""
    count = int(count)
    if count <= 0:
        return np.empty(0, dtype=np.float32)

    idx = np.arange(count, dtype=np.uint64)
    state = np.uint64(seed & _UINT64_MASK) + (idx + np.uint64(1)) * np.uint64(_SPLITMIX64_GAMMA)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(_SPLITMIX64_MUL1)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(_SPLITMIX64_MUL2)
    state = state ^ (state >> np.uint64(31))
    uniforms = (state >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
    return uniforms.astype(np.float32, copy=False)


# ==================== Delayed Feedback Chaotic Sequence ====================


def _resolve_delay_k(delay_k=None):
    if delay_k is None:
        delay_k = getattr(args, "k", 4)
    try:
        k = int(delay_k)
    except Exception:
        k = 4
    return max(1, k)


def _normalize_map_type(map_type=None):
    if map_type is None:
        map_type = getattr(args, "map_type", "logistic")
    normalized = str(map_type).strip().lower().replace("-", "_")
    aliases = {
        "logistic_map": "logistic",
        "tent_map": "tent",
        "sine_map": "sine",
    }
    return aliases.get(normalized, normalized)


def _map_type_id(map_type) -> int:
    map_name = _normalize_map_type(map_type)
    if map_name == "logistic":
        return 0
    if map_name == "tent":
        return 1
    if map_name == "sine":
        return 2
    raise ValueError(
        f"Unsupported delayed-feedback map_type={map_type!r}. "
        "Supported values: logistic, tent, sine."
    )


def _q_mode_id(q_mode) -> int:
    mode = str(q_mode).lower()
    if mode == "floor":
        return 0
    if mode == "ceil":
        return 1
    return 2


def delayed_feedback_seed_map(x, map_type="logistic"):
    """
    Compute f(x) in the delayed-feedback map
    x(n+1) = q_L(mod(a f(x(n)) + b x(n-k), 1)).
    """
    x = np.asarray(x, dtype=np.float64)
    map_name = _normalize_map_type(map_type)

    if map_name == "logistic":
        return x * (1.0 - x)
    if map_name == "tent":
        return np.where(x < 0.5, x, 1.0 - x)
    if map_name == "sine":
        return np.sin(np.pi * x)

    raise ValueError(
        f"Unsupported delayed-feedback map_type={map_type!r}. "
        "Supported values: logistic, tent, sine."
    )


def quantize_qL(value, q_bits=32, q_mode="round"):
    """
    Apply the L-bit digital quantizer q_L used to model finite-precision
    digital chaotic systems.
    """
    q_bits = int(q_bits)
    if q_bits <= 0:
        return value

    scale = float(1 << q_bits)
    mode = str(q_mode).lower()
    arr = np.asarray(value, dtype=np.float64)
    clipped = np.clip(arr, 0.0, 1.0 - (1.0 / scale))

    if mode == "floor":
        quantized = np.floor(clipped * scale) / scale
    elif mode == "ceil":
        quantized = np.ceil(clipped * scale) / scale
    else:
        quantized = np.round(clipped * scale) / scale

    quantized = np.clip(quantized, 0.0, 1.0 - (1.0 / scale))
    if np.isscalar(value):
        return float(quantized)
    return quantized


def _derive_delay_initial_history(k, a=4.0, b=501.0, x0=0.1,
                                  q_bits=32, q_mode="round", map_type="logistic"):
    """
    Build the explicit initial conditions x(0)...x(k) for the paper's
    delayed-feedback system.

    The paper treats these as free initial conditions. We derive a stable,
    deterministic vector from the caller-provided seed-like x0 while keeping
    x(0) anchored to x0 itself.
    """
    k = _resolve_delay_k(k)
    seed = _stable_u64_seed(
        "delay_initial_history",
        float(a),
        float(b),
        int(k),
        float(x0),
        _normalize_map_type(map_type),
    )
    init = _splitmix64_uniform_block(seed, k + 1).astype(np.float64, copy=False)
    if init.size == 0:
        return np.empty(0, dtype=np.float64)

    eps = np.float64(1e-12)
    init = np.clip(init, eps, 1.0 - eps)
    init[0] = np.clip(float(x0) % 1.0, eps, 1.0 - eps)
    init = quantize_qL(init, q_bits=q_bits, q_mode=q_mode)
    return init


def _generate_delayed_feedback_sequence(
    length,
    a=4.0,
    b=501.0,
    k=4,
    x0=0.1,
    initial_history=None,
    q_bits=32,
    q_mode="round",
    map_type="logistic",
):
    a = float(a)
    b = float(b)
    k = _resolve_delay_k(k)
    map_name = _normalize_map_type(map_type)
    length = max(0, int(length))
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    map_id = _map_type_id(map_type)
    q_mode_id = _q_mode_id(q_mode)

    if initial_history is None:
        history = _derive_delay_initial_history(
            k,
            a=a,
            b=b,
            x0=x0,
            q_bits=q_bits,
            q_mode=q_mode,
            map_type=map_type,
        )
    else:
        history = np.asarray(initial_history, dtype=np.float64).reshape(-1)
        if history.size != k + 1:
            raise ValueError(f"initial_history must have exactly {k + 1} elements")
        history = np.clip(history, 1e-12, 1.0 - 1e-12)
        history = quantize_qL(history, q_bits=q_bits, q_mode=q_mode)

    head = 0
    tail = k

    q_bits = int(q_bits)
    q_enabled = q_bits > 0
    if q_enabled:
        q_scale = float(1 << q_bits)
        q_upper = 1.0 - (1.0 / q_scale)

        def quantize_scalar(v):
            if v < 0.0:
                v = 0.0
            elif v > q_upper:
                v = q_upper
            scaled = v * q_scale
            if q_mode_id == 0:
                out = math.floor(scaled) / q_scale
            elif q_mode_id == 1:
                out = math.ceil(scaled) / q_scale
            else:
                out = round(scaled) / q_scale
            if out < 0.0:
                return 0.0
            if out > q_upper:
                return q_upper
            return out
    else:
        def quantize_scalar(v):
            return v

    seq = np.empty(length, dtype=np.float64)
    period = k + 1

    if map_id == 0:
        for i in range(length):
            x_n = float(history[tail])
            x_n_minus_k = float(history[head])
            x_next = quantize_scalar((a * x_n * (1.0 - x_n) + b * x_n_minus_k) % 1.0)
            seq[i] = x_next
            head = (head + 1) % period
            tail = (tail + 1) % period
            history[tail] = x_next
    elif map_id == 1:
        for i in range(length):
            x_n = float(history[tail])
            x_n_minus_k = float(history[head])
            seed_value = x_n if x_n < 0.5 else 1.0 - x_n
            x_next = quantize_scalar((a * seed_value + b * x_n_minus_k) % 1.0)
            seq[i] = x_next
            head = (head + 1) % period
            tail = (tail + 1) % period
            history[tail] = x_next
    else:
        for i in range(length):
            x_n = float(history[tail])
            x_n_minus_k = float(history[head])
            x_next = quantize_scalar((a * math.sin(math.pi * x_n) + b * x_n_minus_k) % 1.0)
            seq[i] = x_next
            head = (head + 1) % period
            tail = (tail + 1) % period
            history[tail] = x_next
    return seq


def _generate_delayed_feedback_parallel_samples(
    num_samples,
    a=4.0,
    b=501.0,
    k=4,
    x0=0.1,
    burn_in=2048,
    decimation=12,
    q_bits=32,
    q_mode="round",
    map_type="logistic",
    chunk_size=32768,
):
    """
    Generate delayed-feedback samples from independent vectorized trajectories.

    This keeps the paper's delayed-feedback recurrence and q_L quantization, but
    avoids a single Python loop over burn_in + num_samples * decimation states.
    """
    num_samples = max(0, int(num_samples))
    if num_samples <= 0:
        return np.empty(0, dtype=np.float64)

    a = float(a)
    b = float(b)
    k = _resolve_delay_k(k)
    burn_in = max(0, int(burn_in))
    decimation = max(1, int(decimation))
    q_bits = max(0, int(q_bits))
    map_name = _normalize_map_type(map_type)
    map_id = _map_type_id(map_name)
    q_mode_id = _q_mode_id(q_mode)
    total_steps = burn_in + decimation
    chunk_size = max(1, int(chunk_size))

    if q_bits > 0:
        q_scale = float(1 << q_bits)
        q_upper = 1.0 - (1.0 / q_scale)

        def quantize_array(values):
            values = np.clip(values, 0.0, q_upper)
            scaled = values * q_scale
            if q_mode_id == 0:
                out = np.floor(scaled) / q_scale
            elif q_mode_id == 1:
                out = np.ceil(scaled) / q_scale
            else:
                out = np.round(scaled) / q_scale
            return np.clip(out, 0.0, q_upper)
    else:
        def quantize_array(values):
            return values

    output = np.empty(num_samples, dtype=np.float64)
    period = k + 1

    for start in range(0, num_samples, chunk_size):
        count = min(chunk_size, num_samples - start)
        seed = _stable_u64_seed(
            "delay_parallel_samples",
            float(a),
            float(b),
            int(k),
            float(x0),
            int(burn_in),
            int(decimation),
            int(q_bits),
            str(q_mode),
            map_name,
            int(start),
            int(count),
        )
        history = _splitmix64_uniform_block(seed, period * count).astype(np.float64, copy=False)
        history = history.reshape(period, count)
        eps = np.float64(1e-12)
        history = np.clip(history, eps, 1.0 - eps)
        per_sample_offset = _splitmix64_uniform_block(seed ^ 0xD1B54A32D192ED03, count).astype(np.float64, copy=False)
        history[0] = np.mod(float(x0) + per_sample_offset, 1.0)
        history[0] = np.clip(history[0], eps, 1.0 - eps)
        history = quantize_array(history)

        head = 0
        tail = k
        x_next = history[tail]
        for _ in range(total_steps):
            x_n = history[tail]
            x_n_minus_k = history[head]
            if map_id == 0:
                seed_values = x_n * (1.0 - x_n)
            elif map_id == 1:
                seed_values = np.where(x_n < 0.5, x_n, 1.0 - x_n)
            else:
                seed_values = np.sin(np.pi * x_n)
            x_next = quantize_array((a * seed_values + b * x_n_minus_k) % 1.0)
            head = (head + 1) % period
            tail = (tail + 1) % period
            history[tail] = x_next

        output[start:start + count] = x_next

    return output


def generate_delayed_feedback_noise(
    shape,
    a=4.0,
    b=501.0,
    k=4,
    x0=0.1,
    burn_in=2048,
    decimation=12,
    q_bits=32,
    q_mode="round",
    map_type="logistic",
):
    shape = torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape, dtype=torch.float32)

    k = _resolve_delay_k(k)
    burn_in = max(0, int(burn_in))
    decimation = max(1, int(decimation))
    q_bits = max(0, int(q_bits))

    sampled = _generate_delayed_feedback_parallel_samples(
        total_elements,
        a=a,
        b=b,
        k=k,
        x0=x0,
        burn_in=burn_in,
        decimation=decimation,
        q_bits=q_bits,
        q_mode=q_mode,
        map_type=map_type,
    )
    noise = torch.from_numpy(sampled.astype(np.float64, copy=False)).clone()
    noise = (2.0 * noise) - 1.0
    noise = noise - noise.mean()
    std = noise.std(unbiased=False)
    if std > 0:
        noise = noise / std
    return noise.reshape(shape).to(dtype=torch.float32)


# ====================  ====================
def compute_fisher_diag(model, dataloader):
    """Fisher - """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fisher_diag = []

    model.eval()
    model = model.to(device)

    for param in model.parameters():
        fisher_diag.append(torch.zeros_like(param).to(device))

    batch_count = 0
    total_samples = 0

    for data, labels in dataloader:
        if batch_count >= 3:
            break
        data, labels = data.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(data)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)

        for i in range(min(5, len(labels))):
            log_prob = log_probs[i, labels[i].item()]
            gradients = torch.autograd.grad(log_prob, model.parameters(), retain_graph=True)

            for j, grad in enumerate(gradients):
                if grad is not None:
                    fisher_diag[j] += grad.pow(2)

        batch_count += 1
        total_samples += min(5, len(data))

    if total_samples > 0:
        for i in range(len(fisher_diag)):
            fisher_diag[i] = fisher_diag[i] / total_samples

    model = model.cpu()
    return fisher_diag


def adaptive_privacy_budget(client_data_sizes, target_epsilon):
    """"""
    total_size = sum(client_data_sizes)
    privacy_budgets = []

    if total_size == 0:
        return [target_epsilon] * len(client_data_sizes)

    for size in client_data_sizes:
        weight = size / total_size
        epsilon = target_epsilon * (weight ** 0.5) * len(client_data_sizes)
        epsilon = min(epsilon, target_epsilon * 2.0)
        epsilon = max(epsilon, target_epsilon * 0.3)
        privacy_budgets.append(epsilon)

    total_budget = sum(privacy_budgets)
    if total_budget > 0:
        scaling_factor = target_epsilon * len(client_data_sizes) / total_budget
        privacy_budgets = [eps * scaling_factor for eps in privacy_budgets]

    return privacy_budgets


def sparsify_gradients(gradients, sparsity=0.4):
    """Placeholder docstring."""
    if sparsity <= 0 or gradients is None:
        return gradients

    sparsified = []
    for grad in gradients:
        if grad is None:
            sparsified.append(None)
            continue

        grad_flat = grad.abs().view(-1)
        k = int((1 - sparsity) * grad_flat.numel())

        if k > 0 and k < grad_flat.numel():
            values, indices = torch.topk(grad_flat, k)
            mask = torch.zeros_like(grad_flat)
            mask[indices] = 1
            mask = mask.view(grad.shape)
            sparsified_grad = grad * mask
        else:
            sparsified_grad = grad

        sparsified.append(sparsified_grad)

    return sparsified


def clip_gradients(gradients, clipping_bound):
    """Placeholder docstring."""
    if clipping_bound <= 0:
        return gradients

    clipped = []
    for grad in gradients:
        if grad is None:
            clipped.append(None)
            continue

        grad_norm = torch.norm(grad)
        if grad_norm > clipping_bound:
            clipped_grad = grad * (clipping_bound / grad_norm)
        else:
            clipped_grad = grad

        clipped.append(clipped_grad)

    return clipped


def compute_noise_multiplier(target_epsilon, target_delta, clipping_bound,
                             batch_size, dataset_size, num_epochs):
    """Placeholder docstring."""
    if target_epsilon <= 0 or dataset_size <= 0:
        return 0.0

    sampling_rate = batch_size / dataset_size
    steps = num_epochs * max(1, dataset_size // batch_size)

    sigma = clipping_bound * math.sqrt(2 * math.log(1.25 / target_delta)) / target_epsilon
    sigma = sigma / math.sqrt(steps * sampling_rate)

    return max(0.0, sigma)


def validate_privacy_params(epsilon, delta, clipping_bound):
    """Placeholder docstring."""
    if epsilon <= 0:
        print(f": epsilon={epsilon} 0?0.0")
        epsilon = 30.0
    elif epsilon < 1.0:
        print(f": epsilon={epsilon} ")
    elif epsilon > 1000.0:
        epsilon = min(epsilon, 1000.0)

    if delta <= 0 or delta >= 1:
        print(f": delta={delta} ?0,1)?e-5")
        delta = 1e-5
    elif delta < 1e-10:
        print(f": delta={delta} ")

    if clipping_bound <= 0:
        print(f": clipping_bound={clipping_bound} 0?.0")
        clipping_bound = 3.0
    elif clipping_bound > 10.0:
        clipping_bound = min(clipping_bound, 10.0)

    return epsilon, delta, clipping_bound



