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

# ==================== Ziggurat Normal Sampler ====================
def _derive_seed_from_uniform_sequence(uniform_sequence, max_seed_samples=8192):
    flat = uniform_sequence.detach().reshape(-1).to(dtype=torch.float64, device='cpu')
    total = int(flat.numel())
    if total <= 0:
        return 1

    if total > max_seed_samples:
        idx = torch.linspace(0, total - 1, steps=max_seed_samples).to(torch.int64)
        sampled = flat[idx]
    else:
        sampled = flat

    sampled = torch.clamp(sampled, min=1e-12, max=1.0 - 1e-12)
    scaled = torch.floor(sampled * ((1 << 53) - 1)).to(torch.int64).numpy().astype(np.uint64, copy=False)

    idx = np.arange(scaled.size, dtype=np.uint64)
    mixed = scaled ^ ((idx + np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9))
    seed = int(np.bitwise_xor.reduce(mixed, dtype=np.uint64))
    seed ^= int((scaled.size * 0x94D049BB133111EB) & ((1 << 64) - 1))
    seed &= (1 << 64) - 1
    if seed == 0:
        seed = 1
    return seed


def generate_ziggurat_gaussian_noise(shape, uniform_sequence):
    shape = torch.Size(shape)
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape, device=uniform_sequence.device, dtype=torch.float32)
    if uniform_sequence.numel() < total_elements:
        raise ValueError("Uniform sequence too short")

    u = uniform_sequence.reshape(-1)[:total_elements].to(dtype=torch.float32)
    u = torch.clamp(u, min=1e-6, max=1.0 - 1e-6)
    noise = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    noise = noise.reshape(shape).to(dtype=torch.float32)
    return noise.to(uniform_sequence.device)


# ==================== 3D-SCM ====================
def generate_3d_scm_sequence(length, a=300, b=150, x0=0.1, y0=0.2, z0=0.3,
                             multiplier=1024, decimation=2):
    x_seq = torch.zeros(length, dtype=torch.float64)
    y_seq = torch.zeros(length, dtype=torch.float64)
    z_seq = torch.zeros(length, dtype=torch.float64)

    x_seq[0], y_seq[0], z_seq[0] = float(x0) % 1.0, float(y0) % 1.0, float(z0) % 1.0

    scale = float(multiplier) if multiplier is not None else 1.0
    scale = min(max(1.0, scale), 4096.0)

    for i in range(length - 1):
        x_next = (a * x_seq[i] * x_seq[i] + b * z_seq[i]) % 1.0
        y_next = (a * y_seq[i] * y_seq[i] + b * x_seq[i]) % 1.0
        z_next = (a * z_seq[i] * z_seq[i] + b * y_seq[i]) % 1.0

        if scale != 1.0:
            x_scramble = (x_next * scale) % 1.0
            y_scramble = (y_next * scale) % 1.0
            z_scramble = (z_next * scale) % 1.0
            x_next = (0.85 * x_next + 0.15 * x_scramble) % 1.0
            y_next = (0.85 * y_next + 0.15 * y_scramble) % 1.0
            z_next = (0.85 * z_next + 0.15 * z_scramble) % 1.0

        x_seq[i + 1] = x_next
        y_seq[i + 1] = y_next
        z_seq[i + 1] = z_next

    return x_seq, y_seq, z_seq


def generate_3d_scm_gaussian_noise(shape, length=20, a=300, b=150,
                                   x0=0.1, y0=0.2, z0=0.3,
                                   multiplier=1024, decimation=2,
                                   burn_in=1024, thin=None, mix_lag=7,
                                   jitter=1e-4, max_direct_uniform=4096):
    total_elements = int(shape.numel())
    if total_elements <= 0:
        return torch.zeros(shape)

    total_uniform = 2 * total_elements
    thin_factor = int(decimation if thin is None else thin)
    thin_factor = max(1, thin_factor)
    burn_in = max(0, int(burn_in))
    lag = max(0, int(mix_lag))
    max_direct_uniform = max(2048, int(max_direct_uniform))

    direct_uniform_target = total_uniform if total_uniform <= max_direct_uniform else max_direct_uniform
    required_length = burn_in + thin_factor * (direct_uniform_target + lag + 64)
    seq_length = max(32, required_length)

    x_seq, y_seq, z_seq = generate_3d_scm_sequence(
        seq_length, a, b, x0, y0, z0, multiplier, decimation
    )

    x, y, z = x_seq[burn_in:], y_seq[burn_in:], z_seq[burn_in:]
    u = torch.remainder(x + 0.41421356237 * y + 0.73205080757 * z, 1.0)

    if lag > 0 and u.numel() > lag:
        u = torch.remainder(u[lag:] + 0.61803398875 * u[:-lag], 1.0)

    u = u[::thin_factor]

    if jitter and float(jitter) > 0:
        u = torch.remainder(u + (torch.rand_like(u) - 0.5) * float(jitter), 1.0)

    check_u = u[:min(int(u.numel()), 4096)]
    quantized = torch.floor(check_u * 4096).to(torch.int64)
    unique_bins = int(torch.unique(quantized).numel())
    if unique_bins < min(256, max(32, int(check_u.numel() // 64))):
        u = torch.remainder(0.8 * u + 0.2 * torch.rand_like(u), 1.0)

    if u.numel() < total_uniform:
        base_len = int(u.numel())
        repeat_times = (total_uniform + base_len - 1) // base_len
        tiled = u.repeat(repeat_times)[:total_uniform]
        phase = torch.remainder(torch.arange(repeat_times, dtype=u.dtype, device=u.device) * 0.7548776662466927, 1.0)
        uniform_vals = torch.remainder(tiled + phase.repeat_interleave(base_len)[:total_uniform], 1.0)
    else:
        uniform_vals = u[:total_uniform]

    pairs = uniform_vals.reshape(-1, 2)
    pairs[:, 1] = torch.remainder(pairs[:, 1] + 0.5 * pairs[:, 0], 1.0)
    uniform_vals = torch.clamp(pairs.reshape(-1), min=1e-10, max=1 - 1e-10)
    
    return generate_ziggurat_gaussian_noise(shape, uniform_vals).to(dtype=torch.float32)


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

                # 获取 DFL 间隙参数
                dfl_decimation = getattr(args, 'dfl_decimation', 2)

                noise = generate_dfl_gaussian_noise(
                    grad.shape,
                    mu=getattr(args, 'dfl_mu', 3.99),
                    alpha=getattr(args, 'dfl_alpha', 0.98),
                    x0=random.random(), x1=random.random(),
                    burn_in=getattr(args, 'dfl_burn_in', 512),
                    decimation=dfl_decimation, # 传入间隙
                    jitter=getattr(args, 'dfl_jitter', 1e-4),
                    max_direct_uniform=getattr(args, 'dfl_max_direct_uniform', 4096)
                ).to(grad_device)

                std = noise.std()
                if std > 0: noise = (noise / std) * sigma
                noisy_gradients.append(clipped_grad + noise)
            except:
                noisy_gradients.append(clipped_grad + generate_random_gaussian_noise_like(clipped_grad) * sigma)
        else:
            noisy_gradients.append(clipped_grad + (generate_random_gaussian_noise_like(clipped_grad) * sigma if sigma > 0 else 0))

    return noisy_gradients


def get_dataset_complexity_factor(name):
    return {'MNIST': 0.35, 'CIFAR10': 0.6, 'SVHN': 0.55, 'EMNIST': 0.45, 'FashionMNIST': 0.45}.get(name, 0.55)


# ==================== DFL (Discrete Fractional Logistic) ====================
# Pre-computed long sequences for fast reuse
_dfl_long_cache = {}  # key: (mu, alpha), value: (seq1, seq2) tuple
_DFL_PRECOMPUTE_LENGTH = 200000  # 限制预生成量为20万点，提供足够的切片空间

def _get_dfl_sequences(mu, alpha, x0, x1, needed_length):
    cache_key = (mu, alpha)
    precompute_len = 5000000  # 【水库总容量】：500 万

    # 如果没有缓存，使用纯 Python 列表进行极速生成（只需约1秒）
    if cache_key not in _dfl_long_cache or len(_dfl_long_cache.get(cache_key, ([], []))[0]) < precompute_len:
        print(f"\n[INFO] 正在极速预生成 {precompute_len} 步混沌池，请稍候约1-2秒...")
        x_list = [0.0] * precompute_len
        y_list = [0.0] * precompute_len
        x_list[0] = float(x0) % 1.0
        y_list[0] = float(x1) % 1.0

        mu_f, alpha_f = float(mu), float(alpha)
        oma = 1.0 - alpha_f

        for n in range(precompute_len - 1):
            prev = n - 1 if n > 0 else 0
            xn, yn = x_list[n], y_list[n]
            xp, yp = x_list[prev], y_list[prev]
            # 纯浮点运算，无调度开销
            x_list[n + 1] = (alpha_f * mu_f * xn * (1.0 - xn) + oma * xp) % 1.0
            y_list[n + 1] = (alpha_f * mu_f * yn * (1.0 - yn) + oma * yp) % 1.0

        _dfl_long_cache[cache_key] = (
            torch.tensor(x_list, dtype=torch.float64),
            torch.tensor(y_list, dtype=torch.float64)
        )
        print("[INFO] 混沌池极速生成完毕！")

    seq1, seq2 = _dfl_long_cache[cache_key]

    # 安全切片：确保需要长度不会超过水库
    if needed_length < len(seq1):
        start_idx1 = int((float(x0) * 104729) % max(1, len(seq1) - needed_length))
        start_idx2 = int((float(x1) * 104729) % max(1, len(seq2) - needed_length))
        return seq1[start_idx1:start_idx1 + needed_length].clone(), \
               seq2[start_idx2:start_idx2 + needed_length].clone()
    else:
        # 理论上不会走到这里，加了保险
        reps = (needed_length + len(seq1) - 1) // len(seq1)
        return seq1.repeat(reps)[:needed_length].clone(), seq2.repeat(reps)[:needed_length].clone()

def generate_dfl_gaussian_noise(shape, mu=3.99, alpha=0.98, x0=0.5, x1=0.6,
                                burn_in=512, decimation=2, jitter=1e-4,
                                max_direct_uniform=4096):
    total_elements = int(shape.numel()) if isinstance(shape, torch.Size) else int(torch.Size(shape).numel())
    if total_elements <= 0: return torch.zeros(shape if isinstance(shape, torch.Size) else torch.Size(shape))

    total_uniform = 2 * total_elements
    thin_factor = max(1, int(decimation))
    burn_in = max(0, int(burn_in))

    # 【关键修复点：抽水桶上限】：每次最多只取 200 万个点。
    # 这样给 500 万的水库留出了整整 300 万的随机滑动窗口，恢复真正的随机性！
    # 同时 200 万的内存开销远低于 500 万，彻底消除内存抖动。
    max_sequence_pool = 2000000
    base_length = min(max_sequence_pool, burn_in + (total_uniform * thin_factor) + 64)
    base_length = max(32, base_length)

    # 直接一次性获取两条基于不同种子 (x0, x1) 的切片
    seq1, seq2 = _get_dfl_sequences(mu, alpha, x0, x1, base_length)

    u = torch.remainder(seq1 + 0.61803398875 * seq2, 1.0)[burn_in:]
    u = u[::thin_factor]

    if jitter > 0:
        u = torch.remainder(u + (torch.rand_like(u) - 0.5) * jitter, 1.0)

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

    noise = generate_ziggurat_gaussian_noise(shape, u_final).to(dtype=torch.float32)

    # 符号翻转保平安
    if needs_sign_flip:
        base_noise_len = blen // 2
        reps_noise = (total_elements + base_noise_len - 1) // base_noise_len
        signs = (torch.randint(0, 2, (reps_noise,), device=noise.device, dtype=torch.float32) * 2 - 1)
        sign_mask = signs.repeat_interleave(base_noise_len)[:total_elements]
        noise = noise * sign_mask.reshape(shape)

    return noise

generate_simple_chaotic_noise = generate_dfl_gaussian_noise