import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=2, help="Number of local epochs")
    parser.add_argument('--global_epoch', type=int, default=15, help="Number of global epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")

    parser.add_argument('--user_sample_rate', type=float, default=1, help="Sample rate for user sampling")

    parser.add_argument('--target_epsilon', type=float, default=1, help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1e-1, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=0.5, help="Gradient clipping bound")

    parser.add_argument('--fisher_threshold', type=float, default=0.4,
                        help="Fisher information threshold for parameter selection")
    parser.add_argument('--lambda_1', type=float, default=0.1, help="Lambda value for EWC regularization term")
    parser.add_argument('--lambda_2', type=float, default=0.05,
                        help="Lambda value for regularization term to control the update magnitude")

    parser.add_argument('--device', type=int, default=0, help='Set the visible CUDA device for calculations')

    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--seed', type=int, default=20260312,
                        help="Global random seed for reproducible comparison")

    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_noise', action='store_true')

    parser.add_argument('--dataset', type=str, default='SVHN')

    parser.add_argument('--dir_alpha', type=float, default=100)

    parser.add_argument('--dirStr', type=str, default='')

    parser.add_argument('--store', action='store_true')

    parser.add_argument('--appendix', type=str, default='')

    parser.add_argument('--sparsity_ratio', type=float, default=0.4,
                        help="Gradient sparsity ratio (default: 0.4)")

    parser.add_argument('--noise_decay', type=float, default=1.0,
                        help="Noise decay rate per epoch (default: 1.0, no decay)")

    parser.add_argument('--noise_decay_type', type=str, default='exponential',
                        choices=['exponential', 'linear', 'step'],
                        help="Type of noise decay: exponential, linear, or step")

    # 混沌差分隐私新增参数
    parser.add_argument('--use_chaotic', action='store_true',
                        help="Use chaotic differential privacy")

    parser.add_argument('--chaotic_factor', type=float, default=0.3,
                        help="Chaotic noise strength factor (default: 0.3)")

    # DFL (Discrete Fractional Logistic) 参数
    parser.add_argument('--dfl_mu', type=float, default=3.99,
                        help="DFL control parameter mu [3.57, 4] (default: 3.99)")
    parser.add_argument('--dfl_alpha', type=float, default=0.98,
                        help="DFL fractional order alpha (0, 1] (default: 0.98)")
    parser.add_argument('--dfl_burn_in', type=int, default=512,
                        help="Burn-in steps before using DFL samples (default: 512)")
    parser.add_argument('--dfl_jitter', type=float, default=1e-4,
                        help="Small jitter added to DFL sequence (default: 1e-4)")
    parser.add_argument('--dfl_max_direct_uniform', type=int, default=4096,
                        help="Max direct DFL uniforms before phase expansion (default: 4096)")
    parser.add_argument('--dfl_decimation', type=int, default=2,
                        help="DFL gap/decimation factor to break correlation (default: 2)")

    # 混沌衰减参数
    parser.add_argument('--chaotic_decay', type=float, default=0.9,
                        help="Decay rate of chaotic factor per epoch (default: 0.9)")

    # DP方法选择
    parser.add_argument('--dp_method', type=str, default='dfl',
                        choices=['none', 'gaussian', 'dfl'],
                        help="Differential privacy method: none, gaussian, dfl")

    # 新增：各DP方法的噪声因子
    parser.add_argument('--sigma_factor_gaussian', type=float, default=0.002,
                        help="Noise multiplier factor for Gaussian DP (default: 0.002)")
    parser.add_argument('--sigma_factor_dfl', type=float, default=0.002,
                        help="Noise multiplier factor for DFL DP (default: 0.002)")

    # Gradient inversion risk simulator (paper-style attack/reconstruction evaluation)
    parser.add_argument('--gir_attack_steps', type=int, default=30,
                        help="Optimization steps per inversion trial (default: 30)")
    parser.add_argument('--gir_attack_trials', type=int, default=2,
                        help="Number of inversion trials for avg/worst leakage (default: 2)")
    parser.add_argument('--gir_attack_lr', type=float, default=0.1,
                        help="Learning rate for inversion optimization (default: 0.1)")
    parser.add_argument('--gir_attack_batch_size', type=int, default=1,
                        help="Batch size used by inversion simulator (default: 1)")
    parser.add_argument('--gir_tv_weight', type=float, default=1e-4,
                        help="Total variation regularization weight in inversion (default: 1e-4)")
    parser.add_argument('--gir_l2_weight', type=float, default=1e-4,
                        help="Extra L2 weight in gradient matching objective (default: 1e-4)")
    parser.add_argument('--gir_eval_interval', type=int, default=0,
                        help="Run simulator every N batches (0 means once near the end of local update)")
    parser.add_argument('--gir_max_evals_per_client_update', type=int, default=1,
                        help="Max number of simulator calls in one local client update (default: 1)")
    args = parser.parse_args()

    return args
