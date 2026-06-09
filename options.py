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
    parser.add_argument('--clipping_bound', type=float, default=2.0, help="Gradient clipping bound")

    parser.add_argument('--device', type=int, default=0, help='Set the visible CUDA device for calculations')

    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--seed', type=int, default=20260313,
                        help="Global random seed for reproducible comparison")

    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_noise', action='store_true')

    parser.add_argument('--dataset', type=str, default='SVHN')

    parser.add_argument('--dir_alpha', type=float, default=100)

    parser.add_argument('--sparsity_ratio', type=float, default=0.0,
                        help="Gradient sparsity ratio (default: 0.0)")

    # Noise kind (5-way taxonomy — replaces old dp_method/use_chaotic/chaotic_factor)
    # gaussian          : true iid N(0,1) noise via torch.rand → inverse-CDF
    # dfl_uniform       : raw chaotic, (u-0.5)*sqrt(12), bounded [-sqrt(3), sqrt(3))
    # dfl_gaussian      : DFL sequence → inverse-CDF → N(0,1)
    # mix_dgauss_gauss  : sqrt(α)·dfl_gaussian + sqrt(1-α)·gaussian
    # mix_dgauss_dchaos : sqrt(α)·dfl_gaussian + sqrt(1-α)·dfl_uniform
    parser.add_argument('--noise_kind', type=str, default='gaussian',
                        choices=['gaussian', 'dfl_uniform', 'dfl_gaussian',
                                 'mix_dgauss_gauss', 'mix_dgauss_dchaos'],
                        help="Noise mechanism (default: gaussian = the formal-DP baseline)")
    parser.add_argument('--mix_alpha', type=float, default=0.5,
                        help="Mix ratio α ∈ [0,1] for mix_* noise kinds (default: 0.5)")

    # DFL (Discrete Fractional Logistic) parameters — used by every non-gaussian noise_kind
    parser.add_argument('--dfl_a', type=float, default=4.0,
                help='DFL control parameter a (default: 4.0)')
    parser.add_argument('--dfl_b', type=float, default=501.0,
                help='DFL feedback parameter b (default: 501.0)')
    parser.add_argument('--dfl_k', type=int, default=3,
                help='DFL delay steps k (default: 3)')
    parser.add_argument('--dfl_burn_in', type=int, default=2048,
                        help="Burn-in steps before using DFL samples (default: 2048)")
    parser.add_argument('--dfl_decimation', type=int, default=11,
                        help="DFL gap/decimation factor to break correlation (default: 11)")

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
