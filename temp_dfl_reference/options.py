import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--local_epoch", type=int, default=2, help="Number of local epochs")
    parser.add_argument("--global_epoch", type=int, default=15, help="Number of global epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    parser.add_argument("--user_sample_rate", type=float, default=1.0, help="Sample rate for user sampling")

    parser.add_argument("--target_epsilon", type=float, default=1.0, help="Target privacy budget epsilon")
    parser.add_argument("--target_delta", type=float, default=1e-1, help="Target privacy budget delta")
    parser.add_argument(
        "--clipping_bound",
        type=float,
        default=2.0,
        help="Client model-update clipping bound for client-level DP-FedAvg",
    )
    parser.add_argument(
        "--train_grad_clip",
        type=float,
        default=0.0,
        help=(
            "Optional local training gradient clipping for numerical stability only. "
            "Set 0 to disable; DP clipping is applied to uploaded model updates."
        ),
    )

    parser.add_argument("--fisher_threshold", type=float, default=0.4,
                        help="Fisher information threshold for parameter selection")
    parser.add_argument("--lambda_1", type=float, default=0.1,
                        help="Lambda value for EWC regularization term")
    parser.add_argument("--lambda_2", type=float, default=0.05,
                        help="Lambda value for regularization term to control the update magnitude")

    parser.add_argument("--device", type=int, default=0, help="Set the visible CUDA device for calculations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=20260312,
                        help="Global random seed for reproducible comparison")

    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--no_noise", action="store_true")

    parser.add_argument("--dataset", type=str, default="SVHN")
    parser.add_argument("--dir_alpha", type=float, default=100)
    parser.add_argument("--dirStr", type=str, default="")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--appendix", type=str, default="")

    parser.add_argument("--sparsity_ratio", type=float, default=0.3,
                        help="Gradient sparsity ratio (default: 0.3)")

    parser.add_argument("--noise_decay", type=float, default=1.0,
                        help="Noise decay rate per epoch (default: 1.0, no decay)")
    parser.add_argument("--noise_decay_type", type=str, default="exponential",
                        choices=["exponential", "linear", "step"],
                        help="Type of noise decay: exponential, linear, or step")

    # Delayed-feedback parameters from the paper
    parser.add_argument("--a", type=float, default=4.0,
                        help="Delayed-feedback control parameter a (default: 4.0)")
    parser.add_argument("--b", type=float, default=501.0,
                        help="Delayed-feedback feedback parameter b (default: 501.0)")
    parser.add_argument("--k", type=int, default=4,
                        help="Delayed-feedback delay steps k (default: 4)")
    parser.add_argument("--burn_in", type=int, default=2048,
                        help="Burn-in steps before using delayed-feedback samples (default: 2048)")
    parser.add_argument("--decimation", type=int, default=12,
                        help="Sampling decimation factor for delayed-feedback sequence (default: 12)")
    parser.add_argument("--q_bits", type=int, default=32,
                        help="L-bit quantization precision q_L for delayed-feedback digital map (default: 32)")
    parser.add_argument("--q_mode", type=str, default="round", choices=["floor", "round", "ceil"],
                        help="Quantization function q_L: floor, round, or ceil (default: round)")
    parser.add_argument("--map_type", type=str, default="logistic", choices=["logistic", "tent", "sine"],
                        help="Seed map f(x) inside delayed feedback: logistic, tent, or sine (default: logistic)")

    parser.add_argument("--dp_method", type=str, default="delayed_feedback",
                        choices=["none", "gaussian", "delayed_feedback"],
                        help="Differential privacy method: none, gaussian, delayed_feedback")

    parser.add_argument("--sigma_factor_gaussian", type=float, default=0.005,
                        help="Noise multiplier factor for Gaussian DP (default: 0.005)")
    parser.add_argument("--sigma_factor_delayed_feedback", type=float, default=0.005,
                        help="Noise multiplier factor for delayed-feedback DP (default: 0.005)")

    # Gradient inversion risk simulator
    parser.add_argument("--gir_attack_steps", type=int, default=24,
                        help="Optimization steps per inversion trial (default: 24)")
    parser.add_argument("--gir_attack_trials", type=int, default=2,
                        help="Number of inversion trials for avg/worst leakage (default: 2)")
    parser.add_argument("--gir_attack_lr", type=float, default=0.1,
                        help="Learning rate for inversion optimization (default: 0.1)")
    parser.add_argument("--gir_attack_batch_size", type=int, default=1,
                        help="Batch size used by inversion simulator (default: 1)")
    parser.add_argument("--gir_tv_weight", type=float, default=1e-4,
                        help="Total variation regularization weight in inversion (default: 1e-4)")
    parser.add_argument("--gir_l2_weight", type=float, default=1e-4,
                        help="Extra L2 weight in gradient matching objective (default: 1e-4)")
    parser.add_argument("--gir_eval_interval", type=int, default=0,
                        help="Run simulator every N batches (0 means once near the end of local update)")
    parser.add_argument("--gir_max_evals_per_client_update", type=int, default=2,
                        help="Max number of simulator calls in one local client update (default: 2)")

    return parser.parse_args()
