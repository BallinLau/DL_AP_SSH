"""
CLI-friendly multi-episode runner for Slurm/sbatch.

This mirrors the notebook `run_multi_episode.ipynb` setup and the existing
`run_multi_episode.py`, but adds CLI switches for quick tests and explicit run
root selection so it can be invoked from a Slurm batch script.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import logging

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
print(f"Added {ROOT} to sys.path for imports")
from config import Config  # noqa: E402
from training.episode import Episode  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from experiments.run_utils import (  # noqa: E402
    resolve_base_dir,
    build_models,
    build_optimizers,
    build_hyperparams,
    ensure_dirs,
    save_models,
    save_stage_df,
    plot_surfaces,
    plot_distributions,
    plot_macro_series,
)
from utils.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged multi-episode training")
    parser.add_argument("--run-root", type=Path, default=None, help="Override output root; default cachedir/timestamp")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per stage")
    parser.add_argument("--n-paths", type=int, default=None, help="Override n_paths for data")
    parser.add_argument("--simulate-horizon", type=int, default=None, help="Override simulate horizon")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu")
    parser.add_argument("--quick-test", action="store_true", help="Shrink workload for smoke tests (n_paths=10, epochs=20, horizon=20)")
    return parser.parse_args()


def configure_hyperparams(args: argparse.Namespace):
    hyperparams = build_hyperparams()
    if args.quick_test:
        hyperparams.n_paths = 50000
        hyperparams.epochs = 20
        hyperparams.simulate_horizon = 15
    if args.n_paths is not None:
        hyperparams.n_paths = args.n_paths
    if args.epochs is not None:
        hyperparams.epochs = args.epochs
    if args.simulate_horizon is not None:
        hyperparams.simulate_horizon = args.simulate_horizon
    return hyperparams


def make_run_root(arg_path: Path | None) -> Path:
    if arg_path is not None:
        return arg_path.expanduser().resolve()
    return ROOT.parent / "cachedir" / datetime.now().strftime("%Y%m%d_%H%M")


def main():
    args = parse_args()
    run_root = make_run_root(args.run_root)
    ensure_dirs(run_root)

    # Configure project-level logger under the run root
    log_dir = run_root / "logs"
    logger = setup_logger('DL-AP', log_dir)
    logger.info(f"Run root: {run_root}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    Config.DEVICE = device

    models = build_models(device, ckpt_dir="/home/fit/zhuyingz/WORK/LiuHao/cachedir/20260228_1055/checkpoints", ckpt_prefix="ep10")
    optimizers = build_optimizers(models)
    hyperparams = configure_hyperparams(args)

    stage_order = [
        ("sdf1", ["sdf_fc1"]),
        ("pv", ["policy_value"]),
        ("fc2", ["fc2"]),
    ]

    summaries = []
    episode = Episode(
            models=models,
            optimizers=optimizers,
            config=Config,
            hyperparams=hyperparams,
            device=device,
            episode_id=0,
        )
    change_circle = 5
    total_length = args.n_episodes
    mode_list = ["2time" if (i % (2*change_circle)) < change_circle else "full" for i in range(total_length)]
    for ep in range(args.n_episodes):
        train_mode = mode_list[ep]  # 根据 mode_list 设置训练模式
        episode.episode_id = ep  # update episode ID for logging/saving
        

        data_kwargs = {
            "n_samples": hyperparams.n_samples,
            "n_paths": hyperparams.n_paths if train_mode == "2time" else min(100, hyperparams.n_paths),
            "group_size": 2 if train_mode == "2time"  else Config.SIMULATE_GROUP_SIZE,
            "n_branches": Config.BRANCH_NUM,
        }
        simulate_kwargs = {"horizon": hyperparams.simulate_horizon} if ep > 0 else {}

        ep_summary = {}
        for stage_name, stage_modules in stage_order:
            summary = episode.run_episode(
                n_epochs=hyperparams.epochs,
                batch_size=hyperparams.batch_size,
                log_interval=50,
                train_modules=stage_modules,
                simulate_kwargs=simulate_kwargs,
                train_mode=train_mode,
                **data_kwargs,
            )
            ep_summary[stage_name] = summary.get("module_summaries", summary)
            save_stage_df(ep, stage_name, resolve_base_dir(run_root, ROOT), episode.df, episode.df_macro, episode.df_sdf)

        save_models(models, ep, resolve_base_dir(run_root, ROOT))

        parent_df = episode.df[episode.df["branch"] <= 0] if "branch" in episode.df.columns else episode.df
        ref_state = {
            "eta": 1.0,
            "i": parent_df["i"].median(),
            "x": parent_df["x"].median(),
            "hatcf": parent_df["Hatcf"].median(),
            "lnkf": parent_df["LnKF"].median(),
        }
        plot_surfaces(ep, models["policy_value"], ref_state, device, resolve_base_dir(run_root, ROOT))
        plot_distributions(ep, episode.df, models["policy_value"], device, resolve_base_dir(run_root, ROOT))
        plot_macro_series(ep, episode.df_macro, resolve_base_dir(run_root, ROOT))

        summaries.append(ep_summary)
        print(f"Episode {ep} done: {ep_summary}")

    print("All episodes done.")
    print(summaries)

    final_sim = SimulateTS(
        models=models,
        config=Config,
        n_paths=hyperparams.n_paths,
        group_size=Config.SIMULATE_GROUP_SIZE,
        branch_num=Config.BRANCH_NUM,
        horizon=hyperparams.simulate_horizon,
        device=device,
    )
    df_firm_sim, df_macro_sim = final_sim.simulate()
    out_dir = resolve_base_dir(run_root, ROOT) / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_firm_sim.to_pickle(out_dir / "final_simulate_firm.pkl")
    df_macro_sim.to_pickle(out_dir / "final_simulate_macro.pkl")
    plot_macro_series(-1, df_macro_sim, resolve_base_dir(run_root, ROOT))


if __name__ == "__main__":
    main()
