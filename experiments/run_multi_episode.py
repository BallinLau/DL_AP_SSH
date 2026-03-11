"""
Multi-episode staged training runner.

Episode 0 uses Sample data.
Episode >=1 uses SimulateTS (generated with weights from previous episode).
Within each episode we call Episode.run_episode once per stage sequence:
    sdf_fc1 -> policy_value -> sdf_fc1 -> fc2
Models/optimizers are continuous across episodes.

Each episode saves:
- checkpoints/ep{episode}_{model}.pt
- data/outputs/ep{episode}_stage_{name}.pkl (+_macro if available)
- experiments/figs/ep{episode}_{p0,pi,bp,q}_{heatmap,surface}.png
- experiments/figs/ep{episode}_{m,bp}_hist.png
"""

import sys
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config, HyperParams  # noqa: E402
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

# Per-run cache directory (set in main). Lives outside code tree at ROOT.parent / cachedir.
RUN_ROOT: Path | None = None


def get_base_dir() -> Path:
    """Return the run-specific base dir if set, otherwise project root."""
    return resolve_base_dir(RUN_ROOT, ROOT)



def plot_stage_losses(all_summaries, figs_dir: Path):
    """
    Plot loss trajectories per stage across episodes, and per-episode bar charts.
    all_summaries: list of ep_summary dicts (one per episode)
    """
    if not all_summaries:
        return
    stage_loss_history = {}
    for ep_idx, ep_summary in enumerate(all_summaries):
        for stage, summary in ep_summary.items():
            losses = summary.get("final_losses", summary)
            for k, v in losses.items():
                stage_loss_history.setdefault(stage, {}).setdefault(k, []).append(v)

    figs_dir.mkdir(parents=True, exist_ok=True)
    eps = list(range(len(all_summaries)))

    # 跨 episode 轨迹
    for stage, loss_dict in stage_loss_history.items():
        for loss_name, values in loss_dict.items():
            padded = values + [float("nan")] * (len(eps) - len(values))
            plt.figure(figsize=(5, 3))
            plt.plot(eps, padded, marker="o")
            plt.xlabel("episode")
            plt.ylabel(loss_name)
            plt.title(f"{stage} - {loss_name}")
            plt.tight_layout()
            plt.savefig(figs_dir / f"loss_{stage}_{loss_name}.png", dpi=150)
            plt.close()

    # 每个 episode 单独画一张（按 stage 汇总）
    for ep_idx, ep_summary in enumerate(all_summaries):
        for stage, summary in ep_summary.items():
            losses = summary.get("final_losses", summary)
            names = list(losses.keys())
            vals = [losses[n] for n in names]
            plt.figure(figsize=(6, 4))
            plt.bar(range(len(names)), vals)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.ylabel("loss")
            plt.title(f"episode {ep_idx} - {stage}")
            plt.tight_layout()
            plt.savefig(figs_dir / f"loss_ep{ep_idx}_{stage}.png", dpi=150)
            plt.close()


def main():
    global RUN_ROOT
    RUN_ROOT = ROOT.parent / "cachedir" / datetime.now().strftime("%Y%m%d_%H%M")
    ensure_dirs(RUN_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device

    n_episodes = 10  # episode 0 + simulate episodes
    models = build_models(device)
    optimizers = build_optimizers(models)
    hyperparams = build_hyperparams()

    summaries = []
    for ep in range(n_episodes):
        episode = Episode(
            models=models,
            optimizers=optimizers,
            config=Config,
            hyperparams=hyperparams,
            device=device,
            episode_id=ep,
        )

        # data settings per episode
        data_kwargs = {
            "n_samples": hyperparams.n_samples,
            "n_paths": hyperparams.n_paths if ep == 0 else min(100, hyperparams.n_paths),
            "group_size": 2 if ep == 0 else Config.SIMULATE_GROUP_SIZE,
            "n_branches": Config.BRANCH_NUM,
        }
        simulate_kwargs = {"horizon": hyperparams.simulate_horizon} if ep > 0 else None

        stage_order = [
            ("sdf1", ["sdf_fc1"]),
            ("pv", ["policy_value"]),
            ("sdf2", ["sdf_fc1"]),
            ("fc2", ["fc2"]),
        ]
        ep_summary = {}
        for stage_name, stage_modules in stage_order:
            summary = episode.run_episode(
                n_epochs=hyperparams.epochs,
                batch_size=hyperparams.batch_size,
                log_interval=50,
                train_modules=stage_modules,
                simulate_kwargs=simulate_kwargs,
                **data_kwargs,
            )
            ep_summary[stage_name] = summary.get("module_summaries", summary)
            # save dfs for this stage
            save_stage_df(ep, stage_name, get_base_dir(), episode.df, episode.df_macro, episode.df_sdf)

        # save models after each episode
        save_models(models, ep, get_base_dir())

        # plots for this episode using latest df
        parent_df = episode.df[episode.df["branch"] <= 0] if "branch" in episode.df.columns else episode.df
        ref_state = {
            "eta": 1.0,
            "i": parent_df["i"].median(),
            "x": parent_df["x"].median(),
            "hatcf": parent_df["Hatcf"].median(),
            "lnkf": parent_df["LnKF"].median(),
        }
        plot_surfaces(ep, models["policy_value"], ref_state, device, get_base_dir())
        plot_distributions(ep, episode.df, models["policy_value"], device, get_base_dir())
        plot_macro_series(ep, episode.df_macro, get_base_dir())

        summaries.append(ep_summary)
        print(f"Episode {ep} done: {ep_summary}")

    print("All episodes done.")
    print(summaries)
    plot_stage_losses(summaries, get_base_dir() / "experiments" / "figs")

    # 额外模拟一次使用最终模型的数据，并导出以便宏观画图
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
    out_dir = get_base_dir() / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_firm_sim.to_pickle(out_dir / "final_simulate_firm.pkl")
    df_macro_sim.to_pickle(out_dir / "final_simulate_macro.pkl")
    plot_macro_series(-1, df_macro_sim, get_base_dir())


if __name__ == "__main__":
    main()
