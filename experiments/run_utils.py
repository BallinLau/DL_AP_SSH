"""
Utilities for multi-episode runs: model/optimizer/hparam builders, I/O helpers, and plots.
"""

from pathlib import Path
from typing import Optional
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config, HyperParams
from models import SDFFC1Combined, PolicyValueModel, FC2Model


def resolve_base_dir(run_root: Optional[Path], project_root: Path) -> Path:
    """Return run_root if provided; otherwise fall back to project_root."""
    return run_root if run_root is not None else project_root


def build_models(device: torch.device, ckpt_dir: Optional[Path | str] = None, ckpt_prefix: str | None = None, strict: bool = True):
    models = {
        "sdf_fc1": SDFFC1Combined(
            sdf_input_dim=Config.FC1_INPUT_DIM,
            fc1_input_dim=Config.FC1_INPUT_DIM,
            sdf_hidden_dims=Config.SDF_HIDDEN_DIMS,
            fc1_hidden_dims=Config.FC1_HIDDEN_DIMS,
            w_hidden_dims=Config.SDF_HIDDEN_DIMS,
        ).to(device),
        "policy_value": PolicyValueModel().to(device),
        "fc2": FC2Model(
            input_dim=Config.FC2_INPUT_DIM,
            hidden_dims=Config.FC2_HIDDEN_DIMS,
            quantile_num=Config.QUANTILE_NUM,
        ).to(device),
    }

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        prefix = f"{ckpt_prefix}_" if ckpt_prefix else ""
        mapping = {
            "sdf_fc1": "sdf_fc1",
            "policy_value": "policy_value",
            "fc2": "fc2",
        }
        for key, stem in mapping.items():
            ckpt_path = ckpt_dir / f"{prefix}{stem}.pt"
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location=device)
                models[key].load_state_dict(state, strict=strict)
                print(f"[build_models] loaded {ckpt_path}")
            else:
                print(f"[build_models] skip missing ckpt: {ckpt_path}")

    return models


def build_optimizers(models):
    opts = {}
    for name, model in models.items():
        opts[name] = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return opts


def build_hyperparams():
    hp = HyperParams(
        n_samples=2000,
        n_paths=200,
        batch_size=512,
        epochs=20,
        simulate_horizon=100,
    )
    hp.lr = 1e-3
    hp.min_lr = 1e-6
    hp.warmup_steps = 0
    hp.max_steps = 20000
    hp.w_sdf = 1.0
    hp.w_p0 = 1.0
    hp.w_pi = 1.0
    hp.w_q = 1.0
    hp.w_fc2 = 1.0
    return hp


def ensure_dirs(base_dir: Path):
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "outputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "experiments" / "figs").mkdir(parents=True, exist_ok=True)


def save_models(models, episode: int, base_dir: Path):
    torch.save(models["sdf_fc1"].state_dict(), base_dir / "checkpoints" / f"ep{episode}_sdf_fc1.pt")
    torch.save(models["policy_value"].state_dict(), base_dir / "checkpoints" / f"ep{episode}_policy_value.pt")
    torch.save(models["fc2"].state_dict(), base_dir / "checkpoints" / f"ep{episode}_fc2.pt")


def save_stage_df(ep: int, name: str, base_dir: Path, df_firm: pd.DataFrame = None, df_macro: pd.DataFrame = None, df_sdf: pd.DataFrame = None):
    out_dir = base_dir / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df = df_firm if df_firm is not None else df_sdf
    if target_df is not None:
        target_df.to_pickle(out_dir / f"ep{ep}_stage_{name}.pkl")
    if df_macro is not None:
        df_macro.to_pickle(out_dir / f"ep{ep}_stage_{name}_macro.pkl")


def plot_surfaces(ep: int, pv_model: PolicyValueModel, ref_state: dict, device: torch.device, base_dir: Path):
    figs_dir = base_dir / "experiments" / "figs"
    b_grid = torch.linspace(0, 1, 50, device=device)
    z_grid = torch.linspace(-4, 4, 50, device=device)
    B, Z = torch.meshgrid(b_grid, z_grid, indexing="ij")

    base = torch.stack(
        [
            B.reshape(-1),
            Z.reshape(-1),
            torch.full_like(B.reshape(-1), ref_state["eta"]),
            torch.full_like(B.reshape(-1), ref_state["i"]),
            torch.full_like(B.reshape(-1), ref_state["x"]),
            torch.full_like(B.reshape(-1), ref_state["hatcf"]),
            torch.full_like(B.reshape(-1), ref_state["lnkf"]),
        ],
        dim=1,
    )
    with torch.no_grad():
        out = pv_model(base)
        P0 = out.P0.reshape(B.shape).cpu().numpy()
        PI = out.PI.reshape(B.shape).cpu().numpy()
        bp = out.bp.reshape(B.shape).cpu().numpy()
        Q = out.Q.reshape(B.shape).cpu().numpy()

    for name, arr in [("p0", P0), ("pi", PI), ("bp", bp), ("q", Q)]:
        plt.figure(figsize=(6, 4))
        cs = plt.contourf(B.cpu().numpy(), Z.cpu().numpy(), arr, levels=30, cmap="viridis")
        plt.colorbar(cs)
        plt.xlabel("b")
        plt.ylabel("z")
        plt.title(f"EP{ep} {name.upper()} heatmap")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ep{ep}_{name}_heatmap.png", dpi=150)
        plt.close()

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(B.cpu().numpy(), Z.cpu().numpy(), arr, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("b")
        ax.set_ylabel("z")
        ax.set_zlabel(name.upper())
        ax.set_title(f"EP{ep} {name.upper()} surface")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ep{ep}_{name}_surface.png", dpi=150)
        plt.close()


def plot_distributions(ep: int, df: pd.DataFrame, pv_model: PolicyValueModel, device: torch.device, base_dir: Path):
    figs_dir = base_dir / "experiments" / "figs"

    if "M" in df.columns:
        plt.figure(figsize=(5, 3))
        df["M"].dropna().hist(bins=40)
        plt.title(f"EP{ep} M distribution")
        plt.xlabel("M")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ep{ep}_m_hist.png", dpi=150)
        plt.close()

    if "branch" in df.columns:
        parent_df = df[df["branch"] <= 0].copy()
    else:
        parent_df = df

    cols = ["b", "z", "ETA", "i", "x", "Hatcf", "LnKF"]
    X = torch.tensor(parent_df[cols].values, device=device, dtype=torch.float32)
    with torch.no_grad():
        bp = pv_model(X).bp.cpu().numpy()

    plt.figure(figsize=(5, 3))
    plt.hist(bp, bins=40)
    plt.title(f"EP{ep} bp distribution (parent states)")
    plt.xlabel("bp")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs_dir / f"ep{ep}_bp_hist.png", dpi=150)
    plt.close()


def plot_macro_series(ep: int, df_macro: pd.DataFrame, base_dir: Path):
    if df_macro is None or df_macro.empty:
        return
    figs_dir = base_dir / "experiments" / "figs"
    if "branch" in df_macro.columns:
        branch_col = df_macro["branch"]
    else:
        branch_col = pd.Series(0, index=df_macro.index)
    branch0 = df_macro[branch_col == -1]
    if branch0.empty:
        branch0 = df_macro
    series_hatc = branch0.groupby("t")["Hatc"].mean().reset_index()
    series_lnk = branch0.groupby("t")["LnK"].mean().reset_index()

    plt.figure(figsize=(6, 3))
    plt.plot(series_hatc["t"], series_hatc["Hatc"], marker="o")
    plt.xlabel("t")
    plt.ylabel("Hatc (avg, branch=0)")
    plt.title(f"EP{ep} Hatc macro series")
    plt.tight_layout()
    plt.savefig(figs_dir / f"ep{ep}_macro_hatc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.plot(series_lnk["t"], series_lnk["LnK"], marker="o")
    plt.xlabel("t")
    plt.ylabel("LnK (avg, branch=0)")
    plt.title(f"EP{ep} LnK macro series")
    plt.tight_layout()
    plt.savefig(figs_dir / f"ep{ep}_macro_lnk.png", dpi=150)
    plt.close()
