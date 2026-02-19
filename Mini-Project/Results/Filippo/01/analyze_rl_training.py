#!/usr/bin/env python3


# Usage: python analyze_rl_training.py --input metrics.csv --outdir figures --window 200 --zip



"""
RL training log visualizer for TurtleBot maze navigation.

Usage:
  python analyze_rl_training.py --input metrics.csv --outdir figures --window 200

Outputs:
  - PNG figures (01_*.png ...), a multi-page PDF report, and summary files.
"""
import os
import json
import argparse
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NUMERIC_HINTS = {
    "episode","reward","success","collision","steps","goal_x","goal_y","dist","min_scan",
    "r_progress","r_time","r_collision_avoid","r_yaw","r_terminal","epsilon"
}

def rolling_mean(s: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    minp = max(1, window // 10)
    return s.rolling(window=window, min_periods=minp).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to metrics.csv")
    ap.add_argument("--outdir", required=True, help="Output folder for figures/report")
    ap.add_argument("--window", type=int, default=200, help="Rolling window (episodes)")
    ap.add_argument("--zip", action="store_true", help="Also create a .zip of outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)

    # type coercion
    for c in df.columns:
        if c in NUMERIC_HINTS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    comp_cols = [c for c in ["r_progress","r_time","r_collision_avoid","r_yaw","r_terminal"] if c in df.columns]
    if comp_cols:
        df["reward_components_sum"] = df[comp_cols].sum(axis=1)
        if "reward" in df.columns:
            df["reward_diff"] = df["reward"] - df["reward_components_sum"]

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    x = df["episode"] if "episode" in df.columns and df["episode"].notna().any() else pd.Series(np.arange(1, len(df)+1), name="episode")

    # Summary JSON
    summary = {
        "rows": int(len(df)),
        "episodes_max": int(x.max()),
        "window_for_rolling_plots": int(args.window),
        "columns": list(df.columns),
    }
    for col in ["reward","success","collision","steps","dist","min_scan","epsilon"]:
        if col in df.columns:
            s = df[col].dropna()
            summary[col] = {
                "mean": float(s.mean()) if len(s) else None,
                "median": float(s.median()) if len(s) else None,
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()) if len(s) else None,
                "max": float(s.max()) if len(s) else None,
            }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Goal summary CSV
    if {"goal_x","goal_y"}.issubset(df.columns):
        goal_summary = df.groupby(["goal_x","goal_y"]).agg(
            episodes=("goal_x","count"),
            success_rate=("success","mean") if "success" in df.columns else ("goal_x","count"),
            collision_rate=("collision","mean") if "collision" in df.columns else ("goal_x","count"),
            reward_mean=("reward","mean") if "reward" in df.columns else ("goal_x","count"),
            reward_median=("reward","median") if "reward" in df.columns else ("goal_x","count"),
            steps_mean=("steps","mean") if "steps" in df.columns else ("goal_x","count"),
            dist_mean=("dist","mean") if "dist" in df.columns else ("goal_x","count"),
        ).reset_index()
        goal_summary.to_csv(os.path.join(args.outdir, "goal_summary.csv"), index=False)

    def finalize_and_save(fig, name, pdf):
        p = os.path.join(args.outdir, name)
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    pdf_path = os.path.join(args.outdir, "training_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Reward
        if "reward" in df.columns:
            fig = plt.figure(figsize=(10.5, 4.6))
            plt.plot(x, df["reward"], linewidth=0.8, alpha=0.25, label="Reward (per episode)")
            plt.plot(x, rolling_mean(df["reward"], args.window), linewidth=2.0, label=f"Reward (rolling mean, window={args.window})")
            plt.title("Episode reward over training")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend(loc="best")
            finalize_and_save(fig, "01_reward_over_time.png", pdf)

        # Rates
        if ("success" in df.columns) or ("collision" in df.columns):
            fig = plt.figure(figsize=(10.5, 4.6))
            if "success" in df.columns:
                plt.plot(x, rolling_mean(df["success"], args.window), linewidth=2.0, label=f"Success rate (rolling mean, window={args.window})")
            if "collision" in df.columns:
                plt.plot(x, rolling_mean(df["collision"], args.window), linewidth=2.0, label=f"Collision rate (rolling mean, window={args.window})")
            plt.title("Rolling outcome rates")
            plt.xlabel("Episode")
            plt.ylabel("Rate")
            plt.ylim(-0.02, 1.02)
            plt.legend(loc="best")
            finalize_and_save(fig, "02_outcome_rates.png", pdf)

        # Steps
        if "steps" in df.columns:
            fig = plt.figure(figsize=(10.5, 4.6))
            plt.plot(x, df["steps"], linewidth=0.8, alpha=0.25, label="Steps (per episode)")
            plt.plot(x, rolling_mean(df["steps"], args.window), linewidth=2.0, label=f"Steps (rolling mean, window={args.window})")
            plt.title("Episode length over training")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            plt.legend(loc="best")
            finalize_and_save(fig, "03_steps_over_time.png", pdf)

        # Distance
        if "dist" in df.columns:
            fig = plt.figure(figsize=(10.5, 4.6))
            plt.plot(x, df["dist"], linewidth=0.8, alpha=0.25, label="Final distance to goal (per episode)")
            plt.plot(x, rolling_mean(df["dist"], args.window), linewidth=2.0, label=f"Final distance (rolling mean, window={args.window})")
            plt.title("Final distance to goal over training")
            plt.xlabel("Episode")
            plt.ylabel("Distance")
            plt.legend(loc="best")
            finalize_and_save(fig, "04_distance_to_goal.png", pdf)

        # Min scan
        if "min_scan" in df.columns:
            fig = plt.figure(figsize=(10.5, 4.6))
            plt.plot(x, df["min_scan"], linewidth=0.8, alpha=0.25, label="Minimum scan distance (per episode)")
            plt.plot(x, rolling_mean(df["min_scan"], args.window), linewidth=2.0, label=f"Minimum scan (rolling mean, window={args.window})")
            plt.title("Minimum obstacle distance (proxy for safety) over training")
            plt.xlabel("Episode")
            plt.ylabel("Minimum scan distance")
            plt.legend(loc="best")
            finalize_and_save(fig, "05_min_scan.png", pdf)

        # Epsilon
        if "epsilon" in df.columns:
            fig = plt.figure(figsize=(10.5, 4.6))
            plt.plot(x, df["epsilon"], linewidth=2.0, label="Epsilon")
            plt.title("Exploration schedule (epsilon) over training")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.legend(loc="best")
            finalize_and_save(fig, "06_epsilon.png", pdf)

        # Components
        if comp_cols:
            fig = plt.figure(figsize=(10.5, 4.6))
            for c in comp_cols:
                plt.plot(x, rolling_mean(df[c], args.window), linewidth=2.0, label=f"{c} (rolling mean)")
            if "reward" in df.columns:
                plt.plot(x, rolling_mean(df["reward"], args.window), linewidth=2.5, label="reward (rolling mean)")
            plt.title("Reward decomposition trends (rolling means)")
            plt.xlabel("Episode")
            plt.ylabel("Value")
            plt.legend(loc="best", ncol=2)
            finalize_and_save(fig, "07_reward_components.png", pdf)

        # Goals
        if {"goal_x","goal_y"}.issubset(df.columns):
            goals = df[["goal_x","goal_y"]].drop_duplicates().dropna().sort_values(["goal_x","goal_y"]).reset_index(drop=True)
            goal_map = {tuple(r): i for i, r in enumerate(goals.to_numpy())}
            goal_id = df.apply(lambda r: goal_map.get((r["goal_x"], r["goal_y"]), np.nan), axis=1)

            fig = plt.figure(figsize=(10.5, 4.6))
            plt.scatter(x, goal_id, s=10, alpha=0.35, label="Sampled goal (id)")
            if "success" in df.columns:
                suc = df["success"] == 1
                plt.scatter(x[suc], goal_id[suc], s=35, marker="x", label="Success episodes")
            plt.yticks(list(goal_map.values()), [f"({gx:.2g}, {gy:.2g})" for gx, gy in goal_map.keys()])
            plt.title("Goal sampling over time (and successes)")
            plt.xlabel("Episode")
            plt.ylabel("Goal (x, y)")
            plt.legend(loc="best")
            finalize_and_save(fig, "08_goal_sampling_timeline.png", pdf)

            if "success" in df.columns:
                goal_rates = df.groupby(["goal_x","goal_y"])["success"].mean().reset_index()
                labels = [f"({r.goal_x:.2g}, {r.goal_y:.2g})" for r in goal_rates.itertuples(index=False)]
                fig = plt.figure(figsize=(10.5, 4.6))
                plt.bar(np.arange(len(goal_rates)), goal_rates["success"].to_numpy())
                plt.xticks(np.arange(len(goal_rates)), labels, rotation=0)
                plt.title("Success rate by goal")
                plt.xlabel("Goal (x, y)")
                plt.ylabel("Success rate")
                plt.ylim(0, max(0.05, float(goal_rates["success"].max()) * 1.2))
                finalize_and_save(fig, "09_goal_success_rate.png", pdf)

        # Reward distribution
        if "reward" in df.columns and ("success" in df.columns or "collision" in df.columns):
            fig = plt.figure(figsize=(10.5, 4.6))
            bins = 60
            plt.hist(df["reward"].dropna(), bins=bins, alpha=0.35, density=True, label="All episodes")
            if "success" in df.columns and df["success"].sum() > 0:
                plt.hist(df.loc[df["success"] == 1, "reward"].dropna(), bins=bins, alpha=0.35, density=True, label="Success episodes")
            if "collision" in df.columns and df["collision"].sum() > 0:
                plt.hist(df.loc[df["collision"] == 1, "reward"].dropna(), bins=bins, alpha=0.35, density=True, label="Collision episodes")
            plt.title("Reward distribution (density)")
            plt.xlabel("Reward")
            plt.ylabel("Density")
            plt.legend(loc="best")
            finalize_and_save(fig, "10_reward_distribution.png", pdf)

        # Residual vs component sum
        if "reward" in df.columns and "reward_components_sum" in df.columns:
            fig = plt.figure(figsize=(10.5, 4.6))
            plt.plot(x, rolling_mean(df["reward_diff"], args.window), linewidth=2.0, label=f"reward - sum(components) (rolling mean, window={args.window})")
            plt.title("Consistency check: reward vs logged components")
            plt.xlabel("Episode")
            plt.ylabel("Difference")
            plt.legend(loc="best")
            finalize_and_save(fig, "11_reward_component_residual.png", pdf)

    if args.zip:
        zip_path = os.path.join(args.outdir, "outputs.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fn in os.listdir(args.outdir):
                if fn.endswith((".png",".pdf",".json",".csv",".py")):
                    zf.write(os.path.join(args.outdir, fn), arcname=fn)

if __name__ == "__main__":
    main()
