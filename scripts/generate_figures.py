"""Generate publication-quality figures for the Local Drift-Adapters paper.

Figure 1: Drift manifold — spatially correlated residuals in the old embedding space.
Figure 2: R@1 vs cluster count K for cross-family and cross-fam+dim pairs.

Usage:
    python scripts/generate_figures.py [--outdir paper/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ── Academic style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ── Figure 1: Drift Manifold ───────────────────────────────────────────────

def _generate_synthetic_drift_data(n_samples: int = 2000):
    """Simulate a heterogeneous embedding space with non-uniform drift.

    Three clusters experience qualitatively different transformations under
    the 'new' model, so a single global Procrustes map leaves spatially
    correlated residuals.
    """
    rng = np.random.default_rng(42)
    n = n_samples // 3

    # Cluster 1 — dense, scientific-like (rotation + shrink)
    c1 = rng.normal(loc=[5, 5], scale=1.0, size=(n, 2))
    theta = np.radians(30)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    c1_new = c1 @ rot * 0.8

    # Cluster 2 — spread-out, conversational (non-linear warp)
    c2 = rng.normal(loc=[-5, -5], scale=2.5, size=(n, 2))
    c2_new = c2 + np.sign(c2) * (c2 ** 2) * 0.05

    # Cluster 3 — elongated, financial (translation)
    t = rng.uniform(0, 10, n)
    c3 = np.column_stack([t - 5, t * 0.5 + 2]) + rng.normal(0, 0.3, size=(n, 2))
    c3_new = c3 + np.array([2, -2])

    X_old = np.vstack([c1, c2, c3])
    X_new = np.vstack([c1_new, c2_new, c3_new])

    # Global Procrustes solution
    U, _, Vt = np.linalg.svd(X_old.T @ X_new)
    W_global = U @ Vt
    residuals = np.linalg.norm(X_new - X_old @ W_global, axis=1)

    return X_old, residuals


def plot_drift_manifold(outdir: Path) -> None:
    X, residuals = _generate_synthetic_drift_data()

    fig, ax = plt.subplots(figsize=(6, 4.8))
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=residuals,
        cmap="magma_r",
        s=12, alpha=0.7, edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(
        r"Global adapter residual $\|f_{\mathrm{new}}(x) - A_{\mathrm{global}}(x)\|_2$",
        rotation=270, labelpad=18, fontsize=10,
    )

    # Region annotations — positioned to avoid overlap
    ax.annotate("Region A\n(rotation + scale)", xy=(5, 4), xytext=(4, -1),
                fontsize=9, weight="bold", color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                ha="center")
    ax.annotate("Region B\n(non-linear warp)", xy=(-7, -8), xytext=(-11, -3),
                fontsize=9, weight="bold", color="black",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                ha="center")
    ax.annotate("Region C\n(translation)", xy=(0, 3), xytext=(-4, 7),
                fontsize=9, weight="bold", color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                ha="center")

    ax.set_xlabel("Embedding dimension 1")
    ax.set_ylabel("Embedding dimension 2")
    ax.set_title("Spatially Correlated Drift: Why Global Adapters Under-Fit",
                 fontweight="bold", fontsize=12)

    out = outdir / "drift_manifold.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ── Figure 2: Performance vs Cluster Count ─────────────────────────────────

def plot_performance_curves(outdir: Path) -> None:
    """Line charts for R@1 vs K from the two sweep tables in the paper."""

    # ── Data from Tables 4 & 5 ──
    k_values = np.array([1, 2, 4, 8, 16, 32, 64])

    # Cross-family: MiniLM → BGE-small (Table 4)
    r1_cross_fam = np.array([0.887, 0.901, 0.922, 0.946, 0.959, 0.964, 0.963])

    # Cross-fam+dim: MiniLM → E5-large (Table 5)
    r1_hard = np.array([0.199, 0.255, 0.319, 0.374, 0.454, 0.523, 0.465])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, r1, label, color, baseline, title in [
        (ax1, r1_cross_fam,
         "Local Affine", "#377eb8", 0.887,
         r"MiniLM $\rightarrow$ BGE-small (384$\rightarrow$384)"),
        (ax2, r1_hard,
         "Local Affine", "#377eb8", 0.199,
         r"MiniLM $\rightarrow$ E5-large (384$\rightarrow$1024)"),
    ]:
        # Smooth interpolation for visual polish
        k_smooth = np.linspace(k_values.min(), k_values.max(), 300)
        spl = make_interp_spline(k_values, r1, k=2)
        r1_smooth = spl(k_smooth)

        ax.plot(k_smooth, r1_smooth, color=color, linestyle="--", alpha=0.35)
        ax.plot(k_values, r1, color=color, marker="o", markersize=5,
                linewidth=1.8, label=label, zorder=3)

        # Global baseline
        ax.axhline(y=baseline, color="#e41a1c", linestyle=":",
                    linewidth=1.2, label=f"Global Affine (K=1)")

        # Mark peak
        best_idx = int(np.argmax(r1))
        ax.annotate(
            f"K={k_values[best_idx]}",
            xy=(k_values[best_idx], r1[best_idx]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)
        ax.set_xlabel("Number of Clusters ($K$)")
        ax.set_ylabel("Recall@1")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(True, which="both", ls="-", alpha=0.15)

    fig.suptitle("Local Adapter Performance vs. Cluster Count",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()

    out = outdir / "performance_vs_k.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--outdir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_drift_manifold(args.outdir)
    plot_performance_curves(args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
