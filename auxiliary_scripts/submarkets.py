import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_stylistic_macc(color=None, split=True, debug=False):
    """
    Stylistic MACC (laying S) with profit area(s) above the curve.
    split=True: two submarkets (mid-x split; low to mid-price, high to y-max).
    split=False: one segment from min to max x, curve up to y-max.
    """
    if color is None:
        color = plt.cm.magma(0.625)

    if debug:
        print(f"Creating stylistic MACC curve (split={split})")

    # Steep at ends, flatter in the middle (U-shaped slope → laying S)
    x = np.linspace(0, 1, 400)
    slope = (x - 0.5) ** 2 + 0.04
    y = np.cumsum(slope)
    y = (y - y.min()) / (y.max() - y.min()) + 0.5
    y_max = float(y.max())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    if split:
        mid_x = 0.66
        mid_price = float(np.interp(mid_x, x, y))
        low = x <= mid_x
        high = x >= mid_x
        ax.fill_between(
            x[low], y[low], mid_price,
            where=y[low] <= mid_price,
            color="#41BCAE", alpha=0.45, linewidth=0, zorder=1,
        )
        ax.fill_between(
            x[high], y[high], y_max,
            where=y[high] <= y_max,
            color="#237067", alpha=0.40, linewidth=0, zorder=1,
        )
        outfile = "results_figures/stylistic_macc.png"
    else:
        ax.fill_between(
            x, y, y_max,
            where=y <= y_max,
            color="#41BCAE", alpha=0.45, linewidth=0, zorder=1,
        )
        outfile = "results_figures/stylistic_macc_reference.png"

    ax.plot(x, y, color=color, lw=2.5, zorder=3)

    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, y_max + 0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    arrow_kw = dict(
        arrowstyle="->",
        mutation_scale=14,
        color="black",
        lw=1.2,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(FancyArrowPatch((0, 0), (1.08, 0), transform=ax.transData, **arrow_kw))
    ax.add_patch(FancyArrowPatch((0, 0), (0, y_max + 0.08), transform=ax.transData, **arrow_kw))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("Fossil CCS/durable CDR capacity [tCO₂ p.a.]", fontsize=14, labelpad=8)
    ax.set_ylabel("CSU price [£/tCO₂]", fontsize=14, labelpad=8)
    ax.xaxis.set_label_coords(0.5, -0.06)
    ax.yaxis.set_label_coords(-0.06, 0.5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=450, bbox_inches="tight")

    if debug:
        print(f"Plot saved to {outfile}")

    return fig


if __name__ == "__main__":
    plot_stylistic_macc(color="black", split=True, debug=True)
    plot_stylistic_macc(color="black", split=False, debug=True)
    plt.show()
