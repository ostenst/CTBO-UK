import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ema_workbench import (
    Model,
    RealParameter,
    ArrayOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments,
)


def simulate_cbam(macc, E0=45, r=0.90, m=1, E=1, gamma=1, g=1, debug=False):
    """
    Calculate profit_cbam for each plant in the MACC.

    Parameters
    ----------
    macc : pd.DataFrame
        Plant data from macc.csv (constant across scenarios).
    m, E, gamma, g : float
        Uncertain input parameters.

    Returns
    -------
    dict with 'profit_cbam' : np.ndarray of shape (n_plants,)
    """
    if debug:
        print(f"simulate_cbam called with m={m}, E={E}, gamma={gamma}, g={g}")
    
    # Load carbon input values for each plant
    cf = macc['ktCO2f'].values
    cl = macc['ktCO2cem'].values
    cp = macc['ktCO2pl'].values
    cb = macc['ktCO2b'].values
    ctot = cf + cl + cp + cb

    # Load MACC cost
    S = macc['MAC'].values
    
    # Calculate profit_cbam for each plant
    profit_cbam = np.zeros(len(macc))
    profit_cbam = (cf+cl+cp)*(1-m)*(E0-E) + ctot*r*(E+gamma-S) - cf*g*gamma*(1-m) # [k€/y]

    if debug:
        print(f"profit_cbam: min={profit_cbam.min():.2f}, max={profit_cbam.max():.2f}")

    return {"profit_cbam": profit_cbam}


def plot_cbam_profit_by_sector(macc, profit_cbam, results_dir="results", debug=False):
    """Box plots of profit_cbam by sector across all scenarios."""
    sectors = macc["sector"].values
    unique_sectors = sorted(macc["sector"].unique())

    magma = plt.cm.magma
    sector_colors = {
        "cement": magma(0.1), "ccgt": magma(0.3), "refinery": magma(0.9),
        "steel": magma(0.7), "drax": magma(0.5), "waste": "#62a7a6",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    colors = []
    for s in unique_sectors:
        mask = sectors == s
        vals = profit_cbam[:, mask].flatten() / 1e3  # k€ → M€
        data.append(vals)
        labels.append(f"{s}\n(n={mask.sum()})")
        colors.append(sector_colors.get(s, "gray"))

    bp = ax.boxplot(data, positions=range(len(unique_sectors)), widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(range(len(unique_sectors)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Profit CBAM [M£/y]", fontsize=13)
    ax.set_title("CBAM profit by sector (all scenarios)", fontsize=14, fontweight="bold")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/cbam_profit_by_sector.png", dpi=200, bbox_inches="tight")
    if debug:
        print(f"Saved to {results_dir}/cbam_profit_by_sector.png")
    return fig


FOCUS_SECTORS = ["ccgt", "steel", "refinery", "cement"]


def plot_cbam_fraction_positive(macc, experiments, profit_cbam,
                                results_dir="results", debug=False):
    """
    For each plant in FOCUS_SECTORS, show the fraction of scenarios with
    positive profit. Horizontal bar chart sorted by fraction, colored by sector.
    """
    sector_mask = np.isin(macc["sector"].values, FOCUS_SECTORS)
    stacks = macc.loc[sector_mask, "stack"].values
    sectors = macc.loc[sector_mask, "sector"].values
    n_scenarios = profit_cbam.shape[0]

    frac_positive = (profit_cbam[:, sector_mask] > 0).sum(axis=0) / n_scenarios

    df = pd.DataFrame({
        "stack": stacks,
        "sector": sectors,
        "frac_positive": frac_positive,
    }).sort_values("frac_positive", ascending=True).reset_index(drop=True)

    magma = plt.cm.magma
    sector_colors = {
        "cement": magma(0.1), "ccgt": magma(0.3), "refinery": magma(0.9),
        "steel": magma(0.7), "drax": magma(0.5), "waste": "#62a7a6",
    }
    colors = [sector_colors.get(s, "gray") for s in df["sector"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.22)))
    ax.barh(range(len(df)), df["frac_positive"] * 100, color=colors,
            edgecolor="black", linewidth=0.3, alpha=0.8)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["stack"], fontsize=8)
    ax.set_xlabel("Scenarios with positive profit [%]", fontsize=13)
    ax.set_xlim(0, 105)
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=sector_colors[s], alpha=0.8,
                       edgecolor="black", linewidth=0.5, label=s.capitalize())
        for s in FOCUS_SECTORS
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    input_cols = [c for c in experiments.columns if c not in ("scenario", "policy", "model")]
    ranges = ", ".join(
        f"{c}=[{experiments[c].min():.1f}–{experiments[c].max():.1f}]"
        for c in input_cols
    )
    ax.set_title(
        f"Fraction of scenarios yielding positive CBAM profit per plant\n"
        f"({n_scenarios} scenarios: {ranges})",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(f"{results_dir}/cbam_fraction_positive.png", dpi=200, bbox_inches="tight")
    if debug:
        print(f"Saved to {results_dir}/cbam_fraction_positive.png")
    return fig


def run_cart(macc, experiments, profit_cbam, results_dir="results", debug=False):
    """
    Run CART scenario discovery on FOCUS_SECTORS plants to find input
    combinations that predict positive profit_cbam.
    """
    from ema_workbench.analysis.cart import CART
    from ema_workbench.analysis.scenario_discovery_util import RuleInductionType

    sector_mask = np.isin(macc["sector"].values, FOCUS_SECTORS)
    filtered_profit = profit_cbam[:, sector_mask]
    n_scenarios, n_plants = filtered_profit.shape

    input_cols = ["m", "E", "gamma", "g"]
    X_scenario = experiments[input_cols]

    x = X_scenario.loc[X_scenario.index.repeat(n_plants)].reset_index(drop=True)
    y = (filtered_profit.flatten() > 0).astype(int)

    if debug:
        sectors_str = ", ".join(FOCUS_SECTORS)
        print(f"CART input: {x.shape[0]} rows ({n_scenarios} scenarios × {n_plants} plants [{sectors_str}])")
        print(f"  Positive profit: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Features: {list(x.columns)}")

    cart = CART(x, y, mass_min=0.05, mode=RuleInductionType.BINARY)
    cart.build_tree()

    if debug:
        print("\n--- CART boxes (terminal leaves) ---")
        for i, (box, stat) in enumerate(zip(cart.boxes, cart.stats)):
            print(f"\nLeaf {i}: {stat}")
            print(box.to_string())

    try:
        fig = cart.show_tree(mplfig=True)
        fig.savefig(f"{results_dir}/cbam_cart.png", dpi=200, bbox_inches="tight")
        if debug:
            print(f"\nTree saved to {results_dir}/cbam_cart.png")
    except Exception as e:
        if debug:
            print(f"\nCould not render tree image (graphviz needed): {e}")

    return cart


def plot_filtered_scenarios(macc, experiments, profit_cbam,
                            m_min=0, E_min=0, gamma_min=0,
                            results_dir="results", debug=False):
    """
    Filter existing results to scenarios where m > m_min, E > E_min,
    gamma > gamma_min (any g). Plot median profit per plant with
    box-whisker spread across matching scenarios.
    """
    scen_mask = (
        (experiments["m"] > m_min) &
        (experiments["E"] > E_min) &
        (experiments["gamma"] > gamma_min)
    )
    n_match = scen_mask.sum()
    if debug:
        print(f"Filtered scenarios: {n_match}/{len(experiments)} "
              f"(m>{m_min}, E>{E_min}, gamma>{gamma_min})")

    sector_mask = np.isin(macc["sector"].values, FOCUS_SECTORS)
    filtered_profit = profit_cbam[scen_mask][:, sector_mask] / 1e3  # k€ → M€

    stacks = macc.loc[sector_mask, "stack"].values
    sectors = macc.loc[sector_mask, "sector"].values
    medians = np.median(filtered_profit, axis=0)

    order = np.argsort(medians)
    stacks = stacks[order]
    sectors = sectors[order]
    medians = medians[order]
    filtered_profit = filtered_profit[:, order]

    magma = plt.cm.magma
    sector_colors = {
        "cement": magma(0.1), "ccgt": magma(0.3), "refinery": magma(0.9),
        "steel": magma(0.7), "drax": magma(0.5), "waste": "#62a7a6",
    }

    n_plants = len(stacks)
    fig, ax = plt.subplots(figsize=(10, max(6, n_plants * 0.22)))

    bp = ax.boxplot(
        [filtered_profit[:, i] for i in range(n_plants)],
        positions=range(n_plants), vert=False, widths=0.6,
        patch_artist=True, showfliers=False,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(sector_colors.get(sectors[i], "gray"))
        patch.set_alpha(0.7)
    for median_line in bp["medians"]:
        median_line.set_color("black")

    ax.set_yticks(range(n_plants))
    ax.set_yticklabels(stacks, fontsize=8)
    ax.set_xlabel("Profit CBAM [M£/y]", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)

    n_pos = (medians > 0).sum()
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=sector_colors[s], alpha=0.7,
                       edgecolor="black", linewidth=0.5, label=s.capitalize())
        for s in FOCUS_SECTORS
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    ax.set_title(
        f"CBAM profit per plant — {n_match} scenarios with m>{m_min}, E>{E_min}, γ>{gamma_min}\n"
        f"{n_pos}/{n_plants} plants with positive median profit",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fname = f"{results_dir}/cbam_filtered_m{m_min}_E{E_min}_g{gamma_min}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    if debug:
        print(f"  {n_pos}/{n_plants} plants with positive median, saved to {fname}")
    return fig


def plot_cart_tree(cart, results_dir="results", debug=False):
    """Render the CART decision tree with magma colormap and compact nodes."""
    from sklearn.tree import plot_tree

    tree = cart.clf
    fig, ax = plt.subplots(figsize=(22, 10))

    annotations = plot_tree(
        tree,
        feature_names=["m", "E", "gamma", "g"],
        class_names=["Negative", "Positive"],
        filled=False,
        rounded=True,
        fontsize=7,
        ax=ax,
        impurity=False,
        proportion=True,
    )

    cmap = plt.cm.magma
    node_values = tree.tree_.value  # shape (n_nodes, 1, 2)
    n_nodes = len(node_values)
    node_idx = 0
    for ann in annotations:
        bbox = ann.get_bbox_patch()
        if bbox is None or node_idx >= n_nodes:
            continue
        n_neg, n_pos = node_values[node_idx, 0]
        frac_pos = n_pos / (n_neg + n_pos) if (n_neg + n_pos) > 0 else 0.5
        color = cmap(frac_pos)
        bbox.set_facecolor(color)
        bbox.set_edgecolor("grey")
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        ann.set_color("white" if luminance < 0.5 else "black")
        node_idx += 1

    fig.tight_layout()
    fig.savefig(f"{results_dir}/cbam_cart_tree.png", dpi=250, bbox_inches="tight")
    if debug:
        print(f"Tree saved to {results_dir}/cbam_cart_tree.png")
    return fig


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    macc = pd.read_csv("results/macc.csv")

    model = Model("CBAM", function=simulate_cbam)

    model.uncertainties = [
        RealParameter("m", 0, 1),       
        RealParameter("E", 0, 400),      
        RealParameter("gamma", 0, 400),   
        RealParameter("g", 0, 1),       
    ]

    model.constants = [
        Constant("macc", macc),
        Constant("E0", 45), # [£/tCO2] initial ETS price
        Constant("r", 0.90), # [-] capture rate
    ]

    model.outcomes = [
        ArrayOutcome("profit_cbam"),
    ]

    n_scenarios = 10000

    results = perform_experiments(
        model,
        n_scenarios,
        uncertainty_sampling=Samplers.LHS,
    )
    experiments, outcomes = results

    experiments.to_csv("results/cbam_experiments.csv", index=False)
    np.save("results/cbam_profit_cbam.npy", outcomes["profit_cbam"])

    print(f"\nCompleted {len(experiments)} experiments")
    print(f"profit_cbam shape: {outcomes['profit_cbam'].shape}")
    print(f"Saved to: results/cbam_experiments.csv, results/cbam_profit_cbam.npy")

    plot_cbam_profit_by_sector(macc, outcomes["profit_cbam"], debug=True)
    plot_cbam_fraction_positive(macc, experiments, outcomes["profit_cbam"], debug=True)
    cart = run_cart(macc, experiments, outcomes["profit_cbam"], debug=True)

    plot_filtered_scenarios(macc, experiments, outcomes["profit_cbam"],
                            debug=True)
    plot_filtered_scenarios(macc, experiments, outcomes["profit_cbam"],
                            m_min=0.77, E_min=141, gamma_min=87, debug=True)

    plot_cart_tree(cart, debug=True)
    plt.show()
