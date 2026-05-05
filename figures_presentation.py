import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import figures_article as fa


def _clear_figure_legends(fig, debug=False):
    n = len(fig.legends)
    for lg in list(fig.legends):
        lg.remove()
    if debug:
        print(f"_clear_figure_legends output: removed={n}")


def _add_inside_legend(ax, handles, labels, loc="upper left", ncol=1, fontsize=9, debug=False):
    if not handles:
        return
    # Keep label order while removing duplicates.
    dedup = dict(zip(labels, handles))
    legend = ax.legend(
        dedup.values(),
        dedup.keys(),
        loc=loc,
        ncol=1,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
    )
    # Keep legend as overlay; don't let layout engine resize panels for it.
    legend.set_in_layout(False)
    if debug:
        print(f"_add_inside_legend output: n_labels={len(dedup)}")


def _save_presentation(fig, output_path, width=14.0, height=8.0, debug=False):
    fig.set_size_inches(width, height, forward=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    if debug:
        print(f"_save_presentation output: file={output_path}, size=({width}, {height})")


def _policy_slug(policy):
    slug = (
        str(policy)
        .replace("£", "GBP")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    return "".join(ch for ch in slug if ch.isalnum() or ch == "_").strip("_").lower()


def figure_policy_focus_presentation(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    policy="£200-Mix",
    savefig=True,
    debug=False,
):
    if debug:
        print("figure_policy_focus_presentation input:", f"policy={policy}")

    years = None
    cases = []
    for results_dir in ["results_baseline", "results_phaseout"]:
        experiments = pd.read_csv(f"{results_dir}/experiments.csv")
        if years is None:
            n_years = fa._load_array(results_dir, "cost_marginal", debug=debug).shape[1]
            years = np.arange(start_year, start_year + n_years, dtype=int)
        cases.append(
            {
                "experiments": experiments,
                "marginal": fa._load_array(results_dir, "cost_marginal", debug=debug) / pounds_to_EUR,
                "ets": fa._load_array(results_dir, "price_ETS", debug=debug) / pounds_to_EUR,
                "csu": fa._load_array(results_dir, "price_CSU", debug=debug) / pounds_to_EUR,
                "fuel_cost": fa._load_array(results_dir, "cost_fuels", debug=debug) / pounds_to_EUR,
                "suppliers": fa._load_array(results_dir, "costs_suppliers", debug=debug),
                "emitters": fa._load_array(results_dir, "costs_emitters", debug=debug),
                "tax": fa._load_array(results_dir, "costs_tax", debug=debug),
                "consumers": fa._load_array(results_dir, "costs_consumers", debug=debug),
                "gas_inc": fa._load_array(results_dir, "gas_increase_abs", debug=debug) / pounds_to_EUR * (100 / 1000),
                "petrol_inc": fa._load_array(results_dir, "petrol_increase_abs", debug=debug),
            }
        )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), sharex=True)
    magma = plt.cm.magma
    color_marginal = magma(0.10)
    color_emitters_ets = magma(0.35)
    color_consumers_fuel = magma(0.60)
    color_suppliers_y = magma(0.85)
    color_tax = "#62a7a6"
    scale_bgbp = 1e-6 / pounds_to_EUR
    use = years <= 2050
    years_plot = years[use]

    for r, case in enumerate(cases):
        mask = case["experiments"]["policy"] == policy
        if mask.sum() == 0:
            continue

        ax = axes[r, 0]
        med, p05, p95 = fa._panel_stats(case["marginal"][mask][:, use], debug=debug)
        ax.plot(years_plot, med, lw=2.3, color=color_marginal, label="Marginal CO2 storage cost")
        ax.fill_between(years_plot, p05, p95, color=color_marginal, alpha=0.2)
        med, p05, p95 = fa._panel_stats(case["ets"][mask][:, use], debug=debug)
        ax.plot(years_plot, med, lw=2.3, color=color_emitters_ets, label="ETS price")
        ax.fill_between(years_plot, p05, p95, color=color_emitters_ets, alpha=0.2)
        med, p05, p95 = fa._panel_stats(case["csu"][mask][:, use], debug=debug)
        ax.plot(years_plot, med, lw=2.3, color=color_suppliers_y, label="CSU price")
        ax.fill_between(years_plot, p05, p95, color=color_suppliers_y, alpha=0.2)
        med, p05, p95 = fa._panel_stats(case["fuel_cost"][mask][:, use], debug=debug)
        ax.plot(years_plot, med, lw=2.3, color=color_consumers_fuel, label="Consumer fuel cost")
        ax.fill_between(years_plot, p05, p95, color=color_consumers_fuel, alpha=0.2)
        ax.set_ylabel("Carbon price [£/tCO2]", fontsize=13)

        ax = axes[r, 1]
        med, p05, p95 = fa._panel_stats(case["suppliers"][mask][:, use], debug=debug)
        ax.plot(years_plot, med * scale_bgbp, lw=2.3, color=color_suppliers_y, label="Fuel supplier cost")
        ax.fill_between(years_plot, p05 * scale_bgbp, p95 * scale_bgbp, color=color_suppliers_y, alpha=0.2)
        med, p05, p95 = fa._panel_stats(case["emitters"][mask][:, use], debug=debug)
        ax.plot(years_plot, med * scale_bgbp, lw=2.3, color=color_emitters_ets, label="Emitter cost")
        ax.fill_between(years_plot, p05 * scale_bgbp, p95 * scale_bgbp, color=color_emitters_ets, alpha=0.2)
        med, p05, p95 = fa._panel_stats(case["tax"][mask][:, use], debug=debug)
        ax.plot(years_plot, med * scale_bgbp, lw=2.3, color=color_tax, label="Emissions tax")
        ax.fill_between(years_plot, p05 * scale_bgbp, p95 * scale_bgbp, color=color_tax, alpha=0.2)
        med, p05, p95 = fa._panel_stats(case["consumers"][mask][:, use], debug=debug)
        ax.plot(years_plot, med * scale_bgbp, lw=2.3, color=color_consumers_fuel, label="Consumer costs")
        ax.fill_between(years_plot, p05 * scale_bgbp, p95 * scale_bgbp, color=color_consumers_fuel, alpha=0.2)
        npv_vals = pd.to_numeric(
            case["experiments"].loc[mask, "NPV_costs_consumers"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        tax_vals = pd.to_numeric(
            case["experiments"].loc[mask, "NPV_costs_tax"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        if npv_vals.size:
            npv_p05 = np.percentile(npv_vals, 5) * scale_bgbp
            npv_p95 = np.percentile(npv_vals, 95) * scale_bgbp
            npv_text = (
                "NPV consumer costs [B£]\n"
                f"p05-p95: {npv_p05:.2f} to {npv_p95:.2f}"
            )
            if tax_vals.size:
                tax_p05 = np.percentile(tax_vals, 5) * scale_bgbp
                tax_p95 = np.percentile(tax_vals, 95) * scale_bgbp
                npv_text += (
                    "\nNPV emissions tax [B£]\n"
                    f"p05-p95: {tax_p05:.2f} to {tax_p95:.2f}"
                )
            ax.text(
                0.04,
                0.66,
                npv_text,
                transform=ax.transAxes,
                va="center",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
            )
        ax.set_ylabel("Annual policy costs [B£ p.a.]", fontsize=13)

        ax = axes[r, 2]
        med, p05, p95 = fa._panel_stats(case["gas_inc"][mask][:, use], debug=debug)
        ax.plot(years_plot, med, lw=2.3, color=magma(0.70), label="Gas price increase")
        ax.fill_between(years_plot, p05, p95, color=magma(0.70), alpha=0.24)
        ax.set_ylabel("Gas price increase [p/kWh]", fontsize=13)

        ax = axes[r, 3]
        med, p05, p95 = fa._panel_stats(case["petrol_inc"][mask][:, use], debug=debug)
        ax.plot(years_plot, med, lw=2.3, color=magma(0.75), label="Petrol price increase")
        ax.fill_between(years_plot, p05, p95, color=magma(0.75), alpha=0.24)
        ax.set_ylabel("Petrol price increase [p/L]", fontsize=13)

    titles = ["Carbon prices", "Annual policy costs", "Gas prices", "Petrol prices"]
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=14)

    for r in range(2):
        for c in range(4):
            ax = axes[r, c]
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.tick_params(labelsize=11)
            ax.set_xlabel("Year", fontsize=13)
            ax.set_xlim(int(years_plot.min()), int(years_plot.max()))
            fa._set_sparse_year_ticks(ax, years_plot, debug=debug)

    for c in range(4):
        axes[0, c].tick_params(labelbottom=False)

    axes[0, 0].text(-0.32, 0.5, "PHASEOUT=False", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12)
    axes[1, 0].text(-0.32, 0.5, "PHASEOUT=True", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12)

    for c in [0, 1]:
        h, l = axes[0, c].get_legend_handles_labels()
        _add_inside_legend(axes[0, c], h, l, loc="upper left", ncol=1, fontsize=9, debug=debug)

    h, l = axes[0, 2].get_legend_handles_labels()
    _add_inside_legend(axes[0, 2], h, l, loc="upper left", ncol=1, fontsize=9, debug=debug)
    h, l = axes[0, 3].get_legend_handles_labels()
    _add_inside_legend(axes[0, 3], h, l, loc="upper left", ncol=1, fontsize=9, debug=debug)

    fig.tight_layout()
    if savefig:
        out = f"{figures_dir}/figure_policy_focus_{_policy_slug(policy)}_presentation.png"
        fig.savefig(out, dpi=450, bbox_inches="tight")
        if debug:
            print(f"figure_policy_focus_presentation output: file={out}")
    return fig


def main(debug=False, policy="ETS-eq"):
    if debug:
        print("main input:", f"debug={debug}, policy={policy}")

    fig4 = fa.figure_4_carbon(savefig=False, debug=debug)
    _clear_figure_legends(fig4, debug=debug)
    if len(fig4.axes) >= 4:
        h_top, l_top = fig4.axes[0].get_legend_handles_labels()
        h_bot, l_bot = fig4.axes[2].get_legend_handles_labels()
        _add_inside_legend(fig4.axes[0], h_top, l_top, loc="upper left", ncol=1, fontsize=10, debug=debug)
        _add_inside_legend(fig4.axes[2], h_bot, l_bot, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig4, "results_figures/figure_4_carbon_presentation.png", width=10, height=8.6, debug=debug)

    fig5 = fa.figure_5_prices(savefig=False, debug=debug, PHASEOUT=False)
    _clear_figure_legends(fig5, debug=debug)
    if len(fig5.axes) >= 6:
        h_top, l_top = fig5.axes[0].get_legend_handles_labels()
        h_bot, l_bot = fig5.axes[3].get_legend_handles_labels()
        _add_inside_legend(fig5.axes[0], h_top, l_top, loc="upper left", ncol=1, fontsize=10, debug=debug)
        _add_inside_legend(fig5.axes[3], h_bot, l_bot, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig5, "results_figures/figure_5_prices_presentation.png", width=10, height=8.6, debug=debug)

    fig6 = fa.figure_6_policyregret(savefig=False, debug=debug)
    _clear_figure_legends(fig6, debug=debug)
    if len(fig6.axes) >= 2:
        h, l = fig6.axes[0].get_legend_handles_labels()
        _add_inside_legend(fig6.axes[0], h, l, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig6, "results_figures/figure_6_policyregret_presentation.png", width=10, height=8.6, debug=debug)

    fig7 = fa.figure_7_gasregret(savefig=False, debug=debug)
    _clear_figure_legends(fig7, debug=debug)
    if len(fig7.axes) >= 2:
        h, l = fig7.axes[0].get_legend_handles_labels()
        _add_inside_legend(fig7.axes[0], h, l, loc="upper left", ncol=1, fontsize=10, debug=debug)
    _save_presentation(fig7, "results_figures/figure_7_gasregret_presentation.png", width=10, height=8.6, debug=debug)

    fig8 = figure_policy_focus_presentation(savefig=False, debug=debug, policy=policy)
    _save_presentation(
        fig8,
        f"results_figures/figure_policy_focus_{_policy_slug(policy)}_presentation.png",
        width=16,
        height=8.8,
        debug=debug,
    )

    plt.show()


if __name__ == "__main__":
    main(debug=True)
