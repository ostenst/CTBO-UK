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


_SIMPLE_PANEL_WIDTH = 14.0
_SIMPLE_PANEL_HEIGHT = 4.8
_SIMPLE_PANEL_DPI = 450
# Margins sized for up to three legend entries (tradeoffs with/without Mix £100).
_SIMPLE_PANEL_MARGINS = dict(left=0.07, right=0.99, top=0.82, bottom=0.20)


def _save_simple_panel_figure(fig, output_path, debug=False):
    """Save 1x3 simple figures at fixed canvas size (no tight bbox crop)."""
    fig.set_size_inches(_SIMPLE_PANEL_WIDTH, _SIMPLE_PANEL_HEIGHT, forward=True)
    fig.subplots_adjust(**_SIMPLE_PANEL_MARGINS)
    fig.savefig(output_path, dpi=_SIMPLE_PANEL_DPI)
    if debug:
        print(
            "_save_simple_panel_figure output:",
            f"file={output_path}, size=({_SIMPLE_PANEL_WIDTH}, {_SIMPLE_PANEL_HEIGHT})",
        )


def _plot_band(ax, x, arr, mask, color, label, debug=False):
    if mask.sum() == 0:
        return
    med, p05, p95 = fa._panel_stats(arr[mask], debug=debug)
    ax.plot(x, med, lw=2.3, color=color, label=label)
    ax.fill_between(x, p05, p95, color=color, alpha=0.2)


def _style_price_panels(axes, years, debug=False):
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=12)
        fa._set_sparse_year_ticks(ax, years, debug=debug)
    axes[0].set_ylabel("Carbon price [£/tCO2]", fontsize=13)
    for c, title in enumerate(
        [
            "Marginal CO₂ storage cost",
            "ETS vs CSU (single-instrument cases)",
            "ETS and CSU (Mix £100)",
        ]
    ):
        axes[c].set_title(title, fontsize=13)
    axes[-1].set_xlabel("Year", fontsize=13)


def simple_prices(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """Three-panel price summary for PHASEOUT=False (x ticks at 2030, 2040, 2050)."""
    results_dir = "results_baseline"
    if debug:
        print(
            "simple_prices input:",
            f"results_dir={results_dir}, figures_dir={figures_dir}",
        )

    experiments = pd.read_csv(f"{results_dir}/experiments.csv")
    n_years = fa._load_array(results_dir, "cost_marginal", debug=debug).shape[1]
    years = np.arange(start_year, start_year + n_years, dtype=int)
    use = years <= 2050
    years = years[use]

    marginal = fa._load_array(results_dir, "cost_marginal", debug=debug)[:, use] / pounds_to_EUR
    ets = fa._load_array(results_dir, "price_ETS", debug=debug)[:, use] / pounds_to_EUR
    csu = fa._load_array(results_dir, "price_CSU", debug=debug)[:, use] / pounds_to_EUR

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    color_marginal = "black"
    color_ets = plt.cm.magma(0.35)
    color_csu = plt.cm.magma(0.85)

    mask_all = np.ones(len(experiments), dtype=bool)
    _plot_band(
        axes[0],
        years,
        marginal,
        mask_all,
        color_marginal,
        "Marginal CO₂ storage cost",
        debug=debug,
    )

    mask_ets = experiments["ETS_SCENARIO"] == "ETS-eq"
    _plot_band(
        axes[1],
        years,
        ets,
        mask_ets,
        color_ets,
        "ETS price (ETS only)",
        debug=debug,
    )
    mask_ctbo = experiments["ETS_SCENARIO"] == "CTBO-only"
    _plot_band(
        axes[1],
        years,
        csu,
        mask_ctbo,
        color_csu,
        "CSU price (CTBO only)",
        debug=debug,
    )

    mask_mix = experiments["ETS_SCENARIO"] == "£100-Mix"
    _plot_band(
        axes[2],
        years,
        ets,
        mask_mix,
        color_ets,
        "ETS price (Mix, targeting £100)",
        debug=debug,
    )
    _plot_band(
        axes[2],
        years,
        csu,
        mask_mix,
        color_csu,
        "CSU price (Mix, targeting £100)",
        debug=debug,
    )

    _style_price_panels(axes, years, debug=debug)
    for ax in axes:
        _add_inside_legend(ax, *ax.get_legend_handles_labels(), loc="upper left", fontsize=9, debug=debug)

    fig.tight_layout()
    if savefig:
        out = f"{figures_dir}/simple_prices_presentation.png"
        fig.savefig(out, dpi=450, bbox_inches="tight")
        if debug:
            print(f"simple_prices output: file={out}")
    return fig


def _annotate_npv_consumers(ax, experiments, mask, scale_bgbp, debug=False):
    npv_vals = pd.to_numeric(
        experiments.loc[mask, "NPV_costs_consumers"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    if npv_vals.size == 0:
        return
    npv_med = np.median(npv_vals) * scale_bgbp
    npv_p05 = np.percentile(npv_vals, 5) * scale_bgbp
    npv_p95 = np.percentile(npv_vals, 95) * scale_bgbp
    text = (
        "NPV consumer costs [B£]\n"
        f"median: {npv_med:.2f}\n"
        f"p05–p95: {npv_p05:.2f} to {npv_p95:.2f}"
    )
    ax.text(
        0.04,
        0.66,
        text,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    if debug:
        print(
            "_annotate_npv_consumers output:",
            f"n={npv_vals.size}, median={npv_med:.3f}, p05={npv_p05:.3f}, p95={npv_p95:.3f}",
        )


def _plot_cost_bands(ax, years, suppliers, emitters, tax, mask, scale_bgbp, debug=False):
    magma = plt.cm.magma
    color_suppliers = magma(0.85)
    color_emitters = magma(0.35)
    color_tax = "#62a7a6"
    for arr, color, label in (
        (suppliers, color_suppliers, "Fuel supplier cost"),
        (emitters, color_emitters, "Emitter cost"),
        (tax, color_tax, "Emissions tax"),
    ):
        _plot_band(ax, years, arr * scale_bgbp, mask, color, label, debug=debug)


def _style_cost_panels(axes, years, debug=False):
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=12)
        fa._set_sparse_year_ticks(ax, years, debug=debug)
    axes[0].set_ylabel("Annual policy costs [B£ p.a.]", fontsize=13)
    for ax, title in zip(
        axes,
        ["CTBO only", "Mix (£100)", "ETS only"],
    ):
        ax.set_title(title, fontsize=13)
    axes[-1].set_xlabel("Year", fontsize=13)


def simple_costs(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """Three-panel policy-cost summary for PHASEOUT=False."""
    results_dir = "results_baseline"
    if debug:
        print(
            "simple_costs input:",
            f"results_dir={results_dir}, figures_dir={figures_dir}",
        )

    experiments = pd.read_csv(f"{results_dir}/experiments.csv")
    n_years = fa._load_array(results_dir, "costs_suppliers", debug=debug).shape[1]
    years = np.arange(start_year, start_year + n_years, dtype=int)
    use = years <= 2050
    years = years[use]

    suppliers = fa._load_array(results_dir, "costs_suppliers", debug=debug)[:, use]
    emitters = fa._load_array(results_dir, "costs_emitters", debug=debug)[:, use]
    tax = fa._load_array(results_dir, "costs_tax", debug=debug)[:, use]
    scale_bgbp = 1e-6 / pounds_to_EUR

    scenarios = ["CTBO-only", "£100-Mix", "ETS-eq"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)

    for ax, scenario in zip(axes, scenarios):
        mask = experiments["ETS_SCENARIO"] == scenario
        _plot_cost_bands(
            ax, years, suppliers, emitters, tax, mask, scale_bgbp, debug=debug
        )
        _annotate_npv_consumers(ax, experiments, mask, scale_bgbp, debug=debug)

    _style_cost_panels(axes, years, debug=debug)
    for ax in axes:
        _add_inside_legend(ax, *ax.get_legend_handles_labels(), loc="upper left", fontsize=9, debug=debug)

    fig.tight_layout()
    if savefig:
        out = f"{figures_dir}/simple_costs_presentation.png"
        fig.savefig(out, dpi=450, bbox_inches="tight")
        if debug:
            print(f"simple_costs output: file={out}")
    return fig


def _regret_stats_from_csv(csv_path, policy, debug=False):
    df = pd.read_csv(csv_path)
    sub = df[df["policy"] == policy]
    year_cols = [c for c in df.columns if c.startswith("year_")]
    years = np.array([int(c.split("_", 1)[1]) for c in year_cols], dtype=int)
    vals = sub[year_cols].to_numpy(dtype=float)
    if vals.size == 0:
        empty = np.array([], dtype=float)
        return years, empty, empty, empty
    p05 = np.nanpercentile(vals, 5, axis=0)
    p50 = np.nanpercentile(vals, 50, axis=0)
    p95 = np.nanpercentile(vals, 95, axis=0)
    if debug:
        print(f"_regret_stats_from_csv output: path={csv_path}, policy={policy}, n_scenarios={len(sub)}")
    return years, p05, p50, p95


def _plot_regret_band(ax, years, p05, p50, p95, color, label, debug=False):
    if years.size == 0:
        return
    ax.plot(years, p50, lw=2.3, color=color, label=label)
    ax.fill_between(years, p05, p95, color=color, alpha=0.2)
    if debug:
        print(f"_plot_regret_band output: label={label}, n_years={years.size}")


def _simple_tradeoffs_figure(
    include_mix=False,
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    output_name="simple_tradeoffs_presentation.png",
    debug=False,
):
    """Three-panel trade-off summary for PHASEOUT=False."""
    results_dir = "results_baseline"
    policy_ets = "ETS-eq"
    policy_mix = "£100-Mix"
    if debug:
        print(
            "_simple_tradeoffs_figure input:",
            f"results_dir={results_dir}, include_mix={include_mix}",
        )

    experiments = pd.read_csv(f"{results_dir}/experiments.csv")
    n_years = fa._load_array(results_dir, "gas_increase_abs", debug=debug).shape[1]
    years = np.arange(start_year, start_year + n_years, dtype=int)
    use = years <= 2050
    years = years[use]

    gas = fa._load_array(results_dir, "gas_increase_abs", debug=debug)[:, use]
    costs_tax = fa._load_array(results_dir, "costs_tax", debug=debug)[:, use]
    gas_scale = (100 / 1000) / pounds_to_EUR  # EUR/MWh -> p/kWh
    scale_bgbp = 1e-6 / pounds_to_EUR  # kEUR/y -> B£ p.a.

    color_ctbo = plt.cm.magma(0.85)
    color_ets = plt.cm.magma(0.35)
    color_mix = plt.cm.magma(0.60)

    fig, axes = plt.subplots(
        1, 3, figsize=(_SIMPLE_PANEL_WIDTH, _SIMPLE_PANEL_HEIGHT), sharex=True
    )

    mask_ctbo = experiments["ETS_SCENARIO"] == "CTBO-only"
    _plot_band(
        axes[0],
        years,
        gas * gas_scale,
        mask_ctbo,
        color_ctbo,
        "CTBO only",
        debug=debug,
    )
    mask_ets = experiments["ETS_SCENARIO"] == policy_ets
    _plot_band(
        axes[0],
        years,
        gas * gas_scale,
        mask_ets,
        color_ets,
        "ETS only",
        debug=debug,
    )

    gas_regret_path = f"{results_dir}/regret_gas_by_scenario_year.csv"
    ry, rp05, rp50, rp95 = _regret_stats_from_csv(gas_regret_path, policy_ets, debug=debug)
    r_use = ry <= 2050
    _plot_regret_band(
        axes[1],
        ry[r_use],
        rp05[r_use] * gas_scale,
        rp50[r_use] * gas_scale,
        rp95[r_use] * gas_scale,
        color_ets,
        "Gas price regret (ETS only)",
        debug=debug,
    )

    _plot_band(
        axes[2],
        years,
        costs_tax * scale_bgbp,
        mask_ets,
        color_ets,
        "Tax revenues (ETS only)" if include_mix else "Tax revenues",
        debug=debug,
    )

    if include_mix:
        mask_mix = experiments["ETS_SCENARIO"] == policy_mix
        _plot_band(
            axes[0],
            years,
            gas * gas_scale,
            mask_mix,
            color_mix,
            "Mix (£100)",
            debug=debug,
        )
        ry, rp05, rp50, rp95 = _regret_stats_from_csv(gas_regret_path, policy_mix, debug=debug)
        _plot_regret_band(
            axes[1],
            ry[r_use],
            rp05[r_use] * gas_scale,
            rp50[r_use] * gas_scale,
            rp95[r_use] * gas_scale,
            color_mix,
            "Gas price regret (Mix £100)",
            debug=debug,
        )
        _plot_band(
            axes[2],
            years,
            costs_tax * scale_bgbp,
            mask_mix,
            color_mix,
            "Tax revenues (Mix £100)",
            debug=debug,
        )

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=12)
        fa._set_sparse_year_ticks(ax, years, debug=debug)

    axes[0].set_ylabel("Gas price increase [p/kWh]", fontsize=13)
    axes[1].set_ylabel("Gas price regret [p/kWh]", fontsize=13)
    axes[2].set_ylabel("Tax revenues [B£ p.a.]", fontsize=13)
    if include_mix:
        titles = [
            "Gas price increase",
            "Gas price regret",
            "Tax revenues",
        ]
    else:
        titles = [
            "Gas price increase",
            "Gas price regret (ETS only)",
            "Tax revenues (ETS only)",
        ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13)
    axes[-1].set_xlabel("Year", fontsize=13)

    _add_inside_legend(axes[0], *axes[0].get_legend_handles_labels(), loc="upper left", fontsize=9, debug=debug)

    if savefig:
        out = f"{figures_dir}/{output_name}"
        _save_simple_panel_figure(fig, out, debug=debug)
    return fig


def simple_tradeoffs(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    return _simple_tradeoffs_figure(
        include_mix=False,
        figures_dir=figures_dir,
        start_year=start_year,
        pounds_to_EUR=pounds_to_EUR,
        savefig=savefig,
        output_name="simple_tradeoffs_presentation.png",
        debug=debug,
    )


def simple_tradeoffs_mix(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """Like simple_tradeoffs, with Mix (£100) series added on each panel."""
    return _simple_tradeoffs_figure(
        include_mix=True,
        figures_dir=figures_dir,
        start_year=start_year,
        pounds_to_EUR=pounds_to_EUR,
        savefig=savefig,
        output_name="simple_tradeoffs_mix100_presentation.png",
        debug=debug,
    )


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

    fig9 = simple_prices(savefig=False, debug=debug)
    _save_presentation(
        fig9,
        "results_figures/simple_prices_presentation.png",
        width=14,
        height=4.8,
        debug=debug,
    )

    fig10 = simple_costs(savefig=False, debug=debug)
    _save_presentation(
        fig10,
        "results_figures/simple_costs_presentation.png",
        width=14,
        height=4.8,
        debug=debug,
    )

    fig11 = simple_tradeoffs(savefig=False, debug=debug)
    _save_simple_panel_figure(
        fig11,
        "results_figures/simple_tradeoffs_presentation.png",
        debug=debug,
    )

    fig12 = simple_tradeoffs_mix(savefig=False, debug=debug)
    _save_simple_panel_figure(
        fig12,
        "results_figures/simple_tradeoffs_mix100_presentation.png",
        debug=debug,
    )

    plt.show()


if __name__ == "__main__":
    main(debug=True)
