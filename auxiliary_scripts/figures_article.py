import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SECTOR_LABELS = {
    "ccgt": "Gas power plant",
    "cement": "Cement stack",
    "refinery": "Refinery stacks",
    "steel": "Steel stacks",
    "waste": "Waste incinerator stack",
    "drax": "Drax stack",
}


def _load_array(results_dir, key, debug=False):
    arr = np.load(f"{results_dir}/outcomes_{key}.npy")
    if debug:
        print(f"_load_array output: results_dir={results_dir}, key={key}, shape={arr.shape}")
    return arr


def _panel_stats(arr, debug=False):
    med = np.nanmedian(arr, axis=0)
    p05 = np.nanpercentile(arr, 5, axis=0)
    p95 = np.nanpercentile(arr, 95, axis=0)
    if debug:
        print(f"_panel_stats output: n_samples={arr.shape[0]}, n_years={arr.shape[1]}")
    return med, p05, p95


def _set_sparse_year_ticks(ax, years, debug=False):
    ticks = [2030, 2040, 2050]
    ax.set_xlim(int(years.min()), int(years.max()))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) if years.min() <= t <= years.max() else "" for t in ticks])
    if debug:
        print(f"_set_sparse_year_ticks output: ticks={ticks}")


def _get_sector_colors(sectors, debug=False):
    magma = plt.cm.magma
    base = {
        "cement": magma(0.10),
        "ccgt": magma(0.30),
        "refinery": magma(0.90),
        "steel": magma(0.70),
        "drax": magma(0.50),
        "waste": "#62a7a6",
    }
    fallback = plt.cm.tab10(np.linspace(0, 1, max(1, len(sectors))))
    colors = {}
    for idx, sector in enumerate(sorted(sectors)):
        colors[sector] = base.get(sector, fallback[idx % len(fallback)])
    if debug:
        print(f"_get_sector_colors output: n_sectors={len(colors)}")
    return colors


def _load_case_data(results_dir, start_year=2025, debug=False):
    experiments = pd.read_csv(f"{results_dir}/experiments.csv")
    supply = _load_array(results_dir, "supply_ktCO2f", debug=debug)
    mandate = _load_array(results_dir, "mandate_ktCO2", debug=debug)
    stored_b = _load_array(results_dir, "stored_ktCO2b", debug=debug)
    stored_d = _load_array(results_dir, "stored_ktCO2daccs", debug=debug)

    years = np.arange(start_year, start_year + supply.shape[1])
    plant_ref = pd.read_csv(f"{results_dir}/plant_reference.csv")
    npv_total = _load_array(results_dir, "plants_NPV_total", debug=debug)
    inv_year = _load_array(results_dir, "plants_investment_year", debug=debug)
    mac = _load_array(results_dir, "plants_MAC", debug=debug)
    cap_ref = pd.read_csv(f"{results_dir}/plants_costbenefit_extended.csv", usecols=["stack", "ktCO2tot_ccs"])
    cap_by_stack = cap_ref.groupby("stack", as_index=True)["ktCO2tot_ccs"].median().to_dict()

    if debug:
        print(f"_load_case_data output: results_dir={results_dir}, n_rows={len(experiments)}")
    return {
        "experiments": experiments,
        "years": years,
        "supply": supply,
        "mandate": mandate,
        "stored_cdr": stored_b + stored_d,
        "plant_ref": plant_ref,
        "npv_total": npv_total,
        "inv_year": inv_year,
        "mac": mac,
        "cap_by_stack": cap_by_stack,
    }


def _plot_carbon_panel(ax, case_data, title, debug=False):
    years = case_data["years"]
    scale = 1000.0
    magma = plt.cm.magma

    med, p05, p95 = _panel_stats(case_data["supply"], debug=debug)
    ax.plot(years, med / scale, lw=2.5, color=magma(0.0), label="Supply (coal, oil, gas)")
    ax.fill_between(years, p05 / scale, p95 / scale, color=magma(0.0), alpha=0.22)

    med, p05, p95 = _panel_stats(case_data["mandate"], debug=debug)
    ax.plot(years, med / scale, lw=2.5, color=magma(0.45), label="Storage (CCS, BECCS, DACCS)")
    ax.fill_between(years, p05 / scale, p95 / scale, color=magma(0.45), alpha=0.22)

    med, p05, p95 = _panel_stats(case_data["stored_cdr"], debug=debug)
    ax.plot(years, med / scale, lw=2.5, color="#62a7a6", label="Storage (BECCS, DACCS)")
    ax.fill_between(years, p05 / scale, p95 / scale, color="#62a7a6", alpha=0.24)

    if title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Carbon flow [MtCO2 p.a.]", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.35)
    _set_sparse_year_ticks(ax, years, debug=debug)


def _plot_npv_panel(ax, case_data, title, pounds_to_EUR=1.15, debug=False):
    plant_ref = case_data["plant_ref"]
    npv_total = case_data["npv_total"]
    inv_year = case_data["inv_year"]
    mac = case_data["mac"]
    cap_by_stack = case_data["cap_by_stack"]
    sectors = plant_ref["sector"].dropna().unique()
    sector_colors = _get_sector_colors(sectors, debug=debug)

    median_npv = np.nanmedian(npv_total, axis=0)
    median_inv = np.nanmedian(inv_year, axis=0)
    median_mac = np.nanmedian(np.abs(mac), axis=0)

    panel = plant_ref.copy()
    panel["NPV_total"] = median_npv
    panel["investment_year"] = median_inv
    panel["MAC"] = median_mac
    panel["ktCO2tot_ccs"] = panel["stack"].map(cap_by_stack)
    panel = panel.dropna(subset=["NPV_total", "investment_year"])
    panel = panel[panel["sector"].str.lower() != "drax"]

    for sector in sorted(panel["sector"].dropna().unique()):
        sector_df = panel[panel["sector"] == sector]
        sizes = np.clip(sector_df["ktCO2tot_ccs"].to_numpy(dtype=float), 1.0, None) * 0.25
        ax.scatter(
            sector_df["investment_year"].to_numpy(dtype=float),
            sector_df["NPV_total"].to_numpy(dtype=float) / 1000.0 / pounds_to_EUR,
            s=sizes,
            c=[sector_colors.get(sector, "grey")],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.45,
            label=SECTOR_LABELS.get(sector, sector),
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel("Investment year", fontsize=15)
    ax.set_ylabel("Investment NPV [M£]", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.35)
    _set_sparse_year_ticks(ax, case_data["years"], debug=debug)


def figure_4_carbon(
    results_dir_baseline="results_baseline",
    results_dir_phaseout="results_phaseout",
    figures_dir="results_figures",
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    if debug:
        print(
            "figure_4_carbon input:",
            f"baseline={results_dir_baseline}, phaseout={results_dir_phaseout}, figures_dir={figures_dir}",
        )

    baseline = _load_case_data(results_dir_baseline, debug=debug)
    phaseout = _load_case_data(results_dir_phaseout, debug=debug)

    fig, axes = plt.subplots(2, 2, figsize=(10, 14.0), sharex=False, sharey=False)

    _plot_carbon_panel(axes[0, 0], baseline, "", debug=debug)
    _plot_carbon_panel(axes[0, 1], phaseout, "", debug=debug)
    _plot_npv_panel(axes[1, 0], baseline, "", pounds_to_EUR=pounds_to_EUR, debug=debug)
    _plot_npv_panel(axes[1, 1], phaseout, "", pounds_to_EUR=pounds_to_EUR, debug=debug)

    # Align top-row carbon y-axis scales.
    top_ymin = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    top_ymax = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(top_ymin, top_ymax)
    axes[0, 1].set_ylim(top_ymin, top_ymax)

    # Align bottom-row NPV y-axis scales.
    bot_ymin = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    bot_ymax = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(bot_ymin, bot_ymax)
    axes[1, 1].set_ylim(bot_ymin, bot_ymax)

    # Align year axis across all panels (top and bottom).
    year_min = int(min(baseline["years"].min(), phaseout["years"].min()))
    year_max = int(max(baseline["years"].max(), phaseout["years"].max()))
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set_xlim(year_min, year_max)
        _set_sparse_year_ticks(ax, np.arange(year_min, year_max + 1), debug=debug)

    # Keep only outer axis labels/tick numbers for a cleaner shared layout.
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False, labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        axes[0, 0].legend(
            dedup.values(),
            dedup.keys(),
            loc="upper left",
            ncol=1,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
        )

    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    if handles2:
        dedup2 = dict(zip(labels2, handles2))
        axes[1, 0].legend(
            dedup2.values(),
            dedup2.keys(),
            loc="upper left",
            ncol=1,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
            title="Sector",
            title_fontsize=11,
        )

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    if savefig:
        fig.savefig(f"{figures_dir}/figure_4_carbon.png", dpi=450, bbox_inches="tight")
    if debug:
        print(f"figure_4_carbon output: file={figures_dir}/figure_4_carbon.png")
    return fig


def figure_5_prices(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
    PHASEOUT=False,
):
    if PHASEOUT:
        results_dir = "results_phaseout"
    else:
        results_dir = "results_baseline"
    if debug:
        print(
            "figure_5_prices input:",
            f"results_dir={results_dir}, figures_dir={figures_dir}",
        )

    scenarios = ["CTBO-only", "£200-Mix", "ETS-eq"]
    experiments = pd.read_csv(f"{results_dir}/experiments.csv")
    years = np.arange(start_year, start_year + _load_array(results_dir, "cost_marginal", debug=debug).shape[1])

    marginal = _load_array(results_dir, "cost_marginal", debug=debug) / pounds_to_EUR
    ets = _load_array(results_dir, "price_ETS", debug=debug) / pounds_to_EUR
    csu = _load_array(results_dir, "price_CSU", debug=debug) / pounds_to_EUR
    fuel = _load_array(results_dir, "cost_fuels", debug=debug) / pounds_to_EUR

    suppliers = _load_array(results_dir, "costs_suppliers", debug=debug)
    emitters = _load_array(results_dir, "costs_emitters", debug=debug)
    tax = _load_array(results_dir, "costs_tax", debug=debug)
    consumers = _load_array(results_dir, "costs_consumers", debug=debug)
    scale_bgbp = 1e-6 / pounds_to_EUR  # kEUR/y -> BGBP/y

    fig, axes = plt.subplots(2, 3, figsize=(10, 14), sharex=True, sharey="row")
    magma = plt.cm.magma
    color_marginal = magma(0.10)
    color_emitters_ets = magma(0.35)
    color_consumers_fuel = magma(0.60)
    color_suppliers_y = magma(0.85)
    color_tax = "#62a7a6"

    for col, scenario in enumerate(scenarios):
        mask = experiments["ETS_SCENARIO"] == scenario
        if mask.sum() == 0:
            continue

        # Upper row: carbon prices
        med, p05, p95 = _panel_stats(marginal[mask], debug=debug)
        axes[0, col].plot(years, med, lw=2.3, color=color_marginal, label="Marginal CO2 storage cost")
        axes[0, col].fill_between(years, p05, p95, color=color_marginal, alpha=0.2)

        med, p05, p95 = _panel_stats(ets[mask], debug=debug)
        axes[0, col].plot(years, med, lw=2.3, color=color_emitters_ets, label="ETS price")
        axes[0, col].fill_between(years, p05, p95, color=color_emitters_ets, alpha=0.2)

        med, p05, p95 = _panel_stats(csu[mask], debug=debug)
        axes[0, col].plot(years, med, lw=2.3, color=color_suppliers_y, label="CSU price")
        axes[0, col].fill_between(years, p05, p95, color=color_suppliers_y, alpha=0.2)

        med, p05, p95 = _panel_stats(fuel[mask], debug=debug)
        axes[0, col].plot(years, med, lw=2.3, color=color_consumers_fuel, label="Consumer fuel cost")
        axes[0, col].fill_between(years, p05, p95, color=color_consumers_fuel, alpha=0.2)

        axes[0, col].grid(True, linestyle="--", alpha=0.35)
        axes[0, col].tick_params(labelsize=12)

        # Lower row: policy costs
        med, p05, p95 = _panel_stats(suppliers[mask], debug=debug)
        axes[1, col].plot(years, med * scale_bgbp, lw=2.3, color=color_suppliers_y, label="Fuel supplier cost")
        axes[1, col].fill_between(years, p05 * scale_bgbp, p95 * scale_bgbp, color=color_suppliers_y, alpha=0.2)

        med, p05, p95 = _panel_stats(emitters[mask], debug=debug)
        axes[1, col].plot(years, med * scale_bgbp, lw=2.3, color=color_emitters_ets, label="Emitter cost")
        axes[1, col].fill_between(years, p05 * scale_bgbp, p95 * scale_bgbp, color=color_emitters_ets, alpha=0.2)

        med, p05, p95 = _panel_stats(tax[mask], debug=debug)
        axes[1, col].plot(years, med * scale_bgbp, lw=2.3, color=color_tax, label="Emissions tax")
        axes[1, col].fill_between(years, p05 * scale_bgbp, p95 * scale_bgbp, color=color_tax, alpha=0.2)

        med, p05, p95 = _panel_stats(consumers[mask], debug=debug)
        axes[1, col].plot(years, med * scale_bgbp, lw=2.3, color=color_consumers_fuel, label="Consumer costs")
        axes[1, col].fill_between(years, p05 * scale_bgbp, p95 * scale_bgbp, color=color_consumers_fuel, alpha=0.2)

        axes[1, col].grid(True, linestyle="--", alpha=0.35)
        axes[1, col].tick_params(labelsize=12)
        axes[1, col].set_xlabel("Year", fontsize=14)

    for c in range(3):
        _set_sparse_year_ticks(axes[0, c], years, debug=debug)
        _set_sparse_year_ticks(axes[1, c], years, debug=debug)

    # Outer labels only (shared axes by row)
    axes[0, 0].set_ylabel("Carbon price [£/tCO2]", fontsize=14)
    axes[1, 0].set_ylabel("Annual policy costs [B£ p.a.]", fontsize=14)
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False, labelleft=False)
    axes[0, 2].tick_params(labelbottom=False, labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    axes[1, 2].tick_params(labelleft=False)

    handles_top, labels_top = axes[0, 0].get_legend_handles_labels()
    handles_bot, labels_bot = axes[1, 0].get_legend_handles_labels()
    if handles_top:
        dedup_top = dict(zip(labels_top, handles_top))
        axes[0, 0].legend(
            dedup_top.values(),
            dedup_top.keys(),
            loc="upper left",
            ncol=1,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
        )
    if handles_bot:
        dedup_bot = dict(zip(labels_bot, handles_bot))
        axes[1, 0].legend(
            dedup_bot.values(),
            dedup_bot.keys(),
            loc="upper left",
            ncol=1,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
        )

    fig.tight_layout(rect=[0.03, 0.06, 0.99, 0.94])
    if savefig:
        fig.savefig(f"{figures_dir}/figure_5_prices.png", dpi=450, bbox_inches="tight")
    if debug:
        print(f"figure_5_prices output: file={figures_dir}/figure_5_prices.png")
    return fig


def _timeseries_regret_summary(
    experiments,
    outcome_arr,
    uncertainty_cols,
    start_year=2025,
    sense="min",
    debug=False,
):
    if debug:
        print(
            "_timeseries_regret_summary input:",
            f"rows={len(experiments)}, shape={outcome_arr.shape}",
        )
    work = experiments.copy()
    work["_policy_name"] = work["policy"].astype(str)
    work["_scenario_id"], _ = pd.factorize(work[uncertainty_cols].apply(tuple, axis=1))
    years = np.arange(start_year, start_year + outcome_arr.shape[1], dtype=int)

    rows = []
    for scenario_id, grp in work.groupby("_scenario_id", sort=True):
        idx = grp.index.to_numpy()
        policies = grp["_policy_name"].to_numpy()
        vals = outcome_arr[idx, :]  # [n_policies, n_years]
        if sense == "min":
            best = np.nanmin(vals, axis=0, keepdims=True)
            reg = vals - best
        elif sense == "max":
            best = np.nanmax(vals, axis=0, keepdims=True)
            reg = best - vals
        else:
            raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
        for i, pol in enumerate(policies):
            row = {"policy": pol}
            row.update({f"year_{int(y)}": float(reg[i, j]) for j, y in enumerate(years)})
            rows.append(row)

    reg_df = pd.DataFrame(rows)
    summary_rows = []
    for policy, grp in reg_df.groupby("policy"):
        for y in years:
            col = f"year_{int(y)}"
            vals = pd.to_numeric(grp[col], errors="coerce").dropna().to_numpy()
            summary_rows.append(
                {
                    "policy": policy,
                    "year": int(y),
                    "p05": float(np.percentile(vals, 5)) if len(vals) else float("nan"),
                    "p50": float(np.percentile(vals, 50)) if len(vals) else float("nan"),
                    "p95": float(np.percentile(vals, 95)) if len(vals) else float("nan"),
                }
            )
    out = pd.DataFrame(summary_rows)
    if debug:
        print(f"_timeseries_regret_summary output: rows={len(out)}")
    return out


def _scalar_regret_by_policy(
    experiments,
    uncertainty_cols,
    scalar_col="NPV_costs_consumers",
    sense="min",
    debug=False,
):
    if debug:
        print(f"_scalar_regret_by_policy input: rows={len(experiments)}, scalar_col={scalar_col}")
    work = experiments.copy()
    work["_policy_name"] = work["policy"].astype(str)
    work["_scenario_id"], _ = pd.factorize(work[uncertainty_cols].apply(tuple, axis=1))
    work[scalar_col] = pd.to_numeric(work[scalar_col], errors="coerce")

    pivot = work.pivot_table(
        index="_scenario_id",
        columns="_policy_name",
        values=scalar_col,
        aggfunc="first",
        observed=True,
    )
    if sense == "min":
        best = pivot.min(axis=1)
        regret_wide = pivot.sub(best, axis=0)
    elif sense == "max":
        best = pivot.max(axis=1)
        regret_wide = pivot.rsub(best, axis=0)
    else:
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    if debug:
        print(f"_scalar_regret_by_policy output: shape={regret_wide.shape}")
    return regret_wide


def figure_6_policyregret(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    if debug:
        print("figure_6_policyregret input: baseline+phaseout")

    uncertainty_cols = [
        "DIFFUSE_END_FRACTION", "DACCS_SCENARIO", "fraction_limestone", "fraction_fossil_waste",
        "drax_efficiency", "drax_efficiency_loss", "capture_rate", "qreb", "pcomp",
        "pcomp_liquefy", "FLH_industry", "FLH_waste", "ccgt_efficiency", "ccgt_efficiency_loss",
        "cgas", "celc", "cstraw", "cliquefy", "cHCN", "cHRSG", "camine", "cstorage",
        "transport_uncertainty", "CAPEX_gasboiler", "CAPEX_bioboiler", "fixate_CAPEX", "CAPEX_m",
        "discount_rate_ccs", "lifetime_ccs", "CEPCI_2025", "NETL_2025",
    ]
    policy_order = ["CTBO-only", "£100-Mix", "£200-Mix", "£300-Mix", "ETS-eq"]
    colors = plt.cm.magma(np.linspace(0.10, 0.85, len(policy_order)))
    scale_bgbp = 1e-6 / pounds_to_EUR  # kEUR/y -> B£/y

    experiments_baseline = pd.read_csv("results_baseline/experiments.csv")
    experiments_phaseout = pd.read_csv("results_phaseout/experiments.csv")
    tax_baseline = np.load("results_baseline/outcomes_costs_tax.npy")
    tax_phaseout = np.load("results_phaseout/outcomes_costs_tax.npy")

    summary_baseline = _timeseries_regret_summary(
        experiments_baseline,
        tax_baseline,
        uncertainty_cols,
        start_year=start_year,
        sense="max",
        debug=debug,
    )
    summary_phaseout = _timeseries_regret_summary(
        experiments_phaseout,
        tax_phaseout,
        uncertainty_cols,
        start_year=start_year,
        sense="max",
        debug=debug,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex="row", sharey="row")
    panel_data = [
        (axes[0, 0], summary_baseline, ""),
        (axes[0, 1], summary_phaseout, ""),
    ]

    for ax, summary, title in panel_data:
        for policy, color in zip(policy_order, colors):
            sub = summary[summary["policy"] == policy].sort_values("year")
            sub = sub[(sub["year"] >= 2025) & (sub["year"] <= 2045)]
            if sub.empty:
                continue
            x = sub["year"].to_numpy()
            p05 = sub["p05"].to_numpy(dtype=float) * scale_bgbp
            p50 = sub["p50"].to_numpy(dtype=float) * scale_bgbp
            p95 = sub["p95"].to_numpy(dtype=float) * scale_bgbp
            ax.plot(x, p50, lw=2.3, color=color, label=policy)
            ax.fill_between(x, p05, p95, color=color, alpha=0.22)
        if title:
            ax.set_title(title, fontsize=15)
        ax.set_xlim(2025, 2045)
        ax.set_xticks([2025, 2035, 2045])
        ax.set_xlabel("Year", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=12)

    axes[0, 0].set_ylabel("Annual regret [B£ p.a.] of foregone tax", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        axes[0, 0].legend(
            dedup.values(),
            dedup.keys(),
            loc="upper left",
            ncol=1,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
        )

    # Bottom row: scalar NPV regret boxplots for the same policies.
    scalar_baseline = _scalar_regret_by_policy(
        experiments_baseline,
        uncertainty_cols,
        scalar_col="NPV_costs_tax",
        sense="max",
        debug=debug,
    )
    scalar_phaseout = _scalar_regret_by_policy(
        experiments_phaseout,
        uncertainty_cols,
        scalar_col="NPV_costs_tax",
        sense="max",
        debug=debug,
    )

    for ax, scalar_df, title in [
        (axes[1, 0], scalar_baseline, ""),
        (axes[1, 1], scalar_phaseout, ""),
    ]:
        cols = [p for p in policy_order if p in scalar_df.columns]
        data = [pd.to_numeric(scalar_df[c], errors="coerce").dropna().to_numpy() * scale_bgbp for c in cols]
        bp = ax.boxplot(data, labels=cols, showfliers=True, patch_artist=True)
        box_colors = plt.cm.magma(np.linspace(0.10, 0.85, max(1, len(data))))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.8)
        if title:
            ax.set_title(title, fontsize=15)
        ax.set_ylabel("NPV regret [B£] of foregone tax", fontsize=14)
        ax.tick_params(axis="x", labelrotation=20, labelsize=11)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Outer labels only for cleaner 2x2 panel.
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    if savefig:
        fig.savefig(f"{figures_dir}/figure_6_policyregret.png", dpi=450, bbox_inches="tight")
    if debug:
        print(f"figure_6_policyregret output: file={figures_dir}/figure_6_policyregret.png")
    return fig


def figure_7_gasregret(
    figures_dir="results_figures",
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    if debug:
        print("figure_7_gasregret input: baseline+phaseout")

    uncertainty_cols = [
        "DIFFUSE_END_FRACTION", "DACCS_SCENARIO", "fraction_limestone", "fraction_fossil_waste",
        "drax_efficiency", "drax_efficiency_loss", "capture_rate", "qreb", "pcomp",
        "pcomp_liquefy", "FLH_industry", "FLH_waste", "ccgt_efficiency", "ccgt_efficiency_loss",
        "cgas", "celc", "cstraw", "cliquefy", "cHCN", "cHRSG", "camine", "cstorage",
        "transport_uncertainty", "CAPEX_gasboiler", "CAPEX_bioboiler", "fixate_CAPEX", "CAPEX_m",
        "discount_rate_ccs", "lifetime_ccs", "CEPCI_2025", "NETL_2025",
    ]
    policy_order = ["CTBO-only", "£100-Mix", "£200-Mix", "£300-Mix", "ETS-eq"]
    colors = plt.cm.magma(np.linspace(0.10, 0.85, len(policy_order)))

    experiments_baseline = pd.read_csv("results_baseline/experiments.csv")
    experiments_phaseout = pd.read_csv("results_phaseout/experiments.csv")
    gas_baseline = np.load("results_baseline/outcomes_gas_increase_abs.npy")
    gas_phaseout = np.load("results_phaseout/outcomes_gas_increase_abs.npy")

    # Convert absolute gas increase EUR/MWh -> p/kWh.
    gas_baseline = gas_baseline / pounds_to_EUR * (100 / 1000)
    gas_phaseout = gas_phaseout / pounds_to_EUR * (100 / 1000)

    regret_baseline = _timeseries_regret_summary(
        experiments_baseline,
        np.load("results_baseline/outcomes_gas_increase_abs.npy"),
        uncertainty_cols,
        start_year=start_year,
        debug=debug,
    )
    regret_phaseout = _timeseries_regret_summary(
        experiments_phaseout,
        np.load("results_phaseout/outcomes_gas_increase_abs.npy"),
        uncertainty_cols,
        start_year=start_year,
        debug=debug,
    )
    # Convert regret summary from EUR/MWh to p/kWh.
    conv = (100 / 1000) / pounds_to_EUR
    for df in (regret_baseline, regret_phaseout):
        df["p05"] = df["p05"] * conv
        df["p50"] = df["p50"] * conv
        df["p95"] = df["p95"] * conv

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey="row")
    top_specs = [
        (axes[0, 0], experiments_baseline, gas_baseline, ""),
        (axes[0, 1], experiments_phaseout, gas_phaseout, ""),
    ]
    for ax, experiments, gas_arr, title in top_specs:
        for policy, color in zip(policy_order, colors):
            mask = experiments["policy"] == policy
            if mask.sum() == 0:
                continue
            years = np.arange(start_year, start_year + gas_arr.shape[1])
            med, p05, p95 = _panel_stats(gas_arr[mask], debug=debug)
            use = (years >= 2025) & (years <= 2050)
            ax.plot(years[use], med[use], lw=2.3, color=color, label=policy)
            ax.fill_between(years[use], p05[use], p95[use], color=color, alpha=0.22)
        if title:
            ax.set_title(title, fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=12)
        y1_min, y1_max = ax.get_ylim()
        ax2 = ax.twinx()
        ax2.set_ylim(y1_min * 11200 / 100, y1_max * 11200 / 100)  # p/kWh -> GBP/year
        ax2.tick_params(labelsize=11)
        if ax is axes[0, 1]:
            ax2.set_ylabel("Household bill increase [£ p.a.]\n - assuming 11.2 MWh p.a.", fontsize=12)

    bottom_specs = [
        (axes[1, 0], regret_baseline, ""),
        (axes[1, 1], regret_phaseout, ""),
    ]
    for ax, summary, title in bottom_specs:
        for policy, color in zip(policy_order, colors):
            sub = summary[summary["policy"] == policy].sort_values("year")
            sub = sub[(sub["year"] >= 2025) & (sub["year"] <= 2050)]
            if sub.empty:
                continue
            x = sub["year"].to_numpy()
            p05 = sub["p05"].to_numpy(dtype=float)
            p50 = sub["p50"].to_numpy(dtype=float)
            p95 = sub["p95"].to_numpy(dtype=float)
            ax.plot(x, p50, lw=2.3, color=color, label=policy)
            ax.fill_between(x, p05, p95, color=color, alpha=0.22)
        if title:
            ax.set_title(title, fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Year", fontsize=14)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set_xlim(2025, 2050)
        ax.set_xticks([2030, 2040, 2050])

    axes[0, 0].set_ylabel("Gas price increase [p/kWh]", fontsize=14)
    axes[1, 0].set_ylabel("Annual regret [p/kWh] of increasing gas prices", fontsize=14)
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False, labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        axes[0, 0].legend(
            dedup.values(),
            dedup.keys(),
            loc="upper left",
            ncol=1,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
        )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    if savefig:
        fig.savefig(f"{figures_dir}/figure_7_gasregret.png", dpi=450, bbox_inches="tight")
    if debug:
        print(f"figure_7_gasregret output: file={figures_dir}/figure_7_gasregret.png")
    return fig


def main(debug=False):
    if debug:
        print("main input: debug=True")
    fig = figure_4_carbon(debug=debug)
    fig = figure_5_prices(debug=debug, PHASEOUT=False)
    fig = figure_6_policyregret(debug=debug)
    fig = figure_7_gasregret(debug=debug)
    plt.show()
    return fig


if __name__ == "__main__":
    main(debug=True)
