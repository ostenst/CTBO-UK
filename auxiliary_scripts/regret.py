import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _select_results_dir(debug=False):
    if debug:
        print("_select_results_dir input: awaiting user selection")
    while True:
        choice = input("Use PHASEOUT scenario? Enter 'y' (PHASEOUT=True) or 'n' (PHASEOUT=False): ").strip().lower()
        if choice in ("y", "yes"):
            results_dir = "results_phaseout"
            break
        if choice in ("n", "no"):
            results_dir = "results_baseline"
            break
        print("Please enter 'y' or 'n'.")
    if debug:
        print(f"_select_results_dir output: results_dir={results_dir}")
    return results_dir


def calculate_regret(
    experiments_df,
    outcomes,
    uncertainty_cols,
    policy_col="policy",
    scalar_metrics=None,
    timeseries_metrics=None,
    start_year=2025,
    debug=False,
):
    if scalar_metrics is None:
        scalar_metrics = [
            {"name": "NPV_costs_tax", "column": "NPV_costs_tax", "sense": "max"},
        ]
    if timeseries_metrics is None:
        timeseries_metrics = [
            {"name": "gas_increase_abs", "outcome_key": "gas_increase_abs", "sense": "min"},
            {"name": "petrol_increase_abs", "outcome_key": "petrol_increase_abs", "sense": "min"},
        ]
    if debug:
        print(
            f"calculate_regret inputs: rows={len(experiments_df)}, "
            f"scalar={[m['name'] for m in scalar_metrics]}, "
            f"timeseries={[m['name'] for m in timeseries_metrics]}"
        )

    work = experiments_df.copy()
    if policy_col not in work.columns:
        raise KeyError(f"policy_col {policy_col!r} not found in experiments dataframe")
    work["_policy_name"] = work[policy_col].astype(str)
    work["_scenario_id"], _ = pd.factorize(work[uncertainty_cols].apply(tuple, axis=1))
    u_lookup = work.groupby("_scenario_id", as_index=False)[uncertainty_cols].first()

    # Scalar regret (e.g., NPV tax income)
    scalar_parts = []
    scalar_summary_rows = []
    for spec in scalar_metrics:
        metric = spec["name"]
        col = spec["column"]
        sense = spec["sense"]
        work[col] = pd.to_numeric(work[col], errors="coerce")

        pivot = work.pivot_table(
            index="_scenario_id", columns="_policy_name", values=col, aggfunc="first", observed=True
        )
        if sense == "min":
            best = pivot.min(axis=1)
            regret_wide = pivot.sub(best, axis=0)
        elif sense == "max":
            best = pivot.max(axis=1)
            regret_wide = pivot.rsub(best, axis=0)
        else:
            raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")

        scalar_parts.append(
            regret_wide.rename(columns={p: f"regret_{metric}__{p}" for p in regret_wide.columns}).reset_index()
        )
        for pol in regret_wide.columns:
            vals = regret_wide[pol].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            scalar_summary_rows.append(
                {
                    "metric": metric,
                    "policy": pol,
                    "mean_regret": float(np.mean(vals)) if vals.size else float("nan"),
                    "median_regret": float(np.median(vals)) if vals.size else float("nan"),
                }
            )

    scalar_regret_by_scenario = u_lookup.copy()
    for part in scalar_parts:
        scalar_regret_by_scenario = scalar_regret_by_scenario.merge(part, on="_scenario_id", how="inner")
    scalar_regret_by_scenario = scalar_regret_by_scenario.drop(columns=["_scenario_id"])
    scalar_regret_summary = pd.DataFrame(scalar_summary_rows).sort_values(["metric", "policy"]).reset_index(drop=True)

    # Time-series regret (fuel price increases)
    timeseries_regret = {}
    timeseries_summary = {}
    years = None
    for spec in timeseries_metrics:
        metric = spec["name"]
        key = spec["outcome_key"]
        sense = spec["sense"]
        arr = np.asarray(outcomes[key], dtype=float)  # shape: [n_experiments, n_years]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D outcome array for {key!r}, got shape {arr.shape}")
        if years is None:
            years = np.arange(start_year, start_year + arr.shape[1], dtype=int)

        regret_rows = []
        summary_rows = []
        for scenario_id, grp in work.groupby("_scenario_id", sort=True):
            idx = grp.index.to_numpy()
            policies = grp["_policy_name"].to_numpy()
            vals = arr[idx, :]  # [n_policies, n_years]
            if sense == "min":
                best = np.nanmin(vals, axis=0, keepdims=True)
                reg = vals - best
            elif sense == "max":
                best = np.nanmax(vals, axis=0, keepdims=True)
                reg = best - vals
            else:
                raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")

            for i, pol in enumerate(policies):
                row = {"_scenario_id": int(scenario_id), "policy": pol}
                row.update({f"year_{int(y)}": float(reg[i, j]) for j, y in enumerate(years)})
                regret_rows.append(row)

        regret_df = pd.DataFrame(regret_rows)
        for pol, grp in regret_df.groupby("policy"):
            for j, y in enumerate(years):
                col = f"year_{int(y)}"
                vals = pd.to_numeric(grp[col], errors="coerce").dropna().to_numpy()
                summary_rows.append(
                    {
                        "metric": metric,
                        "policy": pol,
                        "year": int(y),
                        "p05": float(np.percentile(vals, 5)) if vals.size else float("nan"),
                        "p50": float(np.percentile(vals, 50)) if vals.size else float("nan"),
                        "p95": float(np.percentile(vals, 95)) if vals.size else float("nan"),
                    }
                )

        timeseries_regret[metric] = regret_df
        timeseries_summary[metric] = pd.DataFrame(summary_rows)

    if debug:
        print(
            f"calculate_regret outputs: scalar_by_scenario={scalar_regret_by_scenario.shape}, "
            f"scalar_summary={scalar_regret_summary.shape}, "
            f"timeseries={[k + ':' + str(v.shape) for k, v in timeseries_summary.items()]}"
        )
    return scalar_regret_by_scenario, scalar_regret_summary, timeseries_regret, timeseries_summary


def plot_regret_timeseries(timeseries_summary, metric, figures_dir="results_figures", pounds_to_EUR=1.15, debug=False):
    df = timeseries_summary[metric].copy()
    if df.empty:
        return
    if debug:
        print(f"plot_regret_timeseries inputs: metric={metric}, rows={len(df)}")
    fig, ax = plt.subplots(figsize=(11, 7))
    policies = sorted(df["policy"].unique())
    magma = plt.cm.magma
    colors = magma(np.linspace(0.05, 0.95, max(1, len(policies))))
    y_units = {
        "gas_increase_abs": "pence/kWh",
        "petrol_increase_abs": "cent/L",
        "costs_consumers": "BGBP/y",
        "costs_tax": "BGBP/y",
    }
    for policy, color in zip(policies, colors):
        sub = df[df["policy"] == policy].sort_values("year")
        x = sub["year"].to_numpy()
        p05 = sub["p05"].to_numpy(dtype=float)
        p50 = sub["p50"].to_numpy(dtype=float)
        p95 = sub["p95"].to_numpy(dtype=float)
        if metric == "gas_increase_abs":
            # Match illustrate.py: EUR/MWh -> pence/kWh.
            scale = (100 / 1000) / pounds_to_EUR
            p05 = p05 * scale
            p50 = p50 * scale
            p95 = p95 * scale
        elif metric in ("costs_consumers", "costs_tax"):
            # Match illustrate.py: kEUR/y -> BGBP/y.
            scale = 1e-6 / pounds_to_EUR
            p05 = p05 * scale
            p50 = p50 * scale
            p95 = p95 * scale
        ax.plot(x, p50, linewidth=2.3, color=color, label=policy)
        ax.fill_between(x, p05, p95, color=color, alpha=0.22)

    ax.set_title(f"Regret over time: {metric}", fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(f"Regret [{y_units.get(metric, '-')}] ", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = f"{figures_dir}/regret_timeseries_{metric}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if debug:
        print(f"plot_regret_timeseries output: {out}")


def plot_scalar_regret_boxplot(
    scalar_regret_by_scenario,
    metric="NPV_costs_tax",
    figures_dir="results_figures",
    debug=False,
):
    if debug:
        print(
            f"plot_scalar_regret_boxplot inputs: metric={metric}, "
            f"rows={len(scalar_regret_by_scenario)}"
        )
    prefix = f"regret_{metric}__"
    cols = [c for c in scalar_regret_by_scenario.columns if c.startswith(prefix)]
    cols = sorted(cols)
    if not cols:
        if debug:
            print("plot_scalar_regret_boxplot output: no matching regret columns")
        return

    labels = [c.replace(prefix, "") for c in cols]
    data = [pd.to_numeric(scalar_regret_by_scenario[c], errors="coerce").dropna().to_numpy() for c in cols]
    magma = plt.cm.magma
    colors = magma(np.linspace(0.05, 0.95, max(1, len(data))))

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.8)

    ax.set_title(f"Regret by policy: {metric}", fontsize=16)
    ax.set_xlabel("Policy", fontsize=14)
    ax.set_ylabel("Regret [k€]", fontsize=14)
    ax.tick_params(axis="x", labelrotation=20, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = f"{figures_dir}/regret_boxplot_{metric}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if debug:
        print(f"plot_scalar_regret_boxplot output: {out}")


def debug_timeseries_metric_year(
    experiments,
    outcomes,
    timeseries_regret,
    metric,
    year=2050,
    start_year=2025,
    sense="min",
    debug=False,
):
    if debug:
        print(f"debug_timeseries_metric_year input: metric={metric}, year={year}")
    if metric not in outcomes:
        raise KeyError(f"Metric {metric!r} not found in outcomes.")
    if metric not in timeseries_regret:
        raise KeyError(f"Metric {metric!r} not found in timeseries_regret.")

    arr = np.asarray(outcomes[metric], dtype=float)
    year_idx = int(year - start_year)
    if year_idx < 0 or year_idx >= arr.shape[1]:
        raise ValueError(f"Year {year} outside outcome range [{start_year}, {start_year + arr.shape[1] - 1}]")

    work = experiments.copy()
    work["_policy_name"] = work["policy"].astype(str)
    uncertainty_cols = [c for c in experiments.columns if c not in {"policy", "model", "scenario", "ETS_SCENARIO"} and not c.startswith("NPV_")]
    work["_scenario_id"], _ = pd.factorize(work[uncertainty_cols].apply(tuple, axis=1))

    raw_df = pd.DataFrame(
        {
            "policy": work["_policy_name"].to_numpy(),
            "_scenario_id": work["_scenario_id"].to_numpy(),
            "value": arr[:, year_idx],
        }
    )
    reg_col = f"year_{int(year)}"
    regret_df = timeseries_regret[metric][["policy", "_scenario_id", reg_col]].rename(columns={reg_col: "regret"})

    raw_med = raw_df.groupby("policy")["value"].median().sort_values()
    reg_med = regret_df.groupby("policy")["regret"].median().sort_values()

    if sense == "min":
        best_idx = raw_df.groupby("_scenario_id")["value"].idxmin()
    elif sense == "max":
        best_idx = raw_df.groupby("_scenario_id")["value"].idxmax()
    else:
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    best_by_scenario = raw_df.loc[best_idx, "policy"]
    best_share = (best_by_scenario.value_counts(normalize=True) * 100).sort_values(ascending=False)
    reg_stats = regret_df.groupby("policy")["regret"].agg(
        p05=lambda s: float(np.percentile(s.dropna().to_numpy(), 5)) if len(s.dropna()) else float("nan"),
        p50=lambda s: float(np.percentile(s.dropna().to_numpy(), 50)) if len(s.dropna()) else float("nan"),
        p95=lambda s: float(np.percentile(s.dropna().to_numpy(), 95)) if len(s.dropna()) else float("nan"),
        share_positive=lambda s: float((s > 0).mean() * 100),
    ).sort_index()

    print(f"\nDiagnostics for metric={metric}, year={year}")
    print("\nMedian raw value by policy:")
    print(raw_med.to_string())
    print("\nMedian regret by policy:")
    print(reg_med.to_string())
    print("\nBest-policy share by scenario [%]:")
    print(best_share.to_string())
    print("\nRegret distribution by policy at selected year:")
    print(reg_stats.to_string())


if __name__ == "__main__":
    results_dir = _select_results_dir(debug=False)
    figures_dir = "results_figures"
    debug = False

    experiments = pd.read_csv(f"{results_dir}/experiments.csv")
    outcomes = {
        "gas_increase_abs": np.load(f"{results_dir}/outcomes_gas_increase_abs.npy"),
        "petrol_increase_abs": np.load(f"{results_dir}/outcomes_petrol_increase_abs.npy"),
        "costs_consumers": np.load(f"{results_dir}/outcomes_costs_consumers.npy"),
        "costs_tax": np.load(f"{results_dir}/outcomes_costs_tax.npy"),
    }
    uncertainty_cols = [
        "DIFFUSE_END_FRACTION", "DACCS_SCENARIO", "fraction_limestone", "fraction_fossil_waste",
        "drax_efficiency", "drax_efficiency_loss", "capture_rate", "qreb", "pcomp",
        "pcomp_liquefy", "FLH_industry", "FLH_waste", "ccgt_efficiency",
        "ccgt_efficiency_loss", "cgas", "celc", "cstraw", "cliquefy", "cHCN", "cHRSG",
        "camine", "cstorage", "transport_uncertainty", "CAPEX_gasboiler", "CAPEX_bioboiler",
        "fixate_CAPEX", "CAPEX_m", "discount_rate_ccs", "lifetime_ccs", "CEPCI_2025",
        "NETL_2025",
    ]

    scalar_regret_by_scenario, scalar_regret_summary, timeseries_regret, timeseries_summary = calculate_regret(
        experiments,
        outcomes,
        uncertainty_cols,
        policy_col="policy",
        scalar_metrics=[{"name": "NPV_costs_tax", "column": "NPV_costs_tax", "sense": "max"}],
        timeseries_metrics=[
            {"name": "gas_increase_abs", "outcome_key": "gas_increase_abs", "sense": "min"},
            {"name": "petrol_increase_abs", "outcome_key": "petrol_increase_abs", "sense": "min"},
            {"name": "costs_consumers", "outcome_key": "costs_consumers", "sense": "max"},
            {"name": "costs_tax", "outcome_key": "costs_tax", "sense": "max"},
        ],
        start_year=2025,
        debug=debug,
    )

    scalar_regret_by_scenario.to_csv(f"{results_dir}/regret_by_scenario.csv", index=False)
    scalar_regret_summary.to_csv(f"{results_dir}/regret_summary.csv", index=False)
    timeseries_regret["gas_increase_abs"].to_csv(f"{results_dir}/regret_gas_by_scenario_year.csv", index=False)
    timeseries_regret["petrol_increase_abs"].to_csv(f"{results_dir}/regret_petrol_by_scenario_year.csv", index=False)
    timeseries_regret["costs_consumers"].to_csv(f"{results_dir}/regret_costs_consumers_by_scenario_year.csv", index=False)
    timeseries_regret["costs_tax"].to_csv(f"{results_dir}/regret_costs_tax_by_scenario_year.csv", index=False)
    timeseries_summary["gas_increase_abs"].to_csv(f"{results_dir}/regret_gas_timeseries_summary.csv", index=False)
    timeseries_summary["petrol_increase_abs"].to_csv(f"{results_dir}/regret_petrol_timeseries_summary.csv", index=False)
    timeseries_summary["costs_consumers"].to_csv(f"{results_dir}/regret_costs_consumers_timeseries_summary.csv", index=False)
    timeseries_summary["costs_tax"].to_csv(f"{results_dir}/regret_costs_tax_timeseries_summary.csv", index=False)

    plot_regret_timeseries(timeseries_summary, "gas_increase_abs", figures_dir=figures_dir, debug=debug)
    plot_regret_timeseries(timeseries_summary, "petrol_increase_abs", figures_dir=figures_dir, debug=debug)
    plot_regret_timeseries(timeseries_summary, "costs_consumers", figures_dir=figures_dir, debug=debug)
    plot_regret_timeseries(timeseries_summary, "costs_tax", figures_dir=figures_dir, debug=debug)
    debug_timeseries_metric_year(
        experiments=experiments,
        outcomes=outcomes,
        timeseries_regret=timeseries_regret,
        metric="costs_consumers",
        year=2050,
        start_year=2025,
        sense="min",
        debug=debug,
    )
    plot_scalar_regret_boxplot(
        scalar_regret_by_scenario,
        metric="NPV_costs_tax",
        figures_dir=figures_dir,
        debug=debug,
    )
    print("Wrote scalar regret tables, fuel-regret time-series plots, and NPV-tax regret boxplot.")
