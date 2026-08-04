import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Stack-type suffixes used in characterize_stacks.py (after the final "-").
STACK_TYPE_LABELS = {
    "cement": "Cement",
    "ccgt": "CCGT",
    "waste": "Waste",
    "distillation": "Refinery distillation",
    "HPU": "Refinery HPU",
    "FCC": "Refinery FCC",
    "scattered": "Refinery scattered",
    "power": "Drax (BECCS)",
    "steel_power": "Steel power",
    "blast": "Steel blast furnace",
    "sinter": "Steel sinter",
}

# Drax-only; other -power stacks (e.g. steel) are grouped separately.
BECCS_STACKS = {"Drax-power"}


def _load_array(results_dir, key, debug=False):
    arr = np.load(f"{results_dir}/outcomes_{key}.npy")
    if debug:
        print(f"_load_array output: key={key}, shape={arr.shape}")
    return arr


def _stack_type(stack_name):
    if "-" not in str(stack_name):
        return str(stack_name)
    return str(stack_name).rsplit("-", 1)[-1]


def _stack_aggregate_type(stack_name, sector):
    """Stack aggregate group; steel -power stacks map to Industrial CCS, not BECCS."""
    name = str(stack_name)
    if name in BECCS_STACKS:
        return "power"
    if str(sector) == "steel" and name.endswith("-power"):
        return "steel_power"
    return _stack_type(name)


def _stack_aggregate_types(stack_names, sectors):
    return np.array(
        [_stack_aggregate_type(s, sec) for s, sec in zip(stack_names, sectors)],
        dtype=object,
    )


def _annotate_barh_medians(ax, values, fmt=".0f", debug=False):
    """Label horizontal bars with their median values."""
    x_max = max(float(np.nanmax(values)), 1.0)
    pad = 0.02 * x_max
    for i, val in enumerate(values):
        ax.text(
            float(val) + pad,
            i,
            f"{val:{fmt}}",
            va="center",
            ha="left",
            fontsize=9,
            clip_on=False,
        )
    ax.set_xlim(right=ax.get_xlim()[1] + 0.12 * x_max)
    if debug:
        print(f"_annotate_barh_medians output: n_bars={len(values)}")


def _annotate_boxplot_percentiles(ax, data, fmt=".1f", debug=False):
    """Label each box with median, 5th, and 95th percentile values."""
    y_min, y_max = ax.get_ylim()
    y_pad = 0.01 * (y_max - y_min) if y_max > y_min else 1.0
    for i, vals in enumerate(data, start=1):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        med = float(np.median(arr))
        p05 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        ax.text(
            i,
            med,
            f"med {med:{fmt}}",
            ha="left",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
        )
        ax.text(
            i,
            p05 - y_pad,
            f"p05 {p05:{fmt}}",
            ha="center",
            va="top",
            fontsize=7,
            color="#444444",
        )
        ax.text(
            i,
            p95 + y_pad,
            f"p95 {p95:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#444444",
        )
    if debug:
        print(f"_annotate_boxplot_percentiles output: n_boxes={len(data)}")


def aggregate_ktCO2_by_stack_type(
    results_dir="results_baseline",
    debug=False,
):
    """Sum plants_ktCO2tot_ccs by stack type for each experiment."""
    if debug:
        print(f"aggregate_ktCO2_by_stack_type input: results_dir={results_dir}")

    stacks = _load_array(results_dir, "plants_stack", debug=debug)
    sectors = _load_array(results_dir, "plants_sector", debug=debug)
    ktco2 = _load_array(results_dir, "plants_ktCO2tot_ccs", debug=debug)
    if stacks.shape != ktco2.shape:
        raise ValueError(f"Shape mismatch: plants_stack {stacks.shape}, plants_ktCO2tot_ccs {ktco2.shape}")

    stack_names = np.asarray(stacks[0], dtype=object)
    stack_types = _stack_aggregate_types(stack_names, sectors[0])
    type_order = []
    for st in stack_types:
        if st not in type_order:
            type_order.append(st)

    n_experiments = stacks.shape[0]
    aggregate_rows = []
    for i in range(n_experiments):
        row = {"experiment": i}
        for st in type_order:
            mask = stack_types == st
            row[st] = float(np.nansum(ktco2[i, mask]))
        aggregate_rows.append(row)

    stack_aggregates = pd.DataFrame(aggregate_rows)
    if debug:
        print(
            "aggregate_ktCO2_by_stack_type output:",
            f"experiments={n_experiments}, types={type_order}",
        )
    return stack_aggregates, type_order


def build_stack_scenario_dataframe(
    results_dir="results_baseline",
    pounds_to_EUR=1.15,
    debug=False,
):
    """
    Long-format plant data: one row per (experiment, stack) with stored CO2 volume and MAC.
    """
    if debug:
        print(f"build_stack_scenario_dataframe input: results_dir={results_dir}")

    stacks = _load_array(results_dir, "plants_stack", debug=debug)
    sectors = _load_array(results_dir, "plants_sector", debug=debug)
    ktco2 = _load_array(results_dir, "plants_ktCO2tot_ccs", debug=debug)
    mac = _load_array(results_dir, "plants_MAC", debug=debug)

    experiments_path = f"{results_dir}/experiments.csv"
    try:
        experiments = pd.read_csv(experiments_path)
    except FileNotFoundError:
        experiments = pd.DataFrame({"experiment": np.arange(stacks.shape[0])})

    rows = []
    for i in range(stacks.shape[0]):
        scenario = int(experiments.loc[i, "scenario"]) if "scenario" in experiments.columns else i
        policy = experiments.loc[i, "policy"] if "policy" in experiments.columns else None
        for j in range(stacks.shape[1]):
            stack_name = str(stacks[i, j])
            rows.append(
                {
                    "experiment": i,
                    "scenario": scenario,
                    "policy": policy,
                    "stack": stack_name,
                    "sector": str(sectors[i, j]),
                    "stack_type": _stack_aggregate_type(stack_name, sectors[i, j]),
                    "ktCO2tot_ccs": float(ktco2[i, j]),
                    "MAC": float(mac[i, j]) / pounds_to_EUR,
                }
            )

    stack_df = pd.DataFrame(rows)
    if debug:
        print(
            "build_stack_scenario_dataframe output:",
            f"rows={len(stack_df)}, stacks={stack_df['stack'].nunique()}, "
            f"experiments={stack_df['experiment'].nunique()}",
        )
    return stack_df


def plot_stack_aggregate_medians(
    stack_aggregates,
    type_order,
    figures_dir="results_figures",
    savefig=True,
    debug=False,
):
    """Plot median total ktCO2 capture volume by stack type across experiments."""
    if debug:
        print(
            "plot_stack_aggregate_medians input:",
            f"rows={len(stack_aggregates)}, types={len(type_order)}",
        )

    medians = stack_aggregates[type_order].median(axis=0)
    p05 = stack_aggregates[type_order].quantile(0.05, axis=0)
    p95 = stack_aggregates[type_order].quantile(0.95, axis=0)
    order = medians.sort_values(ascending=True).index.tolist()

    labels = [STACK_TYPE_LABELS.get(st, st) for st in order]
    y = medians[order].to_numpy(dtype=float)
    yerr_lo = (y - p05[order].to_numpy(dtype=float))
    yerr_hi = (p95[order].to_numpy(dtype=float) - y)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(order)))
    ax.barh(
        labels,
        y,
        xerr=[yerr_lo, yerr_hi],
        color=colors,
        alpha=0.9,
        capsize=4,
        error_kw={"elinewidth": 1.2, "alpha": 0.7},
    )
    _annotate_barh_medians(ax, y, fmt=".0f", debug=debug)
    ax.set_xlabel("Total CO₂ capture volume [kt p.a.]", fontsize=14)
    ax.set_title("Median CCS volume by stack type", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    if savefig:
        out = f"{figures_dir}/stack_aggregate_ktCO2_by_stack_type.png"
        fig.savefig(out, dpi=450, bbox_inches="tight")
        if debug:
            print(f"plot_stack_aggregate_medians output: file={out}")
    return fig


# Maps stack-type suffixes (stack aggregates) to policy buckets.
STACK_AGGREGATE_TO_BUCKET = {
    "HPU": "Blue Hydrogen CCS",
    "ccgt": "Power CCS",
    "cement": "Industrial CCS",
    "distillation": "Industrial CCS",
    "FCC": "Industrial CCS",
    "scattered": "Industrial CCS",
    "blast": "Industrial CCS",
    "sinter": "Industrial CCS",
    "steel_power": "Industrial CCS",
    "waste": "Energy from Waste CCS",
}

BUCKET_ORDER = [
    "Blue Hydrogen CCS",
    "Power CCS",
    "Industrial CCS",
    "Energy from Waste CCS",
    "BECCS",
]

def _plant_bucket_mask(stack_names, stack_aggregate_types, bucket):
    """Return boolean mask of plants belonging to a bucket."""
    if bucket == "BECCS":
        return np.array([s in BECCS_STACKS for s in stack_names], dtype=bool)
    stack_types_in_bucket = {
        st for st, b in STACK_AGGREGATE_TO_BUCKET.items() if b == bucket
    }
    return np.isin(stack_aggregate_types, list(stack_types_in_bucket))


def aggregate_buckets_by_experiment(
    results_dir="results_baseline",
    debug=False,
):
    """Sum plants_ktCO2tot_ccs by bucket for each experiment."""
    if debug:
        print(f"aggregate_buckets_by_experiment input: results_dir={results_dir}")

    stacks = _load_array(results_dir, "plants_stack", debug=debug)
    sectors = _load_array(results_dir, "plants_sector", debug=debug)
    ktco2 = _load_array(results_dir, "plants_ktCO2tot_ccs", debug=debug)
    stack_names = np.asarray(stacks[0], dtype=object)
    stack_aggregate_types = _stack_aggregate_types(stack_names, sectors[0])

    rows = []
    for i in range(stacks.shape[0]):
        row = {"experiment": i}
        for bucket in BUCKET_ORDER:
            mask = _plant_bucket_mask(stack_names, stack_aggregate_types, bucket)
            row[bucket] = float(np.nansum(ktco2[i, mask]))
        rows.append(row)

    bucket_aggregates = pd.DataFrame(rows)
    if debug:
        print(
            "aggregate_buckets_by_experiment output:",
            f"experiments={len(bucket_aggregates)}, buckets={BUCKET_ORDER}",
        )
    return bucket_aggregates


def bucket_mac_by_experiment(
    results_dir="results_baseline",
    pounds_to_EUR=1.15,
    debug=False,
):
    """
    Per-experiment median MAC [£/tCO2] within each bucket.
    """
    if debug:
        print(f"bucket_mac_by_experiment input: results_dir={results_dir}")

    stacks = _load_array(results_dir, "plants_stack", debug=debug)
    sectors = _load_array(results_dir, "plants_sector", debug=debug)
    mac = _load_array(results_dir, "plants_MAC", debug=debug)
    stack_names = np.asarray(stacks[0], dtype=object)
    stack_aggregate_types = _stack_aggregate_types(stack_names, sectors[0])

    per_experiment = {bucket: [] for bucket in BUCKET_ORDER}
    for i in range(stacks.shape[0]):
        mac_i = np.asarray(mac[i], dtype=float) / pounds_to_EUR
        for bucket in BUCKET_ORDER:
            mask = _plant_bucket_mask(stack_names, stack_aggregate_types, bucket)
            vals = mac_i[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                per_experiment[bucket].append(float(np.median(vals)))

    if debug:
        sizes = {b: len(v) for b, v in per_experiment.items()}
        print(f"bucket_mac_by_experiment output: sample_sizes={sizes}")
    return per_experiment


def plot_bucket_mac_boxplots(
    mac_by_experiment,
    figures_dir="results_figures",
    savefig=True,
    debug=False,
):
    """Box plots of bucket MAC with whiskers at the 5th and 95th percentiles."""
    if debug:
        print(f"plot_bucket_mac_boxplots input: buckets={BUCKET_ORDER}")

    data = [mac_by_experiment[bucket] for bucket in BUCKET_ORDER]
    labels = list(BUCKET_ORDER)

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2.0},
        boxprops={"facecolor": plt.cm.magma(0.35), "alpha": 0.85},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
    )
    ax.set_ylabel("MAC [£/tCO₂]", fontsize=14)
    ax.set_title("MAC by bucket (5th–95th percentile range)", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _annotate_boxplot_percentiles(ax, data, fmt=".1f", debug=debug)

    fig.tight_layout()
    if savefig:
        out = f"{figures_dir}/bucket_mac_boxplots.png"
        fig.savefig(out, dpi=450, bbox_inches="tight")
        if debug:
            print(f"plot_bucket_mac_boxplots output: file={out}")
    return fig


def plot_bucket_ktCO2_medians(
    bucket_aggregates,
    figures_dir="results_figures",
    savefig=True,
    debug=False,
):
    """Plot median total ktCO2 capture volume by bucket across experiments."""
    if debug:
        print(f"plot_bucket_ktCO2_medians input: rows={len(bucket_aggregates)}")

    medians = bucket_aggregates[BUCKET_ORDER].median(axis=0)
    p05 = bucket_aggregates[BUCKET_ORDER].quantile(0.05, axis=0)
    p95 = bucket_aggregates[BUCKET_ORDER].quantile(0.95, axis=0)
    order = medians.sort_values(ascending=True).index.tolist()

    y = medians[order].to_numpy(dtype=float)
    yerr_lo = y - p05[order].to_numpy(dtype=float)
    yerr_hi = p95[order].to_numpy(dtype=float) - y

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(order)))
    ax.barh(
        order,
        y,
        xerr=[yerr_lo, yerr_hi],
        color=colors,
        alpha=0.9,
        capsize=4,
        error_kw={"elinewidth": 1.2, "alpha": 0.7},
    )
    _annotate_barh_medians(ax, y, fmt=".0f", debug=debug)
    ax.set_xlabel("Total CO₂ capture volume [kt p.a.]", fontsize=14)
    ax.set_title("Median CCS volume by bucket", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    if savefig:
        out = f"{figures_dir}/bucket_ktCO2_medians.png"
        fig.savefig(out, dpi=450, bbox_inches="tight")
        if debug:
            print(f"plot_bucket_ktCO2_medians output: file={out}")
    return fig


def main(
    results_dir="results_baseline",
    figures_dir="results_figures",
    savefig=True,
    debug=False,
):
    if debug:
        print(f"main input: results_dir={results_dir}, figures_dir={figures_dir}")

    stack_scenario_df = build_stack_scenario_dataframe(
        results_dir=results_dir, debug=debug
    )
    stack_scenario_df.to_csv(
        f"{results_dir}/stack_scenario_ktCO2_mac.csv", index=False
    )
    print("\nStack-scenario dataframe (first rows):")
    print(stack_scenario_df.head().to_string(index=False))

    stack_aggregates, type_order = aggregate_ktCO2_by_stack_type(
        results_dir=results_dir, debug=debug
    )
    stack_aggregates.to_csv(
        f"{results_dir}/stack_aggregate_ktCO2_by_stack_type.csv", index=False
    )
    plot_stack_aggregate_medians(
        stack_aggregates,
        type_order,
        figures_dir=figures_dir,
        savefig=savefig,
        debug=debug,
    )

    bucket_aggregates = aggregate_buckets_by_experiment(
        results_dir=results_dir, debug=debug
    )
    bucket_aggregates.to_csv(
        f"{results_dir}/bucket_ktCO2_by_experiment.csv", index=False
    )
    plot_bucket_ktCO2_medians(
        bucket_aggregates,
        figures_dir=figures_dir,
        savefig=savefig,
        debug=debug,
    )

    mac_by_experiment = bucket_mac_by_experiment(results_dir=results_dir, debug=debug)
    plot_bucket_mac_boxplots(
        mac_by_experiment,
        figures_dir=figures_dir,
        savefig=savefig,
        debug=debug,
    )

    plt.show()


if __name__ == "__main__":
    main(debug=True)
