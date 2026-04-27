import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


UNCERTAINTY_COLS = [
    "DIFFUSE_END_FRACTION", "DACCS_SCENARIO", "fraction_limestone", "fraction_fossil_waste",
    "drax_efficiency", "drax_efficiency_loss", "capture_rate", "qreb", "pcomp",
    "pcomp_liquefy", "FLH_industry", "FLH_waste", "ccgt_efficiency", "ccgt_efficiency_loss",
    "cgas", "celc", "cstraw", "cliquefy", "cHCN", "cHRSG", "camine", "cstorage",
    "transport_uncertainty", "CAPEX_gasboiler", "CAPEX_bioboiler", "fixate_CAPEX", "CAPEX_m",
    "discount_rate_ccs", "lifetime_ccs", "CEPCI_2025", "NETL_2025",
]

POLICY_ORDER = ["CTBO-only", "£100-Mix", "£200-Mix", "£300-Mix", "ETS-eq"]


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


def _prepare_xy(df, uncertainty_cols, target_col, debug=False):
    if debug:
        print(f"_prepare_xy input: rows={len(df)}, target_col={target_col}")
    work = df[uncertainty_cols + [target_col]].copy()
    cat_cols = [c for c in uncertainty_cols if work[c].dtype == "object"]
    X = pd.get_dummies(work[uncertainty_cols], columns=cat_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work[target_col], errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if debug:
        print(f"_prepare_xy output: X_shape={X.shape}, y_shape={y.shape}")
    return X, y


def policy_sensitivity_npv_costs_consumers(
    results_dir="results_baseline",
    figures_dir="results_figures",
    target_col="NPV_costs_consumers",
    top_n=12,
    min_rows_per_policy=10,
    n_repeats=12,
    random_state=42,
    debug=False,
):
    if debug:
        print(
            "policy_sensitivity_npv_costs_consumers input:",
            f"results_dir={results_dir}, figures_dir={figures_dir}, target_col={target_col}, min_rows_per_policy={min_rows_per_policy}",
        )

    experiments = pd.read_csv(f"{results_dir}/experiments.csv")
    if "policy" not in experiments.columns:
        raise KeyError("Column 'policy' not found in experiments.csv")
    if target_col not in experiments.columns:
        raise KeyError(f"Column {target_col!r} not found in experiments.csv")

    summary_rows = []
    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(4.2 * len(POLICY_ORDER), 6.5), sharex=False, sharey=False)
    if len(POLICY_ORDER) == 1:
        axes = [axes]

    for ax, policy in zip(axes, POLICY_ORDER):
        subset = experiments[experiments["policy"] == policy].copy()
        if len(subset) < min_rows_per_policy:
            ax.set_title(f"{policy}\n(not enough data)", fontsize=14)
            ax.axis("off")
            continue

        X, y = _prepare_xy(subset, UNCERTAINTY_COLS, target_col, debug=debug)
        if X.empty:
            ax.set_title(f"{policy}\n(no valid data)", fontsize=14)
            ax.axis("off")
            continue

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        model.fit(X, y)

        perm = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
            scoring="r2",
        )
        imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
        top = imp.head(top_n).iloc[::-1]

        ax.barh(top.index, top.values, color=plt.cm.magma(np.linspace(0.2, 0.85, len(top))))
        ax.set_title(policy, fontsize=14)
        ax.set_xlabel("Permutation importance (drop in R²)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)

        for name, value in imp.items():
            summary_rows.append(
                {
                    "policy": policy,
                    "feature": name,
                    "importance_mean": float(value),
                }
            )

    fig.suptitle("Sensitivity of NPV_costs_consumers to uncertainties, by policy", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/sensitivity_npv_costs_consumers_by_policy.png", dpi=300, bbox_inches="tight")

    if len(summary_rows) == 0:
        summary = pd.DataFrame(columns=["policy", "feature", "importance_mean"])
    else:
        summary = (
            pd.DataFrame(summary_rows)
            .sort_values(["policy", "importance_mean"], ascending=[True, False])
            .reset_index(drop=True)
        )
    summary.to_csv(f"{results_dir}/sensitivity_npv_costs_consumers_by_policy.csv", index=False)

    if debug:
        print(
            "policy_sensitivity_npv_costs_consumers output:",
            f"summary_rows={len(summary)},",
            f"figure={figures_dir}/sensitivity_npv_costs_consumers_by_policy.png,",
            f"csv={results_dir}/sensitivity_npv_costs_consumers_by_policy.csv",
        )
    return summary


if __name__ == "__main__":
    results_dir = _select_results_dir(debug=True)
    figures_dir = "results_figures"
    policy_sensitivity_npv_costs_consumers(
        results_dir=results_dir,
        figures_dir=figures_dir,
        target_col="NPV_costs_consumers",
        top_n=10,
        min_rows_per_policy=10,
        n_repeats=12,
        random_state=42,
        debug=True,
    )
    plt.show()