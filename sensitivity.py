import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression


def build_long_dataframe(debug=False):
    """
    Arrange plant constants (C), scenario parameters (X), and plant-level
    outcomes (R) into a single long-format dataframe.

    Each row is one (scenario, plant) combination. NPV_total is a standalone
    column suitable as a regression target for sensitivity analysis.
    """
    C = pd.read_csv("results/plants_clean.csv")
    X = pd.read_csv("results/experiments.csv")
    R = np.load("results/outcomes_plants_NPV_total.npy")
    R_year = np.load("results/outcomes_plants_investment_year.npy")
    ref = pd.read_csv("results/plant_reference.csv")

    if debug:
        print(f"C shape: {C.shape}, X shape: {X.shape}, R shape: {R.shape}, ref shape: {ref.shape}")

    n_scenarios, n_plants = R.shape

    # --- Separate X into model inputs vs. scalar outcomes ---
    # These are the true input columns (uncertainties + levers + metadata)
    scalar_outcome_cols = [
        "gas_increase_2040", "year_DACCS_marginal",
        "NPV_cost_CTBO", "NPV_profit_CTBO", "NPV_cost_ETS", "NPV_profit_ETS",
        "NPV_CTBO_passthrough", "NPV_CfD_passthrough",
        "NPV_ETS_passthrough", "NPV_total_passthrough", "pembroke_CAPEX",
    ]
    input_cols = [c for c in X.columns if c not in scalar_outcome_cols]
    X_inputs = X[input_cols]

    if debug:
        print(f"Input columns ({len(input_cols)}): {input_cols}")
        print(f"Scalar outcome columns ({len(scalar_outcome_cols)}): {scalar_outcome_cols}")

    # --- Build plant constants aligned to ref order ---
    # ref has 110 plants that the model actually uses; match to C via 'stack'
    plant_constants = ref.merge(C, on="stack", how="left", suffixes=("", "_C"))
    # sector appears in both ref and C — keep the ref version, drop duplicate
    if "sector_C" in plant_constants.columns:
        plant_constants = plant_constants.drop(columns=["sector_C"])

    if debug:
        unmatched = plant_constants["ktCO2"].isna().sum()
        print(f"Plants matched: {n_plants - unmatched}/{n_plants}")

    # --- Tile and repeat to long format ---
    # Repeat each scenario row n_plants times (one per plant)
    X_long = X_inputs.loc[X_inputs.index.repeat(n_plants)].reset_index(drop=True)

    # Tile plant constants n_scenarios times (same plant block for each scenario)
    C_long = pd.concat([plant_constants] * n_scenarios, ignore_index=True)

    # Flatten R arrays: scenario 0 × all plants, scenario 1 × all plants, ...
    npv_total = R.flatten()
    inv_year = R_year.flatten()

    df = pd.concat([
        X_long, C_long,
        pd.Series(inv_year, name="investment_year"),
        pd.Series(npv_total, name="NPV_total"),
    ], axis=1)

    if debug:
        print(f"Long dataframe shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    return df


EXCLUDE_COLS = {
    "NPV_total", "investment_year", "scenario", "policy", "model",
    "plant_index", "stack", "site", "sector",
    "latitude", "longitude", "hub",
}

CATEGORICAL_COLS = [
    "DACCS_SCENARIO", "ETS_SCENARIO", "energy_strategy",
    "land_transport", "sea_transport",
]


def compute_src(df, sector, target="NPV_total", top_n=15, debug=False):
    """
    Standardized Regression Coefficients for one sector.
    Returns (src Series sorted by |SRC|, R² of the linear model).
    """
    subset, features = _prepare_features(df, sector, target=target)

    if debug:
        print(f"  {sector}: {len(subset)} rows, {features.shape[1]} features")

    y = subset[target].values
    X_std = StandardScaler().fit_transform(features.values)
    y_std = (y - y.mean()) / y.std()

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_std, y_std)
    r2 = reg.score(X_std, y_std)

    src = pd.Series(reg.coef_, index=features.columns, name="SRC")
    src = src.reindex(src.abs().sort_values(ascending=False).index)

    if debug:
        print(f"  R² = {r2:.3f}")
        print(src.head(top_n).to_string())

    return src, r2


def plot_src_by_sector(df, target="NPV_total", top_n=15, debug=False):
    """Plot horizontal bar charts of top SRCs for each sector."""
    sectors = sorted(df["sector"].dropna().unique())
    n = len(sectors)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()

    for i, sector in enumerate(sectors):
        src, r2 = compute_src(df, sector, target=target, top_n=top_n, debug=debug)
        top = src.head(top_n)
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in top]

        axes[i].barh(range(len(top)), top.values, color=colors)
        axes[i].set_yticks(range(len(top)))
        axes[i].set_yticklabels(top.index, fontsize=9)
        axes[i].invert_yaxis()
        axes[i].set_title(f"{sector}  (R² = {r2:.2f})", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("SRC", fontsize=11)
        axes[i].axvline(0, color="grey", linewidth=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    safe_name = target.replace(" ", "_")
    fig.suptitle(f"Standardized Regression Coefficients — {target} by sector",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"results/sensitivity_src_{safe_name}.png", dpi=200, bbox_inches="tight")
    if debug:
        print(f"\nSaved to results/sensitivity_src_{safe_name}.png")
     


def plot_npv_vs_investment_year(df, debug=False):
    """Box plots of NPV_total grouped by investment year, one subplot per sector."""
    clean = df.dropna(subset=["NPV_total", "investment_year"]).copy()
    clean["investment_year"] = clean["investment_year"].astype(int)

    sectors = sorted(clean["sector"].dropna().unique())
    n = len(sectors)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()

    for i, sector in enumerate(sectors):
        sub = clean[clean["sector"] == sector]
        years = sorted(sub["investment_year"].unique())
        data = [sub.loc[sub["investment_year"] == y, "NPV_total"].values / 1e6
                for y in years]

        bp = axes[i].boxplot(data, positions=range(len(years)), widths=0.6,
                             patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4c72b0")
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")

        axes[i].set_xticks(range(len(years)))
        axes[i].set_xticklabels(years, fontsize=9, rotation=45)
        axes[i].set_xlabel("Investment year", fontsize=11)
        axes[i].set_ylabel("NPV_total [M£]", fontsize=11)

        r = sub[["investment_year", "NPV_total"]].corr().iloc[0, 1]
        axes[i].set_title(f"{sector}  (r = {r:.2f}, n = {len(sub)})",
                          fontsize=13, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("NPV_total vs. investment year by sector",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("results/sensitivity_npv_vs_year.png", dpi=200, bbox_inches="tight")
    if debug:
        print("Saved to results/sensitivity_npv_vs_year.png")
     

def _prepare_features(df, sector, target="NPV_total"):
    """Filter to sector, drop NaN target, encode categoricals, drop zero-variance."""
    subset = df[df["sector"] == sector].dropna(subset=[target]).copy()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    features = subset[feature_cols].copy()

    cat_present = [c for c in CATEGORICAL_COLS if c in feature_cols]
    for c in cat_present:
        features[c] = features[c].fillna("None")

    features = pd.get_dummies(features, columns=cat_present, drop_first=True)
    features = features.select_dtypes(include=[np.number])
    features = features.loc[:, features.std() > 1e-10]
    return subset, features


def compute_logistic(df, sector, top_n=15, debug=False):
    """
    Logistic regression with standardized inputs for P(NPV_total > 0).
    Returns (coef Series sorted by |coef|, accuracy, class balance).
    """
    subset, features = _prepare_features(df, sector)
    y = (subset["NPV_total"].values > 0).astype(int)

    # Skip sectors where all outcomes are the same class
    if y.mean() == 0.0 or y.mean() == 1.0:
        if debug:
            print(f"  {sector}: skipped — all samples in one class (mean={y.mean():.2f})")
        return None, None, y.mean()

    X_std = StandardScaler().fit_transform(features.values)

    clf = LogisticRegression(fit_intercept=True, max_iter=1000, penalty=None)
    clf.fit(X_std, y)
    acc = clf.score(X_std, y)

    coefs = pd.Series(clf.coef_[0], index=features.columns, name="coef")
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    if debug:
        pct_pos = y.mean() * 100
        print(f"  {sector}: n={len(y)}, {pct_pos:.1f}% positive, accuracy={acc:.3f}")
        print(coefs.head(top_n).to_string())

    return coefs, acc, y.mean()


def plot_logistic_by_sector(df, top_n=15, debug=False):
    """Plot logistic regression coefficients for P(NPV_total > 0) per sector."""
    sectors = sorted(df["sector"].dropna().unique())
    n = len(sectors)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()

    plot_idx = 0
    for sector in sectors:
        coefs, acc, pct_pos = compute_logistic(df, sector, top_n=top_n, debug=debug)
        if coefs is None:
            continue
        top = coefs.head(top_n)
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in top]

        axes[plot_idx].barh(range(len(top)), top.values, color=colors)
        axes[plot_idx].set_yticks(range(len(top)))
        axes[plot_idx].set_yticklabels(top.index, fontsize=9)
        axes[plot_idx].invert_yaxis()
        axes[plot_idx].set_title(
            f"{sector}  (acc={acc:.2f}, {pct_pos*100:.0f}% positive)",
            fontsize=13, fontweight="bold")
        axes[plot_idx].set_xlabel("Logistic coefficient (standardized)", fontsize=11)
        axes[plot_idx].axvline(0, color="grey", linewidth=0.5)
        plot_idx += 1

    for j in range(plot_idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Drivers of positive NPV_total (logistic regression) by sector",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("results/sensitivity_logistic.png", dpi=200, bbox_inches="tight")
    if debug:
        print("\nSaved to results/sensitivity_logistic.png")


if __name__ == "__main__":
    df = build_long_dataframe(debug=True)
    plot_logistic_by_sector(df, top_n=15, debug=True)
    plot_src_by_sector(df, target="NPV_total", top_n=15, debug=True)
    plot_src_by_sector(df, target="investment_year", top_n=15, debug=True)
    plt.show()