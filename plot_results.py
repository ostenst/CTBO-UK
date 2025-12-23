"""
Plot results from EMA Workbench experiments.

This script loads the saved experiments and outcomes and generates plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(debug=False):
    """Load experiments and outcomes from saved files."""
    if debug:
        print("Loading results...")
    
    combined_df = pd.read_csv("experiments.csv")
    
    # Load array outcomes
    array_outcomes = {}
    array_names = [
        "gas_increase_pct",
        "fCCS_capacity_vec",
        "BECCS_capacity_vec",
        "DACCS_capacity_vec",
        "CTBO_cost_lev_vec",
        "plant_npv_net",
        "plant_npv_gross",
        "plant_investment_year",
    ]
    for name in array_names:
        try:
            array_outcomes[name] = np.load(f"outcomes_{name}.npy")
            if debug:
                print(f"  Loaded {name}: shape {array_outcomes[name].shape}")
        except FileNotFoundError:
            if debug:
                print(f"  Warning: outcomes_{name}.npy not found")
    
    # Load plant names reference
    plant_names = None
    try:
        plant_names_df = pd.read_csv("plant_names_reference.csv")
        plant_names = plant_names_df['plant_name'].tolist()
        if debug:
            print(f"  Loaded plant_names_reference.csv: {len(plant_names)} plants")
    except FileNotFoundError:
        if debug:
            print("  Warning: plant_names_reference.csv not found")
    
    return combined_df, array_outcomes, plant_names


def plot_boxplot_by_scenario(combined_df, outcome_col, scenario_col="ETS_SCENARIO", 
                              ylabel=None, title=None, debug=False):
    """Plot boxplot of an outcome grouped by scenario."""
    if debug:
        print(f"Plotting boxplot: {outcome_col} by {scenario_col}")
    
    ets_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x=scenario_col, y=outcome_col, 
                order=["Low", "Medium", "High"], palette=ets_colors)
    plt.xlabel('ETS Scenario', fontsize=13)
    plt.ylabel(ylabel or outcome_col, fontsize=13)
    plt.title(title or f'{outcome_col} by ETS Scenario', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()


def plot_timeseries_by_scenario(combined_df, array_data, scenario_col="ETS_SCENARIO",
                                 ylabel=None, title=None, start_year=2025, debug=False):
    """Plot time series with mean and std by scenario."""
    if debug:
        print(f"Plotting time series by {scenario_col}")
    
    ets_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    years = np.arange(start_year, start_year + array_data.shape[1])
    
    plt.figure(figsize=(10, 6))
    for category in ["Low", "Medium", "High"]:
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        category_data = array_data[mask.values]
        
        mean_vals = category_data.mean(axis=0)
        std_vals = category_data.std(axis=0)
        color = ets_colors[category]
        
        plt.plot(years, mean_vals, label=category, linewidth=2, color=color)
        plt.fill_between(years, mean_vals - std_vals, mean_vals + std_vals, 
                         alpha=0.2, color=color)
    
    plt.xlabel('Year', fontsize=13)
    plt.ylabel(ylabel or 'Value', fontsize=13)
    plt.title(title or 'Time Series by ETS Scenario', fontsize=14)
    plt.legend(fontsize=11, title='ETS Scenario')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_capacity_stack(combined_df, array_outcomes, scenario_col="ETS_SCENARIO",
                        start_year=2025, debug=False):
    """Plot stacked area chart of CCS capacities by scenario."""
    if debug:
        print(f"Plotting capacity stack by {scenario_col}")
    
    ets_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    
    fCCS = array_outcomes.get("fCCS_capacity_vec")
    BECCS = array_outcomes.get("BECCS_capacity_vec")
    DACCS = array_outcomes.get("DACCS_capacity_vec")
    
    if fCCS is None or BECCS is None or DACCS is None:
        print("Warning: Missing capacity data for stack plot")
        return
    
    years = np.arange(start_year, start_year + fCCS.shape[1])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for ax, category in zip(axes, ["Low", "Medium", "High"]):
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        
        fCCS_mean = fCCS[mask.values].mean(axis=0)
        BECCS_mean = BECCS[mask.values].mean(axis=0)
        DACCS_mean = DACCS[mask.values].mean(axis=0)
        
        ax.stackplot(years, DACCS_mean, BECCS_mean, fCCS_mean,
                     labels=['DACCS', 'BECCS', 'Fossil CCS'],
                     colors=['#9b59b6', '#27ae60', '#7f8c8d'], alpha=0.8)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_title(f'ETS: {category}', fontsize=13, color=ets_colors[category])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
    
    axes[0].set_ylabel('Capacity (ktCO2/yr)', fontsize=12)
    plt.suptitle('CCS Capacity by Technology and ETS Scenario', fontsize=14)
    plt.tight_layout()


def plot_plant_npv(combined_df, array_outcomes, plant_names, scenario_col="ETS_SCENARIO",
                   debug=False):
    """Plot plant-level NPV by scenario."""
    if debug:
        print(f"Plotting plant NPV by {scenario_col}")
    
    plant_npv_net = array_outcomes.get("plant_npv_net")
    if plant_npv_net is None or plant_names is None:
        print("Warning: Missing plant NPV data")
        return
    
    ets_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    
    # Calculate mean NPV per plant across all experiments
    plt.figure(figsize=(14, 6))
    x = np.arange(len(plant_names))
    width = 0.25
    
    for i, category in enumerate(["Low", "Medium", "High"]):
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        category_npv = plant_npv_net[mask.values]
        mean_npv = np.nanmean(category_npv, axis=0) / 1000  # Convert to MEUR
        
        plt.bar(x + i * width, mean_npv, width, label=category, 
                color=ets_colors[category], alpha=0.8)
    
    plt.xlabel('Plant', fontsize=12)
    plt.ylabel('Mean NPV Net Profit (MEUR)', fontsize=12)
    plt.title('Plant-level NPV by ETS Scenario', fontsize=14)
    plt.xticks(x + width, plant_names, rotation=90, fontsize=7)
    plt.legend(title='ETS Scenario', fontsize=10)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()


def plot_npv_vs_investment_year(combined_df, array_outcomes, scenario_col="ETS_SCENARIO",
                                 debug=False):
    """Plot NPV vs investment year with uncertainty bands by ETS scenario."""
    if debug:
        print(f"Plotting NPV vs investment year by {scenario_col}")
    
    plant_npv_net = array_outcomes.get("plant_npv_net")
    plant_inv_year = array_outcomes.get("plant_investment_year")
    
    if plant_npv_net is None or plant_inv_year is None:
        print("Warning: Missing plant NPV or investment year data")
        return
    
    ets_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    
    plt.figure(figsize=(10, 6))
    
    for category in ["Low", "Medium", "High"]:
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        
        # Get data for this category
        cat_npv = plant_npv_net[mask.values]      # shape: (n_exp, n_plants)
        cat_year = plant_inv_year[mask.values]    # shape: (n_exp, n_plants)
        
        # Flatten and remove NaN pairs
        npv_flat = cat_npv.flatten()
        year_flat = cat_year.flatten()
        valid_mask = ~np.isnan(npv_flat) & ~np.isnan(year_flat)
        npv_valid = npv_flat[valid_mask] / 1000  # Convert to MEUR
        year_valid = year_flat[valid_mask].astype(int)
        
        # Group by year and calculate mean/std
        unique_years = np.sort(np.unique(year_valid))
        mean_npv = []
        std_npv = []
        
        for yr in unique_years:
            yr_npv = npv_valid[year_valid == yr]
            mean_npv.append(np.mean(yr_npv))
            std_npv.append(np.std(yr_npv))
        
        mean_npv = np.array(mean_npv)
        std_npv = np.array(std_npv)
        color = ets_colors[category]
        
        plt.plot(unique_years, mean_npv, label=category, linewidth=2, color=color)
        plt.fill_between(unique_years, mean_npv - std_npv, mean_npv + std_npv,
                         alpha=0.2, color=color)
    
    plt.xlabel('Investment Year', fontsize=13)
    plt.ylabel('NPV Net Profit (MEUR)', fontsize=13)
    plt.title('NPV vs Investment Year by ETS Scenario', fontsize=14)
    plt.legend(fontsize=11, title='ETS Scenario')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_positive_npv_fraction(combined_df, array_outcomes, scenario_col="ETS_SCENARIO",
                                debug=False):
    """Plot fraction of positive NPV vs investment year by ETS scenario."""
    if debug:
        print(f"Plotting positive NPV fraction by {scenario_col}")
    
    plant_npv_net = array_outcomes.get("plant_npv_net")
    plant_inv_year = array_outcomes.get("plant_investment_year")
    
    if plant_npv_net is None or plant_inv_year is None:
        print("Warning: Missing plant NPV or investment year data")
        return
    
    ets_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    
    plt.figure(figsize=(10, 6))
    
    for category in ["Low", "Medium", "High"]:
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        
        # Get data for this category
        cat_npv = plant_npv_net[mask.values]      # shape: (n_exp, n_plants)
        cat_year = plant_inv_year[mask.values]    # shape: (n_exp, n_plants)
        n_exp = cat_npv.shape[0]
        
        # Get all unique years across all experiments
        all_years = cat_year[~np.isnan(cat_year)].astype(int)
        unique_years = np.sort(np.unique(all_years))
        
        # For each experiment, calculate fraction positive per year
        # Then compute mean/std across experiments
        fractions_per_exp = []  # shape will be (n_exp, n_years)
        
        for exp_idx in range(n_exp):
            exp_npv = cat_npv[exp_idx]
            exp_year = cat_year[exp_idx]
            exp_fractions = []
            
            for yr in unique_years:
                yr_mask = exp_year == yr
                if yr_mask.sum() > 0:
                    yr_npv = exp_npv[yr_mask]
                    valid = ~np.isnan(yr_npv)
                    if valid.sum() > 0:
                        frac_positive = (yr_npv[valid] > 0).sum() / valid.sum()
                    else:
                        frac_positive = np.nan
                else:
                    frac_positive = np.nan
                exp_fractions.append(frac_positive)
            
            fractions_per_exp.append(exp_fractions)
        
        fractions_arr = np.array(fractions_per_exp)  # (n_exp, n_years)
        mean_frac = np.nanmean(fractions_arr, axis=0)
        std_frac = np.nanstd(fractions_arr, axis=0)
        
        color = ets_colors[category]
        plt.plot(unique_years, mean_frac, label=category, linewidth=2, color=color)
        plt.fill_between(unique_years, mean_frac - std_frac, mean_frac + std_frac,
                         alpha=0.2, color=color)
    
    plt.xlabel('Investment Year', fontsize=13)
    plt.ylabel('Fraction with Positive NPV', fontsize=13)
    plt.title('Fraction of Plants with Positive NPV vs Investment Year', fontsize=14)
    plt.legend(fontsize=11, title='ETS Scenario')
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    # Load results
    combined_df, array_outcomes, plant_names = load_results(debug=True)

    # In array_outcomes, check how many scenarios have negative gas_increase_pct in any year
    negative_gas_increase_pct = array_outcomes["gas_increase_pct"] < 0
    print(f"Number of scenarios with negative gas increase in any year: {negative_gas_increase_pct.sum()}")
    
    print(f"\nLoaded {len(combined_df)} experiments")
    print(f"Columns: {list(combined_df.columns)}")
    if plant_names:
        print(f"Plants: {len(plant_names)}")
    
    # Plot 1: Boxplot of gas increase in 2040
    plot_boxplot_by_scenario(
        combined_df, 
        outcome_col="gas_increase_pct_2040",
        ylabel="Gas Price Increase (%) in 2040",
        title="Gas Price Increase in 2040 by ETS Scenario"
    )
    
    # Plot 2: Gas increase percentage over time
    if "gas_increase_pct" in array_outcomes:
        plot_timeseries_by_scenario(
            combined_df,
            array_outcomes["gas_increase_pct"],
            ylabel="Gas Price Increase (%)",
            title="Gas Price Increase Over Time by ETS Scenario"
        )
    
    # Plot 3: CTBO levelized cost over time
    if "CTBO_cost_lev_vec" in array_outcomes:
        plot_timeseries_by_scenario(
            combined_df,
            array_outcomes["CTBO_cost_lev_vec"],
            ylabel="CTBO Cost (EUR/tCO2)",
            title="Levelized CTBO Cost Over Time by ETS Scenario"
        )
    
    # Plot 4: CCS capacity stack
    plot_capacity_stack(combined_df, array_outcomes)
    
    # Plot 5: Plant-level NPV
    if plant_names:
        plot_plant_npv(combined_df, array_outcomes, plant_names)
    
    # Plot 6: NPV vs Investment Year
    plot_npv_vs_investment_year(combined_df, array_outcomes)
    
    # Plot 7: Fraction of positive NPV vs Investment Year
    plot_positive_npv_fraction(combined_df, array_outcomes)
    
    plt.show()

