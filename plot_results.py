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
        "CDR_capacity_vec",
        "CTBO_cost_lev_vec",
        "CSU_cost_vec",
        "ets_prices",
        "CTBO_cost_vec",
        "supplied_CO2_vec",
        "total_emissions_vec",
        "ctbo_mandate_vec",
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
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.2), "Medium": magma(0.5), "High": magma(0.8)}
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x=scenario_col, y=outcome_col, 
                order=["Low", "Medium", "High"], palette=ets_colors)
    plt.xlabel('ETS Scenario', fontsize=13)
    plt.ylabel(ylabel or outcome_col, fontsize=13)
    plt.title(title or f'{outcome_col} by ETS Scenario', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()


def plot_timeseries_by_scenario(combined_df, array_data, scenario_col="ETS_SCENARIO",
                                 ylabel=None, title=None, start_year=2025, debug=False, savefig=False):
    """Plot time series with mean and std by scenario."""
    if debug:
        print(f"Plotting time series by {scenario_col}")
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.1), "Medium": magma(0.5), "High": magma(0.9)}
    years = np.arange(start_year, start_year + array_data.shape[1])
    
    plt.figure(figsize=(5, 6))
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
    if savefig:
        plt.savefig(f'4_{title}.png', dpi=300)


def plot_ctbo_total_cost(combined_df, array_outcomes, scenario_col="ETS_SCENARIO",
                          ylabel=None, title=None, start_year=2025, debug=False):
    """Plot CTBO total cost time series with uncertainty bands by scenario."""
    if debug:
        print(f"Plotting CTBO total cost by {scenario_col}")
    
    ctbo_cost = array_outcomes.get("CTBO_cost_vec")/1000
    
    if ctbo_cost is None:
        print("Warning: Missing CTBO_cost_vec data")
        return
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.1), "Medium": magma(0.5), "High": magma(0.9)}
    years = np.arange(start_year, start_year + ctbo_cost.shape[1])
    
    plt.figure(figsize=(5, 6))
    for category in ["Low", "Medium", "High"]:
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        category_data = ctbo_cost[mask.values]
        
        mean_vals = category_data.mean(axis=0)
        std_vals = category_data.std(axis=0)
        color = ets_colors[category]
        
        plt.plot(years, mean_vals, label=category, linewidth=2, color=color)
        plt.fill_between(years, mean_vals - std_vals, mean_vals + std_vals, 
                         alpha=0.2, color=color)
    
    plt.xlabel('Year', fontsize=13)
    plt.ylabel(ylabel or 'Total CTBO Cost (MEUR/yr)', fontsize=13)
    plt.title(title or 'Total CTBO Cost Over Time by ETS Scenario', fontsize=14)
    plt.legend(fontsize=11, title='ETS Scenario')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_csu_cost_with_ets(combined_df, array_outcomes, scenario_col="ETS_SCENARIO",
                            ylabel=None, title=None, start_year=2025, debug=False):
    """Plot CSU cost time series with uncertainty bands and ETS prices overlay."""
    if debug:
        print(f"Plotting CSU cost with ETS prices by {scenario_col}")
    
    csu_cost = array_outcomes.get("CSU_cost_vec")
    ets_prices = array_outcomes.get("ets_prices")
    
    if csu_cost is None:
        print("Warning: Missing CSU_cost_vec data")
        return
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.1), "Medium": magma(0.5), "High": magma(0.9)}
    years = np.arange(start_year, start_year + csu_cost.shape[1])
    
    plt.figure(figsize=(10, 6))
    
    # Plot CSU costs with uncertainty bands
    for category in ["Low", "Medium", "High"]:
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        category_data = csu_cost[mask.values]
        
        mean_vals = category_data.mean(axis=0)
        std_vals = category_data.std(axis=0)
        color = ets_colors[category]
        
        plt.plot(years, mean_vals, label=f'CSU Cost ({category})', linewidth=2, color=color)
        plt.fill_between(years, mean_vals - std_vals, mean_vals + std_vals, 
                         alpha=0.2, color=color)
    
    # Overlay ETS prices
    if ets_prices is not None:
        for category in ["Low", "Medium", "High"]:
            mask = combined_df[scenario_col] == category
            if mask.sum() == 0:
                continue
            category_ets = ets_prices[mask.values]
            mean_ets = category_ets.mean(axis=0)
            color = ets_colors[category]
            
            plt.plot(years, mean_ets, label=f'ETS Price ({category})', 
                     linewidth=2, linestyle='--', color=color, alpha=0.7)
    
    plt.xlabel('Year', fontsize=13)
    plt.ylabel(ylabel or 'Cost (EUR/tCO2)', fontsize=13)
    plt.title(title or 'CSU Cost and ETS Price Over Time by ETS Scenario', fontsize=14)
    plt.legend(fontsize=10, title='Scenario', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_carbon_balance(combined_df, array_outcomes, start_year=2025, debug=False):
    """Plot carbon balance time series (supplied, emissions, mandate) with uncertainty bands."""
    if debug:
        print("Plotting carbon balance")
    
    supplied = array_outcomes.get("supplied_CO2_vec")
    emissions = array_outcomes.get("total_emissions_vec")
    mandate = array_outcomes.get("ctbo_mandate_vec")
    CDR = array_outcomes.get("CDR_capacity_vec")
    
    if supplied is None or emissions is None or mandate is None:
        print("Warning: Missing carbon balance data (supplied_CO2_vec, total_emissions_vec, ctbo_mandate_vec)")
        return
    
    years = np.arange(start_year, start_year + supplied.shape[1])
    
    plt.figure(figsize=(10, 6))
    magma = plt.cm.magma
    
    # Supplied CO2
    median_sup = np.percentile(supplied, 50, axis=0)
    p5_sup = np.percentile(supplied, 5, axis=0)
    p95_sup = np.percentile(supplied, 95, axis=0)
    plt.plot(years, median_sup, label='Supplied CO2', linewidth=2, color='black', linestyle='-')
    plt.fill_between(years, p5_sup, p95_sup, alpha=0.25, color='black')
    
    # Total emissions
    median_em = np.percentile(emissions, 50, axis=0)
    p5_em = np.percentile(emissions, 5, axis=0)
    p95_em = np.percentile(emissions, 95, axis=0)
    plt.plot(years, median_em, label='Gross Emitted CO2', linewidth=2, color=magma(0.30), linestyle='--')
    plt.fill_between(years, p5_em, p95_em, alpha=0.25, color=magma(0.30))
    
    # CTBO mandate
    median_man = np.percentile(mandate, 50, axis=0)
    p5_man = np.percentile(mandate, 5, axis=0)
    p95_man = np.percentile(mandate, 95, axis=0)
    plt.plot(years, median_man, label='CTBO Mandate (Fossil CCS+BECCS+DACCS)', linewidth=2, color='gray', linestyle=':')
    plt.fill_between(years, p5_man, p95_man, alpha=0.25, color='gray')

    # CDR capacity
    median_cdr = np.percentile(CDR, 50, axis=0)
    p5_cdr = np.percentile(CDR, 5, axis=0)
    p95_cdr = np.percentile(CDR, 95, axis=0)
    plt.plot(years, median_cdr, label='BECCS+DACCS', linewidth=2, color=magma(0.60), linestyle='-.')
    plt.fill_between(years, p5_cdr, p95_cdr, alpha=0.25, color=magma(0.60))
    
    plt.xlabel('Year', fontsize=13)
    plt.ylabel('ktCO2/yr', fontsize=13)
    plt.title('Carbon Balance: Supplied CO2, Emissions, and CTBO Mandate', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('2_carbon_balances_multi.png', dpi=300)


def plot_capacity_stack(combined_df, array_outcomes, scenario_col="ETS_SCENARIO",
                        start_year=2025, debug=False):
    """Plot stacked area chart of CCS capacities by scenario."""
    if debug:
        print(f"Plotting capacity stack by {scenario_col}")
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.2), "Medium": magma(0.5), "High": magma(0.8)}
    
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
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.2), "Medium": magma(0.5), "High": magma(0.8)}
    
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
    
    # Create table of NPV statistics (all scenarios combined)
    table_data = []
    for i, plant_name in enumerate(plant_names):
        plant_npv_all = plant_npv_net[:, i]  # All experiments for this plant
        valid_npv = plant_npv_all[~np.isnan(plant_npv_all)]
        
        if len(valid_npv) > 0:
            median_npv = np.median(valid_npv) / 1000  # Convert to MEUR
            p5_npv = np.percentile(valid_npv, 5) / 1000
            p95_npv = np.percentile(valid_npv, 95) / 1000
            frac_positive = (valid_npv > 0).sum() / len(valid_npv)
        else:
            median_npv = np.nan
            p5_npv = np.nan
            p95_npv = np.nan
            frac_positive = np.nan
        
        table_data.append({
            'plant': plant_name,
            'median_npv_MEUR': median_npv,
            'p5_npv_MEUR': p5_npv,
            'p95_npv_MEUR': p95_npv,
            'fraction_positive': frac_positive
        })
    
    npv_table = pd.DataFrame(table_data)
    npv_table.to_csv('plant_npv_statistics.csv', index=False)
    if debug:
        print("  Saved: plant_npv_statistics.csv")



def plot_npv_vs_investment_year(combined_df, array_outcomes, plant_names=None,
                                 plant_filter="fossil", scenario_col="ETS_SCENARIO", debug=False):
    """Scatter plot of NPV vs investment year for individual plants across experiments."""
    if debug:
        print(f"Plotting NPV vs investment year scatter (filter={plant_filter})")
    
    plant_npv_net = array_outcomes.get("plant_npv_net")
    plant_inv_year = array_outcomes.get("plant_investment_year")
    
    if plant_npv_net is None or plant_inv_year is None or plant_names is None:
        print("Warning: Missing plant NPV or investment year data")
        return
    
    # Create plant mask based on filter
    n_plants = plant_npv_net.shape[1]
    plant_mask = np.zeros(n_plants, dtype=bool)
    
    if plant_filter is not None and plant_names is not None:
        for i, name in enumerate(plant_names):
            is_biogenic = name.endswith("W2E") or name.endswith("BECCS")
            if plant_filter == "biogenic":
                plant_mask[i] = is_biogenic
            elif plant_filter == "fossil":
                plant_mask[i] = not is_biogenic
            else:
                plant_mask[i] = True
    else:
        plant_mask[:] = True
    
    # Apply mask to data
    plant_npv_net_filtered = plant_npv_net[:, plant_mask]
    plant_inv_year_filtered = plant_inv_year[:, plant_mask]
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.1), "Medium": magma(0.5), "High": magma(0.9)}
    
    plt.figure(figsize=(10, 6))
    
    # Plot each scenario with different colors
    for category in ["Low", "Medium", "High"]:
        mask = combined_df[scenario_col] == category
        if mask.sum() == 0:
            continue
        
        # Get data for this category
        cat_npv = plant_npv_net_filtered[mask.values]      # shape: (n_exp, n_plants_filtered)
        cat_year = plant_inv_year_filtered[mask.values]    # shape: (n_exp, n_plants_filtered)
        
        # Flatten and remove NaN pairs
        npv_flat = cat_npv.flatten()
        year_flat = cat_year.flatten()
        valid_mask = ~np.isnan(npv_flat) & ~np.isnan(year_flat)
        npv_valid = npv_flat[valid_mask] / 1000  # Convert to MEUR
        year_valid = year_flat[valid_mask].astype(int)
        
        color = ets_colors[category]
        plt.scatter(year_valid, npv_valid, s=10, alpha=0.3, color=color, label=category)
    
    # Title based on filter
    filter_labels = {"biogenic": " (W2E/BECCS only)", "fossil": " (Fossil only)", None: ""}
    title_suffix = filter_labels.get(plant_filter, "")
    
    plt.xlabel('Investment Year', fontsize=13)
    plt.ylabel('NPV Net Profit (MEUR)', fontsize=13)
    plt.title(f'NPV vs Investment Year{title_suffix}', fontsize=14)
    plt.legend(fontsize=11, title='ETS Scenario')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_positive_npv_fraction_boxplot(combined_df, array_outcomes, plant_names=None,
                                        plant_filter=None, ETS_filter=None, debug=False):
    """
    Plot boxplot of fraction of positive NPV vs investment year (all data combined).
    
    Parameters:
        plant_filter: None (all plants), "biogenic" (W2E/BECCS only), or "fossil" (exclude W2E/BECCS)
    """
    if debug:
        print(f"Plotting positive NPV fraction boxplot (filter={plant_filter})")
    
    plant_npv_net = array_outcomes.get("plant_npv_net")
    plant_inv_year = array_outcomes.get("plant_investment_year")
    
    if plant_npv_net is None or plant_inv_year is None:
        print("Warning: Missing plant NPV or investment year data")
        return
    
    # Create ETS mask based on filter
    if ETS_filter is not None:
        ets_mask = combined_df["ETS_SCENARIO"] == ETS_filter
        plant_npv_net = plant_npv_net[ets_mask.values]
        plant_inv_year = plant_inv_year[ets_mask.values]
    
    # Create plant mask based on filter
    n_plants = plant_npv_net.shape[1]
    if plant_filter is not None and plant_names is not None:
        plant_mask = np.zeros(n_plants, dtype=bool)
        for i, name in enumerate(plant_names):
            is_biogenic = name.endswith("W2E") or name.endswith("BECCS")
            if plant_filter == "biogenic":
                plant_mask[i] = is_biogenic
            elif plant_filter == "fossil":
                plant_mask[i] = not is_biogenic
        # Apply mask to data
        plant_npv_net = plant_npv_net[:, plant_mask]
        plant_inv_year = plant_inv_year[:, plant_mask]
    
    n_exp = plant_npv_net.shape[0]
    
    # Get all unique years across all experiments
    all_years = plant_inv_year[~np.isnan(plant_inv_year)].astype(int)
    unique_years = np.sort(np.unique(all_years))
    
    # For each experiment, calculate fraction positive per year
    data_for_boxplot = {yr: [] for yr in unique_years}
    
    for exp_idx in range(n_exp):
        exp_npv = plant_npv_net[exp_idx]
        exp_year = plant_inv_year[exp_idx]
        
        for yr in unique_years:
            yr_mask = exp_year == yr
            if yr_mask.sum() > 0:
                yr_npv = exp_npv[yr_mask]
                valid = ~np.isnan(yr_npv)
                if valid.sum() > 0:
                    frac_positive = (yr_npv[valid] > 0).sum() / valid.sum()
                    data_for_boxplot[yr].append(frac_positive)
    
    # Prepare data for boxplot
    boxplot_data = [data_for_boxplot[yr] for yr in unique_years]
    
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(boxplot_data, positions=unique_years, widths=0.6, patch_artist=True)
    
    # Style the boxplot with different colors based on filter
    colors = {"biogenic": "#27ae60", "fossil": "#7f8c8d", None: "#3498db"}
    box_color = colors.get(plant_filter, "#3498db")
    for box in bp['boxes']:
        box.set(facecolor=box_color, alpha=0.7)
    for median in bp['medians']:
        median.set(color='#e74c3c', linewidth=2)
    
    # Title based on filter
    filter_labels = {"biogenic": " (W2E/BECCS only)", "fossil": " (Fossil only)", None: ""}
    title_suffix = filter_labels.get(plant_filter, "")
    if ETS_filter is not None:
        title_suffix += f" [ETS: {ETS_filter}]"
    
    plt.xlabel('Investment Year', fontsize=13)
    plt.ylabel('Fraction with Positive NPV', fontsize=13)
    plt.title(f'Distribution of Positive NPV Fraction by Investment Year{title_suffix}', fontsize=14)
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis='y')
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
    
    magma = plt.cm.magma
    ets_colors = {"Low": magma(0.2), "Medium": magma(0.5), "High": magma(0.8)}
    
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
            title="Gas Price Increase Over Time by ETS Scenario",
            savefig=True
        )
    
    # Plot 3: CTBO levelized cost over time
    if "CTBO_cost_lev_vec" in array_outcomes:
        plot_timeseries_by_scenario(
            combined_df,
            array_outcomes["CTBO_cost_lev_vec"],
            ylabel="CTBO Cost (EUR/tCO2)",
            title="Levelized CTBO Cost Over Time by ETS Scenario",
            savefig=True
        )
    
    # Plot 4: CTBO total cost over time
    if "CTBO_cost_vec" in array_outcomes:
        print("Plotting CTBO total cost over time")
        plot_ctbo_total_cost(
            combined_df,
            array_outcomes,
            ylabel="Total CTBO Cost (MEUR/yr)",
            title="Total CTBO Cost Over Time by ETS Scenario"
        )
    
    # Plot 5: CSU cost with ETS prices overlay
    if "CSU_cost_vec" in array_outcomes:
        print("Plotting CSU cost with ETS prices overlay")
        plot_csu_cost_with_ets(
            combined_df,
            array_outcomes,
            ylabel="Cost (EUR/tCO2)",
            title="CSU Cost and ETS Price Over Time by ETS Scenario"
        )
    
    # # Plot 4: CCS capacity stack
    # plot_capacity_stack(combined_df, array_outcomes)
    
    # Plot 6: Plant-level NPV
    if plant_names:
        plot_plant_npv(combined_df, array_outcomes, plant_names)
    
    # # Plot 6: NPV vs Investment Year
    plot_npv_vs_investment_year(combined_df, array_outcomes, plant_names, plant_filter="fossil")
    
    # Plot 7: Fraction of positive NPV vs Investment Year (by ETS scenario)
    # plot_positive_npv_fraction(combined_df, array_outcomes)
    
    # # Plot 8: Fraction of positive NPV vs Investment Year (boxplot, all data)
    # plot_positive_npv_fraction_boxplot(combined_df, array_outcomes, plant_names, plant_filter=None, ETS_filter=None)
    
    # # Plot 9: Fraction of positive NPV - biogenic plants only (W2E/BECCS)
    # plot_positive_npv_fraction_boxplot(combined_df, array_outcomes, plant_names, plant_filter="biogenic", ETS_filter=None)
    
    # Plot 10: Fraction of positive NPV - fossil plants only
    plot_positive_npv_fraction_boxplot(combined_df, array_outcomes, plant_names, plant_filter="fossil", ETS_filter=None)
    
    # Plot 11: Carbon balance (supplied, emissions, mandate)
    plot_carbon_balance(combined_df, array_outcomes)
    
    plt.show()

