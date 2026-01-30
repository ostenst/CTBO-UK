import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_carbon_trajectories_uncertainty(results_dir='results', pounds_to_EUR=1.15, debug=False):
    """
    Plot carbon trajectories with uncertainty intervals from EMA workbench results.
    """
    if debug:
        print(f"Loading data from {results_dir}")
    
    # Load array outcomes
    supply_ktCO2f = np.load(f'{results_dir}/outcomes_supply_ktCO2f.npy')
    stored_ktCO2g = np.load(f'{results_dir}/outcomes_stored_ktCO2g.npy')  # = stored_f + stored_cem + stored_pl
    stored_ktCO2b = np.load(f'{results_dir}/outcomes_stored_ktCO2b.npy')
    stored_ktCO2daccs = np.load(f'{results_dir}/outcomes_stored_ktCO2daccs.npy')
    
    if debug:
        print(f"Shape of arrays: {supply_ktCO2f.shape}")
    
    # Calculate combined arrays
    stored_bio_daccs = stored_ktCO2b + stored_ktCO2daccs  # Line 2: biogenic + DACCS
    stored_total = stored_ktCO2g + stored_ktCO2b + stored_ktCO2daccs  # Line 3: all storage
    
    # Years array (from constants)
    START_YEAR = 2025
    END_YEAR = 2055
    years = np.arange(START_YEAR, END_YEAR + 1)
    
    # Calculate statistics (median and percentiles)
    def get_stats(arr):
        median = np.median(arr, axis=0)
        p5 = np.percentile(arr, 5, axis=0)
        p95 = np.percentile(arr, 95, axis=0)
        return median, p5, p95
    
    supply_med, supply_p5, supply_p95 = get_stats(supply_ktCO2f)
    bio_daccs_med, bio_daccs_p5, bio_daccs_p95 = get_stats(stored_bio_daccs)
    total_med, total_p5, total_p95 = get_stats(stored_total)
    
    # Convert to MtCO2
    scale = 1000
    
    # Setup plot
    magma = plt.cm.magma
    dark_magma = magma(0.1)
    green = '#62a7a6'
    purple = '#9F5994'
    fig, ax = plt.subplots(figsize=(7, 6.5))
    
    # Line 1: Supply
    # color1 = magma(0.2)
    ax.plot(years, supply_med / scale, color=dark_magma, linewidth=2.5, label='Fossil fuel supply (coal, oil, gas)')
    ax.fill_between(years, supply_p5 / scale, supply_p95 / scale, color=dark_magma, alpha=0.30)
    
    # Line 2: Biogenic + DACCS storage
    color2 = green
    ax.plot(years, bio_daccs_med / scale, color=color2, linewidth=2.5, label='CDR storage (BECCS, DACCS)')
    ax.fill_between(years, bio_daccs_p5 / scale, bio_daccs_p95 / scale, color=color2, alpha=0.40)
    
    # Line 3: Total storage
    # color3 = magma(0.8)
    ax.plot(years, total_med / scale, color=purple, linewidth=2.5, label='Total storage (fossil CCS, BECCS, DACCS)')
    ax.fill_between(years, total_p5 / scale, total_p95 / scale, color=purple, alpha=0.40)
    
    # Formatting
    # ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Carbon [MtCO₂/year]', fontsize=14)
    # # ax.set_title('Carbon Trajectories with Uncertainty (5th-95th percentile)', fontsize=16)
    # ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(START_YEAR, END_YEAR)
    ax.set_ylim(-20, 300)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/1_carbon_balances.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"Plot saved to {results_dir}/carbon_balances.png")
    
    return fig

def plot_gas_increase_boxplots(results_dir='results', pounds_to_EUR=1.15, ets_scenario=None, debug=False):
    """
    Plot gas_increase_abs as box plots for years 2030, 2035, 2040, 2045, 2050.
    
    Args:
        results_dir: Directory containing results files
        pounds_to_EUR: Conversion rate from pounds to EUR
        ets_scenario: Optional ETS scenario to filter by (e.g., '£200', '£300', '£400'). Default: no filtering
        debug: If True, print debug information
    """
    if debug:
        print(f"Loading data from {results_dir}")
        print(f"ETS scenario filter: {ets_scenario}")
    
    # Load experiments and data
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    gas_increase_abs = np.load(f'{results_dir}/outcomes_gas_increase_abs.npy')  # [€/MWh]
    gas_increase_abs = gas_increase_abs / pounds_to_EUR * (100 / 1000)  # €→£, then £/MWh→pence/kWh
    
    # Filter by ETS scenario if specified
    if ets_scenario is not None:
        mask = experiments['ETS_SCENARIO'] == ets_scenario
        gas_increase_abs = gas_increase_abs[mask]
    
    if debug:
        print(f"Shape of gas_increase_abs: {gas_increase_abs.shape}")
    
    # Years array
    START_YEAR = 2025
    years = np.arange(START_YEAR, START_YEAR + gas_increase_abs.shape[1])
    target_years = [2030, 2035, 2040, 2045, 2050]
    
    # Extract data for target years
    box_data = []
    for year in target_years:
        year_idx = year - START_YEAR
        box_data.append(gas_increase_abs[:, year_idx])
    
    if debug:
        print(f"Target years: {target_years}")
        print(f"Box data shapes: {[d.shape for d in box_data]}")
    
    # Hard-coded box color
    magma = plt.cm.magma
    box_color = magma(0.7)
    median_color = magma(0.1)
    
    fig, ax = plt.subplots(figsize=(6, 5))

    plt.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Create box plots
    bp = ax.boxplot(box_data, positions=range(len(target_years)), patch_artist=True, widths=0.6)
    
    # Style the boxes
    for box in bp['boxes']:
        box.set_facecolor(box_color)
        box.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color(median_color)
        median.set_linewidth(2)
    
    # Formatting
    ax.set_xticks(range(len(target_years)))
    ax.set_xticklabels(target_years, fontsize=12)
    ax.set_ylabel('Gas price increase [pence/kWh]', fontsize=14)
    # ax.set_xlabel('Year', fontsize=14)
    ax.tick_params(labelsize=12)
    y1_min, y1_max = ax.get_ylim()
    ax.set_ylim(y1_min, 4.9)
    
    # Secondary y-axis: annual household bill (11200 kWh avg consumption)
    household_consumption = 11200  # kWh/year
    ax2 = ax.twinx()
    ax2.set_ylabel('Household bill increase [£/year]\n- assuming 11.200 kWh/year', fontsize=14)
    ax2.set_ylim(y1_min * household_consumption / 100, y1_max * household_consumption / 100)  # pence→£
    ax2.tick_params(labelsize=12)
    
    title = 'Gas price increase'
    if ets_scenario is not None:
        title += f' (ETS {ets_scenario})'
    # ax.set_title(title, fontsize=14)
    
    fig.subplots_adjust(left=0.15, right=0.75, top=0.92, bottom=0.12)
    plt.savefig(f'{results_dir}/3_gas_increase_boxplots.png', dpi=450)
    
    if debug:
        print(f"Plot saved to {results_dir}/3_gas_increase_boxplots.png")
    
    return fig

def plot_cost_ctbo_producers_boxplots(results_dir='results', pounds_to_EUR=1.15, ets_scenario=None, debug=False):
    """
    Plot cost_CTBO_producers as box plots for years 2030, 2035, 2040, 2045, 2050.
    
    Args:
        results_dir: Directory containing results files
        pounds_to_EUR: Conversion rate from pounds to EUR
        ets_scenario: Optional ETS scenario to filter by (e.g., '£200', '£300', '£400'). Default: no filtering
        debug: If True, print debug information
    """
    if debug:
        print(f"Loading data from {results_dir}")
        print(f"ETS scenario filter: {ets_scenario}")
    
    # Load experiments and data
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    cost_CTBO_producers = np.load(f'{results_dir}/outcomes_cost_CTBO_producers.npy')  # [k€/yr]
    cost_CTBO_producers = cost_CTBO_producers / 1e6 / pounds_to_EUR  # k€/yr → B£/yr
    
    # Filter by ETS scenario if specified
    if ets_scenario is not None:
        mask = experiments['ETS_SCENARIO'] == ets_scenario
        cost_CTBO_producers = cost_CTBO_producers[mask]
    
    if debug:
        print(f"Shape of cost_CTBO_producers: {cost_CTBO_producers.shape}")
    
    # Years array
    START_YEAR = 2025
    years = np.arange(START_YEAR, START_YEAR + cost_CTBO_producers.shape[1])
    target_years = [2030, 2035, 2040, 2045, 2050]
    
    # Extract data for target years
    box_data = []
    for year in target_years:
        year_idx = year - START_YEAR
        box_data.append(cost_CTBO_producers[:, year_idx])
    
    if debug:
        print(f"Target years: {target_years}")
        print(f"Box data shapes: {[d.shape for d in box_data]}")
    
    # Hard-coded box color
    magma = plt.cm.magma
    box_color = magma(0.1)
    median_color = magma(0.7)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Create box plots
    bp = ax.boxplot(box_data, positions=range(len(target_years)), patch_artist=True, widths=0.6)
    
    # Style the boxes
    for box in bp['boxes']:
        box.set_facecolor(box_color)
        box.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color(median_color)
        median.set_linewidth(2)
    
    # Formatting
    ax.set_xticks(range(len(target_years)))
    ax.set_xticklabels(target_years, fontsize=12)
    ax.set_ylabel('CTBO cost to producers [B£/yr]', fontsize=14)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], 30)
    # ax.set_xlabel('Year', fontsize=14)
    ax.tick_params(labelsize=12)
    
    title = 'CTBO cost to producers'
    if ets_scenario is not None:
        title += f' (ETS {ets_scenario})'
    # ax.set_title(title, fontsize=14)
    
    fig.subplots_adjust(left=0.15, right=0.75, top=0.92, bottom=0.12)
    plt.savefig(f'{results_dir}/3_cost_ctbo_producers_boxplots.png', dpi=450)
    
    if debug:
        print(f"Plot saved to {results_dir}/cost_ctbo_producers_boxplots.png")
    
    return fig

def plot_prices_by_ets(results_dir='results', pounds_to_EUR=1.15, debug=False):
    """
    Plot price_CSU with uncertainty intervals and price_ETS overlay, grouped by ETS_SCENARIO.
    """
    if debug:
        print(f"Loading data from {results_dir}")
    
    # Load experiments and outcomes
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    price_CSU = np.load(f'{results_dir}/outcomes_price_CSU.npy') / pounds_to_EUR  # [€/tCO2] → [£/tCO2]
    price_ETS = np.load(f'{results_dir}/outcomes_price_ETS.npy') / pounds_to_EUR  # [€/tCO2] → [£/tCO2]
    
    if debug:
        print(f"Shape of price_CSU: {price_CSU.shape}")
        print(f"Shape of price_ETS: {price_ETS.shape}")
    
    # Years array
    START_YEAR = 2025
    END_YEAR = 2055
    years = np.arange(START_YEAR, END_YEAR + 1)
    
    # Calculate statistics
    def get_stats(arr):
        median = np.median(arr, axis=0)
        p5 = np.percentile(arr, 5, axis=0)
        p95 = np.percentile(arr, 95, axis=0)
        return median, p5, p95
    
    scenarios = ['£200', '£300', '£400']
    magma = plt.cm.magma
    colors = [magma(0.10), magma(0.40), magma(0.70)]
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    for scenario, color in zip(scenarios, colors):
        mask = experiments['ETS_SCENARIO'] == scenario
        csu_data = price_CSU[mask]
        ets_data = price_ETS[mask]
        
        if len(csu_data) == 0:
            if debug:
                print(f"No data for ETS_SCENARIO={scenario}")
            continue
        
        # Price CSU with uncertainty
        med, p5, p95 = get_stats(csu_data)
        ax.plot(years, med, color=color, linewidth=2.5, label=f'CSU price')
        ax.fill_between(years, p5, p95, color=color, alpha=0.25)
        
        # Price ETS median only (dashed line)
        ets_med = np.median(ets_data, axis=0)
        ax.plot(years, ets_med, color=color, linewidth=2, linestyle='--', alpha=0.8, label=f'ETS {scenario}')
    
    # Formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    # ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Price [£/tCO₂]', fontsize=14)
    # # ax.set_title('CSU Price (solid) and ETS Price (dashed) by Scenario', fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(START_YEAR, 2050)
    
    fig.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.08)  # Fixed margins for consistent x-axis width
    plt.savefig(f'{results_dir}/2_prices_by_ets.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"Plot saved to {results_dir}/prices_by_ets.png")
    
    return fig

def plot_plant_npv_bubbles(results_dir='results', ETS_filter=None, pounds_to_EUR=1.15, debug=False):
    """
    Plot plant-level NPV bubbles: x=median investment year, y=median NPV total,
    size=median ktCO2tot_ccs, color=sector.
    ETS_filter: optional list of ETS scenarios to include, e.g. ['£200', '£300', '£400']
    """
    if debug:
        print(f"Loading plant data from {results_dir}")
    
    # Load plant reference (alphabetically ordered)
    plant_ref = pd.read_csv(f'{results_dir}/plant_reference.csv')
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    
    # Load plant outcome arrays (shape: n_experiments x n_plants)
    investment_year = np.load(f'{results_dir}/outcomes_plants_investment_year.npy')
    npv_total = np.load(f'{results_dir}/outcomes_plants_NPV_total.npy')
    ktCO2tot_ccs = np.load(f'{results_dir}/outcomes_plants_ktCO2tot_ccs.npy')
    
    # Filter by ETS scenario if specified
    if ETS_filter is not None:
        mask = experiments['ETS_SCENARIO'].isin(ETS_filter)
        investment_year = investment_year[mask]
        npv_total = npv_total[mask]
        ktCO2tot_ccs = ktCO2tot_ccs[mask]
        if debug:
            print(f"Filtered to {mask.sum()} experiments with ETS_SCENARIO in {ETS_filter}")
    
    if debug:
        print(f"Shape of plant arrays: {investment_year.shape}")
        print(f"Number of plants: {len(plant_ref)}")
    
    # Calculate median across scenarios (axis=0) for each plant
    med_investment_year = np.nanmedian(investment_year, axis=0)
    med_npv_total = np.nanmedian(npv_total, axis=0)
    med_ktCO2 = np.nanmedian(ktCO2tot_ccs, axis=0)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'stack': plant_ref['stack'],
        'sector': plant_ref['sector'],
        'investment_year': med_investment_year,
        'NPV_total': med_npv_total,
        'ktCO2tot_ccs': med_ktCO2
    })
    
    # Filter out plants with NaN investment year (never invested in any scenario)
    df_valid = df.dropna(subset=['investment_year', 'NPV_total', 'ktCO2tot_ccs'])
    
    if debug:
        print(f"Plants with valid data: {len(df_valid)} / {len(df)}")
    
    if df_valid.empty:
        print("No valid plant data to plot")
        return None
    
    # Hard-coded colors by sector (magma colormap values, waste is green)
    sectors = df_valid['sector'].unique()
    magma = plt.cm.magma
    sector_colors = {
        'cement': magma(0.1),
        'ccgt': magma(0.3),
        'refinery': magma(0.5),
        'steel': magma(0.7),
        'drax': magma(0.9),
        'waste': '#62a7a6'
    }

    # Scale bubble sizes
    size_scale = 0.3
    sizes = df_valid['ktCO2tot_ccs'] * size_scale
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    legend_handles = []
    legend_labels = {'cement': 'Cement', 'waste': 'Waste', 'ccgt': 'Gas power', 'drax': 'Drax', 'refinery': 'Refinery', 'steel': 'Steel'}
    for sector in sectors:
        mask = df_valid['sector'] == sector
        ax.scatter(
            df_valid.loc[mask, 'investment_year'],
            df_valid.loc[mask, 'NPV_total'] / 1000 / pounds_to_EUR,  # Convert k€ to M£
            s=sizes[mask],
            c=[sector_colors[sector]],
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        # Create custom legend handle with fixed size
        handle = plt.scatter([], [], s=80, c=[sector_colors[sector]], alpha=0.7, 
                            edgecolors='black', linewidths=0.5, label=legend_labels[sector])
        legend_handles.append(handle)
    
    # Formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Median investment year', fontsize=14)
    ax.set_ylabel('Median NPV [M£] from all costs and profits', fontsize=14)
    title = 'Plant NPV by investment year (median across scenarios)'
    if ETS_filter:
        title += f'\nETS: {", ".join(ETS_filter)}'
    # # ax.set_title(title, fontsize=16)
    ax.legend(handles=legend_handles, fontsize=12, title_fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(2025, 2050)
    ax.set_xticks(np.arange(2025, 2051, 5))
    y1_min, y1_max = ax.get_ylim()
    ax.set_ylim(-6000, 8000)
    
    fig.subplots_adjust(left=0.17, right=0.95, top=0.97, bottom=0.10)
    plt.savefig(f'{results_dir}/4_plant_npv_total_bubbles.png', dpi=450)
    
    if debug:
        print(f"Plot saved to {results_dir}/plant_npv_total_bubbles.png")
    
    return fig

def plot_plant_npv_csu_bubbles(results_dir='results', ETS_filter=None, exclude_sectors=None, pounds_to_EUR=1.15, debug=False):
    """
    Plot plant-level NPV CSU bubbles: x=median investment year, y=median NPV CSU,
    size=median ktCO2tot_ccs, color=sector.
    ETS_filter: optional list of ETS scenarios to include, e.g. ['£200', '£300', '£400']
    exclude_sectors: optional list of sectors to exclude, e.g. ['drax']
    """
    if debug:
        print(f"Loading plant data from {results_dir}")
    
    # Load plant reference (alphabetically ordered)
    plant_ref = pd.read_csv(f'{results_dir}/plant_reference.csv')
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    
    # Load plant outcome arrays (shape: n_experiments x n_plants)
    investment_year = np.load(f'{results_dir}/outcomes_plants_investment_year.npy')
    npv_csu = np.load(f'{results_dir}/outcomes_plants_NPV_CSU.npy')
    ktCO2tot_ccs = np.load(f'{results_dir}/outcomes_plants_ktCO2tot_ccs.npy')
    
    # Filter by ETS scenario if specified
    if ETS_filter is not None:
        mask = experiments['ETS_SCENARIO'].isin(ETS_filter)
        investment_year = investment_year[mask]
        npv_csu = npv_csu[mask]
        ktCO2tot_ccs = ktCO2tot_ccs[mask]
        if debug:
            print(f"Filtered to {mask.sum()} experiments with ETS_SCENARIO in {ETS_filter}")
    
    if debug:
        print(f"Shape of plant arrays: {investment_year.shape}")
        print(f"Number of plants: {len(plant_ref)}")
    
    # Calculate median across scenarios (axis=0) for each plant
    med_investment_year = np.nanmedian(investment_year, axis=0)
    med_npv_csu = np.nanmedian(npv_csu, axis=0)
    med_ktCO2 = np.nanmedian(ktCO2tot_ccs, axis=0)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'stack': plant_ref['stack'],
        'sector': plant_ref['sector'],
        'investment_year': med_investment_year,
        'NPV_CSU': med_npv_csu,
        'ktCO2tot_ccs': med_ktCO2
    })
    
    # Exclude specified sectors
    if exclude_sectors is not None:
        df = df[~df['sector'].isin(exclude_sectors)]
        if debug:
            print(f"Excluded sectors: {exclude_sectors}, remaining plants: {len(df)}")
            print(f"Median NPV of excluded plants in sectors: {df['NPV_CSU'].median()}")
    
    # Filter out plants with NaN investment year (never invested in any scenario)
    df_valid = df.dropna(subset=['investment_year', 'NPV_CSU', 'ktCO2tot_ccs'])
    
    if debug:
        print(f"Plants with valid data: {len(df_valid)} / {len(df)}")
    
    if df_valid.empty:
        print("No valid plant data to plot")
        return None
    
    # Hard-coded colors by sector (magma colormap values, waste is green)
    sectors = df_valid['sector'].unique()
    magma = plt.cm.magma
    sector_colors = {
        'cement': magma(0.1),
        'ccgt': magma(0.3),
        'refinery': magma(0.5),
        'steel': magma(0.7),
        'drax': magma(0.9),
        'waste': '#62a7a6'
    }

    # Scale bubble sizes
    size_scale = 0.3
    sizes = df_valid['ktCO2tot_ccs'] * size_scale
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    legend_handles = []
    legend_labels = {'cement': 'Cement', 'waste': 'Waste', 'ccgt': 'Gas power', 'drax': 'Drax', 'refinery': 'Refinery', 'steel': 'Steel'}
    for sector in sectors:
        mask = df_valid['sector'] == sector
        ax.scatter(
            df_valid.loc[mask, 'investment_year'],
            df_valid.loc[mask, 'NPV_CSU'] / 1000 / pounds_to_EUR,  # Convert k€ to M£
            s=sizes[mask],
            c=[sector_colors[sector]],
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        # Create custom legend handle with fixed size
        handle = plt.scatter([], [], s=80, c=[sector_colors[sector]], alpha=0.7, 
                            edgecolors='black', linewidths=0.5, label=legend_labels[sector])
        legend_handles.append(handle)
    
    # Formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Median Investment Year', fontsize=14)
    ax.set_ylabel('Median NPV [M£] from CSU costs and profits', fontsize=14)
    title = 'Plant NPV by investment year (median across scenarios)'
    if ETS_filter:
        title += f'\nETS: {", ".join(ETS_filter)}'
    # # ax.set_title(title, fontsize=16)
    # ax.legend(handles=legend_handles, fontsize=12, title_fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(2025, 2050)
    ax.set_xticks(np.arange(2025, 2051, 5))
    y1_min, y1_max = ax.get_ylim()
    ax.set_ylim(-50, 850)
    
    fig.subplots_adjust(left=0.17, right=0.95, top=0.97, bottom=0.10)
    plt.savefig(f'{results_dir}/4_plant_npv_csu_bubbles.png', dpi=450)
    
    if debug:
        print(f"Plot saved to {results_dir}/plant_npv_csu_bubbles.png")
    
    return fig

def plot_plant_npv_ets_bubbles(results_dir='results', ETS_filter=None, exclude_sectors=None, pounds_to_EUR=1.15, debug=False):
    """
    Plot plant-level NPV ETS bubbles: x=median investment year, y=median NPV ETS,
    size=median ktCO2tot_ccs, color=sector.
    ETS_filter: optional list of ETS scenarios to include, e.g. ['£200', '£300', '£400']
    exclude_sectors: optional list of sectors to exclude, e.g. ['drax']
    """
    if debug:
        print(f"Loading plant data from {results_dir}")
    
    # Load plant reference (alphabetically ordered)
    plant_ref = pd.read_csv(f'{results_dir}/plant_reference.csv')
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    
    # Load plant outcome arrays (shape: n_experiments x n_plants)
    investment_year = np.load(f'{results_dir}/outcomes_plants_investment_year.npy')
    npv_ets = np.load(f'{results_dir}/outcomes_plants_NPV_ETS.npy')
    ktCO2tot_ccs = np.load(f'{results_dir}/outcomes_plants_ktCO2tot_ccs.npy')
    
    # Filter by ETS scenario if specified
    if ETS_filter is not None:
        mask = experiments['ETS_SCENARIO'].isin(ETS_filter)
        investment_year = investment_year[mask]
        npv_ets = npv_ets[mask]
        ktCO2tot_ccs = ktCO2tot_ccs[mask]
        if debug:
            print(f"Filtered to {mask.sum()} experiments with ETS_SCENARIO in {ETS_filter}")
    
    if debug:
        print(f"Shape of plant arrays: {investment_year.shape}")
        print(f"Number of plants: {len(plant_ref)}")
    
    # Calculate median across scenarios (axis=0) for each plant
    med_investment_year = np.nanmedian(investment_year, axis=0)
    med_npv_ets = np.nanmedian(npv_ets, axis=0)
    med_ktCO2 = np.nanmedian(ktCO2tot_ccs, axis=0)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'stack': plant_ref['stack'],
        'sector': plant_ref['sector'],
        'investment_year': med_investment_year,
        'NPV_ETS': med_npv_ets,
        'ktCO2tot_ccs': med_ktCO2
    })
    
    # Exclude specified sectors
    if exclude_sectors is not None:
        df = df[~df['sector'].isin(exclude_sectors)]
        if debug:
            print(f"Excluded sectors: {exclude_sectors}, remaining plants: {len(df)}")
    
    # Filter out plants with NaN investment year (never invested in any scenario)
    df_valid = df.dropna(subset=['investment_year', 'NPV_ETS', 'ktCO2tot_ccs'])
    
    if debug:
        print(f"Plants with valid data: {len(df_valid)} / {len(df)}")
    
    if df_valid.empty:
        print("No valid plant data to plot")
        return None
    
    # Hard-coded colors by sector (magma colormap values, waste is green)
    sectors = df_valid['sector'].unique()
    magma = plt.cm.magma
    sector_colors = {
        'cement': magma(0.1),
        'ccgt': magma(0.3),
        'refinery': magma(0.5),
        'steel': magma(0.7),
        'drax': magma(0.9),
        'waste': '#62a7a6'
    }

    # Scale bubble sizes
    size_scale = 0.3
    sizes = df_valid['ktCO2tot_ccs'] * size_scale
    
    fig, ax = plt.subplots(figsize=(6, 6))

    legend_labels = {'cement': 'Cement', 'waste': 'Waste', 'ccgt': 'Gas power', 'drax': 'Drax', 'refinery': 'Refinery', 'steel': 'Steel'}
    legend_handles = []
    for sector in sectors:
        mask = df_valid['sector'] == sector
        ax.scatter(
            df_valid.loc[mask, 'investment_year'],
            df_valid.loc[mask, 'NPV_ETS'] / 1000 / pounds_to_EUR,  # Convert k€ to M£
            s=sizes[mask],
            c=[sector_colors[sector]],
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        # Create custom legend handle with fixed size
        handle = plt.scatter([], [], s=80, c=[sector_colors[sector]], alpha=0.7, 
                            edgecolors='black', linewidths=0.5, label=legend_labels[sector])
        legend_handles.append(handle)
    
    # Formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Median Investment Year', fontsize=14)
    ax.set_ylabel('Median NPV [M£] from ETS costs and profits', fontsize=14)
    title = 'Plant NPV (ETS only) by investment year (median across scenarios)'
    if ETS_filter:
        title += f'\nETS: {", ".join(ETS_filter)}'
    # # ax.set_title(title, fontsize=16)
    # ax.legend(handles=legend_handles, fontsize=12, title_fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(2025, 2050)
    ax.set_xticks(np.arange(2025, 2051, 5))
    
    fig.subplots_adjust(left=0.17, right=0.95, top=0.97, bottom=0.10)
    plt.savefig(f'{results_dir}/4_plant_npv_ets_bubbles.png', dpi=450)
    
    if debug:
        print(f"Plot saved to {results_dir}/plant_npv_ets_bubbles.png")
    
    return fig

def plot_policy_npv_boxplots(results_dir='results', pounds_to_EUR=1.15, debug=False):
    """
    Generate three figures (one per ETS scenario) with box plots of policy NPV costs and profits.
    All figures share the same y-axis limits (determined by the widest range).
    """
    if debug:
        print(f"Loading scalar outcomes from {results_dir}")
    
    # Load scalar outcomes (combined with experiments in experiments.csv)
    outcomes = pd.read_csv(f'{results_dir}/experiments.csv')
    
    if debug:
        print(f"Loaded {len(outcomes)} experiments")
        print(f"Columns: {outcomes.columns.tolist()}")
    
    scenarios = ['£200', '£300', '£400']

    # Labels and styling (reversed so first item appears at top in horizontal boxplot)
    npv_labels = ['Policy cost (CfD eq., -20% cost)', 'Policy cost (CTBO)', 'Fuel supplier costs (CTBO)', 'Emitter and T&S profits (CTBO)'][::-1]

    green = '#62a7a6'
    box_colors = [green, green, green, green][::-1]
    box_alphas = [1, 1, 0.4, 0.4][::-1]
    
    # First pass: calculate data for all scenarios and find global min/max
    scenario_data = {}
    global_min, global_max = float('inf'), float('-inf')
    
    for scenario in scenarios:
        mask = outcomes['ETS_SCENARIO'] == scenario
        data = outcomes[mask]
        
        if len(data) == 0:
            continue
        
        npv_data = [
            data['NPV_cost_CTBO'] * 0.80 / 1e6 / pounds_to_EUR,
            data['NPV_cost_CTBO'] / 1e6 / pounds_to_EUR,
            (data['NPV_cost_CTBO']+data['NPV_profit_CTBO']) / 1e6 / pounds_to_EUR,
            data['NPV_profit_CTBO'] / 1e6 / pounds_to_EUR,
        ]
        scenario_data[scenario] = npv_data
        
        # Update global min/max
        for arr in npv_data:
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
    
    # Add some padding to the limits
    padding = (global_max - global_min) * 0.03
    ylim = (global_min - padding, global_max + padding)
    
    if debug:
        print(f"Global y-axis limits: {ylim}")
    
    # Second pass: create figures with consistent y-axis
    figs = []
    for scenario in scenarios:
        if scenario not in scenario_data:
            if debug:
                print(f"No data for ETS_SCENARIO={scenario}")
            continue
        
        npv_data = scenario_data[scenario][::-1]  # Reverse to match flipped labels/colors

        fig, ax = plt.subplots(figsize=(6.4, 3))


        bp = ax.boxplot(
            npv_data,
            tick_labels=npv_labels,
            patch_artist=True,
            vert=False
        )

        for patch, color, alpha in zip(bp['boxes'], box_colors, box_alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(21.7, color='black', linestyle='--', linewidth=1.5) # Public funding CCS clusters

        ax.set_xlabel('NPV [B£]', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.4, axis='x')
        ax.set_xlim(ylim)
        # ax.legend(loc='upper right', fontsize=12)

        plt.tight_layout()
        filename = f'{results_dir}/5_policy_npv_boxplots_{scenario.replace("£", "")}.png'
        plt.savefig(filename, dpi=450, bbox_inches='tight')
        
        if debug:
            print(f"Plot saved to {filename}")
        
        figs.append(fig)
    
    return figs

if __name__ == "__main__":
    
    plot_carbon_trajectories_uncertainty(debug=True)
    plot_gas_increase_boxplots(debug=True)
    plot_cost_ctbo_producers_boxplots(debug=True)
    plot_prices_by_ets(debug=True)

    plot_plant_npv_bubbles(debug=True)
    plot_plant_npv_csu_bubbles(debug=True, exclude_sectors=['drax']) # 'drax'
    plot_plant_npv_ets_bubbles(debug=True, exclude_sectors=[])
    
    plot_policy_npv_boxplots(debug=True)

    plt.show()
