import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_carbon_trajectories_uncertainty(results_dir='results', debug=False):
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
    green = '#62a7a6'
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Line 1: Supply
    color1 = magma(0.2)
    ax.plot(years, supply_med / scale, color=color1, linewidth=2.5, label='Fossil fuel supply (coal, oil, gas)')
    ax.fill_between(years, supply_p5 / scale, supply_p95 / scale, color=color1, alpha=0.25)
    
    # Line 2: Biogenic + DACCS storage
    color2 = green
    ax.plot(years, bio_daccs_med / scale, color=color2, linewidth=2.5, label='CDR storage (BECCS, DACCS)')
    ax.fill_between(years, bio_daccs_p5 / scale, bio_daccs_p95 / scale, color=color2, alpha=0.25)
    
    # Line 3: Total storage
    color3 = magma(0.8)
    ax.plot(years, total_med / scale, color=color3, linewidth=2.5, label='Total storage (fossil CCS, BECCS, DACCS)')
    ax.fill_between(years, total_p5 / scale, total_p95 / scale, color=color3, alpha=0.25)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Carbon [MtCO₂/year]', fontsize=14)
    ax.set_title('Carbon Trajectories with Uncertainty (5th-95th percentile)', fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(START_YEAR, END_YEAR)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/carbon_balances.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"Plot saved to {results_dir}/carbon_balances.png")
    
    return fig

def plot_gas_increase_by_ets(results_dir='results', debug=False):
    """
    Plot gas_increase_abs with uncertainty intervals, grouped by ETS_SCENARIO.
    """
    
    if debug:
        print(f"Loading data from {results_dir}")
    
    # Load experiments to get ETS_SCENARIO values
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    gas_increase_abs = np.load(f'{results_dir}/outcomes_gas_increase_abs.npy') # [€/MWh]
    pounds_to_EUR = 1.15
    gas_increase_abs = gas_increase_abs / pounds_to_EUR * (100 / 1000)  # €→£, then £/MWh→pence/kWh
    
    if debug:
        print(f"Shape of gas_increase_abs: {gas_increase_abs.shape}")
        print(f"ETS_SCENARIO values: {experiments['ETS_SCENARIO'].unique()}")
    
    # Years array (from constants)
    START_YEAR = 2025
    END_YEAR = 2055
    years = np.arange(START_YEAR, END_YEAR + 1)
    
    # Calculate statistics for each scenario
    def get_stats(arr):
        median = np.median(arr, axis=0)
        p5 = np.percentile(arr, 5, axis=0)
        p95 = np.percentile(arr, 95, axis=0)
        return median, p5, p95
    
    scenarios = ['£200', '£300', '£400']
    magma = plt.cm.magma
    colors = [magma(0.2), magma(0.5), magma(0.8)]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for scenario, color in zip(scenarios, colors):
        mask = experiments['ETS_SCENARIO'] == scenario
        data = gas_increase_abs[mask]
        
        if len(data) == 0:
            if debug:
                print(f"No data for ETS_SCENARIO={scenario}")
            continue
        
        med, p5, p95 = get_stats(data)
        ax.plot(years, med, color=color, linewidth=2.5, label=f'ETS {scenario}')
        ax.fill_between(years, p5, p95, color=color, alpha=0.25)
    
    # Formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Gas price increase [pence/kWh]', fontsize=14)
    ax.set_title('Gas Price Increase by ETS Scenario (5th-95th percentile)', fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(START_YEAR, 2050)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/gas_increase_by_ets.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"Plot saved to {results_dir}/gas_increase_by_ets.png")
    
    return fig

def plot_prices_by_ets(results_dir='results', debug=False):
    """
    Plot price_CSU with uncertainty intervals and price_ETS overlay, grouped by ETS_SCENARIO.
    """
    if debug:
        print(f"Loading data from {results_dir}")
    
    # Load experiments and outcomes
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    price_CSU = np.load(f'{results_dir}/outcomes_price_CSU.npy')  # [€/tCO2]
    price_ETS = np.load(f'{results_dir}/outcomes_price_ETS.npy')  # [€/tCO2]
    
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
    colors = [magma(0.2), magma(0.5), magma(0.8)]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
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
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Price [€/tCO₂]', fontsize=14)
    ax.set_title('CSU Price (solid) and ETS Price (dashed) by Scenario', fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(START_YEAR, 2050)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/prices_by_ets.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"Plot saved to {results_dir}/prices_by_ets.png")
    
    return fig

def plot_plant_npv_bubbles(results_dir='results', ETS_filter=None, debug=False):
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
    
    # Setup colors by sector: magma for all except waste (green)
    sectors = df_valid['sector'].unique()
    non_waste_sectors = [s for s in sectors if s != 'waste']
    magma_colors = plt.cm.magma(np.linspace(0.05, 0.95, len(non_waste_sectors)))
    sector_colors = {s: c for s, c in zip(non_waste_sectors, magma_colors)}
    sector_colors['waste'] = '#62a7a6'  # Green for waste

    # Scale bubble sizes
    size_scale = 0.3
    sizes = df_valid['ktCO2tot_ccs'] * size_scale
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    legend_handles = []
    for sector in sectors:
        mask = df_valid['sector'] == sector
        ax.scatter(
            df_valid.loc[mask, 'investment_year'],
            df_valid.loc[mask, 'NPV_total'] / 1000,  # Convert to M€
            s=sizes[mask],
            c=[sector_colors[sector]],
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        # Create custom legend handle with fixed size
        handle = plt.scatter([], [], s=80, c=[sector_colors[sector]], alpha=0.7, 
                            edgecolors='black', linewidths=0.5, label=sector)
        legend_handles.append(handle)
    
    # Formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Median investment year', fontsize=14)
    ax.set_ylabel('Median NPV [M€] from all costs and profits', fontsize=14)
    title = 'Plant NPV by investment year (median across scenarios)'
    if ETS_filter:
        title += f'\nETS: {", ".join(ETS_filter)}'
    ax.set_title(title, fontsize=16)
    ax.legend(handles=legend_handles, title='Sector', fontsize=11, title_fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(2025, 2050)
    ax.set_xticks(np.arange(2025, 2051, 5))
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plant_npv_total_bubbles.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"Plot saved to {results_dir}/plant_npv_total_bubbles.png")
    
    return fig

def plot_plant_npv_csu_bubbles(results_dir='results', ETS_filter=None, exclude_sectors=None, debug=False):
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
    
    # Filter out plants with NaN investment year (never invested in any scenario)
    df_valid = df.dropna(subset=['investment_year', 'NPV_CSU', 'ktCO2tot_ccs'])
    
    if debug:
        print(f"Plants with valid data: {len(df_valid)} / {len(df)}")
    
    if df_valid.empty:
        print("No valid plant data to plot")
        return None
    
    # Setup colors by sector: magma for all except waste (green)
    sectors = df_valid['sector'].unique()
    non_waste_sectors = [s for s in sectors if s != 'waste']
    magma_colors = plt.cm.magma(np.linspace(0.05, 0.95, len(non_waste_sectors)))
    sector_colors = {s: c for s, c in zip(non_waste_sectors, magma_colors)}
    sector_colors['waste'] = '#62a7a6'  # Green for waste

    # Scale bubble sizes
    size_scale = 0.3
    sizes = df_valid['ktCO2tot_ccs'] * size_scale
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    legend_handles = []
    for sector in sectors:
        mask = df_valid['sector'] == sector
        ax.scatter(
            df_valid.loc[mask, 'investment_year'],
            df_valid.loc[mask, 'NPV_CSU'] / 1000,  # Convert to M€
            s=sizes[mask],
            c=[sector_colors[sector]],
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        # Create custom legend handle with fixed size
        handle = plt.scatter([], [], s=80, c=[sector_colors[sector]], alpha=0.7, 
                            edgecolors='black', linewidths=0.5, label=sector)
        legend_handles.append(handle)
    
    # Formatting
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Median Investment Year', fontsize=14)
    ax.set_ylabel('Median NPV [M€] from CSU costs and profits', fontsize=14)
    title = 'Plant NPV by investment year (median across scenarios)'
    if ETS_filter:
        title += f'\nETS: {", ".join(ETS_filter)}'
    ax.set_title(title, fontsize=16)
    ax.legend(handles=legend_handles, title='Sector', fontsize=11, title_fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.set_xlim(2025, 2050)
    ax.set_xticks(np.arange(2025, 2051, 5))
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plant_npv_csu_bubbles.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"Plot saved to {results_dir}/plant_npv_csu_bubbles.png")
    
    return fig

if __name__ == "__main__":
    
    plot_carbon_trajectories_uncertainty(debug=True)
    plot_gas_increase_by_ets(debug=True)
    plot_prices_by_ets(debug=True)

    plot_plant_npv_bubbles(debug=True)
    plot_plant_npv_csu_bubbles(debug=True, exclude_sectors=['drax'])

    plt.show()
