"""
Carbon Takeback Obligation (CTBO) Policy Simulation

This script simulates the implementation of a CTBO policy in the UK energy system,
analyzing the deployment of CCS, BECCS, and DACCS technologies over time and their
economic impacts on plants and consumers.

Author: Oscar Stenstrom
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Scenario selection
FOAK = True                    # True: First-of-a-kind costs, False: 4th-of-a-kind costs
CTBO_ENABLED = True            # Enable CTBO mandate
ETS = "Low"                   # "High", "Medium", "Low"
DACCS_EXPENSIVE = True         # True: Expensive DACCS (323->281 £/tCO2), False: Cheap (247->152 £/tCO2)
VERBOSE = True                # Print detailed investment decisions
SAVE_PLANT_DATA = True        # Save plant-level results to CSV

# Time parameters
START_YEAR = 2025
END_YEAR = 2055
years = np.arange(START_YEAR, END_YEAR + 1)

# Baseline emissions (2025)
baseline_emissions = {
    'coal': 17,    # [MtCO2/yr]
    'oil': 140,    # [MtCO2/yr]
    'gas': 127     # [MtCO2/yr]
}

# Diffuse emissions reduction trajectory
DIFFUSE_START_FRACTION = 1.0   # 100% of diffuse emissions in START_YEAR
DIFFUSE_END_FRACTION = 0.60    # 40% of diffuse emissions remain in 2050
DIFFUSE_TARGET_YEAR = 2050

# Financial parameters
DISCOUNT_RATE = 0.035          # Real discount rate for NPV calculations [3.5%]
USE_INVESTMENT_YEAR_AS_BASE = False  # NPV base: False=START_YEAR, True=investment year
pounds_to_EUR = 1.15

# Fuel assumptions for consumer cost analysis
fuels = {
    'diesel': {
        'emission_factor': 2.628,  # [kgCO2/litre]
        'price': 143.97  # [pence/litre]
    },
    'petrol': {
        'emission_factor': 2.339,  # [kgCO2/litre]
        'price': 135.07  # [pence/litre]
    },
    'gas': {
        'emission_factor': 0.2039 * 29.3,  # [kgCO2/thrm] (1 thrm = 29.3 kWh)
        'price': 80  # [pence/thrm]
    }
}

# DACCS cost trajectories (£/tCO2)
if DACCS_EXPENSIVE:
    DACCS_2025, DACCS_2050 = 323, 281
else:
    DACCS_2025, DACCS_2050 = 247, 152

# ETS scenario column mapping
ets_column_map = {
    "High": "High Sensitivity - Low Fossil Fuel Prices and High Economic Growth (2024 GBP)",
    "Medium": "Net Zero Strategy Aligned (2024 GBP)",
    "Low": "Low Sensitivity - High Fossil Fuel Prices and Low Economic Growth (2024 GBP)"
}


# ============================================================================
# LOAD DATA AND CALCULATE BASELINES
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load MACC data
macc_4oak = pd.read_csv('macc_4oak.csv')
macc_foak = pd.read_csv('macc_foak.csv')
macc_4oak['invested'] = False
macc_foak['invested'] = False
macc = macc_foak if FOAK else macc_4oak

print(f"Using {'FOAK' if FOAK else '4OAK'} cost scenario")
print(f"ETS scenario: {ETS}")
print(f"DACCS cost: {'Expensive' if DACCS_EXPENSIVE else 'Cheap'}")

# Calculate baseline emissions
cement_emissions = macc_4oak[macc_4oak['site-stack'].str.endswith('-cement')]['ktCO2f_yr_baseline'].sum()
total_emissions_2023 = (baseline_emissions['coal'] + baseline_emissions['oil'] + 
                        baseline_emissions['gas']) * 1000 + cement_emissions
point_sources = macc_4oak['ktCO2f_yr_baseline'].sum()
diffuse_baseline = total_emissions_2023 - point_sources

print(f"\nTotal fossil emissions (2025): {total_emissions_2023:.0f} ktCO2/yr")
print(f"  Point sources: {point_sources:.0f} ktCO2/yr")
print(f"  Diffuse: {diffuse_baseline:.0f} ktCO2/yr")


# ============================================================================
# CREATE TRAJECTORIES
# ============================================================================

print("\n" + "="*80)
print("CREATING TRAJECTORIES")
print("="*80)

# CTBO mandate trajectory: quadratic growth
t = (years - START_YEAR) * 2/5
ctbo_fraction = t**2  # [%]

# Diffuse emissions trajectory
diffuse_fraction = np.where(
    years <= DIFFUSE_TARGET_YEAR,
    DIFFUSE_START_FRACTION - (years - START_YEAR) * ((DIFFUSE_START_FRACTION - DIFFUSE_END_FRACTION) / 
                                                      (DIFFUSE_TARGET_YEAR - START_YEAR)),
    DIFFUSE_END_FRACTION
)
diffuse_emissions = diffuse_baseline * diffuse_fraction

# ETS price trajectory
ets_df = pd.read_csv('data/ETS.csv')
ets_df.columns = ets_df.columns.str.strip()
ets_df.set_index('Year', inplace=True)
ets_column = ets_column_map[ETS]

ets_prices = []
for year in years:
    if year in ets_df.index:
        ets_prices.append(ets_df.loc[year, ets_column] * pounds_to_EUR)
    else:
        ets_prices.append(ets_df.loc[2050, ets_column] * pounds_to_EUR)
ets_prices = np.array(ets_prices)

# DACCS cost trajectory
DACCS_costs = np.where(
    years <= 2050,
    DACCS_2025 + (years - START_YEAR) * ((DACCS_2050 - DACCS_2025) / (2050 - START_YEAR)),
    DACCS_2050
) * pounds_to_EUR

print(f"CTBO mandate in 2050: {ctbo_fraction[np.where(years == 2050)[0][0]]:.0f}%")
print(f"Diffuse emissions in 2050: {diffuse_emissions[np.where(years == 2050)[0][0]]:.0f} ktCO2/yr")
print(f"ETS price in 2050: {ets_prices[np.where(years == 2050)[0][0]]:.1f} EUR/tCO2")
print(f"DACCS cost in 2050: {DACCS_costs[np.where(years == 2050)[0][0]]:.1f} EUR/tCO2")


# ============================================================================
# SIMULATION
# ============================================================================

print("\n" + "="*80)
print("RUNNING SIMULATION")
print("="*80)

# Initialize result vectors
supplied_CO2_vec = []
total_emissions_vec = []
ctbo_mandate_vec = []
fCCS_capacity_vec = []
BECCS_capacity_vec = []
DACCS_capacity_vec = []
marginal_cost_vec = []
CSU_cost_vec = []
CTBO_cost_lev_vec = []

# Initialize plant-level results storage
plant_results = []
first_DACCS_year = None

# Main simulation loop
for i, year in enumerate(years):
    ets_price = ets_prices[i]
    DACCS_cost = DACCS_costs[i]
    diffuse = diffuse_emissions[i]
    
    # Voluntary investments (ETS-driven)
    for idx, plant in macc.iterrows():
        if not plant['invested'] and plant['EUR/tCO2'] < ets_price:
            macc.loc[idx, 'invested'] = True
            macc.loc[idx, 'year_invested'] = year
            if VERBOSE:
                print(f"Year {year}: Voluntary investment in {plant['site-stack']} "
                      f"(cost: {plant['EUR/tCO2']:.0f} < ETS: {ets_price:.0f} EUR/tCO2)")
    
    # Calculate current emissions and capacities
    baseline_emissions_current = macc['ktCO2f_yr_baseline'].where(~macc['invested'], 0).sum()
    residual_emissions = macc['ktCO2f_yr_residual'].where(macc['invested'], 0).sum()
    plant_emissions = baseline_emissions_current + residual_emissions
    total_emissions = plant_emissions + diffuse
    
    supplied_CO2 = macc['ktCO2f_yr_baseline'].sum() + diffuse
    ctbo_mandate = supplied_CO2 * ctbo_fraction[i] / 100
    
    point_fossil_capacity = macc['ktCO2f_yr_captured'].where(macc['invested'], 0).sum()
    point_bio_capacity = macc['ktCO2bio_yr_captured'].where(macc['invested'], 0).sum()
    point_capacity = point_fossil_capacity + point_bio_capacity
    
    DACCS_capacity = 0
    CSU_cost = 0
    marginal_cost = 0
    CTBO_cost = 0
    
    # CTBO-mandated investments
    if CTBO_ENABLED:
        missing_capacity = ctbo_mandate - point_capacity
        
        # Mandate cheapest plants until capacity is met or DACCS becomes cheaper
        j = 0
        while missing_capacity > 0 and j < len(macc):
            plant = macc.iloc[j]
            if not plant['invested']:
                if plant['EUR/tCO2'] > DACCS_cost:
                    if VERBOSE:
                        print(f"Year {year}: Switching to DACCS "
                              f"(cost: {DACCS_cost:.0f} < plant: {plant['EUR/tCO2']:.0f} EUR/tCO2)")
                    break
                
                macc.loc[j, 'invested'] = True
                macc.loc[j, 'year_invested'] = year
                point_fossil_capacity += plant['ktCO2f_yr_captured']
                point_bio_capacity += plant['ktCO2bio_yr_captured']
                point_capacity += plant['ktCO2f_yr_captured'] + plant['ktCO2bio_yr_captured']
                missing_capacity = ctbo_mandate - point_capacity
                
                if VERBOSE:
                    print(f"Year {year}: Mandate {plant['site-stack']} (cost: {plant['EUR/tCO2']:.0f} EUR/tCO2)")
            j += 1
        
        # Calculate costs based on marginal plant
        invested_plants = macc[macc['invested']]
        if len(invested_plants) > 0:
            marginal_plant = invested_plants.loc[invested_plants['EUR/tCO2'].idxmax()]
            marginal_cost = marginal_plant['EUR/tCO2']
            CSU_cost = max(0, marginal_cost - ets_price)
            CTBO_cost = CSU_cost * point_capacity
        
        # Add DACCS if still needed
        if missing_capacity > 0:
            DACCS_capacity = missing_capacity
            marginal_cost = DACCS_cost
            CSU_cost = max(0, marginal_cost - ets_price)
            CTBO_cost = CSU_cost * (point_capacity + DACCS_capacity)
            
            if first_DACCS_year is None:
                first_DACCS_year = year
        
        CTBO_cost_lev = CTBO_cost / supplied_CO2
    
    # Store aggregate results
    supplied_CO2_vec.append(supplied_CO2)
    total_emissions_vec.append(total_emissions)
    ctbo_mandate_vec.append(ctbo_mandate)
    fCCS_capacity_vec.append(point_fossil_capacity)
    BECCS_capacity_vec.append(point_bio_capacity)
    DACCS_capacity_vec.append(DACCS_capacity)
    marginal_cost_vec.append(marginal_cost)
    CSU_cost_vec.append(CSU_cost)
    CTBO_cost_lev_vec.append(CTBO_cost_lev)
    
    # Calculate plant-level costs and profits
    for idx, plant in macc.iterrows():
        # Determine diluted cost based on plant type
        if plant['site-stack'].endswith('W2E') or plant['site-stack'].endswith('BECCS'):
            csu_diluted_cost = 0  # Biogenic sources don't pay CTBO on emissions
        elif plant['site-stack'].endswith('cement'):
            csu_diluted_cost = CTBO_cost_lev * (1 - 0.63)  # 37% from fossil fuels
        else:
            csu_diluted_cost = CTBO_cost_lev # only applies to fossil CO2
        
        # Determine how much to subtract from gross CSU profits
        csu_subtract = max(0, plant['EUR/tCO2'] - ets_price) # positive difference between CCS cost and ETS price
        csu_subtract = csu_subtract + csu_diluted_cost # plus increases in fossil CO2 costs (if any!)
        ctbo_diluted_cost = csu_diluted_cost * plant['ktCO2f_yr_baseline']
        
        # Set values based on investment status
        if plant['invested']:
            investment_year = plant['year_invested']
            CO2_captured_fossil = plant['ktCO2f_yr_captured']
            CO2_captured_bio = plant['ktCO2bio_yr_captured']
            csu_gross_profit = CSU_cost
            csu_net_profit = csu_gross_profit - csu_subtract
            ctbo_fossil_profit = CSU_cost * CO2_captured_fossil
            ctbo_gross_profit = CSU_cost * (CO2_captured_fossil + CO2_captured_bio)
            ctbo_net_profit = csu_net_profit * (CO2_captured_fossil + CO2_captured_bio)
        else:
            investment_year = None
            CO2_captured_fossil = 0
            CO2_captured_bio = 0
            csu_gross_profit = 0
            csu_net_profit = - csu_diluted_cost
            ctbo_fossil_profit = 0
            ctbo_gross_profit = 0
            ctbo_net_profit = - csu_diluted_cost * plant['ktCO2f_yr_baseline']
        
        # Store plant-level results
        plant_results.append({
            'year': year,
            'ETS_price': ets_price,
            'CSU_cost': CSU_cost,
            'investment_year': investment_year,
            'plant': plant['site-stack'],
            'CO2_captured_fossil': CO2_captured_fossil,
            'CO2_captured_bio': CO2_captured_bio,
            'CCS_cost': plant['EUR/tCO2'],
            'marginal_plant': marginal_cost == plant['EUR/tCO2'],
            'csu_diluted_cost': csu_diluted_cost,
            'csu_gross_profit': csu_gross_profit,
            'csu_net_profit': csu_net_profit,
            'ctbo_diluted_cost': ctbo_diluted_cost,
            'ctbo_fossil_profit': ctbo_fossil_profit,
            'ctbo_gross_profit': ctbo_gross_profit,
            'ctbo_net_profit': ctbo_net_profit
        })


# ============================================================================
# RESULTS AND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

idx_2050 = np.where(years == 2050)[0][0]

print(f"\n2050 Aggregate Results:")
print(f"  Supplied CO2f: {supplied_CO2_vec[idx_2050]:.0f} ktCO2/yr")
print(f"  Emitted CO2f: {total_emissions_vec[idx_2050]:.0f} ktCO2/yr")
print(f"  CTBO mandate: {ctbo_mandate_vec[idx_2050]:.0f} ktCO2/yr")
print(f"  Fossil CCS: {fCCS_capacity_vec[idx_2050]:.0f} ktCO2/yr")
print(f"  BECCS: {BECCS_capacity_vec[idx_2050]:.0f} ktCO2/yr")
print(f"  DACCS: {DACCS_capacity_vec[idx_2050]:.0f} ktCO2/yr")
print(f"  CSU cost: {CSU_cost_vec[idx_2050]:.1f} EUR/tCO2")
print(f"  Marginal cost: {marginal_cost_vec[idx_2050]:.1f} EUR/tCO2")
print(f"  CTBO cost levelized: {CTBO_cost_lev_vec[idx_2050]:.1f} EUR/tCO2")

if first_DACCS_year is not None:
    print(f"\nDACCS first needed in: {first_DACCS_year:.0f}")
else:
    print(f"\nDACCS was never needed (all capacity met by point sources)")

# Consumer fuel price impacts
carbon_price = np.array(CTBO_cost_lev_vec) / pounds_to_EUR
emission_factor_diesel = fuels['diesel']['emission_factor']
emission_factor_petrol = fuels['petrol']['emission_factor']
emission_factor_gas = fuels['gas']['emission_factor']
diesel_price = fuels['diesel']['price']
petrol_price = fuels['petrol']['price']
gas_price = fuels['gas']['price']

diesel_increase_abs = carbon_price * (emission_factor_diesel / 1000) * 100
petrol_increase_abs = carbon_price * (emission_factor_petrol / 1000) * 100
gas_increase_abs = carbon_price * (emission_factor_gas / 1000) * 100

diesel_increase_pct = (diesel_increase_abs / diesel_price) * 100
petrol_increase_pct = (petrol_increase_abs / petrol_price) * 100
gas_increase_pct = (gas_increase_abs / gas_price) * 100

print(f"\n2050 Consumer Fuel Price Impacts:")
print(f"  Diesel:  +{diesel_increase_abs[idx_2050]:.2f} pence/litre (+{diesel_increase_pct[idx_2050]:.1f}%)")
print(f"  Petrol:  +{petrol_increase_abs[idx_2050]:.2f} pence/litre (+{petrol_increase_pct[idx_2050]:.1f}%)")
print(f"  Gas:     +{gas_increase_abs[idx_2050]:.2f} pence/thrm (+{gas_increase_pct[idx_2050]:.1f}%)")

# Plant-level analysis
plant_df = pd.DataFrame(plant_results)
print(f"\nPlant-level data collected: {len(plant_df)} plant-year observations")

if len(plant_df) > 0:
    print(f"  Years covered: {plant_df['year'].min():.0f} - {plant_df['year'].max():.0f}")
    print(f"  Number of plants: {plant_df['plant'].nunique()}")
    
    # Calculate NPV for each plant
    plant_npv = []
    for plant_name in plant_df['plant'].unique():
        plant_data = plant_df[plant_df['plant'] == plant_name].copy()
        investment_year = plant_data['investment_year'].max()
        
        if pd.isna(investment_year):
            continue
        
        base_year = investment_year if USE_INVESTMENT_YEAR_AS_BASE else START_YEAR
        plant_data['discount_factor'] = 1 / (1 + DISCOUNT_RATE) ** (plant_data['year'] - base_year)
        
        npv_gross_profit = (plant_data['ctbo_gross_profit'] * plant_data['discount_factor']).sum()
        npv_net_profit = (plant_data['ctbo_net_profit'] * plant_data['discount_factor']).sum()
        npv_diluted_cost = (plant_data['ctbo_diluted_cost'] * plant_data['discount_factor']).sum()
        npv_fossil_profit = (plant_data['ctbo_fossil_profit'] * plant_data['discount_factor']).sum()
        
        plant_npv.append({
            'plant': plant_name,
            'investment_year': investment_year,
            'base_year': base_year,
            'years_operating': len(plant_data),
            'NPV_gross_profit': npv_gross_profit,
            'NPV_net_profit': npv_net_profit,
            'NPV_diluted_cost': npv_diluted_cost,
            'NPV_fossil_profit': npv_fossil_profit
        })
    
    plant_npv_df = pd.DataFrame(plant_npv)
    total_plants = plant_df['plant'].nunique()
    print(f"  Plants that invested: {len(plant_npv_df)} out of {total_plants}")
    
    # CCGT plant summaries
    ccgt_plants = plant_df[plant_df['plant'].str.endswith('CCGT')]
    plant_2030 = ccgt_plants[ccgt_plants['year'] == 2030]
    plant_2040 = ccgt_plants[ccgt_plants['year'] == 2040]
    plant_2050 = ccgt_plants[ccgt_plants['year'] == 2050]
    
    print(f"\n  Total CCGT plant (fossil net) profits in 2030: "
          f"{plant_2030['ctbo_diluted_cost'].sum() - plant_2030['ctbo_fossil_profit'].sum() + plant_2030['ctbo_net_profit'].sum():.0f} kEUR/yr")
    print(f"  Total CCGT plant (fossil net) profits in 2040: "
          f"{plant_2040['ctbo_diluted_cost'].sum() - plant_2040['ctbo_fossil_profit'].sum() + plant_2040['ctbo_net_profit'].sum():.0f} kEUR/yr")
    print(f"  Total CCGT plant (fossil net) profits in 2050: "
          f"{plant_2050['ctbo_diluted_cost'].sum() - plant_2050['ctbo_fossil_profit'].sum() + plant_2050['ctbo_net_profit'].sum():.0f} kEUR/yr")
    
    # NPV summaries
    base_description = "investment year" if USE_INVESTMENT_YEAR_AS_BASE else f"{START_YEAR}"
    print(f"\n  NPV Analysis (discount rate: {DISCOUNT_RATE*100:.1f}%, base: {base_description}):")
    print(f"    Total NPV gross profit (all plants): {plant_npv_df['NPV_gross_profit'].sum():.0f} kEUR")
    print(f"    Total NPV net profit (all plants): {plant_npv_df['NPV_net_profit'].sum():.0f} kEUR")
    print(f"    Total NPV fossil profit (all plants): {plant_npv_df['NPV_fossil_profit'].sum():.0f} kEUR")
    
    # Top 10 and bottom 10 plants
    plant_co2_data = plant_df.groupby('plant')[['CO2_captured_fossil', 'CO2_captured_bio']].max().reset_index()
    plant_npv_with_co2 = plant_npv_df.merge(plant_co2_data, on='plant', how='left')
    
    top_10 = plant_npv_with_co2.nlargest(10, 'NPV_net_profit')
    bottom_10 = plant_npv_with_co2.nsmallest(10, 'NPV_net_profit')
    
    print(f"\n  Top 10 plants by NPV net profit:")
    for idx, row in top_10.iterrows():
        print(f"    {row['plant'][:45]:45s} | NPV: {row['NPV_net_profit']:8,.0f} kEUR | "
              f"Invested: {row['investment_year']:.0f} | CO2f: {row['CO2_captured_fossil']:5.0f} | "
              f"CO2bio: {row['CO2_captured_bio']:5.0f} ktCO2/yr")
    
    print(f"\n  Bottom 10 plants by NPV net profit:")
    for idx, row in bottom_10.iterrows():
        print(f"    {row['plant'][:45]:45s} | NPV: {row['NPV_net_profit']:8,.0f} kEUR | "
              f"Invested: {row['investment_year']:.0f} | CO2f: {row['CO2_captured_fossil']:5.0f} | "
              f"CO2bio: {row['CO2_captured_bio']:5.0f} ktCO2/yr")


# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

# Plot 1: MACC curves
viridis = plt.cm.viridis
plt.figure(figsize=(10, 6))
plt.plot(macc_4oak['ktCO2_yr_cumulative'], macc_4oak['EUR/tCO2'], 
         color=viridis(0.65), label='4th-of-a-kind costs', linewidth=2)
plt.plot(macc_foak['ktCO2_yr_cumulative'], macc_foak['EUR/tCO2'], 
         color='crimson', label='First-of-a-kind costs', linewidth=2)
plt.xlabel('Cumulative Captured CO2 (ktCO2/yr)', fontsize=13)
plt.ylabel('Marginal Abatement Cost (EUR/tCO2)', fontsize=13)
plt.title('MACC of point source CCS (fossil + biogenic)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1_maccs.png', dpi=300)
print("  Saved: 1_maccs.png")

# Plot 2: Emissions and capacities
plt.figure(figsize=(10, 6))

# Stacked area plot for CCS capacities (reversed order so fCCS is on top)
plt.stackplot(years, DACCS_capacity_vec, BECCS_capacity_vec, fCCS_capacity_vec,
              labels=['DACCS', 'BECCS', 'Fossil CCS'],
              colors=[viridis(0.5), viridis(0.65), 'gray'],
              alpha=1.0)

# Line plots on top
plt.plot(years, supplied_CO2_vec, label='Supplied CO2 (O&G, coal, cem)', linewidth=2, color='black')
plt.plot(years, total_emissions_vec, label='Emitted CO2', linewidth=2, color=viridis(0.40))
plt.plot(years, ctbo_mandate_vec, label='CTBO Mandate', linewidth=2, color='black', linestyle='--')

if first_DACCS_year is not None:
    daccs_idx = np.where(years == first_DACCS_year)[0][0]
    plt.plot(first_DACCS_year, DACCS_capacity_vec[daccs_idx], 'o', markersize=6, color=viridis(0.5))
    plt.plot([], [], 'o', markersize=6, color=viridis(0.5), label='Year when DACCS is marginal')

plt.xlabel('Year', fontsize=13)
plt.ylabel('ktCO2/yr', fontsize=13)
plt.title('UK Carbon Balances (CTBO Mandate=Fossil CCS+BECCS+DACCS)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_carbon_balances.png', dpi=300)
print("  Saved: 2_carbon_balances.png")

# Plot 3: Costs
plt.figure(figsize=(10, 6))
plt.plot(years, ets_prices, label='ETS Price', linewidth=2, color='yellow')
plt.plot(years, marginal_cost_vec, label='Marginal CCS Cost', linewidth=2, color='red')
plt.plot(years, CSU_cost_vec, label='CSU Cost', linewidth=2, color='orange')
plt.plot(years, CTBO_cost_lev_vec, label='CTBO Cost (levelized)', linewidth=2, color='black')

if first_DACCS_year is not None:
    daccs_idx = np.where(years == first_DACCS_year)[0][0]
    plt.plot(first_DACCS_year, ets_prices[daccs_idx], 'o', markersize=6, color='yellow')
    plt.plot(first_DACCS_year, marginal_cost_vec[daccs_idx], 'o', markersize=6, color='red')
    plt.plot(first_DACCS_year, CSU_cost_vec[daccs_idx], 'o', markersize=6, color='orange')
    plt.plot(first_DACCS_year, CTBO_cost_lev_vec[daccs_idx], 'o', markersize=6, color='black')
    plt.plot([], [], 'o', markersize=6, color='gray', label='Year when DACCS is marginal')

plt.xlabel('Year', fontsize=13)
plt.ylabel('EUR/tCO2', fontsize=13)
plt.title('CSU cost = Marginal CCS Cost - ETS Price', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_costs.png', dpi=300)
print("  Saved: 3_costs.png")

# Plot 4: Consumer fuel price increases
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
magma = plt.cm.magma

# Absolute increases
ax1.plot(years, diesel_increase_abs, label='Diesel', linewidth=2, color=magma(0.20))
ax1.plot(years, petrol_increase_abs, label='Petrol', linewidth=2, color=magma(0.70))
ax1.plot(years, gas_increase_abs, label='Gas', linewidth=2, color=magma(0.50))

if first_DACCS_year is not None:
    daccs_idx = np.where(years == first_DACCS_year)[0][0]
    ax1.plot(first_DACCS_year, diesel_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.20))
    ax1.plot(first_DACCS_year, petrol_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.70))
    ax1.plot(first_DACCS_year, gas_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.50))
    ax1.plot([], [], 'o', markersize=6, color='gray', label='Year when DACCS is marginal')

idx_2040 = np.where(years == 2040)[0][0]
ax1.annotate(f'CTBO:\n{ctbo_fraction[idx_2040]:.0f}%', 
             xy=(2040, gas_increase_pct[idx_2040]), 
             xytext=(2040, gas_increase_pct[idx_2040] + 5),
             fontsize=11, ha='center')
ax1.annotate(f'CTBO:\n{ctbo_fraction[idx_2050]:.0f}%', 
             xy=(2050, gas_increase_pct[idx_2050]), 
             xytext=(2050, gas_increase_pct[idx_2050] + 5),
             fontsize=11, ha='center')

ax1.set_xlabel('Year', fontsize=13)
ax1.set_ylabel('Price Increase (pence per litre/thrm)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Percentage increases
ax2.plot(years, diesel_increase_pct, label='Diesel', linewidth=2, color=magma(0.20))
ax2.plot(years, petrol_increase_pct, label='Petrol', linewidth=2, color=magma(0.70))
ax2.plot(years, gas_increase_pct, label='Gas', linewidth=2, color=magma(0.50))

if first_DACCS_year is not None:
    ax2.plot(first_DACCS_year, diesel_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.20))
    ax2.plot(first_DACCS_year, petrol_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.70))
    ax2.plot(first_DACCS_year, gas_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.50))
    ax2.plot([], [], 'o', markersize=6, color='gray', label='Year when DACCS is marginal')

ax2.set_xlabel('Year', fontsize=13)
ax2.set_ylabel('Price Increase (%)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('4_price_increases.png', dpi=300)
print("  Saved: 4_price_increases.png")

# Plot 4b: Consumer fuel price increases (2025-2040)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Absolute increases
ax1.plot(years, diesel_increase_abs, label='Diesel', linewidth=2, color=magma(0.20))
ax1.plot(years, petrol_increase_abs, label='Petrol', linewidth=2, color=magma(0.70))
ax1.plot(years, gas_increase_abs, label='Gas', linewidth=2, color=magma(0.50))

if first_DACCS_year is not None and 2025 <= first_DACCS_year <= 2040:
    daccs_idx = np.where(years == first_DACCS_year)[0][0]
    ax1.plot(first_DACCS_year, diesel_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.20))
    ax1.plot(first_DACCS_year, petrol_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.70))
    ax1.plot(first_DACCS_year, gas_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.50))
    ax1.plot([], [], 'o', markersize=6, color='gray', label='Year when DACCS is marginal')

ax1.annotate(f'CTBO:\n{ctbo_fraction[idx_2040]:.0f}%', 
             xy=(2040, gas_increase_pct[idx_2040]), 
             xytext=(2040, gas_increase_pct[idx_2040] + 5),
             fontsize=11, ha='center')

ax1.set_xlim(2025, 2040)

# Set y-limits based on visible data range
idx_start = np.where(years == 2025)[0][0]
idx_end = np.where(years == 2040)[0][0]
abs_data_range = np.concatenate([diesel_increase_abs[idx_start:idx_end+1], 
                                  petrol_increase_abs[idx_start:idx_end+1], 
                                  gas_increase_abs[idx_start:idx_end+1]])
y_min_abs = abs_data_range.min()
y_max_abs = abs_data_range.max()
y_margin_abs = (y_max_abs - y_min_abs) * 0.15
ax1.set_ylim(y_min_abs - y_margin_abs, y_max_abs + y_margin_abs)

ax1.set_xlabel('Year', fontsize=13)
ax1.set_ylabel('Price Increase (pence per litre/thrm)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

# Percentage increases
ax2.plot(years, diesel_increase_pct, label='Diesel', linewidth=2, color=magma(0.20))
ax2.plot(years, petrol_increase_pct, label='Petrol', linewidth=2, color=magma(0.70))
ax2.plot(years, gas_increase_pct, label='Gas', linewidth=2, color=magma(0.50))

if first_DACCS_year is not None and 2025 <= first_DACCS_year <= 2040:
    ax2.plot(first_DACCS_year, diesel_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.20))
    ax2.plot(first_DACCS_year, petrol_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.70))
    ax2.plot(first_DACCS_year, gas_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.50))
    ax2.plot([], [], 'o', markersize=6, color='gray', label='Year when DACCS is marginal')

ax2.set_xlim(2025, 2040)

# Set y-limits based on visible data range
pct_data_range = np.concatenate([diesel_increase_pct[idx_start:idx_end+1], 
                                  petrol_increase_pct[idx_start:idx_end+1], 
                                  gas_increase_pct[idx_start:idx_end+1]])
y_min_pct = pct_data_range.min()
y_max_pct = pct_data_range.max()
y_margin_pct = (y_max_pct - y_min_pct) * 0.15
ax2.set_ylim(y_min_pct - y_margin_pct, y_max_pct + y_margin_pct)
ax2.set_xlabel('Year', fontsize=13)
ax2.set_ylabel('Price Increase (%)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

plt.tight_layout()
plt.savefig('4b_price_increases_2025-2040.png', dpi=300)
print("  Saved: 4b_price_increases_2025-2040.png")

# Plot 4c: Consumer fuel price increases (2035-2055)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Absolute increases
ax1.plot(years, diesel_increase_abs, label='Diesel', linewidth=2, color=magma(0.20))
ax1.plot(years, petrol_increase_abs, label='Petrol', linewidth=2, color=magma(0.70))
ax1.plot(years, gas_increase_abs, label='Gas', linewidth=2, color=magma(0.50))

if first_DACCS_year is not None and 2035 <= first_DACCS_year <= 2055:
    daccs_idx = np.where(years == first_DACCS_year)[0][0]
    ax1.plot(first_DACCS_year, diesel_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.20))
    ax1.plot(first_DACCS_year, petrol_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.70))
    ax1.plot(first_DACCS_year, gas_increase_abs[daccs_idx], 'o', markersize=6, color=magma(0.50))
    ax1.plot([], [], 'o', markersize=6, color='gray', label='Year when DACCS is marginal')

ax1.annotate(f'CTBO:\n{ctbo_fraction[idx_2040]:.0f}%', 
             xy=(2040, gas_increase_pct[idx_2040]), 
             xytext=(2040, gas_increase_pct[idx_2040] + 5),
             fontsize=11, ha='center')
ax1.annotate(f'CTBO:\n{ctbo_fraction[idx_2050]:.0f}%', 
             xy=(2050, gas_increase_pct[idx_2050]), 
             xytext=(2050, gas_increase_pct[idx_2050] + 5),
             fontsize=11, ha='center')

ax1.set_xlim(2035, 2055)

# Set y-limits based on visible data range
idx_start = np.where(years == 2035)[0][0]
idx_end = np.where(years == 2055)[0][0]
abs_data_range = np.concatenate([diesel_increase_abs[idx_start:idx_end+1], 
                                  petrol_increase_abs[idx_start:idx_end+1], 
                                  gas_increase_abs[idx_start:idx_end+1]])
y_min_abs = abs_data_range.min()
y_max_abs = abs_data_range.max()
y_margin_abs = (y_max_abs - y_min_abs) * 0.15
ax1.set_ylim(y_min_abs - y_margin_abs, y_max_abs + y_margin_abs)

ax1.set_xlabel('Year', fontsize=13)
ax1.set_ylabel('Price Increase (pence per litre/thrm)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

# Percentage increases
ax2.plot(years, diesel_increase_pct, label='Diesel', linewidth=2, color=magma(0.20))
ax2.plot(years, petrol_increase_pct, label='Petrol', linewidth=2, color=magma(0.70))
ax2.plot(years, gas_increase_pct, label='Gas', linewidth=2, color=magma(0.50))

if first_DACCS_year is not None and 2035 <= first_DACCS_year <= 2055:
    ax2.plot(first_DACCS_year, diesel_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.20))
    ax2.plot(first_DACCS_year, petrol_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.70))
    ax2.plot(first_DACCS_year, gas_increase_pct[daccs_idx], 'o', markersize=6, color=magma(0.50))
    ax2.plot([], [], 'o', markersize=6, color='gray', label='Year when DACCS is marginal')

ax2.set_xlim(2035, 2055)

# Set y-limits based on visible data range
pct_data_range = np.concatenate([diesel_increase_pct[idx_start:idx_end+1], 
                                  petrol_increase_pct[idx_start:idx_end+1], 
                                  gas_increase_pct[idx_start:idx_end+1]])
y_min_pct = pct_data_range.min()
y_max_pct = pct_data_range.max()
y_margin_pct = (y_max_pct - y_min_pct) * 0.15
ax2.set_ylim(y_min_pct - y_margin_pct, y_max_pct + y_margin_pct)
ax2.set_xlabel('Year', fontsize=13)
ax2.set_ylabel('Price Increase (%)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

plt.tight_layout()
plt.savefig('4c_price_increases_2035-2055.png', dpi=300)
print("  Saved: 4c_price_increases_2035-2055.png")

# Plot 5: Plant-level profits
if len(plant_df) > 0:
    selected_plants = [
        {"Pembroke Power Station-CCGT": "black"},
        {"Runcorn-W2E": "green"},
        {"Hope Cement Works-cement": "red"},
        {"Medway-CCGT": "gray"}
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for plant_dict in selected_plants:
        plant_name, color = list(plant_dict.items())[0]
        plant_data = plant_df[(plant_df['plant'] == plant_name) & (plant_df['year'] <= 2051)]
        
        if len(plant_data) > 0:
            ax1.plot(plant_data['year'], plant_data['ctbo_gross_profit'], 
                    label=plant_name, linewidth=2, color=color)
            ax2.plot(plant_data['year'], plant_data['ctbo_net_profit'], 
                    label=plant_name, linewidth=2, color=color)
            
            investment_year = plant_data['investment_year'].max()
            if not pd.isna(investment_year) and investment_year in plant_data['year'].values:
                invest_data = plant_data[plant_data['year'] == investment_year].iloc[0]
                ax1.plot(investment_year, invest_data['ctbo_gross_profit'], 's', markersize=6, color=color)
                ax2.plot(investment_year, invest_data['ctbo_net_profit'], 's', markersize=6, color=color)
            
            if first_DACCS_year is not None and first_DACCS_year <= 2051 and first_DACCS_year in plant_data['year'].values:
                daccs_data = plant_data[plant_data['year'] == first_DACCS_year].iloc[0]
                ax1.plot(first_DACCS_year, daccs_data['ctbo_gross_profit'], 'o', markersize=6, color=color)
                ax2.plot(first_DACCS_year, daccs_data['ctbo_net_profit'], 'o', markersize=6, color=color)
    
    ax1.plot([], [], 'o', markersize=6, color='black', label='Year when DACCS is marginal')
    ax1.plot([], [], 's', markersize=6, color='black', label='Year when individual plant invested')
    ax1.set_xlabel('Year', fontsize=13)
    ax1.set_ylabel('Plant profits (gross) from selling CO2 (kEUR/yr)', fontsize=13)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot([], [], 'o', markersize=6, color='black', label='Year when DACCS is marginal')
    ax2.plot([], [], 's', markersize=6, color='black', label='Year when individual plant invested')
    ax2.set_xlabel('Year', fontsize=13)
    ax2.set_ylabel('Plant profits (net) from selling CO2 (kEUR/yr)', fontsize=13)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('5_profits.png', dpi=300)
    print("  Saved: 5_profits.png")

# Save plant-level data
if SAVE_PLANT_DATA and len(plant_df) > 0:
    plant_df.to_csv('ctbo_plant_results.csv', index=False)
    plant_npv_df.to_csv('ctbo_plant_npv.csv', index=False)
    print(f"\n  Plant-level data saved to 'ctbo_plant_results.csv'")
    print(f"  Plant NPV data saved to 'ctbo_plant_npv.csv'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

plt.show()
