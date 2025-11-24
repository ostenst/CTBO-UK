import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== INPUT PARAMETERS ====================

# Scenario selection
FOAK = True                    # True: First-of-a-kind costs, False: Nth-of-a-kind costs
CTBO_ENABLED = True            # Enable CTBO mandate
ETS_HIGH = False               # True: High ETS trajectory (85->154 £/tCO2), False: Low (49->74 £/tCO2)
DACCS_EXPENSIVE = True         # True: Expensive DACCS (323->281 £/tCO2), False: Cheap (247->152 £/tCO2)
VERBOSE = False                # Print detailed investment decisions
SAVE_PLANT_DATA = True        # Save plant-level results to CSV

# Time parameters
START_YEAR = 2025
END_YEAR = 2055
years = np.arange(START_YEAR, END_YEAR + 1)

# CTBO trajectory: ctbo_fraction = t^2 where t increases by 2 every 5 years
t = (years - START_YEAR) * 2/5
ctbo_fraction = t**2  # [%]

# Diffuse emissions reduction
DIFFUSE_START_FRACTION = 1.0   # 100% of diffuse emissions in START_YEAR
DIFFUSE_END_FRACTION = 0.50    # 50% of diffuse emissions by 2050
DIFFUSE_TARGET_YEAR = 2050

# Financial parameters
DISCOUNT_RATE = 0.035          # Real discount rate for NPV calculations [3.5%]
USE_INVESTMENT_YEAR_AS_BASE = False  # True: NPV from each plant's investment year, False: NPV from START_YEAR

# Price trajectories (in £/tCO2)
pounds_to_EUR = 1.15
if ETS_HIGH:
    ets_2025, ets_2050 = 85, 154
else:
    ets_2025, ets_2050 = 49, 74

if DACCS_EXPENSIVE:
    DACCS_2025, DACCS_2050 = 323, 281
else:
    DACCS_2025, DACCS_2050 = 247, 152

# ==================== LOAD DATA ====================

macc_4oak = pd.read_csv('macc_4oak.csv')
macc_foak = pd.read_csv('macc_foak.csv')
macc_4oak['invested'] = False
macc_foak['invested'] = False
macc = macc_foak if FOAK else macc_4oak

# Calculate baseline emissions
cement_emissions = macc_4oak[macc_4oak['site-stack'].str.endswith('-cement')]['ktCO2f_yr_baseline'].sum()
total_emissions_2023 = (17 + 140 + 127) * 1000 + cement_emissions  # [ktCO2/yr]
point_sources = macc_4oak['ktCO2f_yr_baseline'].sum()
diffuse_baseline = total_emissions_2023 - point_sources

print(f"Total fossil emissions (2023): {total_emissions_2023:.0f} ktCO2/yr")
print(f"  Point sources: {point_sources:.0f} ktCO2/yr")
print(f"  Diffuse: {diffuse_baseline:.0f} ktCO2/yr")

# ==================== CREATE TRAJECTORIES ====================

# Diffuse emissions trajectory
diffuse_fraction = np.where(
    years <= DIFFUSE_TARGET_YEAR,
    DIFFUSE_START_FRACTION - (years - START_YEAR) * ((DIFFUSE_START_FRACTION - DIFFUSE_END_FRACTION) / (DIFFUSE_TARGET_YEAR - START_YEAR)),
    DIFFUSE_END_FRACTION
)
diffuse_emissions = diffuse_baseline * diffuse_fraction

# ETS price trajectory (convert to EUR/tCO2)
ets_prices = np.where(
    years <= 2050,
    ets_2025 + (years - START_YEAR) * ((ets_2050 - ets_2025) / (2050 - START_YEAR)),
    ets_2050
) * pounds_to_EUR

# DACCS cost trajectory (convert to EUR/tCO2)
DACCS_costs = np.where(
    years <= 2050,
    DACCS_2025 + (years - START_YEAR) * ((DACCS_2050 - DACCS_2025) / (2050 - START_YEAR)),
    DACCS_2050
) * pounds_to_EUR

# ==================== SIMULATION ====================

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
plant_results = []  # List of dicts: {year, plant_name, cost, profit, revenue, ...}

# Track when DACCS is first needed
first_DACCS_year = None

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
                print(f"Year {year}: Voluntary investment in {plant['site-stack']} (cost: {plant['EUR/tCO2']:.0f} < ETS: {ets_price:.0f} EUR/tCO2)")
    
    # Calculate current emissions and capacities
    baseline_emissions = macc['ktCO2f_yr_baseline'].where(~macc['invested'], 0).sum()
    residual_emissions = macc['ktCO2f_yr_residual'].where(macc['invested'], 0).sum()
    plant_emissions = baseline_emissions + residual_emissions
    total_emissions = plant_emissions + diffuse
    
    supplied_CO2 = macc['ktCO2f_yr_baseline'].sum() + diffuse # [ktCO2/yr] NOTE includes cement
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
                        print(f"Year {year}: Switching to DACCS (cost: {DACCS_cost:.0f} < plant: {plant['EUR/tCO2']:.0f} EUR/tCO2)")
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
        
        # Calculate costs
        if j > 0:
            marginal_plant = macc.iloc[j-1]
            marginal_cost = marginal_plant['EUR/tCO2']
            CSU_cost = max(0, marginal_cost - ets_price)
            CTBO_cost = CSU_cost * point_capacity
        
        # Add DACCS if still needed
        if missing_capacity > 0:
            DACCS_capacity = missing_capacity
            marginal_cost = DACCS_cost
            CSU_cost = max(0, marginal_cost - ets_price)
            CTBO_cost = CSU_cost * (point_capacity + DACCS_capacity)
            
            # Record first year DACCS is needed
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
        # Calculate diluted cost (same for invested and non-invested plants)
        if plant['site-stack'].endswith('W2E') or plant['site-stack'].endswith('BECCS'):
            csu_diluted_cost = 0  # Biogenic sources don't pay CTBO on emissions
        elif plant['site-stack'].endswith('cement'):
            csu_diluted_cost = CTBO_cost_lev * (1 - 0.63)  # Only 37% from fossil fuels (rest is calcination)
        else:
            csu_diluted_cost = CTBO_cost_lev  # Pay CTBO on all fossil fuel use
        
        # Set values based on investment status
        if plant['invested']:
            investment_year = plant['year_invested']
            CO2_captured_fossil = plant['ktCO2f_yr_captured'] # [ktCO2/yr]
            CO2_captured_bio = plant['ktCO2bio_yr_captured'] # [ktCO2/yr]
            csu_gross_profit = CSU_cost # [EUR/tCO2f]
            ctbo_fossil_profit = CSU_cost * CO2_captured_fossil # [kEUR/yr]
            ctbo_gross_profit = CSU_cost * (CO2_captured_fossil + CO2_captured_bio) # [kEUR/yr]
        else:
            investment_year = None
            CO2_captured_fossil = 0
            CO2_captured_bio = 0
            csu_gross_profit = 0
            ctbo_fossil_profit = 0
            ctbo_gross_profit = 0
        
        # Calculate derived values
        ctbo_diluted_cost = csu_diluted_cost * plant['ktCO2f_yr_baseline']
        ctbo_net_profit = ctbo_gross_profit - ctbo_diluted_cost
        
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
            'csu_net_profit': csu_gross_profit - csu_diluted_cost,
            'csu_bio_profit': csu_gross_profit,
            'ctbo_diluted_cost': ctbo_diluted_cost,
            'ctbo_fossil_profit': ctbo_fossil_profit,
            'ctbo_gross_profit': ctbo_gross_profit,
            'ctbo_net_profit': ctbo_net_profit
        })

print(f"\n2050 Aggregate Results:")
idx_2050 = np.where(years == 2050)[0][0]
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

# Estimate cost-to-consumer [DESNZ emission factor and fuel price data]
carbon_price = np.array(CTBO_cost_lev_vec) / pounds_to_EUR # [£/tCO2 per year]
emission_factor_diesel = 2.628 # [kgCO2/litre]
emission_factor_petrol = 2.339 # [kgCO2/litre]
emission_factor_gas = 0.2039 # [kgCO2/kWh] 1 thrm <=> 29.3 kWh, or 29.3kWh/thrm
emission_factor_gas = 0.2039 * 29.3 # [kgCO2/thrm]
diesel_price = 143.97 # [pence/litre]
petrol_price = 135.07 # [pence/litre]
gas_price = 80 # [pence/thrm]

# Calculate absolute price increases (convert from £/tCO2 to pence per unit)
diesel_increase_abs = carbon_price * (emission_factor_diesel / 1000) * 100  # [pence/litre]
petrol_increase_abs = carbon_price * (emission_factor_petrol / 1000) * 100  # [pence/litre]
gas_increase_abs = carbon_price * (emission_factor_gas / 1000) * 100  # [pence/thrm]

# Calculate percentage price increases
diesel_increase_pct = (diesel_increase_abs / diesel_price) * 100  # [%]
petrol_increase_pct = (petrol_increase_abs / petrol_price) * 100  # [%]
gas_increase_pct = (gas_increase_abs / gas_price) * 100  # [%]

# Plot fuel price increases
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Absolute increases
ax1.plot(years, diesel_increase_abs, label='Diesel', linewidth=2, color='darkblue')
ax1.plot(years, petrol_increase_abs, label='Petrol', linewidth=2, color='orange')
ax1.plot(years, gas_increase_abs, label='Gas', linewidth=2, color='green')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Absolute Price Increase (pence)', fontsize=12)
ax1.set_title('CTBO Impact on Consumer Fuel Prices: Absolute Increase', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True)

# Percentage increases
ax2.plot(years, diesel_increase_pct, label='Diesel', linewidth=2, color='darkblue')
ax2.plot(years, petrol_increase_pct, label='Petrol', linewidth=2, color='orange')
ax2.plot(years, gas_increase_pct, label='Gas', linewidth=2, color='green')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Percentage Price Increase (%)', fontsize=12)
ax2.set_title('CTBO Impact on Consumer Fuel Prices: Percentage Increase', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True)

plt.tight_layout()

print(f"\n2050 Consumer Fuel Price Impacts:")
print(f"  Diesel:  +{diesel_increase_abs[idx_2050]:.2f} pence/litre (+{diesel_increase_pct[idx_2050]:.1f}%)")
print(f"  Petrol:  +{petrol_increase_abs[idx_2050]:.2f} pence/litre (+{petrol_increase_pct[idx_2050]:.1f}%)")
print(f"  Gas:     +{gas_increase_abs[idx_2050]:.2f} pence/thrm (+{gas_increase_pct[idx_2050]:.1f}%)")


# Convert plant-level results to DataFrame
plant_df = pd.DataFrame(plant_results)
print(f"\nPlant-level data collected: {len(plant_df)} plant-year observations")

if len(plant_df) > 0:
    print(f"  Years covered: {plant_df['year'].min():.0f} - {plant_df['year'].max():.0f}")
    print(f"  Number of plants: {plant_df['plant'].nunique()}")
    
    # Calculate NPV for each plant
    plant_npv = []
    for plant_name in plant_df['plant'].unique():
        plant_data = plant_df[plant_df['plant'] == plant_name].copy()
        # Get investment year (use max to skip None values from non-invested years)
        investment_year = plant_data['investment_year'].max()
        
        # Skip plants that never invested
        if pd.isna(investment_year):
            continue
        
        # Calculate discount factors from base year
        base_year = investment_year if USE_INVESTMENT_YEAR_AS_BASE else START_YEAR
        plant_data['discount_factor'] = 1 / (1 + DISCOUNT_RATE) ** (plant_data['year'] - base_year)
        
        # Calculate discounted cash flows
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
    
    # Print annual summaries (CCGT plants only)
    ccgt_plants = plant_df[plant_df['plant'].str.endswith('CCGT')]
    plant_2030 = ccgt_plants[ccgt_plants['year'] == 2030]
    plant_2040 = ccgt_plants[ccgt_plants['year'] == 2040]
    plant_2050 = ccgt_plants[ccgt_plants['year'] == 2050]
    print(f"\n  Total CCGT plant (fossil net) profits in 2030: {plant_2030['ctbo_diluted_cost'].sum() - plant_2030['ctbo_fossil_profit'].sum() + plant_2030['ctbo_net_profit'].sum():.0f} kEUR/yr")
    print(f"  Total CCGT plant (fossil net) profits in 2040: {plant_2040['ctbo_diluted_cost'].sum() - plant_2040['ctbo_fossil_profit'].sum() + plant_2040['ctbo_net_profit'].sum():.0f} kEUR/yr")
    print(f"  Total CCGT plant (fossil net) profits in 2050: {plant_2050['ctbo_diluted_cost'].sum() - plant_2050['ctbo_fossil_profit'].sum() + plant_2050['ctbo_net_profit'].sum():.0f} kEUR/yr")
    
    # Print NPV summaries
    base_description = "investment year" if USE_INVESTMENT_YEAR_AS_BASE else f"{START_YEAR}"
    print(f"\n  NPV Analysis (discount rate: {DISCOUNT_RATE*100:.1f}%, base: {base_description}):")
    print(f"    Total NPV gross profit (all plants): {plant_npv_df['NPV_gross_profit'].sum():.0f} kEUR")
    print(f"    Total NPV net profit (all plants): {plant_npv_df['NPV_net_profit'].sum():.0f} kEUR")
    print(f"    Total NPV fossil profit (all plants): {plant_npv_df['NPV_fossil_profit'].sum():.0f} kEUR")
    
    # Top 10 and bottom 10 plants by NPV net profit
    # Merge with plant_df to get CO2 capture data (use max to get non-zero invested values)
    plant_co2_data = plant_df.groupby('plant')[['CO2_captured_fossil', 'CO2_captured_bio']].max().reset_index()
    plant_npv_with_co2 = plant_npv_df.merge(plant_co2_data, on='plant', how='left')
    
    top_10 = plant_npv_with_co2.nlargest(10, 'NPV_net_profit')
    bottom_10 = plant_npv_with_co2.nsmallest(10, 'NPV_net_profit')
    
    print(f"\n  Top 10 plants by NPV net profit:")
    for idx, row in top_10.iterrows():
        print(f"    {row['plant'][:45]:45s} | NPV: {row['NPV_net_profit']:8,.0f} kEUR | Invested: {row['investment_year']:.0f} | CO2f: {row['CO2_captured_fossil']:5.0f} | CO2bio: {row['CO2_captured_bio']:5.0f} ktCO2/yr")
    
    print(f"\n  Bottom 10 plants by NPV net profit:")
    for idx, row in bottom_10.iterrows():
        print(f"    {row['plant'][:45]:45s} | NPV: {row['NPV_net_profit']:8,.0f} kEUR | Invested: {row['investment_year']:.0f} | CO2f: {row['CO2_captured_fossil']:5.0f} | CO2bio: {row['CO2_captured_bio']:5.0f} ktCO2/yr")
    
    # Plot CSU profits for top 10 plants
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    top_10_plants = top_10['plant'].tolist()
    for plant_name in top_10_plants:
        plant_data = plant_df[plant_df['plant'] == plant_name]
        ax1.plot(plant_data['year'], plant_data['csu_gross_profit'], label=plant_name[:30], linewidth=2)
        ax2.plot(plant_data['year'], plant_data['csu_net_profit'], label=plant_name[:30], linewidth=2)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('CSU Gross Profit (EUR/tCO2)', fontsize=12)
    ax1.set_title('Top 10 Plants: CSU Gross Profit Over Time', fontsize=14)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True)
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('CSU Net Profit (EUR/tCO2)', fontsize=12)
    ax2.set_title('Top 10 Plants: CSU Net Profit Over Time', fontsize=14)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Plot CTBO profits for top 10 plants (excluding Drax-BECCS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    top_10_plants_no_drax = [p for p in top_10_plants if 'Drax-BECCS' not in p]
    for plant_name in top_10_plants_no_drax:
        plant_data = plant_df[plant_df['plant'] == plant_name]
        ax1.plot(plant_data['year'], plant_data['ctbo_gross_profit'], label=plant_name[:30], linewidth=2)
        ax2.plot(plant_data['year'], plant_data['ctbo_net_profit'], label=plant_name[:30], linewidth=2)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('CTBO Gross Profit (kEUR/yr)', fontsize=12)
    ax1.set_title('Top 10 Plants (excl. Drax): CTBO Gross Profit Over Time', fontsize=14)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True)
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('CTBO Net Profit (kEUR/yr)', fontsize=12)
    ax2.set_title('Top 10 Plants (excl. Drax): CTBO Net Profit Over Time', fontsize=14)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True)
    
    plt.tight_layout()


# ==================== PLOTS ====================

# Plot 1: MACC curves
plt.figure(figsize=(10, 6))
plt.plot(macc_4oak['ktCO2_yr_cumulative'], macc_4oak['EUR/tCO2'], color='blue', label='4OAK')
plt.plot(macc_foak['ktCO2_yr_cumulative'], macc_foak['EUR/tCO2'], color='red', label='FOAK')
plt.xlabel('Cumulative Captured CO2 (ktCO2/yr)', fontsize=12)
plt.ylabel('Marginal Abatement Cost (EUR/tCO2)', fontsize=12)
plt.title('Marginal Abatement Cost Curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()

# Plot 2: Emissions and capacities
plt.figure(figsize=(10, 6))
plt.plot(years, supplied_CO2_vec, label='Supplied CO2', linewidth=2, color='black')
plt.plot(years, total_emissions_vec, label='Emitted CO2', linewidth=2, color='darkgreen')
plt.plot(years, ctbo_mandate_vec, label='CTBO Mandate', linewidth=2, color='black', linestyle='--')
plt.plot(years, fCCS_capacity_vec, label='Fossil CCS', linewidth=2, color='gray')
plt.plot(years, BECCS_capacity_vec, label='BECCS', linewidth=2, color='green')
plt.plot(years, DACCS_capacity_vec, label='DACCS', linewidth=2, color='lightgreen')
plt.xlabel('Year', fontsize=12)
plt.ylabel('ktCO2/yr', fontsize=12)
plt.title('Emissions and Storage Capacity Over Time', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()

# Plot 3: Costs
plt.figure(figsize=(10, 6))
plt.plot(years, ets_prices, label='ETS Price', linewidth=2)
plt.plot(years, marginal_cost_vec, label='Marginal Cost', linewidth=2)
plt.plot(years, CSU_cost_vec, label='CSU Cost', linewidth=2)
plt.plot(years, CTBO_cost_lev_vec, label='CTBO Cost (levelized)', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('EUR/tCO2', fontsize=12)
plt.title('Price Trajectories Over Time', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()

# Plot 4: KPI and CTBO costs
fig, ax1 = plt.subplots(figsize=(10, 6))
KPI = np.array(CSU_cost_vec) / ets_prices
ax1.plot(years, KPI, color='blue', label='CSU/ETS Ratio', linewidth=2)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('CSU/ETS Price Ratio', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(years, CTBO_cost_lev_vec, color='red', label='CTBO Cost', linewidth=2)
ax2.set_ylabel('CTBO Cost (EUR/tCO2)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

plt.title('CSU/ETS Ratio and CTBO Costs Over Time', fontsize=14)
fig.tight_layout()

plt.show()

# Save plant-level data if requested
if SAVE_PLANT_DATA and len(plant_df) > 0:
    plant_df.to_csv('ctbo_plant_results.csv', index=False)
    plant_npv_df.to_csv('ctbo_plant_npv.csv', index=False)
    print(f"\nPlant-level data saved to 'ctbo_plant_results.csv'")
    print(f"Plant NPV data saved to 'ctbo_plant_npv.csv'")

print("\nAnalysis complete!")
