import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TODO:
# (1) Distinguish between fossil and biogenic CO2 (pre and post capture)
# (2) Capture the biogenic CO2 in energy_supply() function
# (3) Adjust for cement CO2
# (4) Implement profits/costs for individual plants

# Load and plot MACC data
macc_4oak = pd.read_csv('macc_4oak.csv')
macc_foak = pd.read_csv('macc_foak.csv')
macc_4oak["invested"] = [False] * len(macc_4oak)
macc_foak["invested"] = [False] * len(macc_foak)

print(macc_4oak.head())
print(macc_foak.head())

plt.plot(macc_4oak['ktCO2_yr_cumulative'], macc_4oak['EUR/tCO2'], color='blue')
plt.plot(macc_foak['ktCO2_yr_cumulative'], macc_foak['EUR/tCO2'], color='red')
plt.xlabel('Cumulative Captured CO2 (ktCO2/yr)')
plt.ylabel('Marginal Abatement Cost (EUR/tCO2)')
plt.title('Marginal Abatement Cost Curve')
plt.legend(['4OAK', 'FOAK'])
plt.grid(True)

# Create exogenous scenarios
# ctbo_fraction = t**2 where t equals 0 in 2025, 2 in 2030, 4 in 2035, etc.
years = np.arange(2025, 2056)  # 2025 to 2055 inclusive
t = (years - 2025) * 2/5  # t increases by 2 every 5 years
ctbo_fraction = t**2
ctbo = pd.DataFrame({
    'year': years,
    't': t,
    'ctbo_fraction': ctbo_fraction
})

# CO2 emissions from IEA UK 2023 data
coal = 17 # [MtCO2/yr]
oil = 140 # [MtCO2/yr]
gas = 127 # [MtCO2/yr]
carbon = (coal + oil + gas)*1000 # [ktCO2/yr]
print("Total carbon emissions in 2023", round(carbon, 0), "ktCO2/yr")
point_sources = np.sum(macc_4oak['ktCO2_yr_baseline']) # [ktCO2/yr]
diffuse = carbon - point_sources # [ktCO2/yr] remaining emissions
print("Diffuse emissions in 2023:", round(diffuse, 0), "ktCO2/yr")

# Decrease diffuse from start_fraction to end_fraction by target_year
start_year = 2025
target_year = 2050
start_fraction = 1.0  # 100% at start_year
end_fraction = 0.5    # 50% at target_year (change this to modify target fraction)

diffuse_percentage = np.where(
    years <= target_year,
    start_fraction - (years - start_year) * ((start_fraction - end_fraction) / (target_year - start_year)),
    end_fraction  # Stay at end_fraction after target_year
)
diffuse_values = diffuse * diffuse_percentage  # [ktCO2/yr]

diffuse_df = pd.DataFrame({
    'year': years,
    'diffuse_percentage': diffuse_percentage * 100,  # Convert to percentage
    'diffuse_ktCO2_yr': diffuse_values
})

print("\nDiffuse Emissions by Year:")
print(diffuse_df)

# Calculate ETS trajectories
# https://www.gov.uk/government/publications/traded-carbon-values-used-for-modelling-purposes-2024/traded-carbon-values-used-for-modelling-purposes-2024
pounds_to_EUR = 1.15
ets_2025 = 49  # [£/tCO2] 49-74 to 85-154
ets_2050 = 74  # [£/tCO2]

# Linear interpolation of ETS prices from 2025 to 2050
ets = np.where(
    years <= 2050,
    ets_2025 + (years - 2025) * ((ets_2050 - ets_2025) / (2050 - 2025)),
    ets_2050  # Stay at 2050 price after 2050
)
ets_df = pd.DataFrame({
    'year': years,
    'ets': ets * pounds_to_EUR
})

print("\nETS Prices by Year:")
print(ets_df)

# ------- Simulate CTBO per year -------
# Simplified analysis: assuming all CO2 is fossil (pessimistic, it will overestimate CTBO capacities), only for the macc_foak plants
# Create dictionaries for efficient lookups
ets_dict = ets_df.set_index('year')['ets'].to_dict()
diffuse_dict = diffuse_df.set_index('year')['diffuse_ktCO2_yr'].to_dict()
ctbo_dict = ctbo.set_index('year')['ctbo_fraction'].to_dict()

# Initialize vectors to store results
supplied_CO2_vec = []
total_emissions_vec = []
ctbo_mandate_vec = []
point_capacity_vec = []
DACCS_capacity_vec = []
marginal_plant_vec = []
CSU_costs_vec = []
CTBO_cost_lev_vec = []
KPI = []

CTBO = True # For a scenario with/without CTBO
CSU_cost = 0
marginal_cost = 0
CTBO_cost_lev = 0

for year in years:
    DACCS_capacity = 0 # [ktCO2/yr] probably doesn't matter where this gets reset

    # Each plant in macc_foak invests if the ETS is high enough
    ets_price = ets_dict[year]  # Get ETS price for this year
    for idx, plant in macc_foak.iterrows():
        if not plant['invested'] and plant['EUR/tCO2'] < ets_price:
            macc_foak.loc[idx, 'invested'] = True
            macc_foak.loc[idx, 'year_invested'] = year
            print(f"// Incentivize plant {plant['site-stack']} in {year} and capture {plant['ktCO2_yr_captured']} ktCO2/yr since ({plant['EUR/tCO2']} < {ets_price}) EUR/tCO2")
        
    # Summarize emissions: baseline from non-invested plants + residuals from invested plants
    baseline_emissions = macc_foak['ktCO2_yr_baseline'].where(~macc_foak['invested'], 0)
    residual_emissions = macc_foak['ktCO2_yr_residual'].where(macc_foak['invested'], 0)
    plant_emissions = baseline_emissions.sum() + residual_emissions.sum()
    diffuse_emissions = diffuse_dict[year]  # Get diffuse emissions for this year
    total_emissions = plant_emissions + diffuse_emissions
    print(f"Total emissions in {year}: {total_emissions:.1f} ktCO2/yr (baseline: {baseline_emissions.sum():.1f}, residual: {residual_emissions.sum():.1f}, diffuse: {diffuse_emissions:.1f})")

    # Get the CTBO fraction for this year
    supplied_CO2 = macc_foak['ktCO2_yr_baseline'].sum() + diffuse_emissions # [ktCO2/yr]
    ctbo_fraction = ctbo_dict[year]/100
    ctbo_mandate = supplied_CO2 * ctbo_fraction

    point_capacity = macc_foak['ktCO2_yr_captured'].where(macc_foak['invested'], 0)
    point_capacity = point_capacity.sum()
    print(f"Storage capacity in {year} is:  {point_capacity:.1f} ktCO2/yr")

    if CTBO:
        missing_capacity = ctbo_mandate - point_capacity
        # go through the macc_foak dataframe and find the plants with the lowest marginal abatement cost
        # and invest in them until the missing capacity is met using a while loop
        i = 0 # Will increase if plants are mandated
        while missing_capacity > 0 and i < len(macc_foak):
            plant = macc_foak.iloc[i]
            if not plant['invested']:
                macc_foak.loc[i, 'invested'] = True
                macc_foak.loc[i, 'year_invested'] = year
                print(f"// Mandate plant {plant['site-stack']} in {year} and capture {plant['ktCO2_yr_captured']} ktCO2/yr with a cost of {plant['EUR/tCO2']} EUR/tCO2")
                point_capacity = point_capacity + plant['ktCO2_yr_captured'] # NOTE: represents point source storage capacity
                missing_capacity = ctbo_mandate - point_capacity
            i += 1
        
        # Marginal plant costs (only if plants were mandated):
        CTBO_cost = 0
        if i > 0:
            marginal_plant = macc_foak.iloc[i-1]
            marginal_cost = marginal_plant['EUR/tCO2']
            CSU_cost = (marginal_plant['EUR/tCO2'] - ets_price) # NOTE: represents point source CSU costs
            print(f"// Marginal plant in year {year} is {marginal_plant['site-stack']} at a cost of {marginal_plant['EUR/tCO2']} EUR/tCO2 => CSU cost: {CSU_cost} EUR/tCO2")
            CTBO_cost = CSU_cost * point_capacity # [EUR/tCO2 * ktCO2/yr = kEUR/yr]

            if i == len(macc_foak):
                print("===>All point sources have invested")

        # If still missing capacity, add DACCS backstop technology
        if missing_capacity > 0:
            DACCS_capacity = missing_capacity
            marginal_cost = 450 # DACCS becomes marginal technology
            CSU_cost = (marginal_cost - ets_price) 
            print(f"===>Adding DACCS capacity which adds to CTBO costs")
            CTBO_cost = CSU_cost * (point_capacity + DACCS_capacity) # [EUR/tCO2 * ktCO2/yr = kEUR/yr] #NOTE I think this gets added twice in the pure DACCS cases...

        CTBO_cost_lev = CTBO_cost / supplied_CO2 # [EUR/tCO2] 

    # Save values to vectors
    supplied_CO2_vec.append(supplied_CO2)
    total_emissions_vec.append(total_emissions)
    ctbo_mandate_vec.append(ctbo_mandate)
    point_capacity_vec.append(point_capacity)
    DACCS_capacity_vec.append(DACCS_capacity)
    marginal_plant_vec.append(marginal_cost)
    CSU_costs_vec.append(CSU_cost)
    CTBO_cost_lev_vec.append(CTBO_cost_lev)
    KPI.append(CSU_cost / ets_price)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(years, supplied_CO2_vec, label='Supplied CO2', linewidth=2)
plt.plot(years, total_emissions_vec, label='Emitted CO2', linewidth=2)
plt.plot(years, ctbo_mandate_vec, label='CTBO Mandate', linewidth=2)
plt.plot(years, point_capacity_vec, label='Storage Capacity', linewidth=2)
plt.plot(years, DACCS_capacity_vec, label='DACCS Capacity', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('ktCO2/yr', fontsize=12)
plt.title('Emissions, CTBO Mandate, and Storage Capacity Over Time', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()

# plot ETS prices and CSU costs
plt.figure(figsize=(10, 6))
plt.plot(years, ets_df['ets'], label='ETS Prices', linewidth=2)
plt.plot(years, marginal_plant_vec, label='Marginal Plant Costs', linewidth=2)
plt.plot(years, CSU_costs_vec, label='CSU Costs', linewidth=2)
plt.plot(years, CTBO_cost_lev_vec, label='CTBO Costs', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('EUR/tCO2', fontsize=12)
plt.title('ETS Prices, CSU Costs, and CTBO Costs Over Time', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()

# plot KPI and CTBO_cost_lev on separate Y axes 
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot KPI on left y-axis
ax1.plot(years, KPI, color='blue', label='KPI', linewidth=2)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('CSU/ETS price ratio', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# Create second y-axis for CTBO costs
ax2 = ax1.twinx()
ax2.plot(years, CTBO_cost_lev_vec, color='red', label='CTBO Costs', linewidth=2)
ax2.set_ylabel('CTBO Costs (EUR/tCO2)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Price Ratio and CTBO Costs Over Time', fontsize=14)
fig.tight_layout()
plt.show()


