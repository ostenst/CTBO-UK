import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def match_transportation(emitter_plant, transport_costs_df, discount_rate=0.035, lifetime=30, debug=False):
    """
    Find the nearest CCS site to an emitter plant and calculate total cost per tonne CO2.
    
    Parameters:
    - emitter_plant: pandas Series with 'Easting' and 'Northing' columns
    - transport_costs_df: DataFrame with CCS transport cost data
    - debug: Boolean to print debug information
    
    Returns:
    - dict: Contains site details and costs
    """
    if debug:
        print(f"Finding nearest CCS site for: {emitter_plant.get('Site', 'Unknown')}")
        print(f"Location: Easting={emitter_plant['Easting']}, Northing={emitter_plant['Northing']}")
    
    # Calculate distances to all sites in transport_costs
    distances = []
    for idx, row in transport_costs_df.iterrows():
        if pd.notna(row['Easting']) and pd.notna(row['Northing']):
            dist = calculate_distance(emitter_plant['Easting'], emitter_plant['Northing'], 
                                    row['Easting'], row['Northing'])
            distances.append({
                'Site ID': row['Site ID'],
                'Site': row['Site'],
                'Easting': row['Easting'],
                'Northing': row['Northing'],
                'Distance': dist
            })
    
    # Find closest site
    distances_df = pd.DataFrame(distances)
    closest_distance_idx = distances_df['Distance'].idxmin()
    closest_site_id = distances_df.loc[closest_distance_idx, 'Site ID']
    closest_site_original = transport_costs_df[transport_costs_df['Site ID'] == closest_site_id].iloc[0]
    
    # Extract cost data
    mass_CO2 = closest_site_original['Total Tonnes of CO2 stored (MtCO2) 2070']
    point_CO2 = closest_site_original['CO2 Point']
    terminal = closest_site_original['Final CO2 Terminal']
    injection_site = closest_site_original['Injection Site']
    onshore_mode = closest_site_original['CO2 Pipeline/Trucking?']
    
    # Calculate onshore costs per tonne CO2
    onshore_opex = float(closest_site_original['CO2 onshore transport opex (£m/y)'])
    onshore_opex = onshore_opex / mass_CO2  # [£/tCO2]
    
    onshore_pipeline = 0
    if onshore_mode == "Pipeline":
        pipeline_capex = float(closest_site_original['CO2 onshore pipeline capex (£m)'])
        annuity_factor = (discount_rate * (1 + discount_rate) ** lifetime) / ((1 + discount_rate) ** lifetime - 1)
        onshore_pipeline = pipeline_capex * annuity_factor
        onshore_pipeline = onshore_pipeline / mass_CO2  # [£/tCO2]
        onshore_opex += onshore_pipeline
    
    injection_opex = float(closest_site_original['CO2 T&S cost from defined point (£/t)'])
    total_cost = onshore_opex + injection_opex
    
    result = {
        'site_id': closest_site_id,
        'site_name': closest_site_original['Site'],
        'distance_m': distances_df.loc[closest_distance_idx, 'Distance'],
        'mass_co2_kt': round(mass_CO2 * 1000, 1),
        'co2_point': point_CO2,
        'terminal': terminal,
        'injection_site': injection_site,
        'onshore_mode': onshore_mode,
        'onshore_cost_per_t': round(onshore_opex, 2),
        'injection_cost_per_t': round(injection_opex, 2),
        'total_cost_per_t': round(total_cost, 1)
    }
    
    if debug:
        print(f"\nClosest site: {result['site_name']} (ID: {result['site_id']})")
        print(f"Distance: {result['distance_m']:.2f} meters")
        print(f"Mass CO2: {result['mass_co2_kt']} ktCO2/y")
        print(f"CO2 point: {result['co2_point']}")
        print(f"Terminal: {result['terminal']}")
        print(f"Injection site: {result['injection_site']}")
        print(f"Onshore mode: {result['onshore_mode']}")
        print(f"Onshore cost: {result['onshore_cost_per_t']} £/tCO2")
        print(f"Injection cost: {result['injection_cost_per_t']} £/tCO2")
        print(f"Total cost: {result['total_cost_per_t']} £/tCO2")
    
    return result

def calculate_distance(easting1, northing1, easting2, northing2):
    """Calculate Euclidean distance between two points"""
    return ((easting1 - easting2)**2 + (northing1 - northing2)**2)**0.5

def approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509):
    a = 2.1673
    b = 0.8092
    c = -0.00332
    n = 0.5291
    m = 0.8391
    
    nCO2 = mCO2*1000 / 44 # [kmolCO2/h] given [tCO2/h]
    n_fluegas = nCO2 / xCO2 # [kmol/h]
    V_fluegas = n_fluegas * 22.4 # [Nm3/h]

    n_largest_absorbers = int( (V_fluegas/1000) // 1613) # The volume limit from [Kim & Leonard, 2025] Table A2
    remaining_V_fluegas = (V_fluegas/1000) % 1613 # [10**3 Nm3/h]

    CAPEX = 0
    for i in range(n_largest_absorbers):
        V_fluegas = 1613 # Add the largest absorbers if needed
        TEC = a + (b * (xCO2)**n + c) * (V_fluegas)**m # NOTE xCO2 as fraction, not percentage
        CAPEX += TEC
    if remaining_V_fluegas > 0:
        TEC = a + (b * (xCO2)**n + c) * (remaining_V_fluegas)**m 
        CAPEX += TEC
    CAPEX = CAPEX * NETL # [MEUR] NETL Methodology
    CAPEX = CAPEX * CEPCI_2025 / CEPCI_2023
    return CAPEX

# Prepare plant data
pounds_to_EUR = 1.15
CEPCI_2025 = 930
CEPCI_2023 = 798.7
fossil_plants = pd.read_csv("data/point_sources_CO2_2022.csv")
fossil_plants['CO2'] = fossil_plants['Emission'] * 3.66
CO2_fossil = fossil_plants['CO2'].sum()

print(f"\nThe 60 largest fossil CO2 emitters (2022):")
largest_plants = fossil_plants.nlargest(60, 'CO2')[['PlantID', 'Site', 'Easting', 'Northing', 'Operator', 'Sector', 'CO2']]
for i, (idx, row) in enumerate(largest_plants.iterrows(), 1):
    print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tCO2/yr")
print("Sum of all point-source fossil emitters: ", round(CO2_fossil*10**-6, 1), " MtCO2/yr")
print("Sum of top 60 fossil emitters: ", round(largest_plants['CO2'].sum()*10**-6, 1), " MtCO2/yr")

# Calculate transport costs for all top 60 fossil emitters
transport_costs = pd.read_csv("data/nzip_balanced_scenario_results.csv", encoding='latin-1')
transport_costs = transport_costs[transport_costs["CO2 Pipeline/Trucking?"] != "No CCS"] 
for idx, plant in largest_plants.iterrows():
    result = match_transportation(plant, transport_costs, discount_rate=0.035, lifetime=30, debug=False)
    largest_plants.loc[idx, 'transport_cost'] = result['total_cost_per_t'] * pounds_to_EUR * CEPCI_2025/CEPCI_2023 # [EUR/tCO2]

# ------------POWER SECTOR------------
# Estimate CAPEX from CO2 flow [t/h] and concentration [%] for each stack
xCO2 = 0.05 # assume CCGT plants
eta_P = 0.49 # [MWel/MWfuel] total efficiency from DUKES data
emission_factor = 0.204 # tCO2/MWhfuel [NZIP, 2020]
power_capacities = pd.read_csv("data/power_capacities_clean.csv")
power_producers = largest_plants[(largest_plants['Sector'] == 'Major power producers') | (largest_plants['Sector'] == 'Minor power producers')].copy()
power_producers = power_producers[power_producers['Site'] != 'Ratcliffe on Soar Power Station'].copy() # Decommissioned coal plant
power_producers = power_producers.merge(power_capacities, left_on='Site', right_on='Power plant name', how='left')

for idx, plant in power_producers.iterrows():
    Pinstalled = plant['Capacity [MW]'] # [MWel]
    Qfuel = Pinstalled / eta_P # [MWfuel]
    mCO2 = Qfuel * emission_factor # [tCO2/h] when at full load
    FLH = plant['CO2'] / mCO2 # [h/y] = [tCO2/yr] / [tCO2/h]

    CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509)
    # need to levelize and apply FOAK factor

# Assume CO2 emissions based on DESNZ projections and CCC scenario
# https://assets.publishing.service.gov.uk/media/6604460f91a320001a82b0fd/uk-greenhouse-gas-emissions-provisional-figures-statistical-release-2023.pdf
# https://assets.publishing.service.gov.uk/media/675c0ca798302e574b915336/eep-report-2023-2050.pdf
# https://www.theccc.org.uk/publication/the-seventh-carbon-budget/

# "Under EEP-ready policies only, emissions [all CO2eq] are projected to fall by 23% between 2022 and 2050."
# In the 7th carbon budget, CO2 emissions fall to negative -40 MtCO2/yr by 2050
CO2_2023 = 300 # [MtCO2/yr]
CO2_plants = 1 # [MtCO2/yr]