import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Check this list also: https://ember-energy.org/latest-insights/the-largest-emitters-in-the-uk-annual-review/
print("This script calculates the costs of industrial plants")

def find_nearest_ccs_site(emitter_plant, transport_costs_df, discount_rate=0.035, lifetime=25, debug=False):
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

# Prepare data
point_sources = pd.read_csv("data/point_sources_CO2_2022.csv")
point_sources['CO2'] = point_sources['Emission'] * 3.66
total_co2 = point_sources['CO2'].sum()

print(f"\nTop 60 largest CO2 emitters (2022):")
top_60_emitters = point_sources.nlargest(60, 'CO2')[['PlantID', 'Site', 'Easting', 'Northing', 'Operator', 'Sector', 'CO2']]
for i, (idx, row) in enumerate(top_60_emitters.iterrows(), 1):
    print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tCO2/yr")
print(f"Sum of top 60 emitters: {top_60_emitters['CO2'].sum():,.0f} tonnes")
print("Sum of all point-source CO2 emissions in 2022: ", round(total_co2*10**-6, 1), " MtCO2/yr")

# Print unique sectors
print(f"\nUnique sectors in point sources data:")
unique_sectors = point_sources['Sector'].unique()
for i, sector in enumerate(sorted(unique_sectors), 1):
    print(f"{i:2d}. {sector}")

# # Select Chemical industry emitters
# chemical_emitters = point_sources[point_sources['Sector'] == 'Chemical industry'].copy()
# print(f"\nChemical industry emitters:")
# if len(chemical_emitters) > 0:
#     for i, (idx, row) in enumerate(chemical_emitters.iterrows(), 1):
#         print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tCO2/yr")
#     print(f"Total Chemical industry CO2 emissions: {chemical_emitters['CO2'].sum():,.0f} tonnes")
# else:
#     print("No emitters found in 'Chemical industry' sector")

# Select Cement emitters
cement_emitters = point_sources[point_sources['Sector'] == 'Cement'].copy()
print(f"\nCement emitters:")
if len(cement_emitters) > 0:
    for i, (idx, row) in enumerate(cement_emitters.iterrows(), 1):
        print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tCO2/yr")
    print(f"Total Cement CO2 emissions: {cement_emitters['CO2'].sum():,.0f} tonnes")
else:
    print("No emitters found in 'Cement' sector")

# Select Port Talbot emitters
port_talbot_emitters = point_sources[point_sources['Site'].str.startswith('Port Talbot', na=False)].copy()
print(f"\nPort Talbot emitters:")
if len(port_talbot_emitters) > 0:
    for i, (idx, row) in enumerate(port_talbot_emitters.iterrows(), 1):
        print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tCO2/yr")
    print(f"Total Port Talbot CO2 emissions: {port_talbot_emitters['CO2'].sum():,.0f} tonnes")
else:
    print("No emitters found starting with 'Port Talbot'")

# Select Scunthorpe emitters
scunthorpe_emitters = point_sources[point_sources['Site'].str.startswith('Scunthorpe', na=False)].copy()
print(f"\nScunthorpe emitters:")
if len(scunthorpe_emitters) > 0:
    for i, (idx, row) in enumerate(scunthorpe_emitters.iterrows(), 1):
        print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tCO2/yr")
    print(f"Total Scunthorpe CO2 emissions: {scunthorpe_emitters['CO2'].sum():,.0f} tonnes")
else:
    print("No emitters found starting with 'Scunthorpe'")

# Calculate transport costs for all top 60 emitters
transport_costs = pd.read_csv("data/nzip_balanced_scenario_results.csv", encoding='latin-1')
transport_costs = transport_costs[transport_costs["CO2 Pipeline/Trucking?"] != "No CCS"] 
for idx, plant in top_60_emitters.iterrows():
    result = find_nearest_ccs_site(plant, transport_costs, discount_rate=0.035, lifetime=25, debug=False)
    plant['transport_cost'] = result['total_cost_per_t'] # [£/tCO2]

# Create DataFrame with only refineries from top 60 emitters
refineries = top_60_emitters[(top_60_emitters['Sector'] == 'Processing & distribution of petroleum products')].copy()
print("These refineries will be considered:")
for i, (idx, row) in enumerate(refineries.iterrows(), 1):
    print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['CO2']:>12,.0f} tCO2/yr")

# Explore CAPEX relationships [Kim & Léonard, 2025]
# TEC = a + (b * xCO2**n + c) * V**m [MEUR, 2023] - note that this is not valid for CCGT if xCO2<5%, or if mCO2>1250 kt/y
a = 2.1673
b = 0.8092
c = -0.00332
n = 0.5291
m = 0.8391

results = []

for idx, refinery in refineries.iterrows():
    total_CO2 = refinery['CO2']
    power_stack = ['power',total_CO2 * 0.298, (3*25 + 8*54)/(25+54)/100] # [stack_name,tCO2/yr, fraction of CO2]
    crackers_stack = ['crackers',total_CO2 * 0.20, 17/100]
    distillation_stack = ['distillation',total_CO2 * 0.17, 11/100]
    smr_stack = ['smr',total_CO2 * 0.118, (8*6 + 24*26)/(6+26)/100]
    remaining_stack = ['remaining',total_CO2 * 0.188, 8/100]

    for stack in [power_stack, crackers_stack, distillation_stack, smr_stack, remaining_stack]:
        FLH = 8400 # [h/y]
        xCO2 = stack[2] # [-]
        mCO2 = stack[1] #[tCO2/yr]

        mCO2 = mCO2 / FLH # [tCO2/h]
        nCO2 = mCO2*1000 / 44 # [kmolCO2/h]
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
        CAPEX = CAPEX * 5.509 # [MEUR] NETL Methodology
        CEPCI_2025 = 930
        CEPCI_2023 = 798.7
        CAPEX = CAPEX * CEPCI_2025 / CEPCI_2023

        capture_rate = 0.95 # [0-1]
        annualized_CAPEX = CAPEX * 0.07 * (1 + 0.07)**25 / ((1 + 0.07)**25 - 1) *10**6# [EUR/y]
        levelized_CAPEX = annualized_CAPEX / (mCO2*FLH*capture_rate) # [EUR/tCO2]
        levelized_CAPEX_FOAK = 1.7553 * levelized_CAPEX # [EUR/tCO2]

        # Calculate transport cost
        pounds_to_EUR = 1.15
        transport_result = find_nearest_ccs_site(refinery, transport_costs, debug=False)
        transport_cost = transport_result['total_cost_per_t'] * pounds_to_EUR
        transport_cost = transport_cost * CEPCI_2025 / CEPCI_2023

        # Estimate energy OPEX (industry)
        Qreb = 3.5 * mCO2*capture_rate*1000/3600 # [MW]
        Qsteam = 0.15 * Qreb # [MW]
        Qelec = 0.30 * Qreb # [MW]
        Qchp = 0.55 * Qreb # [MW]

        cost_steam = 4.1 # [EUR/tsteam @130C] [Ali]
        evaporation_enthalpy = 2257 # [MJ/tsteam]
        cost_steam = cost_steam / evaporation_enthalpy # [EUR/MJ]
        Qsteam = Qsteam * FLH * 3600 # [MJ/y]
        OPEX_steam = cost_steam * Qsteam # [EUR/y]        
        OPEX_steam = OPEX_steam / (stack[1]*capture_rate) # [EUR/tCO2]

        celc = 80 # [EUR/MWh electricity]
        Pelec = Qelec * 0.33 * FLH # [MWh/y]
        OPEX_elec = celc * Pelec # [EUR/y]
        OPEX_elec = OPEX_elec / (stack[1]*capture_rate) # [EUR/tCO2]

        cbio = 65 # [EUR/MWh biomass]
        Qchp = Qchp * FLH # [MWh/y]
        OPEX_chp = cbio * Qchp # [EUR/y]
        OPEX_chp = OPEX_chp / (stack[1]*capture_rate) # [EUR/tCO2]

        OPEXE = OPEX_steam + OPEX_elec + OPEX_chp
   
        total_cost_foak = levelized_CAPEX_FOAK + transport_cost + OPEXE
        total_cost_noak = levelized_CAPEX + transport_cost + OPEXE
        
        # Store results
        results.append({
            'Plant': refinery['Site'],
            'Stack': stack[0],
            'CO2_kt_y': stack[1]/1000,
            'FLH_h': FLH,
            'CAPEX_FOAK': levelized_CAPEX_FOAK,
            'CAPEX_NOAK': levelized_CAPEX,
            'Transport_EUR_tCO2': transport_cost,
            'OPEX_EUR_tCO2': OPEXE,
            'Total_FOAK': total_cost_foak,
            'Total_NOAK': total_cost_noak
        })

# Remove outlier "South Ferriby" from cement_emitters:
cement_emitters = cement_emitters[cement_emitters['Site'] != 'South Ferriby'].copy()
for idx, cement_plant in cement_emitters.iterrows():
    cement_stack = ['cement', cement_plant['CO2'], 0.20] # Tharuns Excel file
    for stack in [cement_stack]:
        FLH = 8400 # [h/y]
        xCO2 = stack[2] # [-]
        mCO2 = stack[1] #[tCO2/yr]

        mCO2 = mCO2 / FLH # [tCO2/h]
        nCO2 = mCO2*1000 / 44 # [kmolCO2/h]
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
        CAPEX = CAPEX * 5.509 # [MEUR] NETL Methodology
        CEPCI_2025 = 930
        CEPCI_2023 = 798.7
        CAPEX = CAPEX * CEPCI_2025 / CEPCI_2023

        capture_rate = 0.95 # [0-1]
        annualized_CAPEX = CAPEX * 0.07 * (1 + 0.07)**25 / ((1 + 0.07)**25 - 1) *10**6# [EUR/y]
        levelized_CAPEX = annualized_CAPEX / (mCO2*FLH*capture_rate) # [EUR/tCO2]
        levelized_CAPEX_FOAK = 1.7553 * levelized_CAPEX # [EUR/tCO2]

        # Calculate transport cost
        pounds_to_EUR = 1.15
        transport_result = find_nearest_ccs_site(cement_plant, transport_costs, debug=False)
        transport_cost = transport_result['total_cost_per_t'] * pounds_to_EUR
        transport_cost = transport_cost * CEPCI_2025 / CEPCI_2023

        # Estimate energy OPEX (industry)
        Qreb = 3.5 * mCO2*capture_rate*1000/3600 # [MW]
        Qsteam = 0.15 * Qreb # [MW]
        Qelec = 0.30 * Qreb # [MW]
        Qchp = 0.55 * Qreb # [MW]

        cost_steam = 4.1 # [EUR/tsteam @130C] [Ali]
        evaporation_enthalpy = 2257 # [MJ/tsteam]
        cost_steam = cost_steam / evaporation_enthalpy # [EUR/MJ]
        Qsteam = Qsteam * FLH * 3600 # [MJ/y]
        OPEX_steam = cost_steam * Qsteam # [EUR/y]        
        OPEX_steam = OPEX_steam / (stack[1]*capture_rate) # [EUR/tCO2]

        celc = 80 # [EUR/MWh electricity]
        Pelec = Qelec * 0.33 * FLH # [MWh/y]
        OPEX_elec = celc * Pelec # [EUR/y]
        OPEX_elec = OPEX_elec / (stack[1]*capture_rate) # [EUR/tCO2]

        cbio = 65 # [EUR/MWh biomass]
        Qchp = Qchp * FLH # [MWh/y]
        OPEX_chp = cbio * Qchp # [EUR/y]
        OPEX_chp = OPEX_chp / (stack[1]*capture_rate) # [EUR/tCO2]

        OPEXE = OPEX_steam + OPEX_elec + OPEX_chp
   
        total_cost_foak = levelized_CAPEX_FOAK + transport_cost + OPEXE
        total_cost_noak = levelized_CAPEX + transport_cost + OPEXE
        
        # Store results
        results.append({
            'Plant': cement_plant['Site'],
            'Stack': stack[0],
            'CO2_kt_y': stack[1]/1000,
            'FLH_h': FLH,
            'CAPEX_FOAK': levelized_CAPEX_FOAK,
            'CAPEX_NOAK': levelized_CAPEX,
            'Transport_EUR_tCO2': transport_cost,
            'OPEX_EUR_tCO2': OPEXE,
            'Total_FOAK': total_cost_foak,
            'Total_NOAK': total_cost_noak
        })

scunthorpe_total = scunthorpe_emitters['CO2'].sum()
chp_stack = scunthorpe_emitters[scunthorpe_emitters['Site']=='Scunthorpe Power Station']['CO2'].iloc[0]
chp_stack = ["chp", chp_stack, 0.296] # [tCO2/yr, fraction of CO2]
stove_stack = scunthorpe_emitters[scunthorpe_emitters['Site']=='Scunthorpe Blast Furnaces']['CO2'].iloc[0] # Assuming this CO2 is emitted from e.g. hot stoves (that connect to the BF)
stove_stack = ["stove", stove_stack, 0.251] # [tCO2/yr, fraction of CO2]
sinter_stack = scunthorpe_emitters[scunthorpe_emitters['Site']=='Scunthorpe Sinter']['CO2'].iloc[0] # Sintering 15%CO2 ish: https://www.sciencedirect.com/science/article/pii/S095965262402852X 
sinter_stack = ["sinter", sinter_stack, 0.15] # [tCO2/yr, fraction of CO2]
scunthorpe_residual = scunthorpe_total - chp_stack[1] - stove_stack[1] - sinter_stack[1] # [tCO2/yr]

for stack in [chp_stack, stove_stack, sinter_stack]:
    FLH = 8400 # [h/y]
    xCO2 = stack[2] # [-]
    mCO2 = stack[1] #[tCO2/yr]

    mCO2 = mCO2 / FLH # [tCO2/h]
    nCO2 = mCO2*1000 / 44 # [kmolCO2/h]
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
    CAPEX = CAPEX * 5.509 # [MEUR] NETL Methodology
    CEPCI_2025 = 930
    CEPCI_2023 = 798.7
    CAPEX = CAPEX * CEPCI_2025 / CEPCI_2023

    capture_rate = 0.95 # [0-1]
    annualized_CAPEX = CAPEX * 0.07 * (1 + 0.07)**25 / ((1 + 0.07)**25 - 1) *10**6# [EUR/y]
    levelized_CAPEX = annualized_CAPEX / (mCO2*FLH*capture_rate) # [EUR/tCO2]
    levelized_CAPEX_FOAK = 1.7553 * levelized_CAPEX # [EUR/tCO2]

    # Calculate transport cost
    pounds_to_EUR = 1.15
    scunthorpe_power_station = scunthorpe_emitters[scunthorpe_emitters['Site']=='Scunthorpe Power Station'].iloc[0]
    transport_result = find_nearest_ccs_site(scunthorpe_power_station, transport_costs, debug=False)
    transport_cost = transport_result['total_cost_per_t'] * pounds_to_EUR
    transport_cost = transport_cost * CEPCI_2025 / CEPCI_2023

    # Estimate energy OPEX (industry)
    Qreb = 3.5 * mCO2*capture_rate*1000/3600 # [MW]
    Qsteam = 0.15 * Qreb # [MW]
    Qelec = 0.30 * Qreb # [MW]
    Qchp = 0.55 * Qreb # [MW]

    cost_steam = 4.1 # [EUR/tsteam @130C] [Ali]
    evaporation_enthalpy = 2257 # [MJ/tsteam]
    cost_steam = cost_steam / evaporation_enthalpy # [EUR/MJ]
    Qsteam = Qsteam * FLH * 3600 # [MJ/y]
    OPEX_steam = cost_steam * Qsteam # [EUR/y]        
    OPEX_steam = OPEX_steam / (stack[1]*capture_rate) # [EUR/tCO2]

    celc = 80 # [EUR/MWh electricity]
    Pelec = Qelec * 0.33 * FLH # [MWh/y]
    OPEX_elec = celc * Pelec # [EUR/y]
    OPEX_elec = OPEX_elec / (stack[1]*capture_rate) # [EUR/tCO2]

    cbio = 65 # [EUR/MWh biomass]
    Qchp = Qchp * FLH # [MWh/y]
    OPEX_chp = cbio * Qchp # [EUR/y]
    OPEX_chp = OPEX_chp / (stack[1]*capture_rate) # [EUR/tCO2]

    OPEXE = OPEX_steam + OPEX_elec + OPEX_chp
   
    total_cost_foak = levelized_CAPEX_FOAK + transport_cost + OPEXE
    total_cost_noak = levelized_CAPEX + transport_cost + OPEXE
        
    # Store results
    results.append({
        'Plant': "Scunthorpe",
        'Stack': stack[0],
        'CO2_kt_y': stack[1]/1000,
        'FLH_h': FLH,
        'CAPEX_FOAK': levelized_CAPEX_FOAK,
        'CAPEX_NOAK': levelized_CAPEX,
        'Transport_EUR_tCO2': transport_cost,
        'OPEX_EUR_tCO2': OPEXE,
        'Total_FOAK': total_cost_foak,
        'Total_NOAK': total_cost_noak
    })


# Create and display results table
if results:
    print(f"{'Plant Name':<30} | {'CO2':<8} | {'FLH':<6} | {'FOAK':<8} | {'NOAK':<8} | {'Transport':<9} | {'OPEX':<8} | {'TotalFOAK':<9} | {'TotalNOAK':<9}")
    print(f"{'':<30} | {'kt/y':<8} | {'h/y':<6} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<9} | {'EUR/tCO2':<8} | {'EUR/tCO2':<9} | {'EUR/tCO2':<9}")
    print("-" * 120)
    
    for result in results:
        print(f"{result['Plant']:<30} | {result['CO2_kt_y']:>8.1f} | {result['FLH_h']:>6.0f} | {result['CAPEX_FOAK']:>8.1f} | {result['CAPEX_NOAK']:>8.1f} | {result['Transport_EUR_tCO2']:>9.1f} | {result['OPEX_EUR_tCO2']:>8.1f} | {result['Total_FOAK']:>9.1f} | {result['Total_NOAK']:>9.1f}")
    
    # Summary statistics
    print("-" * 120)
    avg_capture_foak = np.mean([r['CAPEX_FOAK'] for r in results])
    avg_capture_noak = np.mean([r['CAPEX_NOAK'] for r in results])
    avg_transport = np.mean([r['Transport_EUR_tCO2'] for r in results])
    avg_opex = np.mean([r['OPEX_EUR_tCO2'] for r in results])
    avg_total_foak = np.mean([r['Total_FOAK'] for r in results])
    avg_total_noak = np.mean([r['Total_NOAK'] for r in results])
    
    print(f"{'AVERAGE':<30} | {'':<6} | {'':<8} | {'':<6} | {avg_capture_foak:>8.1f} | {avg_capture_noak:>8.1f} | {avg_transport:>9.1f} | {avg_opex:>8.1f} | {avg_total_foak:>9.1f} | {avg_total_noak:>9.1f}")
    
    print(f"\nSummary:")
    print(f"Number of plant-stacks analyzed: {len(results)}")
    print(f"Total CO2 emissions: {sum([r['CO2_kt_y'] for r in results]):,.1f} ktCO2/y")
    print(f"Average total CCS cost (FOAK): {avg_total_foak:.1f} EUR/tCO2")
    print(f"Average total CCS cost (NOAK): {avg_total_noak:.1f} EUR/tCO2")
    print(f"Cost range (FOAK): {min([r['Total_FOAK'] for r in results]):.1f} - {max([r['Total_FOAK'] for r in results]):.1f} EUR/tCO2")
    print(f"Cost range (NOAK): {min([r['Total_NOAK'] for r in results]):.1f} - {max([r['Total_NOAK'] for r in results]):.1f} EUR/tCO2")

print("\nWe are missing 1 ethylene (petrochemical) plant named Fife.")
    