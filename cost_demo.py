import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Check this list also: https://ember-energy.org/latest-insights/the-largest-emitters-in-the-uk-annual-review/
print("This script calculates the costs of: ")
print("1) CCGT - Pembroke Power Station")
print("2) CCGT - Rocksavage Power")
print("3) Refinery - Fawley")
print("4) Refinery - Lindsey")
print("5) Cement - Hope")
print("6) Cement - Cauldon")
print("7) Steel - Scunthorpe")
print("8) Petrochemical - Billingham")
print("9) Petrochemical - Grangemouth")
print("10) W2E - Runcorn")
print("11) W2E - Dudley")
print("12) Biopower - Drax")

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

# Calculate transport costs for all top 60 emitters
transport_costs = pd.read_csv("data/nzip_balanced_scenario_results.csv", encoding='latin-1')
transport_costs = transport_costs[transport_costs["CO2 Pipeline/Trucking?"] != "No CCS"] 
for idx, plant in top_60_emitters.iterrows():
    result = find_nearest_ccs_site(plant, transport_costs, discount_rate=0.035, lifetime=25, debug=False)
    plant['transport_cost'] = result['total_cost_per_t'] # [£/tCO2]

# Create DataFrame with only power producers from top 60 emitters
power_producers = top_60_emitters[(top_60_emitters['Sector'] == 'Major power producers') | (top_60_emitters['Sector'] == 'Minor power producers')].copy()

# Explore CAPEX relationships [Kim & Léonard, 2025]
# TEC = a + (b * xCO2**n + c) * V**m [MEUR, 2023] - note that this is not valid for CCGT if xCO2<5%, or if mCO2>1250 kt/y
a = 2.1673
b = 0.8092
c = -0.00332
n = 0.5291
m = 0.8391

# Manual capacity data for specific plants
capacity_data = {
    'Pembroke Power Station': 2200,
    'Staythorpe Power Station': 1828,
    'Didcot B': 1450,
    'West Burton B CCGT Power Station': 1332,
    'Saltend': 1200,
    'Immingham': 1252, # "VPI" in DUKES 5.11 data
    'Marchwood  Power  Station': 898,
    'Carrington CCGT': 884, 
    'Seabank': 1234,
    'Grain CCGT': 1517,
    'South Humber': 1365, # "EP SHB LTD" in data
    'Spalding': 950,
    'Connahs Quay Power Station': 1380,
    'Peterhead Power Station': 1180,
    'Keadby': 735,
    'Langage Energy Centre': 905,
    'Kilroot': 700, # Missing in DUKES 5.11 data, taken from Google
    'Ballylumford': 616,
    'Damhead Creek Power Station': 805,
    'Little Barford': 735,
    'Enfield Power Station': 408,
    'Coolkeeragh': 413,
    'Medway': 755,
    'Kings Lynn Power Station': 395,
    'Shoreham Power Station': 420,
    'Great Yarmouth Power Station': 420,
    'Cottam Development Centre': 455,
    'Coryton': 800,
    'Rye House': 715,
    'Rocksavage': 810,
}

# Calculate costs for all power producers with capacity data
print(f"\n" + "="*100)
print("CCS COST ANALYSIS FOR POWER PRODUCERS")
print("="*100)

# Initialize results storage
results = []

for plant_name, capacity in capacity_data.items():
    # Find the plant in the power producers
    plant_mask = power_producers['Site'] == plant_name
    if not plant_mask.any():
        print(f"Plant not found: {plant_name}")
        continue
    
    plant = power_producers[plant_mask].iloc[0]
    
    # Calculate costs using the same methodology
    xCO2 = 0.05 # assume CCGT plants
    Pinstalled = capacity # MW
    eta_P = 0.49 # [MWel/MWfuel] total efficiency from DUKES data
    Qfuel = Pinstalled / eta_P # [MWfuel]
    emission_factor = 0.204 # tCO2/MWhfuel [NZIP, 2020]
    FLH = plant['CO2'] / (Qfuel * emission_factor) # [h/y] = tCO2/yr / (tCO2/h)

    mCO2 = Qfuel * emission_factor # [tCO2/h] when at full load
    nCO2 = mCO2*1000 / 44 # [kmolCO2/h]
    n_fluegas = nCO2 / xCO2 # [kmol/h]
    V_fluegas = n_fluegas * 22.4 # [Nm3/h]

    TEC = a + (b * (xCO2)**n + c) * (V_fluegas/1000)**m # NOTE xCO2 as fraction, not percentage
    CAPEX = TEC * 5.509 # [MEUR] NETL Methodology

    capture_rate = 0.95 # [0-1]
    annualized_CAPEX = CAPEX * 0.07 * (1 + 0.07)**25 / ((1 + 0.07)**25 - 1) *10**6# [EUR/y]
    levelized_CAPEX = annualized_CAPEX / (plant['CO2']*capture_rate) # [EUR/tCO2]
    levelized_CAPEX_FOAK = 1.7553 * levelized_CAPEX # [EUR/tCO2]

    # Calculate transport cost
    pounds_to_EUR = 1.15
    transport_result = find_nearest_ccs_site(plant, transport_costs, debug=False)
    transport_cost = transport_result['total_cost_per_t'] * pounds_to_EUR

    # Estimate energy OPEX (CCGT)
    Pgas = Qfuel * 0.35 # [MW] Harvey GT lecture, and about 10% are lost as <120C flue gas heat
    Qsteam = Qfuel * 0.51 # [MW] 
    Prankine = Pinstalled - Pgas
    eta_steam = Prankine/Qsteam # [-]

    Qreb = 3.5 * mCO2*1000/3600 # [MW]
    Plost = Qreb * eta_steam # [MW]
    Prankine = Prankine - Plost # [MW]

    Plost = Plost * FLH # [MWh/y]
    OPEXE = Plost * 80*pounds_to_EUR # [EUR/y] assumed electricity price
    OPEXE = OPEXE / (plant['CO2']*capture_rate) # [EUR/tCO2]
    
    total_cost_foak = levelized_CAPEX_FOAK + transport_cost + OPEXE
    total_cost_noak = levelized_CAPEX + transport_cost + OPEXE
    
    # Store results
    results.append({
        'Plant': plant_name,
        'Capacity_MW': capacity,
        'CO2_kt_y': plant['CO2']/1000,
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
    print(f"{'Plant Name':<30} | {'Cap':<6} | {'CO2':<8} | {'FLH':<6} | {'FOAK':<8} | {'NOAK':<8} | {'Transport':<9} | {'OPEX':<8} | {'TotalFOAK':<9} | {'TotalNOAK':<9}")
    print(f"{'':<30} | {'MW':<6} | {'kt/y':<8} | {'h/y':<6} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<9} | {'EUR/tCO2':<8} | {'EUR/tCO2':<9} | {'EUR/tCO2':<9}")
    print("-" * 120)
    
    for result in results:
        print(f"{result['Plant']:<30} | {result['Capacity_MW']:>6.0f} | {result['CO2_kt_y']:>8.1f} | {result['FLH_h']:>6.0f} | {result['CAPEX_FOAK']:>8.1f} | {result['CAPEX_NOAK']:>8.1f} | {result['Transport_EUR_tCO2']:>9.1f} | {result['OPEX_EUR_tCO2']:>8.1f} | {result['Total_FOAK']:>9.1f} | {result['Total_NOAK']:>9.1f}")
    
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
    print(f"Number of plants analyzed: {len(results)}")
    print(f"Total capacity: {sum([r['Capacity_MW'] for r in results]):,.0f} MW")
    print(f"Total CO2 emissions: {sum([r['CO2_kt_y'] for r in results]):,.1f} ktCO2/y")
    print(f"Average total CCS cost (FOAK): {avg_total_foak:.1f} EUR/tCO2")
    print(f"Average total CCS cost (NOAK): {avg_total_noak:.1f} EUR/tCO2")
    print(f"Cost range (FOAK): {min([r['Total_FOAK'] for r in results]):.1f} - {max([r['Total_FOAK'] for r in results]):.1f} EUR/tCO2")
    print(f"Cost range (NOAK): {min([r['Total_NOAK'] for r in results]):.1f} - {max([r['Total_NOAK'] for r in results]):.1f} EUR/tCO2")
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv('ccs_cost_analysis_results.csv', index=False)
    print(f"\nResults saved to: ccs_cost_analysis_results.csv")

print(f"\nPower producers in top 60 emitters:")
print(f"Number of plants: {len(power_producers)}")
print(f"Total CO2 emissions: {power_producers['CO2'].sum():,.0f} tCO2/yr")
print(f"Percentage of top 60: {power_producers['CO2'].sum() / top_60_emitters['CO2'].sum() * 100:.1f}%")
