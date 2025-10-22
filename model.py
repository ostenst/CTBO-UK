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

def levelize_MEUR(CAPEX, annual_CO2, capture_rate=0.95, discount_rate=0.07, lifetime=25):
    annualized_CAPEX = CAPEX * discount_rate * (1 + discount_rate)**lifetime / ((1 + discount_rate)**lifetime - 1) *10**6 # [EUR/y]
    levelized_CAPEX = annualized_CAPEX / (annual_CO2*capture_rate) # [EUR/tCO2]
    return levelized_CAPEX
    
def energy_supply(mCO2, capture_rate, FLH, qreb, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, evaporation_enthalpy=2257):
                # Estimate energy penalty
                mCO2_captured = mCO2 * capture_rate # [tCO2/h] 
                mCO2_residual = mCO2 * (1-capture_rate) # [tCO2/h] 

                Qreb = qreb * mCO2_captured*1000/3600 # [MW]
                Qsteam = qsteam * Qreb # [MW]
                Qelc = qelc * Qreb # [MW]
                Qchp = qchp * Qreb # [MW]

                # Steam OPEX
                cost_steam = csteam / evaporation_enthalpy # [EUR/MJ]
                Qsteam = Qsteam * FLH * 3600 # [MJ/y]
                OPEX_steam = cost_steam * Qsteam # [EUR/y]        
                OPEX_steam = OPEX_steam / (annual_CO2*capture_rate) # [EUR/tCO2]

                # Electricity OPEX
                Pelec = Qelc * elc_eff * FLH # [MWh/y]
                OPEX_elec = celc * Pelec # [EUR/y]
                OPEX_elec = OPEX_elec / (annual_CO2*capture_rate) # [EUR/tCO2]

                # Biomass CHP OPEX (no CAPEX currently, and the CO2 is not captured)
                Qchp = Qchp * FLH # [MWh/y]
                OPEX_chp = cbio * Qchp # [EUR/y]
                OPEX_chp = OPEX_chp / (annual_CO2*capture_rate) # [EUR/tCO2]

                OPEXE = OPEX_steam + OPEX_elec + OPEX_chp
                CAPEX_CHP = 0
                biogenic_fraction = 0.1
                return mCO2_captured, biogenic_fraction, mCO2_residual, OPEXE, CAPEX_CHP

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
results = []
xCO2 = 0.05 # assume CCGT plants
eta_P = 0.49 # [MWel/MWfuel] total efficiency from DUKES data
emission_factor = 0.204 # tCO2/MWhfuel [NZIP, 2020]
FOAK = 1.7553 # [-] check foak.py for details
capture_rate = 0.95 # [-]
discount_rate = 0.07 # [-]
lifetime = 25 # [y]
gas_eff = 0.35 # [-] Harvey GT lecture, and about 10% are lost as <120C flue gas heat
steam_eff = 0.51 # [-]
qreb = 3.5 # [MJ/kg]
celc = 80 # [EUR/MWh]

power_capacities = pd.read_csv("data/power_capacities_clean.csv")
power_producers = largest_plants[(largest_plants['Sector'] == 'Major power producers') | (largest_plants['Sector'] == 'Minor power producers')].copy()
power_producers = power_producers[power_producers['Site'] != 'Ratcliffe on Soar Power Station'].copy() # Decommissioned coal plant
power_producers = power_producers.merge(power_capacities, left_on='Site', right_on='Power plant name', how='left')

for idx, plant in power_producers.iterrows():
    # Estimate CAPEX from CO2 flow [tCO2/yr] and concentration [%]
    Pinstalled = plant['Capacity [MW]'] # [MWel]
    Qfuel = Pinstalled / eta_P # [MWf]
    mCO2 = Qfuel * emission_factor # [tCO2/h]
    FLH = plant['CO2'] / mCO2 # [h/y] 

    CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
    levelized_CAPEX = levelize_MEUR(CAPEX, plant['CO2'], capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
    CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
    CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2] 

    # Estimate energy penalty
    mCO2_captured = mCO2 * capture_rate # [tCO2/h]
    mCO2_residual = mCO2 * (1-capture_rate) # [tCO2/h]

    Pgas = Qfuel * gas_eff # [MWel]
    Qsteam = Qfuel * steam_eff # [MWth] 
    Prankine = Pinstalled - Pgas
    eta_steam = Prankine/Qsteam # [-]

    Qreb = qreb * mCO2_captured*1000/3600 # [MW]
    Plost = Qreb * eta_steam # [MW]
    Prankine = Prankine - Plost # [MW]

    Plost = Plost * FLH # [MWh/y]
    OPEXE = Plost * celc # [EUR/y] 
    OPEXE = OPEXE / (mCO2_captured*FLH) # [EUR/tCO2]

    results.append({
        'site-stack': plant['Site'] + '-' + 'CCGT',
        'annual_CO2': plant['CO2']/1000, # [ktCO2/yr]
        'captured_CO2': mCO2_captured*FLH/1000, # [ktCO2/yr]
        'biogenic_fraction': 0, # [-] 
        'residual_CO2': mCO2_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
        'FLH': FLH, # [h/y]
        'CAPEX_4OAK': CAPEX_4OAK, # [EUR/tCO2]
        'CAPEX_FOAK': CAPEX_FOAK, # [EUR/tCO2]
        'OPEXE': OPEXE, # [EUR/tCO2]
        'transtorage': plant['transport_cost'], # [EUR/tCO2]
        'total_4OAK': CAPEX_4OAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
        'total_FOAK': CAPEX_FOAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
    })

# ------------INDUSTRY SECTOR------------
FLH = 8500 # [h/y]
qsteam = 0.15 # [-] fraction of Qreb covered by steam
qelc = 0.30 # [-] fraction of Qreb covered by electricity
qchp = 0.55 # [-] fraction of Qreb covered by biomass CHP
csteam = 4.1 # [EUR/tsteam @130C] [Ali]
evaporation_enthalpy = 2257 # [MJ/tsteam]
elc_eff = 0.33 # [-] efficiency of electrified reboiler
cbio = 65 # [EUR/MWh biomass]

refineries = largest_plants[(largest_plants['Sector'] == 'Processing & distribution of petroleum products')].copy()
scunthorpe_stacks = largest_plants[largest_plants['Site'].str.startswith('Scunthorpe', na=False)].copy()
# chp_stack = scunthorpe_stacks[scunthorpe_stacks['Site']=='Scunthorpe Power Station'].copy()
# stove_stack = scunthorpe_stacks[scunthorpe_stacks['Site']=='Scunthorpe Blast Furnaces'].copy()
# sinter_stack = scunthorpe_stacks[scunthorpe_stacks['Site']=='Scunthorpe Sinter'].copy()
# scunthorpe_total = scunthorpe_stacks['CO2'].sum()
cement_plants = largest_plants[largest_plants['Sector'] == 'Cement'].copy()

industrial_plants = pd.concat([
    refineries,
    scunthorpe_stacks,
    cement_plants
], ignore_index=True)


for idx, plant in industrial_plants.iterrows():
    # Determine the number of stacks and calculate their CCS cost
    if plant['Sector'] == 'Processing & distribution of petroleum products':
        annual_CO2 = plant['CO2']
        stacks = {
            'power': [annual_CO2 * 0.298, (3*25 + 8*54)/(25+54)/100], # [tCO2/yr, fraction of CO2]
            'crackers': [annual_CO2 * 0.20, 17/100],
            'distillation': [annual_CO2 * 0.17, 11/100],
            'smr': [annual_CO2 * 0.118, (8*6 + 24*26)/(6+26)/100],
            'remaining': [annual_CO2 * 0.188, 8/100],
        }
        for stack_name, stack_data in stacks.items():
            annual_CO2 = stack_data[0]  # [tCO2/yr]
            xCO2 = stack_data[1]  # [-]
            mCO2 = annual_CO2 / FLH  # [tCO2/h]

            CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
            levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2, capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
            CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
            CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2]

            mCO2_captured, biogenic_fraction, mCO2_residual, OPEXE, CAPEX_CHP = energy_supply(mCO2, capture_rate, FLH, qreb, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, evaporation_enthalpy=2257)
            
            results.append({
                'site-stack': plant['Site'] + '-' + stack_name,
                'annual_CO2': annual_CO2/1000, # [ktCO2/yr]
                'captured_CO2': mCO2_captured*FLH/1000, # [ktCO2/yr]
                'biogenic_fraction': biogenic_fraction, # [-] 
                'residual_CO2': mCO2_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
                'FLH': FLH, # [h/y]
                'CAPEX_4OAK': CAPEX_4OAK, # [EUR/tCO2]
                'CAPEX_FOAK': CAPEX_FOAK, # [EUR/tCO2]
                'OPEXE': OPEXE, # [EUR/tCO2]
                'transtorage': plant['transport_cost'], # [EUR/tCO2]
                'total_4OAK': CAPEX_4OAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
                'total_FOAK': CAPEX_FOAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
            })

    if plant['Sector'] == 'Iron & steel industries':
        if plant['Site'] == 'Scunthorpe Power Station':
            stack_name = 'chp'
            annual_CO2 = plant['CO2']  # [tCO2/yr]
            xCO2 = 0.296  # [-]
        if plant['Site'] == 'Scunthorpe Blast Furnaces':
            stack_name = 'stove'
            annual_CO2 = plant['CO2']  
            xCO2 = 0.251  
        if plant['Site'] == 'Scunthorpe Sinter':
            stack_name = 'sinter'
            annual_CO2 = plant['CO2']  
            xCO2 = 0.15  
        mCO2 = annual_CO2 / FLH  # [tCO2/h]
        
        CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
        levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2, capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
        CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
        CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2]

        mCO2_captured, biogenic_fraction, mCO2_residual, OPEXE, CAPEX_CHP = energy_supply(mCO2, capture_rate, FLH, qreb, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, evaporation_enthalpy=2257)
            
        results.append({
            'site-stack': plant['Site'] + '-' + stack_name,
            'annual_CO2': annual_CO2/1000, # [ktCO2/yr]
            'captured_CO2': mCO2_captured*FLH/1000, # [ktCO2/yr]
            'biogenic_fraction': biogenic_fraction, # [-] 
            'residual_CO2': mCO2_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
            'FLH': FLH, # [h/y]
            'CAPEX_4OAK': CAPEX_4OAK, # [EUR/tCO2]
            'CAPEX_FOAK': CAPEX_FOAK, # [EUR/tCO2]
            'OPEXE': OPEXE, # [EUR/tCO2]
            'transtorage': plant['transport_cost'], # [EUR/tCO2]
            'total_4OAK': CAPEX_4OAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
            'total_FOAK': CAPEX_FOAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
            })

    if plant['Sector'] == 'Cement':
        annual_CO2 = plant['CO2']  # [tCO2/yr]
        xCO2 = 0.20  # [-]
        mCO2 = annual_CO2 / FLH  # [tCO2/h]
        
        CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
        levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2, capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
        CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
        CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2]
        
        mCO2_captured, biogenic_fraction, mCO2_residual, OPEXE, CAPEX_CHP = energy_supply(mCO2, capture_rate, FLH, qreb, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, evaporation_enthalpy=2257)

        results.append({
            'site-stack': plant['Site'] + '-' + 'cement',
            'annual_CO2': annual_CO2/1000, # [ktCO2/yr]
            'captured_CO2': mCO2_captured*FLH/1000, # [ktCO2/yr]
            'biogenic_fraction': biogenic_fraction, # [-] 
            'residual_CO2': mCO2_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
            'FLH': FLH, # [h/y]
            'CAPEX_4OAK': CAPEX_4OAK, # [EUR/tCO2]
            'CAPEX_FOAK': CAPEX_FOAK, # [EUR/tCO2]
            'OPEXE': OPEXE, # [EUR/tCO2]
            'transtorage': plant['transport_cost'], # [EUR/tCO2]
            'total_4OAK': CAPEX_4OAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
            'total_FOAK': CAPEX_FOAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
            })

# Create marginal abatement cost curve
if results:
    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    
    # Sort by total_4OAK cost (ascending order for MACC)
    results_df = results_df.sort_values('total_4OAK')
    
    # Calculate cumulative captured CO2
    results_df['cumulative_captured_CO2'] = results_df['captured_CO2'].cumsum()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['cumulative_captured_CO2'], results_df['total_4OAK'], 
             'o-', linewidth=2, markersize=6, color='blue', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Cumulative Captured CO2 (ktCO2/yr)', fontsize=14)
    plt.ylabel('Marginal Abatement Cost (EUR/tCO2)', fontsize=14)
    plt.title('Marginal Abatement Cost Curve (4OAK)', fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Format axes
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    
    # Add some statistics
    total_captured = results_df['cumulative_captured_CO2'].iloc[-1]
    avg_cost = results_df['total_4OAK'].mean()
    min_cost = results_df['total_4OAK'].min()
    max_cost = results_df['total_4OAK'].max()
    
    plt.text(0.02, 0.98, f'Total Captured: {total_captured:.0f} ktCO2/yr\n'
                         f'Average Cost: {avg_cost:.1f} EUR/tCO2\n'
                         f'Cost Range: {min_cost:.1f} - {max_cost:.1f} EUR/tCO2',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    print(f"\nMarginal Abatement Cost Curve Summary:")
    print(f"Total CO2 capture potential: {total_captured:.0f} ktCO2/yr")
    print(f"Average abatement cost: {avg_cost:.1f} EUR/tCO2")
    print(f"Cost range: {min_cost:.1f} - {max_cost:.1f} EUR/tCO2")

    # Create FOAK MACC curve
    # Sort by FOAK cost for proper MACC ordering
    results_df_foak = results_df.sort_values('total_FOAK').copy()
    results_df_foak['cumulative_captured_CO2_foak'] = results_df_foak['captured_CO2'].cumsum()
    
    plt.figure(figsize=(12, 8))
    plt.plot(results_df_foak['cumulative_captured_CO2_foak'], results_df_foak['total_FOAK'], 
             's-', linewidth=2, markersize=6, color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Cumulative Captured CO2 (ktCO2/yr)', fontsize=14)
    plt.ylabel('Marginal Abatement Cost (EUR/tCO2)', fontsize=14)
    plt.title('Marginal Abatement Cost Curve (FOAK)', fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Format axes
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    
    # Add FOAK statistics
    total_captured_foak = results_df_foak['cumulative_captured_CO2_foak'].iloc[-1]
    avg_cost_foak = results_df_foak['total_FOAK'].mean()
    min_cost_foak = results_df_foak['total_FOAK'].min()
    max_cost_foak = results_df_foak['total_FOAK'].max()
    
    plt.text(0.02, 0.98, f'Total Captured: {total_captured_foak:.0f} ktCO2/yr\n'
                         f'Average Cost: {avg_cost_foak:.1f} EUR/tCO2\n'
                         f'Cost Range: {min_cost_foak:.1f} - {max_cost_foak:.1f} EUR/tCO2',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    print(f"\nFOAK Marginal Abatement Cost Curve Summary:")
    print(f"Total CO2 capture potential: {total_captured_foak:.0f} ktCO2/yr")
    print(f"Average abatement cost: {avg_cost_foak:.1f} EUR/tCO2")
    print(f"Cost range: {min_cost_foak:.1f} - {max_cost_foak:.1f} EUR/tCO2")

# Create and display results table
if results:
    print(f"{'Site-Stack':<30} | {'annual_CO2':<8} | {'captured_CO2':<8} | {'biogenic_fraction':<8} | {'residual_CO2':<8} | {'FLH':<6} | {'CAPEX_4OAK':<8} | {'CAPEX_FOAK':<8} | {'OPEXE':<8} | {'transtorage':<8} | {'total_4OAK':<8} | {'total_FOAK':<8}")
    print(f"{'':<30} | {'ktCO2/yr':<8} | {'ktCO2/yr':<8} | {' - ':<8} | {'ktCO2/yr':<8} | {'h/y':<6} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8}")
    print("-" * 150)
    for result in results:
        print(f"{result['site-stack']:<30} | {result['annual_CO2']:>8.1f} | {result['captured_CO2']:>8.1f} | {result['biogenic_fraction']:>8.1f} | {result['residual_CO2']:>8.1f} | {result['FLH']:>6.0f} | {result['CAPEX_4OAK']:>8.1f} | {result['CAPEX_FOAK']:>8.1f} | {result['OPEXE']:>8.1f} | {result['transtorage']:>8.1f} | {result['total_4OAK']:>8.1f} | {result['total_FOAK']:>8.1f}")
    print("-" * 150)
    print(f"Total CO2 emissions: {sum([r['annual_CO2'] for r in results]):,.1f} ktCO2/y")
    print(f"Total captured CO2: {sum([r['captured_CO2'] for r in results]):,.1f} ktCO2/y")

# Assume CO2 emissions based on DESNZ projections and CCC scenario
# https://assets.publishing.service.gov.uk/media/6604460f91a320001a82b0fd/uk-greenhouse-gas-emissions-provisional-figures-statistical-release-2023.pdf
# https://assets.publishing.service.gov.uk/media/675c0ca798302e574b915336/eep-report-2023-2050.pdf
# https://www.theccc.org.uk/publication/the-seventh-carbon-budget/

# "Under EEP-ready policies only, emissions [all CO2eq] are projected to fall by 23% between 2022 and 2050."
# In the 7th carbon budget, CO2 emissions fall to negative -40 MtCO2/yr by 2050
CO2_2023 = 300 # [MtCO2/yr]
CO2_plants = 1 # [MtCO2/yr]
plt.show()

print("TODO: Move the Grangemouth Power Station from the refineries to the power sector")
print("TODO: Remove the duplicate Fawley refinery (maybe also move to power sector?)")
print("TODO: Also capture the CO2 from biomass CHP - add to captured volumes and CAPEX!")