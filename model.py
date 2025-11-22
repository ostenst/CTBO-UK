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
    
def energy_supply(mCO2, capture_rate, FLH, qreb, pcompr, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, emission_factor, evaporation_enthalpy=2257, maximize_beccs=False):
                # Estimate energy penalty
                mCO2f_captured = mCO2 * capture_rate # [tCO2/h] 
                mCO2f_residual = mCO2 * (1-capture_rate) # [tCO2/h] 

                Qreb = qreb * mCO2f_captured*1000/3600 # [MW]
                Pcompr = pcompr * mCO2f_captured*1000/3600 # [MW]

                # Assuming heat demand is covered by these fractions
                Qsteam = qsteam * Qreb # [MW]
                Qelc = qelc * Qreb # [MW]
                Qchp = qchp * Qreb # [MW]

                # Steam OPEX
                cost_steam = csteam / evaporation_enthalpy # [EUR/MJ]
                Qsteam = Qsteam * FLH * 3600 # [MJ/y]
                OPEX_steam = cost_steam * Qsteam # [EUR/y]        
                OPEX_steam = OPEX_steam / (annual_CO2*capture_rate) # [EUR/tCO2]

                # Electricity OPEX
                Pelec = (Qelc * elc_eff + Pcompr)* FLH # [MWh/y]
                OPEX_elec = celc * Pelec # [EUR/y]
                OPEX_elec = OPEX_elec / (annual_CO2*capture_rate) # [EUR/tCO2]

                # Biomass CHP OPEX (simplify: all CHP goes to heat, but we capture the self-generated CO2) 
                if maximize_beccs:
                    Qfuel = Qchp / (1 - capture_rate*emission_factor*(qreb/3600*1000)) # Set up this equation for the heat part, where Qfuel_heat_tot=qchp*Qreb+Qselfcapture
                    OPEX_chp = cbio * Qfuel * FLH # [EUR/y]
                    mCO2bio_captured = Qfuel*emission_factor*capture_rate # [tCO2/h]
                else:
                    Qfuel = Qchp # [MW]
                    OPEX_chp = cbio * Qfuel * FLH # [EUR/y]
                    mCO2bio_captured = 0
                OPEX_chp = OPEX_chp / (annual_CO2*capture_rate) # [EUR/tCO2]

                OPEXE = OPEX_steam + OPEX_elec + OPEX_chp
                return mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel # Includes biogenic captured CO2 and Qchp for CHPCAPEX

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

# Prepare tentative W2E data (lacking transport locations etc)
w2e_plants = pd.read_csv("data/w2e_plants.csv")

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
qreb = 3.5 # [MJ/kgCO2]
pcompr = 0.37 # [MJ/kgCO2] power compression penalty from Tharun plant and system paper
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
    OPEXE += pcompr * 1000/3600*celc # [MJ/kgCO2 => MWh/tCO2 => EUR/tCO2] power compression penalty from Tharun plant and system paper

    results.append({
        'site-stack': plant['Site'] + '-' + 'CCGT',
        'annual_CO2': plant['CO2']/1000, # [ktCO2/yr]
        'biogenic': 0, # [-] fraction of baseline
        'captured_CO2f': mCO2_captured*FLH/1000, # [ktCO2/yr]
        'captured_CO2bio': 0, # [ktCO2/yr] 
        'residual_CO2f': mCO2_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
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
emission_factor_bio = 0.3318 # tCO2/MWhfuel biomass [NZIP, 2020]

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

            mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(mCO2, capture_rate, FLH, qreb, pcompr, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, emission_factor_bio, evaporation_enthalpy=2257)
            mCO2 = mCO2f_captured + mCO2bio_captured

            CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] NOTE the capture rate/CO2%(if biomass chp) inconsistency
            levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2, capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
            CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
            CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2]

            results.append({
                'site-stack': plant['Site'] + '-' + stack_name,
                'annual_CO2': annual_CO2/1000, # [ktCO2/yr]
                'biogenic': 0, # [-] fraction of baseline
                'captured_CO2f': mCO2f_captured*FLH/1000, # [ktCO2/yr]
                'captured_CO2bio': mCO2bio_captured*FLH/1000, # [ktCO2/yr] 
                'residual_CO2f': mCO2f_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
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
        
        mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(mCO2, capture_rate, FLH, qreb, pcompr, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, emission_factor_bio,evaporation_enthalpy=2257)
        mCO2 = mCO2f_captured + mCO2bio_captured

        CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
        levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2, capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
        CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
        CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2]
 
        results.append({
            'site-stack': plant['Site'] + '-' + stack_name,
            'annual_CO2': annual_CO2/1000, # [ktCO2/yr]
            'biogenic': 0, # [-] fraction of baseline
            'captured_CO2f': mCO2f_captured*FLH/1000, # [ktCO2/yr]
            'captured_CO2bio': mCO2bio_captured*FLH/1000, # [ktCO2/yr] 
            'residual_CO2f': mCO2f_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
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
        
        mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(mCO2, capture_rate, FLH, qreb, pcompr, qsteam, qelc, qchp, csteam, elc_eff, cbio, celc, emission_factor_bio,evaporation_enthalpy=2257)
        mCO2 = mCO2f_captured + mCO2bio_captured

        CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
        levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2, capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
        CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
        CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2]
        
        results.append({
            'site-stack': plant['Site'] + '-' + 'cement',
            'annual_CO2': annual_CO2/1000, # [ktCO2/yr]
            'biogenic': 0, # [-] fraction of baseline
            'captured_CO2f': mCO2f_captured*FLH/1000, # [ktCO2/yr]
            'captured_CO2bio': mCO2bio_captured*FLH/1000, # [ktCO2/yr] 
            'residual_CO2f': mCO2f_residual*FLH/1000, # [ktCO2/yr] count only fossil residuals
            'FLH': FLH, # [h/y]
            'CAPEX_4OAK': CAPEX_4OAK, # [EUR/tCO2]
            'CAPEX_FOAK': CAPEX_FOAK, # [EUR/tCO2]
            'OPEXE': OPEXE, # [EUR/tCO2]
            'transtorage': plant['transport_cost'], # [EUR/tCO2]
            'total_4OAK': CAPEX_4OAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
            'total_FOAK': CAPEX_FOAK+OPEXE+plant['transport_cost'], # [EUR/tCO2]
            })

# ------------W2E SECTOR------------
emission_factor = 0.98 # [tCO2/twaste] Tolvik report
w2e_plants['CO2'] = w2e_plants['Capacity 2023 [ktpa]']*1000 * emission_factor # [tCO2/yr]
FLH = 8760 * 0.866 # [h/y] 
LHV = 9.52 # [MJ/kg]
fossil = 0.465 # fossil content of the CO2 emitted

eta_elec = 605 # [kWh/twaste] *3.6 to get MJ
eta_elec = 2178 # [MJ/twaste]
eta_heat = 110 # [kWh/twaste] *3.6 to get MJ
eta_heat = 396 # [MJ/twaste]
lost_heat = LHV*1000 - eta_elec - eta_heat # [MJ/twaste]

for idx, plant in w2e_plants.iterrows():

    mCO2 = plant['CO2'] / FLH # [tCO2/h]
    xCO2 = 0.11 # [-] NOTE: I must recalculate this for W2E
    CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
    levelized_CAPEX = levelize_MEUR(CAPEX, plant['CO2'], capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
    CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
    CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2] 

    # Estimate energy penalty
    Qreb = qreb * mCO2*capture_rate*1000/3600 # [MW] NOTE: assuming that waste heat is used
    evaporation_enthalpy = 2257 # [MJ/tsteam]
    cost_steam = csteam / evaporation_enthalpy # [EUR/MJ]
    Qsteam = Qreb * FLH * 3600 # [MJ/y] assuming all is covered by recovered steam
    OPEX_steam = cost_steam * Qsteam # [EUR/y]        
    OPEXE = OPEX_steam / (plant['CO2']*capture_rate) # [EUR/tCO2]
    OPEXE += pcompr * 1000/3600*celc # [MJ/kgCO2 => MWh/tCO2 => EUR/tCO2] power compression penalty from Tharun plant and system paper

    # Dummy transport costs
    transport_cost = 30 # [EUR/tCO2] NOTE: must calculate later using geographic distance
    transport_cost = transport_cost * CEPCI_2025 / CEPCI_2023

    results.append({
        'site-stack': plant['Name'] + '-' + 'W2E',
        'annual_CO2': plant['CO2']/1000, # [ktCO2/yr]
        'biogenic': (1-fossil), # [-] fraction of baseline
        'captured_CO2f': mCO2*capture_rate*FLH/1000 * fossil, # [ktCO2/yr]
        'captured_CO2bio': mCO2*capture_rate*FLH/1000 * (1-fossil), # [ktCO2/yr] 
        'residual_CO2f': mCO2*(1-capture_rate)*fossil*FLH/1000, # [ktCO2/yr] count only fossil residuals
        'FLH': FLH, # [h/y]
        'CAPEX_4OAK': CAPEX_4OAK, # [EUR/tCO2]
        'CAPEX_FOAK': CAPEX_FOAK, # [EUR/tCO2]
        'OPEXE': OPEXE, # [EUR/tCO2]
        'transtorage': transport_cost, # [EUR/tCO2]
        'total_4OAK': CAPEX_4OAK+OPEXE+transport_cost, # [EUR/tCO2]
        'total_FOAK': CAPEX_FOAK+OPEXE+transport_cost, # [EUR/tCO2]
    })

# ------------ DRAX BECCS ------------
Drax_CO2 = 11500000 # [tCO2/yr]
Pinstalled = 2580 # [MW]
eta_P = 33/(1-0.24) # ~44% efficiency before a CCS retrofit, which leads to 33%
Qfuel = Pinstalled / (eta_P/100) # [MWfuel]
FLH = Drax_CO2 / (Qfuel * emission_factor) # [h/y] = tCO2/yr / (tCO2/h)

# CAPEX
xCO2 = 0.13 # [-] NOTE: I guessed this
mCO2 = Drax_CO2 / FLH # [tCO2/h]
CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509) # [MEUR] note the capture rate inconsistency
levelized_CAPEX = levelize_MEUR(CAPEX, Drax_CO2, capture_rate=capture_rate, discount_rate=discount_rate, lifetime=lifetime) # [EUR/tCO2]
CAPEX_4OAK = levelized_CAPEX # [EUR/tCO2]
CAPEX_FOAK = FOAK * CAPEX_4OAK # [EUR/tCO2] 

# OPEXE
profit_baseline = Qfuel * eta_P/100 * FLH * celc # [EUR/y]
profit_BECCS = Qfuel * eta_P/100*(1-0.24) * FLH * celc  # [EUR/y]
difference = profit_baseline - profit_BECCS # [EUR/y]
OPEXE = difference / (Drax_CO2*capture_rate) # [EUR/tCO2]
OPEXE += pcompr * 1000/3600*celc # [MJ/kgCO2 => MWh/tCO2 => EUR/tCO2] power compression penalty from Tharun plant and system paper

# Dummy transport costs
transport_cost = 30 # [EUR/tCO2] NOTE: must calculate later using geographic distance
transport_cost = transport_cost * CEPCI_2025 / CEPCI_2023

results.append({
    'site-stack': 'Drax-BECCS',
    'annual_CO2': Drax_CO2/1000, # [ktCO2/yr]
    'biogenic': 1, # [-] fraction of baseline
    'captured_CO2f': 0, # [ktCO2/yr]
    'captured_CO2bio': mCO2*capture_rate*FLH/1000, # [ktCO2/yr] 
    'residual_CO2f': 0, # [ktCO2/yr] count only fossil residuals
    'FLH': FLH, # [h/y]
    'CAPEX_4OAK': CAPEX_4OAK, # [EUR/tCO2]
    'CAPEX_FOAK': CAPEX_FOAK, # [EUR/tCO2]
    'OPEXE': OPEXE, # [EUR/tCO2]
    'transtorage': transport_cost, # [EUR/tCO2]
    'total_4OAK': CAPEX_4OAK+OPEXE+transport_cost, # [EUR/tCO2]
    'total_FOAK': CAPEX_FOAK+OPEXE+transport_cost, # [EUR/tCO2]
})

# For every item in results, add 44 SEK/tCO2 for amine makeup to the total_4OAK and total_FOAK. Assumption from Ramboll.
for result in results:
    result['total_4OAK'] += 44 / 10.96 # [SEK/tCO2] to [EUR/tCO2]
    result['total_FOAK'] += 44 / 10.96

# Create marginal abatement cost curve
if results:
    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    
    # Sort by total_4OAK cost (ascending order for MACC)
    results_df = results_df.sort_values('total_4OAK')
    
    # Calculate cumulative captured CO2
    results_df['captured_total'] = results_df['captured_CO2f'] + results_df['captured_CO2bio']
    results_df['cumulative_captured'] = results_df['captured_total'].cumsum()

    # Save MACC curve data to CSV files
    macc_4oak_df = pd.DataFrame({
        'site-stack': results_df['site-stack'],
        'ktCO2f_yr_baseline': results_df['annual_CO2']*(1-results_df['biogenic']),
        'ktCO2bio_yr_baseline': results_df['annual_CO2']*results_df['biogenic'],
        'ktCO2f_yr_captured': results_df['captured_CO2f'],
        'ktCO2bio_yr_captured': results_df['captured_CO2bio'],
        'ktCO2_yr_cumulative': results_df['cumulative_captured'],
        'ktCO2f_yr_residual': results_df['residual_CO2f'],
        'EUR/tCO2': results_df['total_4OAK']
    })
    macc_4oak_df.to_csv('macc_4oak.csv', index=False)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create step function data for horizontal lines (like macc.py)
    x_steps = []
    y_steps = []
    
    for i in range(len(results_df) - 1):
        # Each plant is a horizontal line from current to next cumulative capacity
        x_steps.extend([results_df['cumulative_captured'].iloc[i], results_df['cumulative_captured'].iloc[i+1]])
        y_steps.extend([results_df['total_4OAK'].iloc[i+1], results_df['total_4OAK'].iloc[i+1]])  # Use the cost of the current plant
    
    plt.plot(x_steps, y_steps, 'b-', linewidth=3, color='blue', alpha=0.7)
    
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
    total_captured = results_df['cumulative_captured'].iloc[-1]
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
    results_df_foak['cumulative_captured'] = results_df_foak['captured_total'].cumsum()

    macc_foak_df = pd.DataFrame({
        'site-stack': results_df_foak['site-stack'],
        'ktCO2f_yr_baseline': results_df_foak['annual_CO2']*(1-results_df_foak['biogenic']),
        'ktCO2bio_yr_baseline': results_df_foak['annual_CO2']*results_df_foak['biogenic'],
        'ktCO2f_yr_captured': results_df_foak['captured_CO2f'],
        'ktCO2bio_yr_captured': results_df_foak['captured_CO2bio'],
        'ktCO2_yr_cumulative': results_df_foak['cumulative_captured'],
        'ktCO2f_yr_residual': results_df_foak['residual_CO2f'],
        'EUR/tCO2': results_df_foak['total_FOAK']
    })
    macc_foak_df.to_csv('macc_foak.csv', index=False)
    
    plt.figure(figsize=(12, 8))
    
    # Create step function data for horizontal lines (like macc.py)
    x_steps_foak = []
    y_steps_foak = []
    
    for i in range(len(results_df_foak) - 1):
        # Each plant is a horizontal line from current to next cumulative capacity
        x_steps_foak.extend([results_df_foak['cumulative_captured'].iloc[i], results_df_foak['cumulative_captured'].iloc[i+1]])
        y_steps_foak.extend([results_df_foak['total_FOAK'].iloc[i+1], results_df_foak['total_FOAK'].iloc[i+1]])  # Use the cost of the current plant
    
    plt.plot(x_steps_foak, y_steps_foak, 'r-', linewidth=3, color='red', alpha=0.7)
    
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
    total_captured_foak = results_df_foak['cumulative_captured'].iloc[-1]
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
    print(f"{'Site-Stack':<40} | {'annual_CO2':<8} | {'captured_CO2f':<8} | {'captured_CO2bio':<8} | {'residual_CO2f':<8} | {'FLH':<6} | {'CAPEX_4OAK':<8} | {'CAPEX_FOAK':<8} | {'OPEXE':<8} | {'transtorage':<8} | {'total_4OAK':<8} | {'total_FOAK':<8}")
    print(f"{'':<40} | {'ktCO2/yr':<8} | {'ktCO2/yr':<8} | {' - ':<8} | {'ktCO2/yr':<8} | {'h/y':<6} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8}")
    print("-" * 150)
    for result in results:
        print(f"{result['site-stack']:<40} | {result['annual_CO2']:>8.1f} | {result['captured_CO2f']:>8.1f} | {result['captured_CO2bio']:>8.1f} | {result['residual_CO2f']:>8.1f} | {result['FLH']:>6.0f} | {result['CAPEX_4OAK']:>8.1f} | {result['CAPEX_FOAK']:>8.1f} | {result['OPEXE']:>8.1f} | {result['transtorage']:>8.1f} | {result['total_4OAK']:>8.1f} | {result['total_FOAK']:>8.1f}")
    print("-" * 150)
    print(f"Total CO2 emissions: {sum([r['annual_CO2'] for r in results]):,.1f} ktCO2/y")
    print(f"Total captured CO2: {sum([r['captured_CO2f'] for r in results]):,.1f} ktCO2/y")

print(len(results))
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

print("\n Should probably NOT capture the extra bio CO2 from industries - its complicated and increases costs... MAYBE add it as an uncertainty lever later!")
print("Deconflicting Fossil Fuel Abatement, Industrial Competitiveness, and Consumer Costs through a UK Carbon Takeback Obligation")