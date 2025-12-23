"""
Combined MACC and CTBO Policy Simulation for UK CCS

This script calculates the Marginal Abatement Cost Curve (MACC) for UK CCS
and simulates a Carbon Takeback Obligation (CTBO) policy in one step.

Author: Oscar Stenstrom
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def match_transportation(emitter_plant, transport_costs_df, discount_rate=0.035, lifetime=30, debug=False):
    """Find the nearest CCS site to an emitter plant and calculate total cost per tonne CO2."""
    if debug:
        print(f"Finding nearest CCS site for: {emitter_plant.get('Site', 'Unknown')}")
    
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
    
    distances_df = pd.DataFrame(distances)
    closest_distance_idx = distances_df['Distance'].idxmin()
    closest_site_id = distances_df.loc[closest_distance_idx, 'Site ID']
    closest_site_original = transport_costs_df[transport_costs_df['Site ID'] == closest_site_id].iloc[0]
    
    mass_CO2 = closest_site_original['Total Tonnes of CO2 stored (MtCO2) 2070']
    onshore_mode = closest_site_original['CO2 Pipeline/Trucking?']
    
    onshore_opex = float(closest_site_original['CO2 onshore transport opex (£m/y)']) / mass_CO2
    
    if onshore_mode == "Pipeline":
        pipeline_capex = float(closest_site_original['CO2 onshore pipeline capex (£m)'])
        annuity_factor = (discount_rate * (1 + discount_rate) ** lifetime) / ((1 + discount_rate) ** lifetime - 1)
        onshore_pipeline = pipeline_capex * annuity_factor / mass_CO2
        onshore_opex += onshore_pipeline
    
    injection_opex = float(closest_site_original['CO2 T&S cost from defined point (£/t)'])
    total_cost = onshore_opex + injection_opex
    
    return {
        'site_id': closest_site_id,
        'site_name': closest_site_original['Site'],
        'distance_m': distances_df.loc[closest_distance_idx, 'Distance'],
        'total_cost_per_t': round(total_cost, 1)
    }


def calculate_distance(easting1, northing1, easting2, northing2, debug=False):
    """Calculate Euclidean distance between two points in meters."""
    return ((easting1 - easting2)**2 + (northing1 - northing2)**2)**0.5


def latlon_to_easting_northing(latitude, longitude, debug=False):
    """Convert WGS84 lat/lon to British National Grid Easting/Northing (EPSG:27700)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    return easting, northing


def approximate_CAPEX(mCO2, xCO2, fixed_rate, CEPCI_curr, CEPCI_base=798.7, capture_rate=0.90, NETL=5.509, debug=False):
    """Estimate CAPEX for CO2 capture using Kim & Leonard (2025) correlation."""
    if debug:
        print(f"approximate_CAPEX inputs: mCO2={mCO2}, xCO2={xCO2}")
    
    a, b, c, n, m = 2.1673, 0.8092, -0.00332, 0.5291, 0.8391
    
    nCO2 = mCO2 * 1000 / 44
    n_fluegas = nCO2 / xCO2
    V_fluegas = n_fluegas * 22.4
    
    n_largest_absorbers = int((V_fluegas/1000) // 1613)
    remaining_V_fluegas = (V_fluegas/1000) % 1613
    
    CAPEX = 0
    for i in range(n_largest_absorbers):
        TEC = a + (b * (xCO2)**n + c) * (1613)**m
        CAPEX += TEC
    
    if remaining_V_fluegas > 0:
        TEC = a + (b * (xCO2)**n + c) * (remaining_V_fluegas)**m
        CAPEX += TEC
    
    OPEX_fixed = CAPEX * fixed_rate
    CAPEX = CAPEX * NETL * CEPCI_curr / CEPCI_base
    
    return CAPEX, OPEX_fixed


def levelize_MEUR(CAPEX, annual_CO2, capture_rate=0.95, discount_rate=0.07, lifetime=25, debug=False):
    """Convert CAPEX [MEUR] to levelized cost [EUR/tCO2]."""
    annualized_CAPEX = CAPEX * discount_rate * (1 + discount_rate)**lifetime / ((1 + discount_rate)**lifetime - 1) * 10**6
    levelized_CAPEX = annualized_CAPEX / (annual_CO2 * capture_rate)
    return levelized_CAPEX


def energy_supply(mCO2, cap_rate, FLH, qreb_val, pcompr_val, qsteam_val, qelc_val, qchp_val,
                  csteam_val, elc_eff_val, cbio_val, celc_val, emission_factor, annual_CO2,
                  evap_enthalpy=2257, maximize_beccs=False, debug=False):
    """Calculate energy penalty costs (OPEXE) for CCS."""
    mCO2f_captured = mCO2 * cap_rate
    mCO2f_residual = mCO2 * (1 - cap_rate)
    
    Qreb = qreb_val * mCO2f_captured * 1000 / 3600
    Pcompr = pcompr_val * mCO2f_captured * 1000 / 3600
    
    Qsteam_energy = qsteam_val * Qreb
    Qelc = qelc_val * Qreb
    Qchp = qchp_val * Qreb
    
    cost_steam = csteam_val / evap_enthalpy
    Qsteam_annual = Qsteam_energy * FLH * 3600
    OPEX_steam = cost_steam * Qsteam_annual / (annual_CO2 * cap_rate)
    
    Pelec = (Qelc * elc_eff_val + Pcompr) * FLH
    OPEX_elec = celc_val * Pelec / (annual_CO2 * cap_rate)
    
    if maximize_beccs:
        Qfuel = Qchp / (1 - cap_rate * emission_factor * (qreb_val/3600*1000))
        mCO2bio_captured = Qfuel * emission_factor * cap_rate
    else:
        Qfuel = Qchp
        mCO2bio_captured = 0
    
    OPEX_chp = cbio_val * Qfuel * FLH / (annual_CO2 * cap_rate)
    OPEXE = OPEX_steam + OPEX_elec + OPEX_chp
    
    return mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel


def create_result_entry(site_stack, annual_CO2, biogenic, captured_CO2f, captured_CO2bio,
                        residual_CO2f, FLH, CAPEX_4OAK, CAPEX_FOAK, OPEXE, OPEX_fixed, transtorage, distance_km):
    """Create a standardized result dictionary entry."""
    return {
        'site-stack': site_stack,
        'annual_CO2': annual_CO2 / 1000,
        'biogenic': biogenic,
        'captured_CO2f': captured_CO2f,
        'captured_CO2bio': captured_CO2bio,
        'residual_CO2f': residual_CO2f,
        'FLH': FLH,
        'CAPEX_4OAK': CAPEX_4OAK,
        'CAPEX_FOAK': CAPEX_FOAK,
        'OPEXE': OPEXE,
        'OPEX_fixed': OPEX_fixed,
        'transtorage': transtorage,
        'total_4OAK': CAPEX_4OAK + OPEXE + OPEX_fixed + transtorage,
        'total_FOAK': CAPEX_FOAK + OPEXE + OPEX_fixed + transtorage,
        'distance_km': distance_km
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(debug=False):
    """Load all required CSV files once. Returns a dictionary of dataframes."""
    if debug:
        print("\n" + "="*80)
        print("LOADING DATA FILES")
        print("="*80)
    
    data = {}
    
    # Load fossil plants
    data['fossil_plants'] = pd.read_csv("data/point_sources_CO2_2022.csv")
    data['fossil_plants']['CO2'] = data['fossil_plants']['Emission'] * 3.66
    if debug:
        print(f"  Loaded fossil_plants: {len(data['fossil_plants'])} rows")
    
    # Load transport costs
    data['transport_costs'] = pd.read_csv("data/nzip_balanced_scenario_results.csv", encoding='latin-1')
    data['transport_costs'] = data['transport_costs'][data['transport_costs']["CO2 Pipeline/Trucking?"] != "No CCS"]
    if debug:
        print(f"  Loaded transport_costs: {len(data['transport_costs'])} rows")
    
    # Load W2E plants
    data['w2e_plants'] = pd.read_csv("data/w2e_plants.csv")
    data['w2e_plants'] = data['w2e_plants'][data['w2e_plants']['No.'] != 40]  # Remove Shetland Islands
    if debug:
        print(f"  Loaded w2e_plants: {len(data['w2e_plants'])} rows")
    
    # Load power capacities
    data['power_capacities'] = pd.read_csv("data/power_capacities_clean.csv")
    if debug:
        print(f"  Loaded power_capacities: {len(data['power_capacities'])} rows")
    
    # Load ETS prices (for non-linear ETS option)
    data['ets_prices'] = pd.read_csv('data/ETS.csv')
    data['ets_prices'].columns = data['ets_prices'].columns.str.strip()
    data['ets_prices'].set_index('Year', inplace=True)
    if debug:
        print(f"  Loaded ets_prices: {len(data['ets_prices'])} rows")
    
    return data


# ============================================================================
# COMBINED SIMULATION
# ============================================================================

def combined_simulation(
    data,
    # --- MACC Configuration ---
    HALF=False,                     # [-] bool: use every 2nd plant from top-60 emitters
    pounds_to_EUR=1.15,             # [EUR/GBP] currency conversion factor
    CEPCI_2025=930,                 # [-] Chemical Engineering Plant Cost Index (2025)
    CEPCI_2023=798.7,               # [-] Chemical Engineering Plant Cost Index (2023 reference)
    FOAK_MULTIPLIER=1.7553,         # [-] First-of-a-kind cost multiplier
    NETL=5.509,                     # [-] NETL methodology CAPEX multiplier
    capture_rate=0.95,              # [-] CO2 capture rate (0-1)
    discount_rate_ccs=0.07,         # [-] Discount rate for CCS CAPEX annualization (0-1)
    lifetime_ccs=25,                # [years] CCS project lifetime
    qreb=3.5,                       # [MJ/kgCO2] Specific reboiler heat duty
    pcompr=0.37,                    # [MJ/kgCO2] Specific compression power
    celc=300,                       # [EUR/MWh] Electricity cost
    cbio=120,                       # [EUR/MWh] Biomass fuel cost
    csteam=4.1,                     # [EUR/tsteam] Steam cost at ~130°C
    amine_cost=44,                  # [SEK/tCO2] Amine makeup cost
    sek_to_eur=0.091,               # [EUR/SEK] currency conversion factor
    fixed=0.04,                     # [-] Fixed OPEX as fraction of CAPEX per annum
    power_ccgt=None,                # dict: {xCO2 [-], eta_P [MWel/MWfuel], emission_factor [tCO2/MWhfuel], gas_eff [-], steam_eff [-]}
    w2e_config=None,                # dict: {emission_factor [tCO2/twaste], FLH [h/yr], fossil_fraction [-]}
    drax_config=None,               # dict: {CO2 [tCO2/yr], Pinstalled [MW], eta_P [%], coordinates (lat, lon)}
    qsteam=0.15,                    # [-] Fraction of reboiler heat from steam (0-1)
    qelc=0.30,                      # [-] Fraction of reboiler heat from electricity (0-1)
    qchp=0.55,                      # [-] Fraction of reboiler heat from biomass CHP (0-1)
    elc_eff=0.33,                   # [-] Efficiency of electric reboiler (0-1)
    evaporation_enthalpy=2257,      # [MJ/tsteam] Enthalpy of vaporization
    emission_factor_bio=0.3318,     # [tCO2/MWhfuel] Biomass emission factor
    # --- Sector-specific parameters ---
    FLH_industry=8500,              # [h/yr] Full load hours for industrial plants
    refinery_stacks=None,           # dict: stack fractions and xCO2 values
    scunthorpe_xCO2=None,           # dict: xCO2 values for iron & steel stacks
    cement_xCO2=0.20,               # [-] CO2 concentration in cement flue gas (0-1)
    w2e_xCO2=0.11,                  # [-] CO2 concentration in W2E flue gas (0-1)
    drax_xCO2=0.13,                 # [-] CO2 concentration in Drax flue gas (0-1)
    drax_efficiency_penalty=0.24,   # [-] Efficiency penalty from CCS on Drax (0-1)
    extra_transtorage=30,           # [EUR/tCO2] Additional transport & storage cost
    # --- CTBO Configuration ---
    USE_FOAK=False,                 # [-] bool: use FOAK costs instead of 4OAK
    CTBO_ENABLED=True,              # [-] bool: enable CTBO mandate
    ETS_SCENARIO="High",            # [-] str: "High", "Medium", or "Low"
    DACCS_EXPENSIVE=True,           # [-] bool: use expensive DACCS trajectory
    VERBOSE=True,                   # [-] bool: print investment decisions
    START_YEAR=2025,                # [year] Simulation start year
    END_YEAR=2055,                  # [year] Simulation end year
    baseline_emissions=None,        # dict: {coal, oil, gas} in [MtCO2/yr]
    DIFFUSE_START_FRACTION=1.0,     # [-] Fraction of diffuse emissions at START_YEAR (0-1)
    DIFFUSE_END_FRACTION=0.20,      # [-] Fraction of diffuse emissions at DIFFUSE_TARGET_YEAR (0-1)
    DIFFUSE_TARGET_YEAR=2050,       # [year] Target year for diffuse emissions reduction
    DISCOUNT_RATE=0.035,            # [-] Real discount rate for NPV calculations (0-1)
    USE_INVESTMENT_YEAR_AS_BASE=False,  # [-] bool: NPV base year = investment year vs START_YEAR
    fuels=None,                     # dict: {fuel: {emission_factor [kgCO2/L or kgCO2/therm], price [pence/L or pence/therm]}}
    ETS_LINEAR=True,                # [-] bool: use linear ETS trajectory vs CSV data
    ets_linear_start=45,            # [GBP/tCO2] ETS price at START_YEAR
    ets_linear_end=None,            # dict: {"Low", "Medium", "High"} in [GBP/tCO2] at 2050
    CTBO_growth_factor=0.4,         # [-] Quadratic growth factor for CTBO mandate (t = years * factor)
    cement_process_fraction=0.63,   # [-] Fraction of cement CO2 from process (vs fuel)
    debug=True                      # [-] bool: print debug information
):
    """
    Run combined MACC calculation and CTBO simulation.
    
    Parameters:
        data: Dictionary of pre-loaded dataframes from load_data()
        All other parameters have sensible defaults.
    
    Returns:
        Dictionary containing all simulation results
    """
    
    # Set default dicts if not provided
    if power_ccgt is None:
        power_ccgt = {'xCO2': 0.05, 'eta_P': 0.49, 'emission_factor': 0.204, 'gas_eff': 0.35, 'steam_eff': 0.51}
    if w2e_config is None:
        w2e_config = {'emission_factor': 0.98, 'FLH': 8760 * 0.866, 'fossil_fraction': 0.465}
    if drax_config is None:
        drax_config = {'CO2': 11500000, 'Pinstalled': 2580, 'eta_P': 33 / (1 - drax_efficiency_penalty), 'coordinates': (53.738710, -0.993030)}
    if baseline_emissions is None:
        baseline_emissions = {'coal': 17, 'oil': 140, 'gas': 127}
    if fuels is None:
        fuels = {
            'diesel': {'emission_factor': 2.628, 'price': 143.97}, # https://commonslibrary.parliament.uk/research-briefings/cbp-9714/
            'petrol': {'emission_factor': 2.339, 'price': 135.07}, # kgCO2/litre, p/litre
            'gas': {'emission_factor': 0.2039 * 29.3, 'price': 6.3*29.3} # kgCO2/thrm (1 thrm = 29.3 kWh), p/therm
        }
    if ets_linear_end is None:
        ets_linear_end = {"Low": 85, "Medium": 125, "High": 155} # {"Low": 85, "Medium": 125, "High": 155} CCC 7thCB uses 409 GBP by 2050
    if refinery_stacks is None:
        # {stack_name: [fraction_of_total_CO2, xCO2_concentration]}
        refinery_stacks = {
            'power': [0.298, (3*25 + 8*54)/(25+54)/100],
            'crackers': [0.20, 17/100],
            'distillation': [0.17, 11/100],
            'smr': [0.118, (8*6 + 24*26)/(6+26)/100],
            'remaining': [0.188, 8/100],
        }
    if scunthorpe_xCO2 is None:
        scunthorpe_xCO2 = {
            'Scunthorpe Power Station': ('chp', 0.296),
            'Scunthorpe Blast Furnaces': ('stove', 0.251),
            'Scunthorpe Sinter': ('sinter', 0.15),
        }
    
    # DACCS costs
    if DACCS_EXPENSIVE:
        DACCS_2025, DACCS_2050 = 391, 391 # 323, 281 7th CB
    else:
        DACCS_2025, DACCS_2050 = 322, 322 # 247, 152
    
    years = np.arange(START_YEAR, END_YEAR + 1)
    
    if debug:
        print("\n" + "="*80)
        print("RUNNING COMBINED SIMULATION")
        print("="*80)
        print(f"  HALF={HALF}, USE_FOAK={USE_FOAK}, ETS_SCENARIO={ETS_SCENARIO}")
        print(f"  DACCS_EXPENSIVE={DACCS_EXPENSIVE}, CTBO_ENABLED={CTBO_ENABLED}")
    
    # ========================================================================
    # MACC CALCULATION
    # ========================================================================
    
    if debug:
        print("\n--- CALCULATING MACC ---")
    
    results = []
    
    # Prepare largest plants
    fossil_plants = data['fossil_plants'].copy()
    largest_plants = fossil_plants.nlargest(60, 'CO2')[['PlantID', 'Site', 'Easting', 'Northing', 'Operator', 'Sector', 'CO2']]
    largest_plants = largest_plants[largest_plants['Site'] != "Elgin PUQ"]
    largest_plants.loc[largest_plants['Site'] == 'Grangemouth Power Station', 'Sector'] = 'Major power producers'
    
    # Combine Fawley Refinery entries
    fawley_mask = largest_plants['Site'] == 'Fawley Refinery'
    if fawley_mask.sum() > 1:
        fawley_total_co2 = largest_plants.loc[fawley_mask, 'CO2'].sum()
        first_fawley_idx = largest_plants[fawley_mask].index[0]
        largest_plants = largest_plants[~fawley_mask | (largest_plants.index == first_fawley_idx)].copy()
        largest_plants.loc[first_fawley_idx, 'CO2'] = fawley_total_co2
    
    # Calculate full point source emissions before HALF filtering
    largest_plants_full = largest_plants.copy()
    full_point_source_emissions = largest_plants_full['CO2'].sum() / 1000  # [ktCO2/yr]
    full_cement_emissions = largest_plants_full[largest_plants_full['Sector'] == 'Cement']['CO2'].sum() / 1000
    
    if HALF:
        largest_plants = largest_plants.iloc[::2]
    
    transport_costs = data['transport_costs'].copy()
    
    # Calculate transport costs for largest plants
    for idx, plant in largest_plants.iterrows():
        result = match_transportation(plant, transport_costs, discount_rate=0.035, lifetime=30)
        largest_plants.loc[idx, 'transport_cost'] = result['total_cost_per_t'] * pounds_to_EUR * CEPCI_2025/CEPCI_2023
        largest_plants.loc[idx, 'distance_km'] = result['distance_m'] / 1000
    
    # Prepare W2E plants
    w2e_plants = data['w2e_plants'].copy()
    w2e_plants['Easting'] = 0.0
    w2e_plants['Northing'] = 0.0
    w2e_plants['transport_cost'] = 0.0
    
    for idx, plant in w2e_plants.iterrows():
        easting, northing = latlon_to_easting_northing(plant['Latitude'], plant['Longitude'])
        w2e_plants.loc[idx, 'Easting'] = easting
        w2e_plants.loc[idx, 'Northing'] = northing
        plant_with_coords = w2e_plants.loc[idx]
        result = match_transportation(plant_with_coords, transport_costs, discount_rate=0.035, lifetime=30)
        w2e_plants.loc[idx, 'transport_cost'] = result['total_cost_per_t'] * pounds_to_EUR * CEPCI_2025/CEPCI_2023
        w2e_plants.loc[idx, 'distance_km'] = result['distance_m'] / 1000
    
    # --- POWER SECTOR ---
    xCO2 = power_ccgt['xCO2']
    eta_P = power_ccgt['eta_P']
    emission_factor = power_ccgt['emission_factor']
    gas_eff = power_ccgt['gas_eff']
    steam_eff = power_ccgt['steam_eff']
    
    power_capacities = data['power_capacities'].copy()
    power_producers = largest_plants[
        (largest_plants['Sector'] == 'Major power producers') |
        (largest_plants['Sector'] == 'Minor power producers')
    ].copy()
    power_producers = power_producers[power_producers['Site'] != 'Ratcliffe on Soar Power Station'].copy()
    power_producers = power_producers.merge(power_capacities, left_on='Site', right_on='Power plant name', how='left')
    
    for idx, plant in power_producers.iterrows():
        Pinstalled = plant['Capacity [MW]']
        Qfuel = Pinstalled / eta_P
        mCO2 = Qfuel * emission_factor
        FLH = plant['CO2'] / mCO2
        
        CAPEX, OPEX_fixed_val = approximate_CAPEX(mCO2, xCO2, fixed, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=NETL)
        levelized_CAPEX = levelize_MEUR(CAPEX, plant['CO2'], capture_rate, discount_rate_ccs, lifetime_ccs)
        CAPEX_4OAK = levelized_CAPEX
        CAPEX_FOAK = FOAK_MULTIPLIER * CAPEX_4OAK
        OPEX_fixed_val = OPEX_fixed_val * 10**6 / (plant['CO2'] * capture_rate)
        
        mCO2_captured = mCO2 * capture_rate
        mCO2_residual = mCO2 * (1 - capture_rate)
        
        Pgas = Qfuel * gas_eff
        Qsteam_power = Qfuel * steam_eff
        Prankine = Pinstalled - Pgas
        eta_steam = Prankine / Qsteam_power
        
        Qreb_power = qreb * mCO2_captured * 1000 / 3600
        Plost = Qreb_power * eta_steam * FLH
        
        OPEXE = Plost * celc / (mCO2_captured * FLH)
        OPEXE += pcompr * 1000/3600 * celc
        
        results.append(create_result_entry(
            f"{plant['Site']}-CCGT",
            plant['CO2'], 0,
            mCO2_captured * FLH / 1000, 0,
            mCO2_residual * FLH / 1000,
            FLH, CAPEX_4OAK, CAPEX_FOAK, OPEXE, OPEX_fixed_val,
            plant['transport_cost'], plant['distance_km']
        ))
    
    # --- INDUSTRY SECTOR ---
    refineries = largest_plants[largest_plants['Sector'] == 'Processing & distribution of petroleum products'].copy()
    scunthorpe_stacks = largest_plants[largest_plants['Site'].str.startswith('Scunthorpe', na=False)].copy()
    cement_plants = largest_plants[largest_plants['Sector'] == 'Cement'].copy()
    industrial_plants = pd.concat([refineries, scunthorpe_stacks, cement_plants], ignore_index=True)
    
    for idx, plant in industrial_plants.iterrows():
        if plant['Sector'] == 'Processing & distribution of petroleum products':
            annual_CO2 = plant['CO2']
            
            for stack_name, stack_params in refinery_stacks.items():
                annual_CO2_stack = annual_CO2 * stack_params[0]
                xCO2_stack = stack_params[1]
                mCO2 = annual_CO2_stack / FLH_industry
                
                mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(
                    mCO2, capture_rate, FLH_industry, qreb, pcompr, qsteam, qelc, qchp,
                    csteam, elc_eff, cbio, celc, emission_factor_bio, annual_CO2_stack, evaporation_enthalpy
                )
                mCO2_total = mCO2f_captured + mCO2bio_captured
                
                CAPEX, OPEX_fixed_val = approximate_CAPEX(mCO2_total, xCO2_stack, fixed, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=NETL)
                levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2_stack, capture_rate, discount_rate_ccs, lifetime_ccs)
                CAPEX_4OAK = levelized_CAPEX
                CAPEX_FOAK = FOAK_MULTIPLIER * CAPEX_4OAK
                OPEX_fixed_val = OPEX_fixed_val * 10**6 / (plant['CO2'] * capture_rate)
                
                results.append(create_result_entry(
                    f"{plant['Site']}-{stack_name}",
                    annual_CO2_stack, 0,
                    mCO2f_captured * FLH_industry / 1000,
                    mCO2bio_captured * FLH_industry / 1000,
                    mCO2f_residual * FLH_industry / 1000,
                    FLH_industry, CAPEX_4OAK, CAPEX_FOAK, OPEXE, OPEX_fixed_val,
                    plant['transport_cost'], plant['distance_km']
                ))
        
        if plant['Sector'] == 'Iron & steel industries':
            if plant['Site'] in scunthorpe_xCO2:
                stack_name, xCO2_stack = scunthorpe_xCO2[plant['Site']]
            else:
                continue
            
            annual_CO2_stack = plant['CO2']
            mCO2 = annual_CO2_stack / FLH_industry
            
            mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(
                mCO2, capture_rate, FLH_industry, qreb, pcompr, qsteam, qelc, qchp,
                csteam, elc_eff, cbio, celc, emission_factor_bio, annual_CO2_stack, evaporation_enthalpy
            )
            mCO2_total = mCO2f_captured + mCO2bio_captured
            
            CAPEX, OPEX_fixed_val = approximate_CAPEX(mCO2_total, xCO2_stack, fixed, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=NETL)
            levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2_stack, capture_rate, discount_rate_ccs, lifetime_ccs)
            CAPEX_4OAK = levelized_CAPEX
            CAPEX_FOAK = FOAK_MULTIPLIER * CAPEX_4OAK
            OPEX_fixed_val = OPEX_fixed_val * 10**6 / (plant['CO2'] * capture_rate)
            
            results.append(create_result_entry(
                f"{plant['Site']}-{stack_name}",
                annual_CO2_stack, 0,
                mCO2f_captured * FLH_industry / 1000,
                mCO2bio_captured * FLH_industry / 1000,
                mCO2f_residual * FLH_industry / 1000,
                FLH_industry, CAPEX_4OAK, CAPEX_FOAK, OPEXE, OPEX_fixed_val,
                plant['transport_cost'], plant['distance_km']
            ))
        
        if plant['Sector'] == 'Cement':
            annual_CO2_stack = plant['CO2']
            xCO2_stack = cement_xCO2
            mCO2 = annual_CO2_stack / FLH_industry
            
            mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(
                mCO2, capture_rate, FLH_industry, qreb, pcompr, qsteam, qelc, qchp,
                csteam, elc_eff, cbio, celc, emission_factor_bio, annual_CO2_stack, evaporation_enthalpy
            )
            mCO2_total = mCO2f_captured + mCO2bio_captured
            
            CAPEX, OPEX_fixed_val = approximate_CAPEX(mCO2_total, xCO2_stack, fixed, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=NETL)
            levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2_stack, capture_rate, discount_rate_ccs, lifetime_ccs)
            CAPEX_4OAK = levelized_CAPEX
            CAPEX_FOAK = FOAK_MULTIPLIER * CAPEX_4OAK
            OPEX_fixed_val = OPEX_fixed_val * 10**6 / (plant['CO2'] * capture_rate)
            
            results.append(create_result_entry(
                f"{plant['Site']}-cement",
                annual_CO2_stack, 0,
                mCO2f_captured * FLH_industry / 1000,
                mCO2bio_captured * FLH_industry / 1000,
                mCO2f_residual * FLH_industry / 1000,
                FLH_industry, CAPEX_4OAK, CAPEX_FOAK, OPEXE, OPEX_fixed_val,
                plant['transport_cost'], plant['distance_km']
            ))
    
    # --- WASTE-TO-ENERGY SECTOR ---
    emission_factor_w2e = w2e_config['emission_factor']
    FLH_w2e = w2e_config['FLH']
    fossil_fraction = w2e_config['fossil_fraction']
    
    w2e_plants['CO2'] = w2e_plants['Capacity 2023 [ktpa]'] * 1000 * emission_factor_w2e
    
    for idx, plant in w2e_plants.iterrows():
        mCO2 = plant['CO2'] / FLH_w2e
        xCO2_w2e = w2e_xCO2
        
        CAPEX, OPEX_fixed_val = approximate_CAPEX(mCO2, xCO2_w2e, fixed, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=NETL)
        levelized_CAPEX = levelize_MEUR(CAPEX, plant['CO2'], capture_rate, discount_rate_ccs, lifetime_ccs)
        CAPEX_4OAK = levelized_CAPEX
        CAPEX_FOAK = FOAK_MULTIPLIER * CAPEX_4OAK
        OPEX_fixed_val = OPEX_fixed_val * 10**6 / (plant['CO2'] * capture_rate)
        
        Qreb_w2e = qreb * mCO2 * capture_rate * 1000 / 3600
        cost_steam = csteam / evaporation_enthalpy
        Qsteam_annual = Qreb_w2e * FLH_w2e * 3600
        OPEX_steam = cost_steam * Qsteam_annual / (plant['CO2'] * capture_rate)
        OPEXE = OPEX_steam + pcompr * 1000/3600 * celc
        
        results.append(create_result_entry(
            f"{plant['Name']}-W2E",
            plant['CO2'], 1 - fossil_fraction,
            mCO2 * capture_rate * FLH_w2e / 1000 * fossil_fraction,
            mCO2 * capture_rate * FLH_w2e / 1000 * (1 - fossil_fraction),
            mCO2 * (1 - capture_rate) * fossil_fraction * FLH_w2e / 1000,
            FLH_w2e, CAPEX_4OAK, CAPEX_FOAK, OPEXE, OPEX_fixed_val,
            plant['transport_cost'], plant['distance_km']
        ))
    
    # --- DRAX BECCS ---
    Drax_CO2 = drax_config['CO2']
    Pinstalled_drax = drax_config['Pinstalled']
    eta_P_drax = drax_config['eta_P']
    emission_factor = power_ccgt['emission_factor']
    
    Qfuel_drax = Pinstalled_drax / (eta_P_drax / 100)
    FLH_drax = Drax_CO2 / (Qfuel_drax * emission_factor)
    
    xCO2_drax = drax_xCO2
    mCO2_drax = Drax_CO2 / FLH_drax
    CAPEX_drax, OPEX_fixed_val = approximate_CAPEX(mCO2_drax, xCO2_drax, fixed, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=NETL)
    levelized_CAPEX_drax = levelize_MEUR(CAPEX_drax, Drax_CO2, capture_rate, discount_rate_ccs, lifetime_ccs)
    CAPEX_4OAK_drax = levelized_CAPEX_drax
    CAPEX_FOAK_drax = FOAK_MULTIPLIER * CAPEX_4OAK_drax
    OPEX_fixed_val = OPEX_fixed_val * 10**6 / (Drax_CO2 * capture_rate)
    
    profit_baseline = Qfuel_drax * eta_P_drax/100 * FLH_drax * celc
    profit_BECCS = Qfuel_drax * eta_P_drax/100 * (1 - drax_efficiency_penalty) * FLH_drax * celc
    difference = profit_baseline - profit_BECCS
    OPEXE_drax = difference / (Drax_CO2 * capture_rate)
    OPEXE_drax += pcompr * 1000/3600 * celc
    
    drax_coordinates = drax_config['coordinates']
    drax_easting, drax_northing = latlon_to_easting_northing(drax_coordinates[0], drax_coordinates[1])
    drax_plant = pd.Series({'Easting': drax_easting, 'Northing': drax_northing, 'Name': 'Drax'})
    result_drax = match_transportation(drax_plant, transport_costs, discount_rate=0.035, lifetime=30)
    transport_cost_drax = result_drax['total_cost_per_t'] * pounds_to_EUR * CEPCI_2025/CEPCI_2023
    drax_distance = result_drax['distance_m'] / 1000
    
    results.append(create_result_entry(
        'Drax-BECCS',
        Drax_CO2, 1,
        0, mCO2_drax * capture_rate * FLH_drax / 1000,
        0, FLH_drax,
        CAPEX_4OAK_drax, CAPEX_FOAK_drax, OPEXE_drax, OPEX_fixed_val,
        transport_cost_drax, drax_distance
    ))
    
    # --- POST-PROCESSING ---
    for result in results:
        result['total_4OAK'] += amine_cost * sek_to_eur + extra_transtorage
        result['total_FOAK'] += amine_cost * sek_to_eur + extra_transtorage
        result['transtorage'] += extra_transtorage
    
    # Create MACC dataframes
    results_df = pd.DataFrame(results)
    
    # 4OAK MACC
    results_df_4oak = results_df.sort_values('total_4OAK').copy()
    results_df_4oak['captured_total'] = results_df_4oak['captured_CO2f'] + results_df_4oak['captured_CO2bio']
    results_df_4oak['cumulative_captured'] = results_df_4oak['captured_total'].cumsum()
    
    macc_4oak = pd.DataFrame({
        'site-stack': results_df_4oak['site-stack'],
        'ktCO2f_yr_baseline': results_df_4oak['annual_CO2'] * (1 - results_df_4oak['biogenic']),
        'ktCO2bio_yr_baseline': results_df_4oak['annual_CO2'] * results_df_4oak['biogenic'],
        'ktCO2f_yr_captured': results_df_4oak['captured_CO2f'],
        'ktCO2bio_yr_captured': results_df_4oak['captured_CO2bio'],
        'ktCO2_yr_cumulative': results_df_4oak['cumulative_captured'],
        'ktCO2f_yr_residual': results_df_4oak['residual_CO2f'],
        'EUR/tCO2': results_df_4oak['total_4OAK']
    })
    
    # FOAK MACC
    results_df_foak = results_df.sort_values('total_FOAK').copy()
    results_df_foak['captured_total'] = results_df_foak['captured_CO2f'] + results_df_foak['captured_CO2bio']
    results_df_foak['cumulative_captured'] = results_df_foak['captured_total'].cumsum()
    
    macc_foak = pd.DataFrame({
        'site-stack': results_df_foak['site-stack'],
        'ktCO2f_yr_baseline': results_df_foak['annual_CO2'] * (1 - results_df_foak['biogenic']),
        'ktCO2bio_yr_baseline': results_df_foak['annual_CO2'] * results_df_foak['biogenic'],
        'ktCO2f_yr_captured': results_df_foak['captured_CO2f'],
        'ktCO2bio_yr_captured': results_df_foak['captured_CO2bio'],
        'ktCO2_yr_cumulative': results_df_foak['cumulative_captured'],
        'ktCO2f_yr_residual': results_df_foak['residual_CO2f'],
        'EUR/tCO2': results_df_foak['total_FOAK']
    })
    
    if debug:
        total_captured = macc_4oak['ktCO2_yr_cumulative'].iloc[-1]
        avg_cost = macc_4oak['EUR/tCO2'].mean()
        print(f"\nMACC Summary (4OAK):")
        print(f"  Total CO2 capture potential bio+fossil: {total_captured:.0f} ktCO2/yr")
        print(f"  Average abatement cost: {avg_cost:.1f} EUR/tCO2")
        print(f"  Cost range: {macc_4oak['EUR/tCO2'].min():.1f} - {macc_4oak['EUR/tCO2'].max():.1f} EUR/tCO2")
    
    # ========================================================================
    # CTBO SIMULATION
    # ========================================================================
    
    if debug:
        print("\n--- RUNNING CTBO SIMULATION ---")
        print(f"  Using {'FOAK' if USE_FOAK else '4OAK'} cost scenario")
        print(f"  ETS scenario: {ETS_SCENARIO}")
        print(f"  DACCS cost: {'Expensive' if DACCS_EXPENSIVE else 'Cheap'}")
    
    # Select MACC
    macc_4oak_sim = macc_4oak.copy()
    macc_foak_sim = macc_foak.copy()
    macc_4oak_sim['invested'] = False
    macc_foak_sim['invested'] = False
    macc = macc_foak_sim if USE_FOAK else macc_4oak_sim
    
    macc_fossil_CO2 = macc['ktCO2f_yr_baseline'].sum() 

    # Calculate baseline emissions
    total_emissions_2023 = (baseline_emissions['coal'] + baseline_emissions['oil'] +
                            baseline_emissions['gas']) * 1000 + full_cement_emissions
    diffuse_baseline = total_emissions_2023 - full_point_source_emissions # NOTE: same no matter if HALF=True or False
    
    # Create trajectories
    t = (years - START_YEAR) * CTBO_growth_factor
    ctbo_fraction = t**2
    
    diffuse_fraction = np.where(
        years <= DIFFUSE_TARGET_YEAR,
        DIFFUSE_START_FRACTION - (years - START_YEAR) * ((DIFFUSE_START_FRACTION - DIFFUSE_END_FRACTION) /
                                                          (DIFFUSE_TARGET_YEAR - START_YEAR)),
        DIFFUSE_END_FRACTION
    )
    diffuse_emissions = diffuse_baseline * diffuse_fraction
    
    if ETS_LINEAR:
        ets_end = ets_linear_end[ETS_SCENARIO]
        ets_prices = np.where(
            years <= 2050,
            ets_linear_start + (years - START_YEAR) * ((ets_end - ets_linear_start) / (2050 - START_YEAR)),
            ets_end
        ) * pounds_to_EUR
    else:
        ets_df = data['ets_prices'].copy()
        ets_column_map = {
            "High": "High Sensitivity - Low Fossil Fuel Prices and High Economic Growth (2024 GBP)",
            "Medium": "Net Zero Strategy Aligned (2024 GBP)",
            "Low": "Low Sensitivity - High Fossil Fuel Prices and Low Economic Growth (2024 GBP)"
        }
        ets_column = ets_column_map[ETS_SCENARIO]
        ets_prices = []
        for year in years:
            if year in ets_df.index:
                ets_prices.append(ets_df.loc[year, ets_column] * pounds_to_EUR)
            else:
                ets_prices.append(ets_df.loc[2050, ets_column] * pounds_to_EUR)
        ets_prices = np.array(ets_prices)
    
    DACCS_costs = np.where(
        years <= 2050,
        DACCS_2025 + (years - START_YEAR) * ((DACCS_2050 - DACCS_2025) / (2050 - START_YEAR)),
        DACCS_2050
    ) * pounds_to_EUR
    
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
        total_emissions_current = plant_emissions + diffuse
        
        supplied_CO2 = macc['ktCO2f_yr_baseline'].sum() + diffuse
        ctbo_mandate = supplied_CO2 * ctbo_fraction[i] / 100
        
        point_fossil_capacity = macc['ktCO2f_yr_captured'].where(macc['invested'], 0).sum()
        point_bio_capacity = macc['ktCO2bio_yr_captured'].where(macc['invested'], 0).sum()
        point_capacity = point_fossil_capacity + point_bio_capacity
        
        DACCS_capacity = 0
        CSU_cost = 0
        marginal_cost = 0
        CTBO_cost = 0
        CTBO_cost_lev = 0
        
        # CTBO-mandated investments
        if CTBO_ENABLED:
            missing_capacity = ctbo_mandate - point_capacity
            
            j = 0
            while missing_capacity > 0 and j < len(macc):
                plant = macc.iloc[j]
                if not plant['invested']:
                    if plant['EUR/tCO2'] > DACCS_cost:
                        if VERBOSE:
                            print(f"Year {year}: Switching to DACCS "
                                  f"(cost: {DACCS_cost:.0f} < plant: {plant['EUR/tCO2']:.0f} EUR/tCO2)")
                        break
                    
                    macc.loc[macc.index[j], 'invested'] = True
                    macc.loc[macc.index[j], 'year_invested'] = year
                    point_fossil_capacity += plant['ktCO2f_yr_captured']
                    point_bio_capacity += plant['ktCO2bio_yr_captured']
                    point_capacity += plant['ktCO2f_yr_captured'] + plant['ktCO2bio_yr_captured']
                    missing_capacity = ctbo_mandate - point_capacity
                    
                    if VERBOSE:
                        print(f"Year {year}: Mandate {plant['site-stack']} (cost: {plant['EUR/tCO2']:.0f} EUR/tCO2)")
                j += 1
            
            invested_plants = macc[macc['invested']]
            if len(invested_plants) > 0:
                marginal_plant = invested_plants.loc[invested_plants['EUR/tCO2'].idxmax()]
                marginal_cost = marginal_plant['EUR/tCO2']
                CSU_cost = max(0, marginal_cost - ets_price)
                CTBO_cost = CSU_cost * point_capacity
            
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
        total_emissions_vec.append(total_emissions_current)
        ctbo_mandate_vec.append(ctbo_mandate)
        fCCS_capacity_vec.append(point_fossil_capacity)
        BECCS_capacity_vec.append(point_bio_capacity)
        DACCS_capacity_vec.append(DACCS_capacity)
        marginal_cost_vec.append(marginal_cost)
        CSU_cost_vec.append(CSU_cost)
        CTBO_cost_lev_vec.append(CTBO_cost_lev)
        
        # Calculate plant-level costs and profits
        for idx, plant in macc.iterrows():
            if plant['site-stack'].endswith('W2E') or plant['site-stack'].endswith('BECCS'):
                csu_diluted_cost = 0
            elif plant['site-stack'].endswith('cement'):
                csu_diluted_cost = CTBO_cost_lev * (1 - cement_process_fraction)
            else:
                csu_diluted_cost = CTBO_cost_lev
            
            csu_subtract = max(0, plant['EUR/tCO2'] - ets_price)                # if invested, this diff must be covered by someting, i.e., CSU profits
            csu_subtract = csu_subtract + csu_diluted_cost                      # also internalize fossil fuel costs
            ctbo_diluted_cost = csu_diluted_cost * plant['ktCO2f_yr_baseline']  # internalized fuel costs
            
            # NOTE: think about what PROFITS and COSTS we can actually include in NPV calculations... for example, ETS avoided? Draw MACC!
            # Why are fossil plants profiting MORE when the ETS price is very high? => Internalized costs are low
            # I think the NPV calculations refer to only CSUs: the profits of selling CSUs (above the ETS price... so the NPV of "engaging with the CTBO-CSU program") and the costs of internalizing CSUs...
            # This is the "additional" NPV provided by a CTBO!
            # Why do fossil emitters benefit from high ETS prices? Because the dark blue area decreases, i.e., a greater fraction of the CSU cost is pure profit, rather than internalized costs.
            # We could make an alternative NPV calculation including CCS costs and ETS profits - however, it does not match the investment logic of MACCcost<ETSprice
            if plant['invested']:
                investment_year = plant['year_invested']
                CO2_captured_fossil = plant['ktCO2f_yr_captured']
                CO2_captured_bio = plant['ktCO2bio_yr_captured']
                csu_gross_profit = CSU_cost
                csu_net_profit = csu_gross_profit - csu_subtract # [EUR/tCO2]
                ctbo_fossil_profit = CSU_cost * CO2_captured_fossil
                ctbo_gross_profit = CSU_cost * (CO2_captured_fossil + CO2_captured_bio)
                ctbo_net_profit = csu_net_profit * (CO2_captured_fossil + CO2_captured_bio) # [kEUR/yr] total profit 
            else:
                investment_year = None
                CO2_captured_fossil = 0
                CO2_captured_bio = 0
                csu_gross_profit = 0
                csu_net_profit = -csu_diluted_cost
                ctbo_fossil_profit = 0
                ctbo_gross_profit = 0
                ctbo_net_profit = -csu_diluted_cost * plant['ktCO2f_yr_baseline']
            
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
    # Calculate NPV for each plant
    plant_results = pd.DataFrame(plant_results)
    plant_npv = []
    for plant_name in plant_results['plant'].unique():
        plant_data = plant_results[plant_results['plant'] == plant_name].copy()
        investment_year = plant_data['investment_year'].max()
        if pd.isna(investment_year):
            continue

        base_year = investment_year if USE_INVESTMENT_YEAR_AS_BASE else START_YEAR
        plant_data['discount_factor'] = 1 / (1 + DISCOUNT_RATE) ** (plant_data['year'] - base_year)
        npv_gross_profit = (plant_data['ctbo_gross_profit'] * plant_data['discount_factor']).sum()
        npv_net_profit = (plant_data['ctbo_net_profit'] * plant_data['discount_factor']).sum()

        plant_npv.append({
            'plant': plant_name,
            'npv_gross_profit': npv_gross_profit,
            'npv_net_profit': npv_net_profit,
            'investment_year': investment_year,
        })

    results = {}
    results['macc_4oak'] = macc_4oak
    results['macc_foak'] = macc_foak
    results['macc'] = macc
    results['years'] = years
    results['ctbo_fraction'] = ctbo_fraction
    results['ets_prices'] = ets_prices
    results['DACCS_costs'] = DACCS_costs
    results['supplied_CO2_vec'] = supplied_CO2_vec
    results['total_emissions_vec'] = total_emissions_vec
    results['ctbo_mandate_vec'] = ctbo_mandate_vec
    results['fCCS_capacity_vec'] = fCCS_capacity_vec
    results['BECCS_capacity_vec'] = BECCS_capacity_vec
    results['DACCS_capacity_vec'] = DACCS_capacity_vec
    results['marginal_cost_vec'] = marginal_cost_vec
    results['CSU_cost_vec'] = CSU_cost_vec
    results['CTBO_cost_lev_vec'] = CTBO_cost_lev_vec
    results['plant_results'] = plant_results
    results['first_DACCS_year'] = first_DACCS_year
    results['pounds_to_EUR'] = pounds_to_EUR
    results['fuels'] = fuels
    results['START_YEAR'] = START_YEAR
    results['DISCOUNT_RATE'] = DISCOUNT_RATE
    results['USE_INVESTMENT_YEAR_AS_BASE'] = USE_INVESTMENT_YEAR_AS_BASE
    results['plant_npv'] = plant_npv
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("COMBINED MACC + CTBO SIMULATION")
    print("="*80)
    
    # Load data once
    data = load_data(debug=True)
    
    # Run simulation with default parameters (or override as needed)
    results = combined_simulation(data)
    
    # Extract results
    macc_4oak = results['macc_4oak']
    macc_foak = results['macc_foak']
    macc = results['macc']
    years = results['years']
    ctbo_fraction = results['ctbo_fraction']
    ets_prices = results['ets_prices']
    DACCS_costs = results['DACCS_costs']
    supplied_CO2_vec = results['supplied_CO2_vec']
    total_emissions_vec = results['total_emissions_vec']
    ctbo_mandate_vec = results['ctbo_mandate_vec']
    fCCS_capacity_vec = results['fCCS_capacity_vec']
    BECCS_capacity_vec = results['BECCS_capacity_vec']
    DACCS_capacity_vec = results['DACCS_capacity_vec']
    marginal_cost_vec = results['marginal_cost_vec']
    CSU_cost_vec = results['CSU_cost_vec']
    CTBO_cost_lev_vec = results['CTBO_cost_lev_vec']
    plant_results = results['plant_results']
    first_DACCS_year = results['first_DACCS_year']
    pounds_to_EUR = results['pounds_to_EUR']
    fuels = results['fuels']
    START_YEAR = results['START_YEAR']
    DISCOUNT_RATE = results['DISCOUNT_RATE']
    USE_INVESTMENT_YEAR_AS_BASE = results['USE_INVESTMENT_YEAR_AS_BASE']
    plant_npv = results['plant_npv']
    
    # ============================================================================
    # RESULTS
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
    
    # ============================================================================
    # PLOTS
    # ============================================================================
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    viridis = plt.cm.viridis
    magma = plt.cm.magma
    
    # Plot 1: MACC curves
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
    plt.stackplot(years, DACCS_capacity_vec, BECCS_capacity_vec, fCCS_capacity_vec,
                  labels=['DACCS', 'BECCS', 'Fossil CCS'],
                  colors=[viridis(0.5), viridis(0.65), 'gray'],
                  alpha=1.0)
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
    
    # Plot 4: Consumer fuel price increases (full range)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
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

    # Plot 5: Plant-level NPV
    plant_npv_df = pd.DataFrame(plant_npv) # Omit Drax-BECCS for clarity
    plant_npv_df = plant_npv_df[plant_npv_df['plant'] != 'Drax-BECCS']
    print("\n" + "="*80)
    print(f"Plant-level NPV: {len(plant_npv_df)} plants")
    plt.figure(figsize=(14, 6))
    x = range(len(plant_npv_df))
    # plt.bar(x, plant_npv_df['npv_gross_profit']/1000, label='Gross Profit', color='red', alpha=0.5)
    plt.bar(x, plant_npv_df['npv_net_profit']/1000, label='Net Profit', color='green', alpha=0.5)
    plt.xlabel('Plant', fontsize=13)
    plt.ylabel('NPV (kEUR)', fontsize=13)
    plt.title('Plant-level NPV', fontsize=14)
    plt.legend(fontsize=11)
    plt.xticks(x, plant_npv_df['plant'], rotation=90, fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 6: Plant-level NPV vs investment year
    plt.figure(figsize=(10, 6))
    colors = ['green' if p.endswith('W2E') or p.endswith('BECCS') else 'gray' for p in plant_npv_df['plant']]
    plt.scatter(plant_npv_df['investment_year'], plant_npv_df['npv_net_profit']/1000, c=colors, alpha=0.5)
    plt.scatter([], [], color='green', alpha=0.5, label='W2E / BECCS')
    plt.scatter([], [], color='gray', alpha=0.5, label='Other (fossil)')
    plt.xlabel('Investment Year', fontsize=13)
    plt.ylabel('NPV (kEUR)', fontsize=13)
    plt.title('Plant-level NPV vs Investment Year', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('6_npv_vs_investment_year.png', dpi=300)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    plt.show()
