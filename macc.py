"""
Marginal Abatement Cost Curve (MACC) Analysis for UK CCS

This script calculates the cost of carbon capture and storage (CCS) for major UK CO2 emitters
across multiple sectors: power generation, industrial facilities, waste-to-energy, and BECCS.

Author: Oscar Stenstrom
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import geopandas as gpd


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def match_transportation(emitter_plant, transport_costs_df, discount_rate=0.035, lifetime=30, debug=False):
    """
    Find the nearest CCS site to an emitter plant and calculate total cost per tonne CO2.
    
    Parameters:
        emitter_plant: pandas Series with 'Easting' and 'Northing' columns
        transport_costs_df: DataFrame with CCS transport cost data
        discount_rate: Discount rate for annualization (default: 0.035)
        lifetime: Project lifetime in years (default: 30)
        debug: If True, print debug information
    
    Returns:
        dict: Contains site details and costs
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
    onshore_mode = closest_site_original['CO2 Pipeline/Trucking?']
    
    # Calculate onshore costs per tonne CO2
    onshore_opex = float(closest_site_original['CO2 onshore transport opex (£m/y)']) / mass_CO2
    
    if onshore_mode == "Pipeline":
        pipeline_capex = float(closest_site_original['CO2 onshore pipeline capex (£m)'])
        annuity_factor = (discount_rate * (1 + discount_rate) ** lifetime) / ((1 + discount_rate) ** lifetime - 1)
        onshore_pipeline = pipeline_capex * annuity_factor / mass_CO2
        onshore_opex += onshore_pipeline
    
    injection_opex = float(closest_site_original['CO2 T&S cost from defined point (£/t)'])
    total_cost = onshore_opex + injection_opex
    
    result = {
        'site_id': closest_site_id,
        'site_name': closest_site_original['Site'],
        'distance_m': distances_df.loc[closest_distance_idx, 'Distance'],
        'mass_co2_kt': round(mass_CO2 * 1000, 1),
        'co2_point': closest_site_original['CO2 Point'],
        'terminal': closest_site_original['Final CO2 Terminal'],
        'injection_site': closest_site_original['Injection Site'],
        'onshore_mode': onshore_mode,
        'onshore_cost_per_t': round(onshore_opex, 2),
        'injection_cost_per_t': round(injection_opex, 2),
        'total_cost_per_t': round(total_cost, 1)
    }
    
    if debug:
        print(f"\nClosest site: {result['site_name']} (ID: {result['site_id']}) of emitter: {emitter_plant['Name']}")
        print(f"Distance: {result['distance_m']:.2f} meters")
        print(f"Total cost: {result['total_cost_per_t']} £/tCO2")
    
    return result


def calculate_distance(easting1, northing1, easting2, northing2, debug=False):
    """Calculate Euclidean distance between two points in meters"""
    if debug:
        print(f"Calculating distance: ({easting1}, {northing1}) to ({easting2}, {northing2})")
    return ((easting1 - easting2)**2 + (northing1 - northing2)**2)**0.5


def latlon_to_easting_northing(latitude, longitude, debug=False):
    """Convert WGS84 lat/lon to British National Grid Easting/Northing (EPSG:27700)"""
    if debug:
        print(f"Converting: Lat={latitude}, Lon={longitude}")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    if debug:
        print(f"Result: Easting={easting}, Northing={northing}")
    return easting, northing


def approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023=798.7, capture_rate=0.90, NETL=5.509, debug=False):
    """
    Estimate CAPEX for CO2 capture using Kim & Leonard (2025) correlation.
    
    Parameters:
        mCO2: CO2 mass flow rate [tCO2/h]
        xCO2: CO2 concentration in flue gas [-]
        CEPCI_2025: Chemical Engineering Plant Cost Index for 2025
        CEPCI_2023: Chemical Engineering Plant Cost Index for 2023 (default: 798.7)
        capture_rate: CO2 capture rate (default: 0.90)
        NETL: NETL methodology multiplier (default: 5.509)
        debug: If True, print inputs and outputs
        
    Returns:
        float: Total CAPEX in MEUR
    """
    if debug:
        print(f"approximate_CAPEX inputs: mCO2={mCO2}, xCO2={xCO2}")
    
    # Correlation coefficients from Kim & Leonard (2025)
    a, b, c, n, m = 2.1673, 0.8092, -0.00332, 0.5291, 0.8391
    
    # Convert CO2 flow to flue gas volume
    nCO2 = mCO2 * 1000 / 44  # [kmolCO2/h]
    n_fluegas = nCO2 / xCO2  # [kmol/h]
    V_fluegas = n_fluegas * 22.4  # [Nm3/h]
    
    # Split into multiple absorbers if needed (max 1613 × 10^3 Nm3/h per absorber)
    n_largest_absorbers = int((V_fluegas/1000) // 1613)
    remaining_V_fluegas = (V_fluegas/1000) % 1613  # [10^3 Nm3/h]
    
    CAPEX = 0
    for i in range(n_largest_absorbers):
        TEC = a + (b * (xCO2)**n + c) * (1613)**m
        CAPEX += TEC
    
    if remaining_V_fluegas > 0:
        TEC = a + (b * (xCO2)**n + c) * (remaining_V_fluegas)**m
        CAPEX += TEC
    
    CAPEX = CAPEX * NETL * CEPCI_2025 / CEPCI_2023  # [MEUR]
    
    if debug:
        print(f"approximate_CAPEX output: CAPEX={CAPEX} MEUR")
    
    return CAPEX


def levelize_MEUR(CAPEX, annual_CO2, capture_rate=0.95, discount_rate=0.07, lifetime=25, debug=False):
    """
    Convert CAPEX [MEUR] to levelized cost [EUR/tCO2].
    
    Parameters:
        CAPEX: Capital expenditure [MEUR]
        annual_CO2: Annual CO2 emissions [tCO2/yr]
        capture_rate: CO2 capture rate (default: 0.95)
        discount_rate: Discount rate (default: 0.07)
        lifetime: Project lifetime in years (default: 25)
        debug: If True, print inputs and outputs
        
    Returns:
        float: Levelized CAPEX [EUR/tCO2]
    """
    if debug:
        print(f"levelize_MEUR inputs: CAPEX={CAPEX}, annual_CO2={annual_CO2}")
    
    annualized_CAPEX = CAPEX * discount_rate * (1 + discount_rate)**lifetime / ((1 + discount_rate)**lifetime - 1) * 10**6
    levelized_CAPEX = annualized_CAPEX / (annual_CO2 * capture_rate)
    
    if debug:
        print(f"levelize_MEUR output: {levelized_CAPEX} EUR/tCO2")
    
    return levelized_CAPEX


def energy_supply(mCO2, capture_rate, FLH, qreb, pcompr, qsteam, qelc, qchp, csteam, elc_eff, 
                  cbio, celc, emission_factor, annual_CO2, evaporation_enthalpy=2257, 
                  maximize_beccs=False, debug=False):
    """
    Calculate energy penalty costs (OPEXE) for CCS including steam, electricity, and biomass CHP.
    
    Parameters:
        mCO2: CO2 mass flow rate [tCO2/h]
        capture_rate: CO2 capture rate [-]
        FLH: Full load hours [h/y]
        qreb: Specific reboiler heat duty [MJ/kgCO2]
        pcompr: Specific compression power [MJ/kgCO2]
        qsteam: Fraction of Qreb covered by steam [-]
        qelc: Fraction of Qreb covered by electricity [-]
        qchp: Fraction of Qreb covered by biomass CHP [-]
        csteam: Steam cost [EUR/tsteam]
        elc_eff: Efficiency of electrified reboiler [-]
        cbio: Biomass cost [EUR/MWh]
        celc: Electricity cost [EUR/MWh]
        emission_factor: Biomass emission factor [tCO2/MWhfuel]
        annual_CO2: Annual CO2 emissions [tCO2/yr]
        evaporation_enthalpy: Enthalpy of vaporization [MJ/tsteam] (default: 2257)
        maximize_beccs: If True, maximize biogenic CO2 capture (default: False)
        debug: If True, print inputs and outputs
        
    Returns:
        tuple: (mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel)
    """
    if debug:
        print(f"energy_supply inputs: mCO2={mCO2}, capture_rate={capture_rate}")
    
    mCO2f_captured = mCO2 * capture_rate  # [tCO2/h]
    mCO2f_residual = mCO2 * (1 - capture_rate)  # [tCO2/h]
    
    Qreb = qreb * mCO2f_captured * 1000 / 3600  # [MW]
    Pcompr = pcompr * mCO2f_captured * 1000 / 3600  # [MW]
    
    # Energy sources
    Qsteam = qsteam * Qreb  # [MW]
    Qelc = qelc * Qreb  # [MW]
    Qchp = qchp * Qreb  # [MW]
    
    # Steam OPEX
    cost_steam = csteam / evaporation_enthalpy  # [EUR/MJ]
    Qsteam_annual = Qsteam * FLH * 3600  # [MJ/y]
    OPEX_steam = cost_steam * Qsteam_annual / (annual_CO2 * capture_rate)  # [EUR/tCO2]
    
    # Electricity OPEX
    Pelec = (Qelc * elc_eff + Pcompr) * FLH  # [MWh/y]
    OPEX_elec = celc * Pelec / (annual_CO2 * capture_rate)  # [EUR/tCO2]
    
    # Biomass CHP OPEX
    if maximize_beccs:
        Qfuel = Qchp / (1 - capture_rate * emission_factor * (qreb/3600*1000))
        mCO2bio_captured = Qfuel * emission_factor * capture_rate  # [tCO2/h]
    else:
        Qfuel = Qchp  # [MW]
        mCO2bio_captured = 0
    
    OPEX_chp = cbio * Qfuel * FLH / (annual_CO2 * capture_rate)  # [EUR/tCO2]
    
    OPEXE = OPEX_steam + OPEX_elec + OPEX_chp
    
    if debug:
        print(f"energy_supply output: OPEXE={OPEXE} EUR/tCO2")
    
    return mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel


def create_result_entry(site_stack, annual_CO2, biogenic, captured_CO2f, captured_CO2bio, 
                       residual_CO2f, FLH, CAPEX_4OAK, CAPEX_FOAK, OPEXE, transtorage, distance_km):
    """Create a standardized result dictionary entry"""
    return {
        'site-stack': site_stack,
        'annual_CO2': annual_CO2 / 1000,  # [ktCO2/yr]
        'biogenic': biogenic,
        'captured_CO2f': captured_CO2f,  # [ktCO2/yr]
        'captured_CO2bio': captured_CO2bio,  # [ktCO2/yr]
        'residual_CO2f': residual_CO2f,  # [ktCO2/yr]
        'FLH': FLH,
        'CAPEX_4OAK': CAPEX_4OAK,  # [EUR/tCO2]
        'CAPEX_FOAK': CAPEX_FOAK,  # [EUR/tCO2]
        'OPEXE': OPEXE,  # [EUR/tCO2]
        'transtorage': transtorage,  # [EUR/tCO2]
        'total_4OAK': CAPEX_4OAK + OPEXE + transtorage,  # [EUR/tCO2]
        'total_FOAK': CAPEX_FOAK + OPEXE + transtorage,  # [EUR/tCO2]
        'distance_km': distance_km
    }


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Economic parameters
pounds_to_EUR = 1.15
CEPCI_2025 = 930
CEPCI_2023 = 798.7
FOAK = 1.7553  # First-of-a-kind multiplier (from foak.py)

# CCS parameters
capture_rate = 0.95
discount_rate = 0.07
lifetime = 25  # [years]
qreb = 3.5  # [MJ/kgCO2]
pcompr = 0.37  # [MJ/kgCO2]

# Energy and makeup costs
celc = 80  # [EUR/MWh]
cbio = 65  # [EUR/MWh]
csteam = 4.1  # [EUR/tsteam @130C]
amine_cost = 44  # [SEK/tCO2]
sek_to_eur = 0.091

# Power sector (CCGT) assumptions
power_ccgt = {
    'xCO2': 0.05,  # CCGT plants CO2 concentration
    'eta_P': 0.49,  # [MWel/MWfuel] total efficiency from DUKES data
    'emission_factor': 0.204,  # [tCO2/MWhfuel] from NZIP (2020)
    'gas_eff': 0.35,  # Gas turbine efficiency
    'steam_eff': 0.51  # Steam cycle efficiency
}

# Waste-to-Energy (W2E) sector assumptions
w2e = {
    'emission_factor': 0.98,  # [tCO2/twaste] Tolvik report
    'FLH': 8760 * 0.866,  # [h/y]
    'fossil_fraction': 0.465  # Fossil content of W2E CO2
}

# Drax BECCS assumptions
drax = {
    'CO2': 11500000,  # [tCO2/yr]
    'Pinstalled': 2580,  # [MW]
    'eta_P': 33 / (1 - 0.24)  # ~44% before CCS, 33% after
}

# Energy supply fractions for industrial sector
qsteam = 0.15  # Fraction of Qreb covered by steam
qelc = 0.30  # Fraction of Qreb covered by electricity
qchp = 0.55  # Fraction of Qreb covered by biomass CHP
elc_eff = 0.33  # Efficiency of electrified reboiler
evaporation_enthalpy = 2257  # [MJ/tsteam]
emission_factor_bio = 0.3318  # [tCO2/MWhfuel]


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

print("\n" + "="*80)
print("LOADING AND PREPARING DATA")
print("="*80)

# Load fossil fuel point sources
fossil_plants = pd.read_csv("data/point_sources_CO2_2022.csv")
fossil_plants['CO2'] = fossil_plants['Emission'] * 3.66
CO2_fossil = fossil_plants['CO2'].sum()

# Select top 60 emitters
largest_plants = fossil_plants.nlargest(60, 'CO2')[['PlantID', 'Site', 'Easting', 'Northing', 'Operator', 'Sector', 'CO2']]

print(f"\nThe 60 largest fossil CO2 emitters (2022):")
for i, (idx, row) in enumerate(largest_plants.iterrows(), 1):
    print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tCO2/yr")

print(f"\nSum of all point-source fossil emitters: {CO2_fossil*10**-6:.1f} MtCO2/yr")
print(f"Sum of top 60 fossil emitters: {largest_plants['CO2'].sum()*10**-6:.1f} MtCO2/yr")

# Clean data: remove outliers and handle duplicates
largest_plants = largest_plants[largest_plants['Site'] != "Elgin PUQ"]  # North Sea location
largest_plants.loc[largest_plants['Site'] == 'Grangemouth Power Station', 'Sector'] = 'Major power producers'

# Combine duplicate Fawley Refinery entries
fawley_mask = largest_plants['Site'] == 'Fawley Refinery'
if fawley_mask.sum() > 1:
    fawley_total_co2 = largest_plants.loc[fawley_mask, 'CO2'].sum()
    first_fawley_idx = largest_plants[fawley_mask].index[0]
    largest_plants = largest_plants[~fawley_mask | (largest_plants.index == first_fawley_idx)].copy()
    largest_plants.loc[first_fawley_idx, 'CO2'] = fawley_total_co2

# Load transport cost data
transport_costs = pd.read_csv("data/nzip_balanced_scenario_results.csv", encoding='latin-1')
transport_costs = transport_costs[transport_costs["CO2 Pipeline/Trucking?"] != "No CCS"]

# Calculate transport costs for all plants
print("\nCalculating transport costs for largest plants...")
for idx, plant in largest_plants.iterrows():
    result = match_transportation(plant, transport_costs, discount_rate=0.035, lifetime=30, debug=False)
    largest_plants.loc[idx, 'transport_cost'] = result['total_cost_per_t'] * pounds_to_EUR * CEPCI_2025/CEPCI_2023
    largest_plants.loc[idx, 'distance_km'] = result['distance_m'] / 1000

# Load W2E plants
w2e_plants = pd.read_csv("data/w2e_plants.csv")
w2e_plants = w2e_plants[w2e_plants['No.'] != 40]  # Remove Shetland Islands plant

# Convert coordinates and calculate transport costs for W2E plants
print("Processing W2E plants...")
w2e_plants['Easting'] = 0.0
w2e_plants['Northing'] = 0.0
w2e_plants['transport_cost'] = 0.0

for idx, plant in w2e_plants.iterrows():
    easting, northing = latlon_to_easting_northing(plant['Latitude'], plant['Longitude'], debug=False)
    w2e_plants.loc[idx, 'Easting'] = easting
    w2e_plants.loc[idx, 'Northing'] = northing
    plant_with_coords = w2e_plants.loc[idx]
    result = match_transportation(plant_with_coords, transport_costs, discount_rate=0.035, lifetime=30, debug=False)
    w2e_plants.loc[idx, 'transport_cost'] = result['total_cost_per_t'] * pounds_to_EUR * CEPCI_2025/CEPCI_2023
    w2e_plants.loc[idx, 'distance_km'] = result['distance_m'] / 1000


# ============================================================================
# POWER SECTOR
# ============================================================================

print("\n" + "="*80)
print("POWER SECTOR")
print("="*80)

results = []

# Extract power CCGT parameters
xCO2 = power_ccgt['xCO2']
eta_P = power_ccgt['eta_P']
emission_factor = power_ccgt['emission_factor']
gas_eff = power_ccgt['gas_eff']
steam_eff = power_ccgt['steam_eff']

# Load power capacities
power_capacities = pd.read_csv("data/power_capacities_clean.csv")
power_producers = largest_plants[
    (largest_plants['Sector'] == 'Major power producers') | 
    (largest_plants['Sector'] == 'Minor power producers')
].copy()
power_producers = power_producers[power_producers['Site'] != 'Ratcliffe on Soar Power Station'].copy()
power_producers = power_producers.merge(power_capacities, left_on='Site', right_on='Power plant name', how='left')

for idx, plant in power_producers.iterrows():
    Pinstalled = plant['Capacity [MW]']
    Qfuel = Pinstalled / eta_P
    mCO2 = Qfuel * emission_factor  # [tCO2/h]
    FLH = plant['CO2'] / mCO2  # [h/y]
    
    # CAPEX
    CAPEX = approximate_CAPEX(mCO2, xCO2, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=5.509)
    levelized_CAPEX = levelize_MEUR(CAPEX, plant['CO2'], capture_rate, discount_rate, lifetime)
    CAPEX_4OAK = levelized_CAPEX
    CAPEX_FOAK = FOAK * CAPEX_4OAK
    
    # OPEXE - Energy penalty
    mCO2_captured = mCO2 * capture_rate
    mCO2_residual = mCO2 * (1 - capture_rate)
    
    Pgas = Qfuel * gas_eff
    Qsteam = Qfuel * steam_eff
    Prankine = Pinstalled - Pgas
    eta_steam = Prankine / Qsteam
    
    Qreb_power = qreb * mCO2_captured * 1000 / 3600  # [MW]
    Plost = Qreb_power * eta_steam * FLH  # [MWh/y]
    
    OPEXE = Plost * celc / (mCO2_captured * FLH)
    OPEXE += pcompr * 1000/3600 * celc  # Compression penalty
    
    results.append(create_result_entry(
        f"{plant['Site']}-CCGT",
        plant['CO2'], 0,
        mCO2_captured * FLH / 1000, 0,
        mCO2_residual * FLH / 1000,
        FLH, CAPEX_4OAK, CAPEX_FOAK, OPEXE,
        plant['transport_cost'], plant['distance_km']
    ))


# ============================================================================
# INDUSTRY SECTOR
# ============================================================================

print("\n" + "="*80)
print("INDUSTRY SECTOR")
print("="*80)

FLH_industry = 8500  # [h/y]

# Select industrial plants
refineries = largest_plants[largest_plants['Sector'] == 'Processing & distribution of petroleum products'].copy()
scunthorpe_stacks = largest_plants[largest_plants['Site'].str.startswith('Scunthorpe', na=False)].copy()
cement_plants = largest_plants[largest_plants['Sector'] == 'Cement'].copy()
industrial_plants = pd.concat([refineries, scunthorpe_stacks, cement_plants], ignore_index=True)

# Process each industrial plant
for idx, plant in industrial_plants.iterrows():
    
    # REFINERIES
    if plant['Sector'] == 'Processing & distribution of petroleum products':
        annual_CO2 = plant['CO2']
        stacks = {
            'power': [annual_CO2 * 0.298, (3*25 + 8*54)/(25+54)/100],
            'crackers': [annual_CO2 * 0.20, 17/100],
            'distillation': [annual_CO2 * 0.17, 11/100],
            'smr': [annual_CO2 * 0.118, (8*6 + 24*26)/(6+26)/100],
            'remaining': [annual_CO2 * 0.188, 8/100],
        }
        
        for stack_name, stack_data in stacks.items():
            annual_CO2_stack = stack_data[0]
            xCO2_stack = stack_data[1]
            mCO2 = annual_CO2_stack / FLH_industry
            
            mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(
                mCO2, capture_rate, FLH_industry, qreb, pcompr, qsteam, qelc, qchp,
                csteam, elc_eff, cbio, celc, emission_factor_bio, annual_CO2_stack, evaporation_enthalpy
            )
            mCO2_total = mCO2f_captured + mCO2bio_captured
            
            CAPEX = approximate_CAPEX(mCO2_total, xCO2_stack, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=5.509)
            levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2_stack, capture_rate, discount_rate, lifetime)
            CAPEX_4OAK = levelized_CAPEX
            CAPEX_FOAK = FOAK * CAPEX_4OAK
            
            results.append(create_result_entry(
                f"{plant['Site']}-{stack_name}",
                annual_CO2_stack, 0,
                mCO2f_captured * FLH_industry / 1000,
                mCO2bio_captured * FLH_industry / 1000,
                mCO2f_residual * FLH_industry / 1000,
                FLH_industry, CAPEX_4OAK, CAPEX_FOAK, OPEXE,
                plant['transport_cost'], plant['distance_km']
            ))
    
    # IRON & STEEL
    if plant['Sector'] == 'Iron & steel industries':
        if plant['Site'] == 'Scunthorpe Power Station':
            stack_name, xCO2_stack = 'chp', 0.296
        elif plant['Site'] == 'Scunthorpe Blast Furnaces':
            stack_name, xCO2_stack = 'stove', 0.251
        elif plant['Site'] == 'Scunthorpe Sinter':
            stack_name, xCO2_stack = 'sinter', 0.15
        
        annual_CO2_stack = plant['CO2']
        mCO2 = annual_CO2_stack / FLH_industry
        
        mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(
            mCO2, capture_rate, FLH_industry, qreb, pcompr, qsteam, qelc, qchp,
            csteam, elc_eff, cbio, celc, emission_factor_bio, annual_CO2_stack, evaporation_enthalpy
        )
        mCO2_total = mCO2f_captured + mCO2bio_captured
        
        CAPEX = approximate_CAPEX(mCO2_total, xCO2_stack, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=5.509)
        levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2_stack, capture_rate, discount_rate, lifetime)
        CAPEX_4OAK = levelized_CAPEX
        CAPEX_FOAK = FOAK * CAPEX_4OAK
        
        results.append(create_result_entry(
            f"{plant['Site']}-{stack_name}",
            annual_CO2_stack, 0,
            mCO2f_captured * FLH_industry / 1000,
            mCO2bio_captured * FLH_industry / 1000,
            mCO2f_residual * FLH_industry / 1000,
            FLH_industry, CAPEX_4OAK, CAPEX_FOAK, OPEXE,
            plant['transport_cost'], plant['distance_km']
        ))
    
    # CEMENT
    if plant['Sector'] == 'Cement':
        annual_CO2_stack = plant['CO2']
        xCO2_stack = 0.20
        mCO2 = annual_CO2_stack / FLH_industry
        
        mCO2f_captured, mCO2bio_captured, mCO2f_residual, OPEXE, Qfuel_tot = energy_supply(
            mCO2, capture_rate, FLH_industry, qreb, pcompr, qsteam, qelc, qchp,
            csteam, elc_eff, cbio, celc, emission_factor_bio, annual_CO2_stack, evaporation_enthalpy
        )
        mCO2_total = mCO2f_captured + mCO2bio_captured
        
        CAPEX = approximate_CAPEX(mCO2_total, xCO2_stack, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=5.509)
        levelized_CAPEX = levelize_MEUR(CAPEX, annual_CO2_stack, capture_rate, discount_rate, lifetime)
        CAPEX_4OAK = levelized_CAPEX
        CAPEX_FOAK = FOAK * CAPEX_4OAK
        
        results.append(create_result_entry(
            f"{plant['Site']}-cement",
            annual_CO2_stack, 0,
            mCO2f_captured * FLH_industry / 1000,
            mCO2bio_captured * FLH_industry / 1000,
            mCO2f_residual * FLH_industry / 1000,
            FLH_industry, CAPEX_4OAK, CAPEX_FOAK, OPEXE,
            plant['transport_cost'], plant['distance_km']
        ))


# ============================================================================
# WASTE-TO-ENERGY SECTOR
# ============================================================================

print("\n" + "="*80)
print("WASTE-TO-ENERGY SECTOR")
print("="*80)

# Extract W2E parameters
emission_factor_w2e = w2e['emission_factor']
FLH_w2e = w2e['FLH']
fossil_fraction = w2e['fossil_fraction']

w2e_plants['CO2'] = w2e_plants['Capacity 2023 [ktpa]'] * 1000 * emission_factor_w2e

for idx, plant in w2e_plants.iterrows():
    mCO2 = plant['CO2'] / FLH_w2e
    xCO2_w2e = 0.11
    
    # CAPEX
    CAPEX = approximate_CAPEX(mCO2, xCO2_w2e, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=5.509)
    levelized_CAPEX = levelize_MEUR(CAPEX, plant['CO2'], capture_rate, discount_rate, lifetime)
    CAPEX_4OAK = levelized_CAPEX
    CAPEX_FOAK = FOAK * CAPEX_4OAK
    
    # OPEXE - Assuming waste heat is used for reboiler
    Qreb_w2e = qreb * mCO2 * capture_rate * 1000 / 3600  # [MW]
    cost_steam = csteam / evaporation_enthalpy  # [EUR/MJ]
    Qsteam_annual = Qreb_w2e * FLH_w2e * 3600  # [MJ/y]
    OPEX_steam = cost_steam * Qsteam_annual / (plant['CO2'] * capture_rate)
    OPEXE = OPEX_steam + pcompr * 1000/3600 * celc
    
    results.append(create_result_entry(
        f"{plant['Name']}-W2E",
        plant['CO2'], 1 - fossil_fraction,
        mCO2 * capture_rate * FLH_w2e / 1000 * fossil_fraction,
        mCO2 * capture_rate * FLH_w2e / 1000 * (1 - fossil_fraction),
        mCO2 * (1 - capture_rate) * fossil_fraction * FLH_w2e / 1000,
        FLH_w2e, CAPEX_4OAK, CAPEX_FOAK, OPEXE,
        plant['transport_cost'], plant['distance_km']
    ))


# ============================================================================
# DRAX BECCS
# ============================================================================

print("\n" + "="*80)
print("DRAX BECCS")
print("="*80)

# Extract Drax parameters
Drax_CO2 = drax['CO2']
Pinstalled_drax = drax['Pinstalled']
eta_P_drax = drax['eta_P']

Qfuel_drax = Pinstalled_drax / (eta_P_drax / 100)
FLH_drax = Drax_CO2 / (Qfuel_drax * emission_factor)

# CAPEX
xCO2_drax = 0.13
mCO2_drax = Drax_CO2 / FLH_drax
CAPEX_drax = approximate_CAPEX(mCO2_drax, xCO2_drax, CEPCI_2025, CEPCI_2023, capture_rate=0.90, NETL=5.509)
levelized_CAPEX_drax = levelize_MEUR(CAPEX_drax, Drax_CO2, capture_rate, discount_rate, lifetime)
CAPEX_4OAK_drax = levelized_CAPEX_drax
CAPEX_FOAK_drax = FOAK * CAPEX_4OAK_drax

# OPEXE - Lost revenue from efficiency penalty
profit_baseline = Qfuel_drax * eta_P_drax/100 * FLH_drax * celc
profit_BECCS = Qfuel_drax * eta_P_drax/100 * (1 - 0.24) * FLH_drax * celc
difference = profit_baseline - profit_BECCS
OPEXE_drax = difference / (Drax_CO2 * capture_rate)
OPEXE_drax += pcompr * 1000/3600 * celc

# Transport costs
drax_coordinates = (53.738710, -0.993030)
drax_easting, drax_northing = latlon_to_easting_northing(drax_coordinates[0], drax_coordinates[1])
drax_plant = pd.Series({'Easting': drax_easting, 'Northing': drax_northing, 'Name': 'Drax'})
result_drax = match_transportation(drax_plant, transport_costs, discount_rate=0.035, lifetime=30, debug=False)
transport_cost_drax = result_drax['total_cost_per_t'] * pounds_to_EUR * CEPCI_2025/CEPCI_2023
drax_distance = result_drax['distance_m'] / 1000

results.append(create_result_entry(
    'Drax-BECCS',
    Drax_CO2, 1,
    0, mCO2_drax * capture_rate * FLH_drax / 1000,
    0, FLH_drax,
    CAPEX_4OAK_drax, CAPEX_FOAK_drax, OPEXE_drax,
    transport_cost_drax, drax_distance
))


# ============================================================================
# POST-PROCESSING
# ============================================================================

# Add amine makeup cost (assumption from Ramboll)
for result in results:
    result['total_4OAK'] += amine_cost * sek_to_eur # [SEK/tCO2] to [EUR/tCO2]
    result['total_FOAK'] += amine_cost * sek_to_eur


# ============================================================================
# MACC CURVE GENERATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING MACC CURVES")
print("="*80)

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_4OAK')
    results_df['captured_total'] = results_df['captured_CO2f'] + results_df['captured_CO2bio']
    results_df['cumulative_captured'] = results_df['captured_total'].cumsum()
    
    # Save 4OAK data
    macc_4oak_df = pd.DataFrame({
        'site-stack': results_df['site-stack'],
        'ktCO2f_yr_baseline': results_df['annual_CO2'] * (1 - results_df['biogenic']),
        'ktCO2bio_yr_baseline': results_df['annual_CO2'] * results_df['biogenic'],
        'ktCO2f_yr_captured': results_df['captured_CO2f'],
        'ktCO2bio_yr_captured': results_df['captured_CO2bio'],
        'ktCO2_yr_cumulative': results_df['cumulative_captured'],
        'ktCO2f_yr_residual': results_df['residual_CO2f'],
        'EUR/tCO2': results_df['total_4OAK']
    })
    macc_4oak_df.to_csv('macc_4oak.csv', index=False)
    
    # Plot 4OAK MACC
    plt.figure(figsize=(12, 8))
    x_steps, y_steps = [], []
    for i in range(len(results_df) - 1):
        x_steps.extend([results_df['cumulative_captured'].iloc[i], results_df['cumulative_captured'].iloc[i+1]])
        y_steps.extend([results_df['total_4OAK'].iloc[i+1], results_df['total_4OAK'].iloc[i+1]])
    
    plt.plot(x_steps, y_steps, 'b-', linewidth=3, alpha=0.7)
    plt.xlabel('Cumulative Captured CO2 (ktCO2/yr)', fontsize=14)
    plt.ylabel('Marginal Abatement Cost (EUR/tCO2)', fontsize=14)
    plt.title('Marginal Abatement Cost Curve (4OAK)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    total_captured = results_df['cumulative_captured'].iloc[-1]
    avg_cost = results_df['total_4OAK'].mean()
    min_cost = results_df['total_4OAK'].min()
    max_cost = results_df['total_4OAK'].max()
    
    plt.text(0.02, 0.98, f'Total Captured: {total_captured:.0f} ktCO2/yr\n'
                         f'Average Cost: {avg_cost:.1f} EUR/tCO2\n'
                         f'Cost Range: {min_cost:.1f} - {max_cost:.1f} EUR/tCO2',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=12)
    plt.tight_layout()
    
    print(f"\nMarginal Abatement Cost Curve Summary:")
    print(f"Total CO2 capture potential: {total_captured:.0f} ktCO2/yr")
    print(f"Average abatement cost: {avg_cost:.1f} EUR/tCO2")
    print(f"Cost range: {min_cost:.1f} - {max_cost:.1f} EUR/tCO2")
    
    # FOAK MACC
    results_df_foak = results_df.sort_values('total_FOAK').copy()
    results_df_foak['cumulative_captured'] = results_df_foak['captured_total'].cumsum()
    
    macc_foak_df = pd.DataFrame({
        'site-stack': results_df_foak['site-stack'],
        'ktCO2f_yr_baseline': results_df_foak['annual_CO2'] * (1 - results_df_foak['biogenic']),
        'ktCO2bio_yr_baseline': results_df_foak['annual_CO2'] * results_df_foak['biogenic'],
        'ktCO2f_yr_captured': results_df_foak['captured_CO2f'],
        'ktCO2bio_yr_captured': results_df_foak['captured_CO2bio'],
        'ktCO2_yr_cumulative': results_df_foak['cumulative_captured'],
        'ktCO2f_yr_residual': results_df_foak['residual_CO2f'],
        'EUR/tCO2': results_df_foak['total_FOAK']
    })
    macc_foak_df.to_csv('macc_foak.csv', index=False)
    
    plt.figure(figsize=(12, 8))
    x_steps_foak, y_steps_foak = [], []
    for i in range(len(results_df_foak) - 1):
        x_steps_foak.extend([results_df_foak['cumulative_captured'].iloc[i], results_df_foak['cumulative_captured'].iloc[i+1]])
        y_steps_foak.extend([results_df_foak['total_FOAK'].iloc[i+1], results_df_foak['total_FOAK'].iloc[i+1]])
    
    plt.plot(x_steps_foak, y_steps_foak, 'r-', linewidth=3, alpha=0.7)
    plt.xlabel('Cumulative Captured CO2 (ktCO2/yr)', fontsize=14)
    plt.ylabel('Marginal Abatement Cost (EUR/tCO2)', fontsize=14)
    plt.title('Marginal Abatement Cost Curve (FOAK)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    total_captured_foak = results_df_foak['cumulative_captured'].iloc[-1]
    avg_cost_foak = results_df_foak['total_FOAK'].mean()
    min_cost_foak = results_df_foak['total_FOAK'].min()
    max_cost_foak = results_df_foak['total_FOAK'].max()
    
    plt.text(0.02, 0.98, f'Total Captured: {total_captured_foak:.0f} ktCO2/yr\n'
                         f'Average Cost: {avg_cost_foak:.1f} EUR/tCO2\n'
                         f'Cost Range: {min_cost_foak:.1f} - {max_cost_foak:.1f} EUR/tCO2',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=12)
    plt.tight_layout()
    
    print(f"\nFOAK Marginal Abatement Cost Curve Summary:")
    print(f"Total CO2 capture potential: {total_captured_foak:.0f} ktCO2/yr")
    print(f"Average abatement cost: {avg_cost_foak:.1f} EUR/tCO2")
    print(f"Cost range: {min_cost_foak:.1f} - {max_cost_foak:.1f} EUR/tCO2")


# ============================================================================
# RESULTS TABLE
# ============================================================================

print("\n" + "="*80)
print("RESULTS TABLE")
print("="*80)

if results:
    print(f"{'Site-Stack':<30} | {'annual_CO2':<8} | {'captured_CO2f':<8} | {'captured_CO2bio':<8} | "
          f"{'residual_CO2f':<8} | {'FLH':<6} | {'CAPEX_4OAK':<8} | {'CAPEX_FOAK':<8} | {'OPEXE':<8} | "
          f"{'transtorage':<8} | {'total_4OAK':<8} | {'total_FOAK':<8} | {'distance_km':<8}")
    print(f"{'':<40} | {'ktCO2/yr':<8} | {'ktCO2/yr':<8} | {' - ':<8} | {'ktCO2/yr':<8} | {'h/y':<6} | "
          f"{'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | "
          f"{'EUR/tCO2':<8} | {'km':<8}")
    print("-" * 150)
    
    for result in results:
        print(f"{result['site-stack']:<40} | {result['annual_CO2']:>8.1f} | {result['captured_CO2f']:>8.1f} | "
              f"{result['captured_CO2bio']:>8.1f} | {result['residual_CO2f']:>8.1f} | {result['FLH']:>6.0f} | "
              f"{result['CAPEX_4OAK']:>8.1f} | {result['CAPEX_FOAK']:>8.1f} | {result['OPEXE']:>8.1f} | "
              f"{result['transtorage']:>8.1f} | {result['total_4OAK']:>8.1f} | {result['total_FOAK']:>8.1f} | "
              f"{result['distance_km']:>8.2f}")
    
    print("-" * 150)
    print(f"Total CO2 emissions: {sum([r['annual_CO2'] for r in results]):,.1f} ktCO2/y")
    print(f"Total captured CO2f: {sum([r['captured_CO2f'] for r in results]):,.1f} ktCO2/y")

print(f"\nTotal number of entries: {len(results)}")


# ============================================================================
# MAPPING
# ============================================================================

print("\n" + "="*80)
print("CREATING MAP OF ALL PLANTS")
print("="*80)

# Prepare data for mapping
drax_data = pd.DataFrame([{
    'Site': 'Drax',
    'CO2': Drax_CO2,
    'Easting': drax_easting,
    'Northing': drax_northing,
    'distance_km': drax_distance,
    'Type': 'BECCS'
}])

largest_plants_map = largest_plants[['Site', 'CO2', 'Easting', 'Northing', 'distance_km']].copy()
largest_plants_map['Type'] = 'Fossil'

w2e_plants_map = w2e_plants[['Name', 'CO2', 'Easting', 'Northing', 'distance_km']].copy()
w2e_plants_map = w2e_plants_map.rename(columns={'Name': 'Site'})
w2e_plants_map['Type'] = 'W2E'

all_plants = pd.concat([largest_plants_map, w2e_plants_map, drax_data], ignore_index=True)
all_plants = all_plants[
    all_plants['Easting'].notna() & 
    all_plants['Northing'].notna() &
    all_plants['distance_km'].notna()
].copy()

print(f"Total plants to map: {len(all_plants)}")
print(f"  - Fossil: {len(largest_plants_map)}")
print(f"  - W2E: {len(w2e_plants_map)}")
print(f"  - BECCS: {len(drax_data)}")

# Load shapefile and create GeoDataFrame
europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")
plants_gdf = gpd.GeoDataFrame(
    all_plants, 
    geometry=gpd.points_from_xy(all_plants['Easting'], all_plants['Northing'], crs="EPSG:27700")
).to_crs("EPSG:4326")

# Create map
fig, ax = plt.subplots(1, 1, figsize=(12, 15))
ax.set_aspect(1.90)

europe.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.45)

scatter = ax.scatter(
    plants_gdf.geometry.x, 
    plants_gdf.geometry.y,
    s=plants_gdf['CO2']/750,
    c=plants_gdf['distance_km'],
    cmap='viridis',
    vmin=0,
    vmax=plants_gdf['distance_km'].max(),
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('Distance to Nearest CCS Site (km)', fontsize=14)

ax.set_title('UK CCS Candidate Plants\nBubble size proportional to annual CO₂ emissions, colored by distance to nearest CCS site', 
             fontsize=15, fontweight='bold')
ax.set_xlim(-9, 3)
ax.set_ylim(49.5, 59.5)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True, alpha=0.3)

# Statistics
total_co2 = plants_gdf['CO2'].sum() / 1e6
n_plants = len(plants_gdf)
avg_distance = plants_gdf['distance_km'].mean()
max_distance = plants_gdf['distance_km'].max()
max_distance_plant = plants_gdf.loc[plants_gdf['distance_km'].idxmax()]

print(f"\nPlant with maximum distance: {max_distance_plant['Site']} ({max_distance_plant['Type']}) at {max_distance:.1f} km")

stats_text = f'Total plants: {n_plants}\n'
stats_text += f'Total CO₂: {total_co2:.1f} MtCO₂/yr\n'
stats_text += f'Avg. distance: {avg_distance:.1f} km\n'
stats_text += f'Max distance: {max_distance:.1f} km'

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
        fontsize=12)

plt.tight_layout()
plt.savefig('map_ccs_distances.png', dpi=400, bbox_inches='tight')
print(f"\nMap saved as 'map_ccs_distances.png'")

plt.show()

