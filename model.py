

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


def simulate_CTBO(

    largest_plants = None,
    w2e_plants = None,

    # Scenarios:

    # Uncertainties:
    pounds_to_EUR = 1.15,
    CEPCI_2025 = 930,
    CEPCI_2023 = 798.7,
    FOAK = 1.7553,  # First-of-a-kind multiplier (from foak.py)

    capture_rate = 0.95,
    discount_rate = 0.07,
    lifetime = 25,  # [years]
    qreb = 3.5,  # [MJ/kgCO2]
    pcompr = 0.37,  # [MJ/kgCO2]

    celc = 80,  # [EUR/MWh]
    cbio = 65,  # [EUR/MWh]
    csteam = 4.1,  # [EUR/tsteam @130C]
    amine_cost = 44,  # [SEK/tCO2]
    sek_to_eur = 0.091,

    power_ccgt = {
        'xCO2': 0.05,  # CCGT plants CO2 concentration
        'eta_P': 0.49,  # [MWel/MWfuel] total efficiency from DUKES data
        'emission_factor': 0.204,  # [tCO2/MWhfuel] from NZIP (2020)
        'gas_eff': 0.35,  # Gas turbine efficiency
        'steam_eff': 0.51  # Steam cycle efficiency
    },

    w2e = {
        'emission_factor': 0.98,  # [tCO2/twaste] Tolvik report
        'FLH': 8760 * 0.866,  # [h/y]
        'fossil_fraction': 0.465  # Fossil content of W2E CO2
    },

    drax = {
        'CO2': 11500000,  # [tCO2/yr]
        'Pinstalled': 2580,  # [MW]
        'eta_P': 33 / (1 - 0.24)  # ~44% before CCS, 33% after
    },

    qsteam = 0.15,  # Fraction of Qreb covered by steam
    qelc = 0.30,  # Fraction of Qreb covered by electricity
    qchp = 0.55,  # Fraction of Qreb covered by biomass CHP
    elc_eff = 0.33,  # Efficiency of electrified reboiler
    evaporation_enthalpy = 2257,  # [MJ/tsteam]
    emission_factor_bio = 0.3318,  # [tCO2/MWhfuel]
    
    # Levers:
    decision = "amine" # ["ref", "amine","oxy","clc"],
 
    # Constants:

):
    results = {
        "regret" : 1
    }

    return results


if __name__ == "__main__":

    # These are needed here, unless we repeat transport calculations in each scenario
    pounds_to_EUR = 1.15
    CEPCI_2025 = 930
    CEPCI_2023 = 798.7

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


    dict = simulate_CTBO(largest_plants, w2e_plants)
    print("Simulation completed")