import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def capture_condition(stack, x):

    # Adapt the script depending on energy strategy
    if stack['energy_strategy'] == 'Class III-NG':
        ktCO2f = stack['ktCO2']
        ktCO2cem = 0
        ktCO2b = 0
        FLH = x['FLH_industry']

        # Qfuel = mCO2*CR*qreb + mCO2_chp*CR*qreb, and Qfuel=mCO2_chp/emission_factor_gas (Sunny et al., 2022)
        mCO2 = ktCO2f * 1000 / FLH # [tCO2/h]
        qreb = x['qreb'] / 3.6 # [MWh/tCO2]
        Qgas_boiler = mCO2*x['capture_rate']*qreb / (1 - x['emission_factor_gas']*x['capture_rate']*qreb) # [MW]
        Qbio_boiler = 0
        mCO2_extra = Qgas_boiler * x['emission_factor_gas'] # [tCO2/h]
        mCO2_total = mCO2 + mCO2_extra # [tCO2/h]

        # Mix the two streams of CO2
        n = mCO2/44 / stack['xCO2'] # [Mmol/h]  
        nextra = mCO2_extra/44 / x['xCO2_gasboiler'] # [Mmol/h]
        nmix = n + nextra # [Mmol/h]
        nCO2 = mCO2_total / 44 # [Mmol/h]
        xCO2 = nCO2 / nmix # [-]

        # Calculate OPEXE
        OPEXE = Qgas_boiler * x['cgas'] # [€/h]
        OPEXE = OPEXE / mCO2_total # [€/tCO2]
        power_demand = x['pcapture']   
        if stack['land_transport'] == 'truck' or pd.notna(stack['sea_transport']):
            power_demand += x['pcomp_liquefy'] 
        else:
            power_demand += x['pcomp'] 
        OPEXE += power_demand/3.6 * x['celc'] # [€/tCO2]

        ktCO2_dict = {
            'ktCO2f': ktCO2f,
            'ktCO2cem': ktCO2cem,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': mCO2_extra * FLH /1000,
            'ktCO2f_ccs': mCO2_total * FLH /1000 * x['capture_rate'],
            'ktCO2cem_ccs': 0,
            'ktCO2b_ccs': 0,
            'ktCO2f_res': mCO2_total * FLH /1000 * (1 - x['capture_rate']),
            'ktCO2cem_res': 0,
        }

        return ktCO2_dict, FLH, mCO2_total, xCO2, OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Class I-HCN':
        ktCO2f = stack['ktCO2']
        ktCO2cem = 0
        ktCO2b = 0
        FLH = x['FLH_industry']

        mCO2 = ktCO2f * 1000 / FLH # [tCO2/h]
        qreb = x['qreb'] / 3.6 # [MWh/tCO2]
        Qreb = mCO2*x['capture_rate']*qreb # [MW]

        # Calculate OPEXE
        Qgas_boiler = 0
        Qbio_boiler = 0
        qsteam = x['qlatent_steam'] / 3.6 # [kWh/t steam]
        steam_demand = qreb / (qsteam/1000) # [tsteam/tCO2]
        OPEXE = steam_demand * x['cHCN'] # [€/tCO2]

        power_demand = x['pcapture']   
        if stack['land_transport'] == 'truck' or pd.notna(stack['sea_transport']):
            power_demand += x['pcomp_liquefy'] 
        else:
            power_demand += x['pcomp'] 
        OPEXE += power_demand/3.6 * x['celc'] # [€/tCO2]

        ktCO2_dict = {
                'ktCO2f': ktCO2f,
                'ktCO2cem': ktCO2cem,
                'ktCO2b': ktCO2b,
                'ktCO2f_inc': 0,
                'ktCO2f_ccs': ktCO2f * x['capture_rate'],
                'ktCO2cem_ccs': 0,
                'ktCO2b_ccs': 0,
                'ktCO2f_res': ktCO2f * (1 - x['capture_rate']),
                'ktCO2cem_res': 0,
            }

        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 
    
    else:
        ktCO2_dict = {
            'ktCO2f': 1,
            'ktCO2cem': 2,
            'ktCO2b': 3,
            'ktCO2f_inc': 4,
            'ktCO2f_ccs': 5,
            'ktCO2cem_ccs': 6,
            'ktCO2b_ccs': 7,
            'ktCO2f_res': 8,
            'ktCO2cem_res': 9,
        }
        FLH = 10000,
        mCO2 = 10000,
        xCO2 = 10000,
        OPEXE = 10000,
        Qgas_boiler = 10000,
        Qbio_boiler = 10000,

        return ktCO2_dict, FLH, mCO2, xCO2, OPEXE, Qgas_boiler, Qbio_boiler 

def simulate_ctbo(
    # Constants
    plants_clean,
    DEFOSSILIZE = False,
    ASSUME_FOAK = True,
    DISCOUNT_RATE = 0.035,
    CTBO_QUADRATIC = 0.4,

    emission_factor_gas = 0.204, # [tCO2/MWh] NZIP
    emission_factor_waste = 0.98, # [tCO2/t waste] Tolvik
    emission_factor_pellets = 0.358, # [tCO2/t biopellets] calc. from Emenike et al. (2020)
    emission_factor_straw = 0.353, # [tCO2/t biostraw] 
    LHV_pellets = 17.8, # [MJ/kg]
    LHV_straw = 16.4, # [MJ/kg]
    qlatent_steam = 2163, # [MJ/t] @3bar

    xCO2_gasboiler = 0.08,
    xCO2_bioboiler = 0.14,

    # Uncertainties
    capture_rate = 0.95,
    qreb = 3.5, # [GJ/tCO2]
    pcapture = 0.05, # [MJ/kgCO2] Kumar et al. (2023)
    pcomp = 9.85/37.31, # [MW/(kgCO2/s)=MJ/kgCO2] @20bar Deng et al. (2019) 
    pcomp_liquefy = 12.00/37.31, # [MJ/kgCO2] @20bar

    FLH_industry = 8500,
    FLH_waste = 8760*0.866,

    cgas = 40, # [€/MWh]
    celc = 250, # [€/MWh]
    cpellets = 200, # [€/t biopellets]
    cstraw = 150, # [€/t biostraw]
    cliquefy = 7, # [€/tCO2] @20bar includes CAPEX excludes electricity OPEX
    cHCN = 6, # [€/t steam] Biermann et al. (2022)
    cHRSG = 10, # [€/t steam]

    # Levers

):
    # Store assumptions
    x = {
        'capture_rate': capture_rate,
        'qreb': qreb,
        'pcapture': pcapture,
        'pcomp': pcomp,
        'pcomp_liquefy': pcomp_liquefy,

        'FLH_industry': FLH_industry,
        'FLH_waste': FLH_waste,

        'cgas': cgas,
        'celc': celc,
        'cpellets': cpellets,
        'cstraw': cstraw,
        'cliquefy': cliquefy,
        'cHCN': cHCN,
        'cHRSG': cHRSG,

        'emission_factor_gas': emission_factor_gas,
        'emission_factor_waste': emission_factor_waste,
        'emission_factor_pellets': emission_factor_pellets,
        'emission_factor_straw': emission_factor_straw,
        'LHV_pellets': LHV_pellets,
        'LHV_straw': LHV_straw,
        'qlatent_steam': qlatent_steam,
        'xCO2_gasboiler': xCO2_gasboiler,
        'xCO2_bioboiler': xCO2_bioboiler,
    }
    
    # Specify whether plants defossilize
    if DEFOSSILIZE:
        plants_clean = plants_clean[~plants_clean['sector'].isin(['steel', 'refinery'])]
        ccgt_plants = plants_clean[plants_clean['sector'] == 'ccgt']
        ccgt_even = ccgt_plants.iloc[::2]
        plants_clean = pd.concat([plants_clean[plants_clean['sector'] != 'ccgt'], ccgt_even])

    MACC = pd.DataFrame(columns=[
        'sector', 'site', 'stack', 'ktCO2f', 'ktCO2cem', 'ktCO2b',
        'MAC', 'CAPEX', 'OPEX', 'year_invest', 
        'ktCO2f_inc', 'ktCO2f_ccs', 'ktCO2cem_ccs', 'ktCO2b_ccs', 'ktCO2f_res', 'ktCO2cem_res', 
        ])

    print(plants_clean)

    for index, stack in plants_clean.iterrows():

        # Calculate energy and carbon balances of CO2 capture, compression, and liquefaction
        ktCO2_dict, FLH, mCO2_total, xCO2, OPEXE, Qgas_boiler, Qbio_boiler = capture_condition(stack, x)

        # Add amine makeup costs

        # Calculate CAPEX (of amine plant and Qboilers)

        # Calculate T&S costs

        # Store results as a MACC row

    results = {}
    return results

if __name__ == "__main__":
    plants_clean = pd.read_csv('results/plants_clean.csv')
    results = simulate_ctbo(plants_clean)
    print(results)