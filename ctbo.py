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
    
    if stack['energy_strategy'] == 'Biomass':
        ktCO2f = stack['ktCO2'] * (1 - x['fraction_limestone'])
        ktCO2cem = stack['ktCO2'] * x['fraction_limestone']
        ktCO2b = 0
        FLH = x['FLH_industry']

        # Qfuel = mCO2*CR*qreb + mCO2_chp*CR*qreb, and Qfuel=mCO2_chp/emission_factor_gas (Sunny et al., 2022)
        mCO2 = (ktCO2f + ktCO2cem) * 1000 / FLH # [tCO2/h]
        qreb = x['qreb'] / 3.6 # [MWh/tCO2]
        Qgas_boiler = 0
        Qbio_boiler = mCO2*x['capture_rate']*qreb / (1 - x['emission_factor_straw']*x['capture_rate']*qreb) # [MW]
        mCO2_extra = Qbio_boiler * x['emission_factor_straw'] # [tCO2/h]
        mCO2_total = mCO2 + mCO2_extra # [tCO2/h]

        # Mix the two streams of CO2
        n = mCO2/44 / stack['xCO2'] # [Mmol/h]  
        nextra = mCO2_extra/44 / x['xCO2_bioboiler'] # [Mmol/h]
        nmix = n + nextra # [Mmol/h]
        nCO2 = mCO2_total / 44 # [Mmol/h]
        xCO2 = nCO2 / nmix # [-]

        # Calculate OPEXE
        OPEXE = Qbio_boiler * x['cstraw'] / (x['LHV_straw']/3.6) # [€/h]
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
            'ktCO2f_inc': 0,
            'ktCO2f_ccs': ktCO2f * x['capture_rate'],
            'ktCO2cem_ccs': ktCO2cem * x['capture_rate'],
            'ktCO2b_ccs': mCO2_extra * FLH /1000 * x['capture_rate'],
            'ktCO2f_res': ktCO2f * (1 - x['capture_rate']),
            'ktCO2cem_res': ktCO2cem * (1 - x['capture_rate']),
        }

        return ktCO2_dict, FLH, mCO2_total, xCO2, OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Class I-HCN' or stack['energy_strategy'] == 'Class I-HRSG':
        ktCO2f = stack['ktCO2']
        ktCO2cem = 0
        ktCO2b = 0
        FLH = x['FLH_industry']
        mCO2 = ktCO2f * 1000 / FLH # [tCO2/h]
        qreb = x['qreb'] / 3.6 # [MWh/tCO2]

        # Calculate OPEXE
        Qgas_boiler = 0
        Qbio_boiler = 0
        qsteam = x['qlatent_steam'] / 3.6 # [kWh/t steam]
        steam_demand = qreb / (qsteam/1000) # [tsteam/tCO2]
        if stack['energy_strategy'] == 'Class I-HCN':
            OPEXE = steam_demand * x['cHCN'] # [€/tCO2]
        elif stack['energy_strategy'] == 'Class I-HRSG':
            OPEXE = steam_demand * x['cHRSG'] # [€/tCO2]
        else:
            OPEXE = None

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
    
    if stack['energy_strategy'] == 'Drax':
        ktCO2f = 0
        ktCO2cem = 0
        ktCO2b = stack['ktCO2']
        Qfuel = x['drax_capacity'] / (x['drax_efficiency']/100) # [MW]
        FLH = ktCO2b*1000 / (Qfuel * x['emission_factor_pellets'] * x['LHV_pellets'])
        mCO2 = ktCO2b * 1000 / FLH # [tCO2/h]

        # Calculate OPEXE
        Qgas_boiler = 0
        Qbio_boiler = 0
        profit_baseline = Qfuel * x['drax_efficiency']/100 * FLH * x['celc'] # [€/y]
        profit_BECCS = Qfuel * x['drax_efficiency']/100 * (1 - x['drax_efficiency_loss']) * FLH * x['celc']  # [€/y]
        difference = profit_baseline - profit_BECCS # [€/y]
        OPEXE = difference / (ktCO2b * 1000 * x['capture_rate']) # [€/tCO2]

        power_demand = x['pcapture']   
        if stack['land_transport'] == 'truck' or pd.notna(stack['sea_transport']):
            power_demand += x['pcomp_liquefy'] 
        else:
            power_demand += x['pcomp'] 
        OPEXE += power_demand/3.6 * x['celc'] # [€/tCO2]

        ktCO2_dict = {
            'ktCO2f': 0,
            'ktCO2cem': 0,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': 0,
            'ktCO2f_ccs': 0,
            'ktCO2cem_ccs': 0,
            'ktCO2b_ccs': ktCO2b * x['capture_rate'],
            'ktCO2f_res': 0,
            'ktCO2cem_res': 0,
        }

        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Waste-HCN':
        ktCO2f = stack['ktCO2'] * x['fraction_fossil_waste']
        ktCO2cem = 0
        ktCO2b = stack['ktCO2'] * (1 - x['fraction_fossil_waste'])
        FLH = x['FLH_waste']
        mCO2 = (ktCO2b + ktCO2f) * 1000 / FLH # [tCO2/h]

        # Calculate OPEXE
        Qgas_boiler = 0
        Qbio_boiler = 0
        qreb = x['qreb'] / 3.6 # [MWh/tCO2]
        steam_demand = qreb / (x['qlatent_steam']/1000) # [tsteam/tCO2]
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
            'ktCO2b_ccs': ktCO2b * x['capture_rate'],
            'ktCO2f_res': ktCO2f * (1 - x['capture_rate']),
            'ktCO2cem_res': 0,
        }

        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 

    if stack['energy_strategy'] == 'LP steam':
        ktCO2f = stack['ktCO2']
        ktCO2cem = 0
        ktCO2b = 0
        Qfuel = stack['ccgt_capacity'] / (x['ccgt_efficiency']) # [MW]
        mCO2 = Qfuel * x['emission_factor_gas'] # [tCO2/h]
        FLH = ktCO2f * 1000 / mCO2 # [h/y] 

        # Calculate OPEXE
        Qgas_boiler = 0
        Qbio_boiler = 0
        Plost = x['ccgt_efficiency_loss'] * stack['ccgt_capacity'] # [MW]
        OPEXE = Plost * x['celc'] / (mCO2 * x['capture_rate']) # [€/tCO2]
        power_demand = 0 # Sherman data already includes capture plant power
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

def approximate_CAPEX(mCO2, xCO2, fixed_rate, CEPCI_curr, CEPCI_base=798.7, NETL=5.509, debug=False):
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

def simulate_ctbo(
    # Constants
    plants_clean,
    DEFOSSILIZE = False,
    ASSUME_FOAK = True,
    DISCOUNT_RATE = 0.035,
    CTBO_QUADRATIC = 0.4,

    emission_factor_gas = 0.204, # [tCO2/MWh] NZIP
    emission_factor_waste = 0.98, # [tCO2/t waste] Tolvik
    emission_factor_pellets = 0.358, # [tCO2/MWh] calc. from Emenike et al. (2020)
    emission_factor_straw = 0.353, # [tCO2/MWh] 
    LHV_pellets = 17.8, # [MJ/kg]
    LHV_straw = 16.4, # [MJ/kg]
    qlatent_steam = 2163, # [MJ/t] @3bar

    xCO2_gasboiler = 0.08,
    xCO2_bioboiler = 0.14,

    fraction_limestone = 0.60, # [-] Biermann (2022)
    fraction_fossil_waste = 0.465, # [-] Tolvik

    drax_capacity = 2580, # [MW] DUKES 5.11 DESNZ
    drax_efficiency = 33/(1-0.24), # [MWelc/MWfuel] Donnison et al. (2020)
    drax_efficiency_loss = 0.24, # [-] Donnison et al. (2020)

    # Uncertainties
    capture_rate = 0.95,
    qreb = 3.5, # [GJ/tCO2]
    pcapture = 0.05, # [MJ/kgCO2] Kumar et al. (2023)
    pcomp = 9.85/37.31, # [MW/(kgCO2/s)=MJ/kgCO2] @20bar Deng et al. (2019) 
    pcomp_liquefy = 12.00/37.31, # [MJ/kgCO2] @20bar

    FLH_industry = 8500,
    FLH_waste = 8760*0.866,
    ccgt_efficiency = 0.49, # [MWelc/MWfuel] DUKES DESNZ  
    ccgt_efficiency_loss = (67.3 - 13.260)/420, # [MWlost/MWbaseline] based on Sherman FEED study, using LP steam, subtracting the compression work

    cgas = 40, # [€/MWh]
    celc = 250, # [€/MWh]
    cpellets = 200, # [€/t biopellets]
    cstraw = 150, # [€/t biostraw]
    cliquefy = 7, # [€/tCO2] @20bar includes CAPEX excludes electricity OPEX
    cHCN = 6, # [€/t steam] Biermann et al. (2022)
    cHRSG = 10, # [€/t steam]
    camine = 4, # [€/tCO2] NOTE: No ref currently!

    fixate_CAPEX = 0.03, # [-] of CAPEX
    CEPCI_2023 = 898.7,
    CEPCI_2025 = 930,
    NETL_2025 = 5.509,
    discount_rate_ccs = 0.07,
    lifetime_ccs = 25,

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
        'ccgt_efficiency': ccgt_efficiency,
        'ccgt_efficiency_loss': ccgt_efficiency_loss,

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
        'fraction_limestone': fraction_limestone,
        'fraction_fossil_waste': fraction_fossil_waste,

        'drax_capacity': drax_capacity,
        'drax_efficiency': drax_efficiency,
        'drax_efficiency_loss': drax_efficiency_loss,
    }
    
    # Specify whether plants defossilize. And omit low-concentration refinery stacks that have NaN as energy_strategy.
    if DEFOSSILIZE:
        plants_clean = plants_clean[~plants_clean['sector'].isin(['steel', 'refinery'])]
        ccgt_plants = plants_clean[plants_clean['sector'] == 'ccgt']
        ccgt_even = ccgt_plants.iloc[::2]
        plants_clean = pd.concat([plants_clean[plants_clean['sector'] != 'ccgt'], ccgt_even])
    plants_clean = plants_clean[plants_clean['energy_strategy'].notna()]
    print("Number of plants =", len(plants_clean))
    print("Total CO2 capture capacity =", plants_clean['ktCO2'].sum(), "ktCO2/y")

    MACC = pd.DataFrame(columns=[
        'sector', 'site', 'stack', 'ktCO2f', 'ktCO2cem', 'ktCO2b',
        'MAC', 'CAPEX', 'OPEX', 'year_invest', 
        'ktCO2f_inc', 'ktCO2f_ccs', 'ktCO2cem_ccs', 'ktCO2b_ccs', 'ktCO2f_res', 'ktCO2cem_res', 
        ])

    # Print the refineries only
    print(plants_clean[plants_clean['sector'] == 'cement'])

    for index, stack in plants_clean.iterrows():

        # Calculate energy and carbon balances of CO2 capture, compression, and liquefaction
        ktCO2_dict, FLH, mCO2_total, xCO2, OPEXE, Qgas_boiler, Qbio_boiler = capture_condition(stack, x)
        print("\nOPEXE =", OPEXE, "€/tCO2 for this stack =", stack['stack'])

        # Add amine makeup costs
        OPEX = OPEXE + camine
        print("OPEX =", OPEX, "€/tCO2 for this stack =", stack['stack'])

        # Calculate CAPEX (of amine plant and Qboilers)
        CAPEX, OPEX_fixed = approximate_CAPEX(mCO2_total, xCO2, fixate_CAPEX, CEPCI_2025, CEPCI_base=798.7, NETL=5.509, debug=False) # [MEUR], [MEUR/yr]
        tCO2_total = mCO2_total * FLH # [tCO2/y]
        CAPEX_levelized = levelize_MEUR(CAPEX, tCO2_total, capture_rate, discount_rate_ccs, lifetime_ccs) # [€/tCO2]
        OPEX_fixed = (OPEX_fixed * 10**6) / tCO2_total # [€/tCO2]
        print("CAPEX_levelized =", CAPEX_levelized, "€/tCO2 for this stack =", stack['stack'])
        print("OPEX_fixed =", OPEX_fixed, "€/tCO2 for this stack =", stack['stack'])
        print("----------> CAPEX looks low... please check!")


        # Calculate T&S costs

        # Store results as a MACC row

    results = {}
    return results

if __name__ == "__main__":
    plants_clean = pd.read_csv('results/plants_clean.csv')
    results = simulate_ctbo(plants_clean)
    print(results)