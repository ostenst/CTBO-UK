import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def capture_condition(stack, x, liquefy=False):

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
        OPEXE = OPEXE / (mCO2_total * x['capture_rate']) # [€/tCO2]
        power_demand = x['pcapture']   
        if liquefy:
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
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2b_ccs']

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
        OPEXE = OPEXE / (mCO2_total * x['capture_rate']) # [€/tCO2]
        power_demand = x['pcapture']   
        if liquefy:
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
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2b_ccs']

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
        if liquefy:
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
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2b_ccs']
        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Drax':
        ktCO2f = 0
        ktCO2cem = 0
        ktCO2b = stack['ktCO2']
        Qfuel = x['drax_capacity'] / (x['drax_efficiency']/100) # [MW]
        FLH = ktCO2b*1000 / (Qfuel * x['emission_factor_pellets']) 
        mCO2 = ktCO2b * 1000 / FLH # [tCO2/h]

        # Calculate OPEXE
        Qgas_boiler = 0
        Qbio_boiler = 0
        profit_baseline = Qfuel * x['drax_efficiency']/100 * FLH * x['celc'] # [€/y]
        profit_BECCS = Qfuel * x['drax_efficiency']/100 * (1 - x['drax_efficiency_loss']) * FLH * x['celc']  # [€/y]
        difference = profit_baseline - profit_BECCS # [€/y]
        OPEXE = difference / (ktCO2b * 1000 * x['capture_rate']) # [€/tCO2]

        power_demand = x['pcapture']   
        if liquefy:
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
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2b_ccs']
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
        if liquefy:
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
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2b_ccs']
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
        if liquefy:
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
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2b_ccs']
        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 

def approximate_CAPEX(mCO2, xCO2, fixate_CAPEX, CEPCI_2025, CEPCI_base=798.7, NETL=5.509, debug=False):
    """Estimate CAPEX for CO2 capture using Kim & Leonard (2025) correlation."""
    if debug:
        print(f"approximate_CAPEX inputs: mCO2={mCO2}, xCO2={xCO2}")
    
    a, b, c, n, m = 2.1673, 0.8092, -0.00332, 0.5291, 0.8391
    
    nCO2 = mCO2 * 1000 / 44 # [kmolCO2/h]
    n_fluegas = nCO2 / xCO2
    V_fluegas = n_fluegas * 22.4 # [Nm3/h]
    
    n_largest_absorbers = int((V_fluegas/1000) // 1613)
    remaining_V_fluegas = (V_fluegas/1000) % 1613
    
    CAPEX = 0
    for i in range(n_largest_absorbers):
        TEC = a + (b * (xCO2)**n + c) * (1613)**m
        CAPEX += TEC
    
    if remaining_V_fluegas > 0:
        TEC = a + (b * (xCO2)**n + c) * (remaining_V_fluegas)**m
        CAPEX += TEC

    OPEX_fixed = CAPEX * fixate_CAPEX
    CAPEX = CAPEX * NETL * CEPCI_2025 / CEPCI_base
    
    return CAPEX, OPEX_fixed

def auxiliary_CAPEX(Qgas_boiler, Qbio_boiler, x, liquefy=False):
    CAPEX_gasboiler = Qgas_boiler * x['CAPEX_gasboiler'] # [M€]
    CAPEX_bioboiler = Qbio_boiler * x['CAPEX_bioboiler'] # [M€]

    if liquefy:
        CAPEX_liquefaction = x['cliquefy'] # [€/tCO2]
    else:
        CAPEX_liquefaction = 0
    return CAPEX_gasboiler+CAPEX_bioboiler, CAPEX_liquefaction

def levelize_MEUR(CAPEX, annual_CO2, capture_rate=0.95, discount_rate=0.07, lifetime=25, debug=False):
    """Convert CAPEX [MEUR] to levelized cost [EUR/tCO2]."""
    annualized_CAPEX = CAPEX * discount_rate * (1 + discount_rate)**lifetime / ((1 + discount_rate)**lifetime - 1) * 10**6
    levelized_CAPEX = annualized_CAPEX / (annual_CO2 * capture_rate)
    return levelized_CAPEX

def onshore_OPEX(stack, tCO2_total, capture_rate, x):
    # Check if truck or pipeline land transport (Ouvrey et al., 2024)
    distance = stack['km_hub'] # [km]
    if stack['land_transport'] == 'truck':
        a1, a2 = 0.15, 5.58 
        UC = a1 + a2 / distance # [€/(t*km)]
        cost = UC * distance # [€/tCO2]
    elif stack['land_transport'] == 'pipeline':
        a1, a2, a3, a4 = 0.02, 260, 0.07, -0.61
        UC = a1 + a2 * (distance / 1)**a3 * (tCO2_total*capture_rate / 1)**a4 # [€/(t*km)]
        cost = UC * distance # [€/tCO2]
    else:
        raise ValueError(f"Invalid land transport type: {stack['land_transport']}")
    return cost * (1 + x['transport_uncertainty'])

def shipping_OPEX(stack, transport_hubs, x):
    # If the stack uses sea transport, calculate the shipping cost (Ouvrey et al., 2024)
    if pd.isna(stack['sea_transport']):
        return 0
    hub = stack['hub']
    distance = transport_hubs.loc[transport_hubs['hub_name'] == hub, 'km_shipping'].values[0] # [km]
    if pd.isna(distance):
        return 0
    
    a1, a2 = 0.05, 20.8
    UC = a1 + a2 / distance # [€/(t*km)]
    cost = UC * distance # [€/tCO2]
    return cost * (1 + x['transport_uncertainty'])

def offshore_OPEX(stack, transport_hubs, x):
    # Calculate offshore pipeline costs and CO2 storage costs (Ouvrey et al., 2024) and CATF (2025)
    hub = stack['hub']
    if hub == 'Pembroke' or hub == 'Bristol' or hub == 'Dublin':
        storage_hub = 'Hynet'
    elif hub == 'London':
        storage_hub = 'Bacton'
    else:
        storage_hub = hub
    
    distance = transport_hubs.loc[transport_hubs['hub_name'] == storage_hub, 'km_offshore'].values[0] # [km]
    if pd.isna(distance):
        raise ValueError(f"Invalid transport-storage hub: {storage_hub}")
    ktCO2_storage = transport_hubs.loc[transport_hubs['hub_name'] == storage_hub, 'ktCO2'].values[0] # [ktCO2/y] assumed scale (CATF, 2025)
    
    a1, a2, a3, a4 = 0.02, 58.3, 0.07, -0.51
    UC = a1 + a2 * (distance / 1)**a3 * (ktCO2_storage*1000 / 1)**a4 # [€/(t*km)]
    cost = UC * distance # [€/tCO2]
    return cost * (1 + x['transport_uncertainty']) + x['cstorage']

def adjust_outliers(MACC, capture_rate, x):
    """
    Adjust MACC for outliers:
    1. Set Padeswood-cement MAC to 0 (subsidized)
    2. Add new-build waste incinerator with CCS (MAC=0)
    3. Add new-build CCGT with CCS (MAC=0)
    """
    # 1. Padeswood Cement has been subsidized - force MAC to zero
    mask = MACC['stack'] == 'Padeswood-cement'
    if mask.any():
        MACC.loc[mask, 'MAC'] = 0
        print("Adjusted MAC for Padeswood-cement to 0 €/tCO2")

    # 2. Add new-built waste incinerator with CCS (Protos)
    protos_ktco2 = 370  # [ktCO2/y]
    protos_fossil = protos_ktco2 * x['fraction_fossil_waste']
    protos_bio = protos_ktco2 * (1 - x['fraction_fossil_waste'])
    protos_residual_fossil = protos_fossil / capture_rate * (1 - capture_rate)
    
    MACC.loc[len(MACC)] = {
        'sector': 'waste',
        'site': 'Protos',
        'stack': 'Protos-waste',
        'ktCO2f': protos_fossil / capture_rate,
        'ktCO2cem': 0,
        'ktCO2b': protos_bio / capture_rate,
        'MAC': 0,  # New-build, MAC=0
        'CAPEX': 0,
        'OPEX': 0,
        'invested': False,
        'year_invest': None,
        'ktCO2f_inc': 0,
        'ktCO2f_ccs': protos_fossil,
        'ktCO2cem_ccs': 0,
        'ktCO2b_ccs': protos_bio,
        'ktCO2tot_ccs': protos_ktco2,
        'ktCO2f_res': protos_residual_fossil,
        'ktCO2cem_res': 0,
    }

    # 3. Add new-built CCGT with CCS (Teeside)
    teeside_ktco2 = 2000  # [ktCO2/y]
    teeside_residual_fossil = teeside_ktco2 / capture_rate * (1 - capture_rate)
    
    MACC.loc[len(MACC)] = {
        'sector': 'ccgt',
        'site': 'Teeside',
        'stack': 'Teeside-ccgt',
        'ktCO2f': teeside_ktco2 / capture_rate,
        'ktCO2cem': 0,
        'ktCO2b': 0,
        'MAC': 0,  # New-build, MAC=0
        'CAPEX': 0,
        'OPEX': 0,
        'invested': False,
        'year_invest': None,
        'ktCO2f_inc': 0,
        'ktCO2f_ccs': teeside_ktco2,
        'ktCO2cem_ccs': 0,
        'ktCO2b_ccs': 0,
        'ktCO2tot_ccs': teeside_ktco2,
        'ktCO2f_res': teeside_residual_fossil,
        'ktCO2cem_res': 0,
    }

    return MACC

def simulate_ctbo(
    # Constants
    plants_clean,
    transport_hubs,
    save_macc = False,

    DEFOSSILIZE = False,
    ASSUME_FOAK = True,
    DISCOUNT_RATE = 0.035,
    CTBO_QUADRATIC = 0.4,
    FOAK_CALIBRATION = 1.6379, # from calibrate_foak.py
    ETS_START = 45, # [£/tCO2]
    ETS_SCENARIO = '£300', # [£/tCO2] 200, 300, 400
    DACCS_SCENARIO = '£391', # [£/tCO2] 322, 391, 7th Carbon Budget
    
    START_YEAR = 2025,
    END_YEAR = 2055,
    DIFFUSE_END_YEAR = 2050,
    DIFFUSE_END_FRACTION = 0.5,

    coal_2023 = 17, # [MtCO2] IEA (2025)
    oil_2023 = 139, # [MtCO2] 
    gas_2023 = 127, # [MtCO2] 

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
    CAPEX_gasboiler = (0.04+0.27)/2, # [M€/MW] Danish Energy Agency
    CAPEX_bioboiler = (0.81+1.15)/2, # [M€/MW] Danish Energy Agency

    fixate_CAPEX = 0.03, # [-] of CAPEX
    CEPCI_2023 = 898.7,
    CEPCI_2025 = 930,
    NETL_2025 = 5.509,
    discount_rate_ccs = 0.07,
    lifetime_ccs = 25,
    pounds_to_EUR = 1.15,

    transport_uncertainty = 0.15, # [-] 
    cstorage = 25, # [€/tCO2] CATF (2025)

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
        'CAPEX_gasboiler': CAPEX_gasboiler,
        'CAPEX_bioboiler': CAPEX_bioboiler,
        'transport_uncertainty': transport_uncertainty,
        'cstorage': cstorage,

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
    # Calculate the point-source carbon supply by subtracting waste emissions, Drax, and limestone emissions
    total_ktCO2 = plants_clean['ktCO2'].sum()
    print(f"Total CO2 capture capacity = {total_ktCO2:.2f} ktCO2/y")
    waste_ktCO2 = plants_clean[plants_clean['sector'] == 'waste']['ktCO2'].sum()
    drax_ktCO2 = plants_clean[plants_clean['sector'] == 'drax']['ktCO2'].sum()
    cement_ktCO2 = plants_clean[plants_clean['sector'] == 'cement']['ktCO2'].sum() * fraction_limestone
    pointsources_ktCO2f = total_ktCO2 - waste_ktCO2 - drax_ktCO2 - cement_ktCO2 # [ktCO2f] supplied and emitted in 2023
    print(f"Point-source carbon supply = {pointsources_ktCO2f:.2f} ktCO2/y")
    
    # Specify whether plants defossilize. Omit low-concentration refinery stacks that have NaN as energy_strategy.
    if DEFOSSILIZE:
        plants_clean = plants_clean[~plants_clean['sector'].isin(['steel', 'refinery'])]
        ccgt_plants = plants_clean[plants_clean['sector'] == 'ccgt']
        ccgt_even = ccgt_plants.iloc[::2]
        plants_clean = pd.concat([plants_clean[plants_clean['sector'] != 'ccgt'], ccgt_even])
    plants_clean = plants_clean[plants_clean['energy_strategy'].notna()]
    print("Number of plants =", len(plants_clean))
    print("Total CO2 capture capacity =", plants_clean['ktCO2'].sum(), "ktCO2/y")

    # ------------------- CONSTRUCT THE MACC ---------------------
    # NOTE: Maybe include PLASTIC and CEMENT in the mandate? To ensure full CTBO coverage when fraction=100%?
    # NOTE: ALTERNATIVE: Calculate only O,G,C emissions, implying that waste have zero fossil emissions, and cement as 33% emissions...
    MACC = pd.DataFrame(columns=[
        'sector', 'site', 'stack', 'ktCO2f', 'ktCO2cem', 'ktCO2b',
        'MAC', 'CAPEX', 'OPEX', 'invested', 'year_invest', 
        'ktCO2f_inc', 'ktCO2f_ccs', 'ktCO2cem_ccs', 'ktCO2b_ccs', 'ktCO2tot_ccs', 'ktCO2f_res', 'ktCO2cem_res', 
        ])

    OPEX_results = {}
    CAPEX_results = {}
    TS_results = {}
    for index, stack in plants_clean.iterrows():

        # Calculate energy and carbon balances of CO2 capture, compression, and liquefaction
        LIQUEFY = (stack['land_transport'] == 'truck' or pd.notna(stack['sea_transport']))
        ktCO2_dict, FLH, mCO2_total, xCO2, OPEXE, Qgas_boiler, Qbio_boiler = capture_condition(stack, x, liquefy=LIQUEFY) # [ktCO2/y], [tCO2/h generated]

        # Add amine makeup costs
        OPEX = OPEXE + camine

        # Calculate CAPEX (of amine plant and Qboilers)
        tCO2_total = mCO2_total * FLH # [tCO2/y generated]
        CAPEX, OPEX_fixed = approximate_CAPEX(mCO2_total, xCO2, fixate_CAPEX, CEPCI_2025, CEPCI_base=CEPCI_2025, NETL=NETL_2025, debug=False) # [M€], [M€/yr]
        if ASSUME_FOAK:
            CAPEX = CAPEX * FOAK_CALIBRATION
        OPEX_fixed = (OPEX_fixed * 10**6) / tCO2_total # [€/tCO2]
        CAPEX_boilers, CAPEX_liquefaction = auxiliary_CAPEX(Qgas_boiler, Qbio_boiler, x, liquefy=LIQUEFY) # [M€], [€/tCO2]
        CAPEX += CAPEX_boilers
        CAPEX_levelized = levelize_MEUR(CAPEX, tCO2_total, capture_rate, discount_rate_ccs, lifetime_ccs) # [€/tCO2]
        CAPEX_levelized += CAPEX_liquefaction # NOTE: Later, when separating CAPEX and OPEX in the MACC NPV calculations, we consider liquefaction an OPEX.
        
        # Calculate T&S costs
        onshore_cost = onshore_OPEX(stack, tCO2_total, capture_rate, x)
        shipping_cost = shipping_OPEX(stack, transport_hubs, x)
        offshore_cost = offshore_OPEX(stack, transport_hubs, x)
        OPEX_transtorage = onshore_cost + shipping_cost + offshore_cost

        # Store results in dictionaries
        MAC = OPEX + CAPEX_levelized + OPEX_transtorage
        OPEX_results[stack['stack']] = OPEX
        CAPEX_results[stack['stack']] = CAPEX_levelized
        TS_results[stack['stack']] = OPEX_transtorage

        # Store results as a MACC row
        MACC.loc[len(MACC)] = {
            'sector': stack['sector'],    
            'site': stack['site'], 
            'stack': stack['stack'],  
            'ktCO2f': ktCO2_dict['ktCO2f'], 
            'ktCO2cem': ktCO2_dict['ktCO2cem'], 
            'ktCO2b': ktCO2_dict['ktCO2b'], 
            'MAC': MAC, 
            'CAPEX': CAPEX,
            'OPEX': OPEX + OPEX_transtorage, 
            'invested': False,
            'year_invest': None, 
            'ktCO2f_inc': ktCO2_dict['ktCO2f_inc'], 
            'ktCO2f_ccs': ktCO2_dict['ktCO2f_ccs'], 
            'ktCO2cem_ccs': ktCO2_dict['ktCO2cem_ccs'], 
            'ktCO2b_ccs': ktCO2_dict['ktCO2b_ccs'], 
            'ktCO2tot_ccs': ktCO2_dict['ktCO2tot_ccs'],
            'ktCO2f_res': ktCO2_dict['ktCO2f_res'], 
            'ktCO2cem_res': ktCO2_dict['ktCO2cem_res'],
        }
    MACC = adjust_outliers(MACC, capture_rate, x)
    MACC = MACC.sort_values(by='MAC', ascending=True)

    print(MACC[MACC['sector'] == 'steel'][['stack', 'MAC', 'ktCO2tot_ccs', 'invested', 'ktCO2f', 'ktCO2f_inc', 'ktCO2f_ccs', 'ktCO2f_res']])
    for sector in MACC['sector'].unique():
        median_MAC = MACC[MACC['sector'] == sector]['MAC'].median()
        print(f"Median MAC for sector {sector} = {median_MAC:.2f} €/tCO2")

    if save_macc:
        MACC.to_csv('results/macc.csv', index=False)
        plot_macc(MACC, savefig=save_macc)

    # ------------------- SIMULATE THE CTBO ---------------------
    # Calculate diffuse carbon trajectories by subtracting point-source carbon from total supply
    years = np.arange(START_YEAR, END_YEAR + 1)
    supply_ktCO2f = (coal_2023 + oil_2023 + gas_2023) * 1000 
    diffuse_ktCO2f = supply_ktCO2f - pointsources_ktCO2f    # [ktCO2] suppled and emitted in 2023
    diffuse_fraction = np.where(
        years <= DIFFUSE_END_YEAR,
        1.0 - (years - START_YEAR) * ((1.0 - DIFFUSE_END_FRACTION) / (DIFFUSE_END_YEAR - START_YEAR)),
        DIFFUSE_END_FRACTION
    )
    diffuse_trajectory = diffuse_ktCO2f * diffuse_fraction

    # Calculate ETS and CTBO policy trajectories
    if DACCS_SCENARIO == '£322':
        cost_DACCS = 322 * pounds_to_EUR
    elif DACCS_SCENARIO == '£391':
        cost_DACCS = 391 * pounds_to_EUR
    if ETS_SCENARIO == '£200':
        ETS_END = 200 
    elif ETS_SCENARIO == '£300':
        ETS_END = 300 
    elif ETS_SCENARIO == '£400':
        ETS_END = 400
    ets_trajectory = np.where(
        years <= DIFFUSE_END_YEAR,
        ETS_START + (years - START_YEAR) * ((ETS_END - ETS_START) / (DIFFUSE_END_YEAR - START_YEAR)),
        ETS_END
    ) * pounds_to_EUR
    ctbo_trajectory = ((years - START_YEAR) * CTBO_QUADRATIC)**2 / 100

    # Initialize results arrays for carbon (indifferent of cement/plastic), policies, and plants
    _supply_ktCO2f = []
    _emitted_ktCO2f = []
    _mandate_ktCO2 = []
    _stored_ktCO2f = []
    _stored_ktCO2b = []
    _stored_ktCO2daccs = []

    _cost_marginal = []
    _price_ETS = []
    _price_CSU = []
    _cost_CTBO_producers = [] # Total cost rectangle
    _cost_CSU_embedded = [] # Total cost rectangle / supply of CO2
    _cost_CTBO_policy = [] # Area under MACC
    _profit_CTBO_policy = [] # Area above MACC
    _cost_ETS_policy = []
    _profit_ETS_policy = []

    _diesel_increase = [] # Absolute increases
    _petrol_increase = [] 
    _gas_increase = [] 
    _plants_costbenefit = []

    gas_increase_2040 = None
    year_DACCS_marginal = None

    # Simulate the CTBO
    stored_ktCO2daccs = 0
    cost_marginal = 0
    cost_CTBO_producers = 0
    cost_CSU_embedded = 0

    for i, year in enumerate(years):

        diffuse_supply = diffuse_trajectory[i]
        ets_price = ets_trajectory[i] # NOTE: It could here be possible to re-shuffle the MAC each year based on ETS-priced extra fuel
        ctbo_fraction = ctbo_trajectory[i]

        # Plants invest voluntarily if ETS price > MAC (adjusted by any increased fossil costs)
        for idx, plant in MACC.iterrows():
            if not plant['invested']:

                if plant['ktCO2f_inc'] > 0:
                    costs_extra = plant['ktCO2f_inc']/plant['ktCO2f'] * (1 - capture_rate) * ets_price # [€/tCO2] extra fossil costs from natural gas
                else:
                    costs_extra = 0
                
                if plant['MAC']+costs_extra < ets_price:
                    MACC.loc[idx, 'invested'] = True
                    MACC.loc[idx, 'year_invest'] = year
                    print(f"Voluntary investment in {plant['stack']} in {year}")
                    print(f"   ETS price = {ets_price:.0f} €/tCO2 and MAC = {plant['MAC']:.0f} €/tCO2")
                    print(f"   Extra fossil costs = {costs_extra:.0f} €/tCO2")

        # Calculate current carbon balances 
        total_ktCO2f = MACC['ktCO2f'].sum() + MACC['ktCO2f_inc'].where(MACC['invested'], 0).sum() # [ktCO2f]
        waste_ktCO2f = MACC[MACC['sector'] == 'waste']['ktCO2f'].sum()
        pointsource_supply = total_ktCO2f - waste_ktCO2f
        supply_ktCO2f = pointsource_supply + diffuse_supply
        print(" ")
        print(supply_ktCO2f)

        pointsource_emissions = MACC['ktCO2f'].where(~MACC['invested'], 0).sum() + MACC['ktCO2cem'].where(~MACC['invested'], 0).sum()
        pointsource_residuals = MACC['ktCO2f_res'].where(MACC['invested'], 0).sum() + MACC['ktCO2cem_res'].where(MACC['invested'], 0).sum()
        emitted_ktCO2f = pointsource_emissions + pointsource_residuals + diffuse_supply
        print(emitted_ktCO2f)

        cdr = MACC['ktCO2b_ccs'].where(MACC['invested'], 0).sum() + stored_ktCO2daccs
        print(cdr)
        print("CDR does not add up to CO2 emissions since the mandate doesnt concern limestone or plastic... how to handle?")

        # Mandate CTBO compliance
        ctbo_mandate = supply_ktCO2f * ctbo_fraction
        stored_ktCO2f = MACC['ktCO2f_ccs'].where(MACC['invested'], 0).sum() + MACC['ktCO2cem_ccs'].where(MACC['invested'], 0).sum()
        stored_ktCO2b = MACC['ktCO2b_ccs'].where(MACC['invested'], 0).sum()

        stored_missing = ctbo_mandate - (stored_ktCO2f + stored_ktCO2b + stored_ktCO2daccs)
        j = 0
        print(" Endless loop? ")
        while stored_missing > 0 and j < len(MACC):
            plant = MACC.iloc[j]
            if not plant['invested']:
                print(plant['stack'])

                # Consider this plant for the CTBO if it's chepar than DACCS (ignoring extra ETS costs)
                if plant['MAC'] > cost_DACCS:
                    print(f"DACCS is cheaper than plant {plant['stack']}: {plant['MAC']:.0f} €/tCO2 > {cost_DACCS:.0f} €/tCO2")
                    break
                MACC.loc[MACC.index[j], 'invested'] = True
                MACC.loc[MACC.index[j], 'year_invest'] = year
                stored_ktCO2f += plant['ktCO2f_ccs'] + plant['ktCO2cem_ccs']
                stored_ktCO2b += plant['ktCO2b_ccs']
                stored_missing = ctbo_mandate - (stored_ktCO2f + stored_ktCO2b + stored_ktCO2daccs)
            j += 1
        
        # Calculate costs
        plants_invested = MACC[MACC['invested']]
        if len(plants_invested) > 0:
            plant_marginal = plants_invested.loc[plants_invested['MAC'].idxmax()]
            cost_marginal = plant_marginal['MAC']

        if stored_missing > 0:
            stored_ktCO2daccs = stored_missing
            cost_marginal = cost_DACCS
            if year_DACCS_marginal is None:
                year_DACCS_marginal = year

        cost_CSU = max(0, cost_marginal - ets_price) # [€/tCO2]
        cost_CTBO_producers = cost_CSU * (stored_ktCO2f + stored_ktCO2b + stored_ktCO2daccs) # [k€/y] 
        cost_CSU_embedded = cost_CTBO_producers / supply_ktCO2f # [€/tCO2]
        print('CSU =',cost_CSU_embedded)
        
        _supply_ktCO2f.append(supply_ktCO2f)
        _emitted_ktCO2f.append(emitted_ktCO2f)
        _mandate_ktCO2.append(ctbo_mandate)
        _stored_ktCO2f.append(stored_ktCO2f)
        _stored_ktCO2b.append(stored_ktCO2b)
        _stored_ktCO2daccs.append(stored_ktCO2daccs)
        _cost_marginal.append(cost_marginal)
        _price_ETS.append(ets_price)
        _price_CSU.append(cost_CSU)
        _cost_CTBO_producers.append(cost_CTBO_producers)
        _cost_CSU_embedded.append(cost_CSU_embedded)

    # Plot carbon trajectories
    plot_carbon_trajectories(
        years, 
        _supply_ktCO2f, 
        _emitted_ktCO2f, 
        _mandate_ktCO2, 
        _stored_ktCO2f, 
        _stored_ktCO2b, 
        _stored_ktCO2daccs,
        savefig=save_macc
    )
    
    # Plot cost trajectories
    plot_cost_trajectories(
        years,
        _cost_marginal,
        _price_ETS,
        _price_CSU,
        _cost_CSU_embedded,
        savefig=save_macc
    )
    plt.show()

    results = {}
    return results

def plot_opex_distributions_by_sector(plants_clean, opex_results, bins=30, ncols=2, debug=False):
    """
    Create one distribution plot per sector using the computed OPEX values.
    """
    if debug:
        print(
            "plot_opex_distributions_by_sector inputs:",
            f"plants={len(plants_clean)}, opex_entries={len(opex_results)}, bins={bins}, ncols={ncols}",
        )

    opex_df = plants_clean[['sector', 'stack']].copy()
    opex_df['OPEX'] = opex_df['stack'].map(opex_results)
    opex_df = opex_df.dropna(subset=['sector', 'OPEX'])

    sectors = sorted(opex_df['sector'].dropna().unique())
    if not sectors:
        if debug:
            print("plot_opex_distributions_by_sector output: sectors_plotted=0")
        return 0

    ncols = max(1, min(ncols, len(sectors)))
    nrows = math.ceil(len(sectors) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    if isinstance(axes, np.ndarray):
        axes_flat = list(axes.flatten())
    else:
        axes_flat = [axes]

    plotted = 0
    for idx, sector in enumerate(sectors):
        ax = axes_flat[idx]
        sector_data = opex_df.loc[opex_df['sector'] == sector, 'OPEX']
        hist_bins = max(1, min(bins, len(sector_data)))
        single_value = sector_data.nunique() == 1
        hist_kwargs = dict(density=True, alpha=0.55, color='tab:blue', label='Histogram')
        if single_value:
            value = sector_data.iloc[0]
            hist_kwargs['range'] = (value - 0.5, value + 0.5)
        try:
            ax.hist(sector_data, bins=hist_bins, **hist_kwargs)
        except ValueError:
            hist_kwargs['bins'] = 1
            hist_kwargs['range'] = (sector_data.min() - 0.5, sector_data.max() + 0.5)
            ax.hist(sector_data, **hist_kwargs)

        if len(sector_data) > 1:
            try:
                kde = gaussian_kde(sector_data)
                x_values = np.linspace(sector_data.min(), sector_data.max(), 200)
                ax.plot(x_values, kde(x_values), color='tab:orange', lw=2, label='KDE')
            except Exception:
                pass

        ax.set_title(f"OPEX distribution — {sector}", fontsize=16)
        ax.set_xlabel("OPEX (€/tCO2)", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)
        plotted += 1

    for extra_ax in axes_flat[len(sectors):]:
        extra_ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    if debug:
        print(f"plot_opex_distributions_by_sector output: sectors_plotted={plotted}")
    return plotted

def plot_capex_distributions_by_sector(plants_clean, capex_results, bins=30, ncols=2, debug=False):
    """
    Create one distribution plot per sector using the computed levelized CAPEX values.
    """
    if debug:
        print(
            "plot_capex_distributions_by_sector inputs:",
            f"plants={len(plants_clean)}, capex_entries={len(capex_results)}, bins={bins}, ncols={ncols}",
        )

    capex_df = plants_clean[['sector', 'stack']].copy()
    capex_df['CAPEX'] = capex_df['stack'].map(capex_results)
    capex_df = capex_df.dropna(subset=['sector', 'CAPEX'])

    sectors = sorted(capex_df['sector'].dropna().unique())
    if not sectors:
        if debug:
            print("plot_capex_distributions_by_sector output: sectors_plotted=0")
        return 0

    adjusted_ncols = max(1, min(ncols, len(sectors)))
    nrows = math.ceil(len(sectors) / adjusted_ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=adjusted_ncols, figsize=(6 * adjusted_ncols, 5 * nrows))
    if isinstance(axes, np.ndarray):
        axes_flat = list(axes.flatten())
    else:
        axes_flat = [axes]

    plotted = 0
    for idx, sector in enumerate(sectors):
        ax = axes_flat[idx]
        sector_data = capex_df.loc[capex_df['sector'] == sector, 'CAPEX']
        hist_bins = max(1, min(bins, len(sector_data)))
        single_value = sector_data.nunique() == 1
        hist_kwargs = dict(density=True, alpha=0.55, color='tab:green', label='Histogram')
        if single_value:
            value = sector_data.iloc[0]
            hist_kwargs['range'] = (value - 0.5, value + 0.5)
        try:
            ax.hist(sector_data, bins=hist_bins, **hist_kwargs)
        except ValueError:
            hist_kwargs['bins'] = 1
            hist_kwargs['range'] = (sector_data.min() - 0.5, sector_data.max() + 0.5)
            ax.hist(sector_data, **hist_kwargs)

        if len(sector_data) > 1:
            try:
                kde = gaussian_kde(sector_data)
                x_values = np.linspace(sector_data.min(), sector_data.max(), 200)
                ax.plot(x_values, kde(x_values), color='tab:orange', lw=2, label='KDE')
            except Exception:
                pass

        ax.set_title(f"Levelized CAPEX distribution — {sector}", fontsize=16)
        ax.set_xlabel("CAPEX (€/tCO2)", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)
        plotted += 1

    for extra_ax in axes_flat[len(sectors):]:
        extra_ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    if debug:
        print(f"plot_capex_distributions_by_sector output: sectors_plotted={plotted}")
    return plotted

def plot_ts_costs_by_sector(plants_clean, ts_results, bins=30, ncols=2, debug=False):
    """
    Plot total transport & storage costs per sector.
    """
    if debug:
        print(
            "plot_ts_costs_by_sector inputs:",
            f"plants={len(plants_clean)}, ts_entries={len(ts_results)}, bins={bins}, ncols={ncols}",
        )

    ts_df = plants_clean[['sector', 'stack']].copy()
    ts_df['TS'] = ts_df['stack'].map(ts_results)
    ts_df = ts_df.dropna(subset=['sector', 'TS'])

    sectors = sorted(ts_df['sector'].dropna().unique())
    if not sectors:
        if debug:
            print("plot_ts_costs_by_sector output: sectors_plotted=0")
        return 0

    adjusted_ncols = max(1, min(ncols, len(sectors)))
    nrows = math.ceil(len(sectors) / adjusted_ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=adjusted_ncols, figsize=(6 * adjusted_ncols, 5 * nrows))
    if isinstance(axes, np.ndarray):
        axes_flat = list(axes.flatten())
    else:
        axes_flat = [axes]

    plotted = 0
    for idx, sector in enumerate(sectors):
        ax = axes_flat[idx]
        sector_data = ts_df.loc[ts_df['sector'] == sector, 'TS']
        hist_bins = max(1, min(bins, len(sector_data)))
        single_value = sector_data.nunique() == 1
        hist_kwargs = dict(density=True, alpha=0.55, color='tab:purple', label='Histogram')
        if single_value:
            value = sector_data.iloc[0]
            hist_kwargs['range'] = (value - 0.5, value + 0.5)
        try:
            ax.hist(sector_data, bins=hist_bins, **hist_kwargs)
        except ValueError:
            hist_kwargs['bins'] = 1
            hist_kwargs['range'] = (sector_data.min() - 0.5, sector_data.max() + 0.5)
            ax.hist(sector_data, **hist_kwargs)

        if len(sector_data) > 1:
            try:
                kde = gaussian_kde(sector_data)
                x_values = np.linspace(sector_data.min(), sector_data.max(), 200)
                ax.plot(x_values, kde(x_values), color='tab:orange', lw=2, label='KDE')
            except Exception:
                pass

        ax.set_title(f"Transport & storage distribution — {sector}", fontsize=16)
        ax.set_xlabel("T&S cost (€/tCO2)", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)
        plotted += 1

    for extra_ax in axes_flat[len(sectors):]:
        extra_ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    if debug:
        print(f"plot_ts_costs_by_sector output: sectors_plotted={plotted}")
    return plotted

def plot_macc(macc, savefig=False, debug=False):
    """
    Plot the MACC curve with cumulative ktCO2tot_ccs on the x axis and MAC on the y axis.
    """
    magma = plt.cm.magma
    if debug:
        print(
            "plot_macc inputs:",
            f"rows={len(macc)}, columns={list(macc.columns)}",
        )

    macc_plot = macc[['stack', 'ktCO2tot_ccs', 'MAC']].dropna(subset=['ktCO2tot_ccs', 'MAC']).copy()
    if macc_plot.empty:
        if debug:
            print("plot_macc output: no data to plot")
        return 0

    macc_plot = macc_plot.sort_values(by='MAC')
    macc_plot['cumulative_kt'] = macc_plot['ktCO2tot_ccs'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(macc_plot['cumulative_kt']/1000, macc_plot['MAC'], where='pre', color=magma(0.5), linewidth=2.5)
    ax.set_xlabel("Cumulative MtCO₂ CCS", fontsize=14)
    ax.set_ylabel("MAC [€/tCO₂] of CCS/BECCS", fontsize=14)
    ax.set_title("Marginal Abatement Cost Curve", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    if savefig:
        plt.savefig('results/macc_curve.png', dpi=450, bbox_inches='tight')

    if debug:
        print("plot_macc output:", macc_plot[['cumulative_kt', 'MAC']].tail(1))

    return len(macc_plot)

def plot_carbon_trajectories(years, supply, emitted, mandate, stored_f, stored_b, stored_daccs, savefig=False, debug=False):
    """
    Plot carbon supply, emissions, mandate, and storage trajectories over time.
    """
    if debug:
        print(f"plot_carbon_trajectories inputs: years={len(years)}, supply={len(supply)}, emitted={len(emitted)}")
    
    # Convert lists to arrays and scale to MtCO2
    supply_arr = np.array(supply) / 1000
    emitted_arr = np.array(emitted) / 1000
    mandate_arr = np.array(mandate) / 1000
    stored_f_arr = np.array(stored_f) / 1000
    stored_removal_arr = (np.array(stored_b) + np.array(stored_daccs)) / 1000
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot supply, emitted, and mandate
    ax.plot(years, supply_arr, linewidth=2.5, label='Supply', color='tab:blue', linestyle='-')
    ax.plot(years, emitted_arr, linewidth=2.5, label='Emitted', color='tab:red', linestyle='-')
    ax.plot(years, mandate_arr, linewidth=2.5, label='CTBO Mandate', color='tab:purple', linestyle='--')
    
    # Plot stored fossil and stored removals
    ax.plot(years, stored_f_arr, linewidth=2.5, label='Stored Fossil CO₂', color='tab:orange', linestyle='-')
    ax.plot(years, stored_removal_arr, linewidth=2.5, label='Stored Removals (Bio+DACCS)', color='tab:green', linestyle='-')
    
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("MtCO₂", fontsize=14)
    ax.set_title("Carbon Trajectories under CTBO", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    if savefig:
        plt.savefig('results/carbon_trajectories.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"plot_carbon_trajectories output: plotted {len(years)} years")
    
    return len(years)

def plot_cost_trajectories(years, marginal_cost, ets_price, csu_price, csu_embedded, savefig=False, debug=False):
    """
    Plot cost and price trajectories: marginal cost, ETS price, CSU price, and embedded CSU cost.
    """
    if debug:
        print(f"plot_cost_trajectories inputs: years={len(years)}, marginal_cost={len(marginal_cost)}")
    
    # Convert lists to arrays
    marginal_arr = np.array(marginal_cost)
    ets_arr = np.array(ets_price)
    csu_arr = np.array(csu_price)
    csu_emb_arr = np.array(csu_embedded)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot all four cost/price lines
    ax.plot(years, marginal_arr, linewidth=2.5, label='Marginal Cost', color='tab:blue', linestyle='-')
    ax.plot(years, ets_arr, linewidth=2.5, label='ETS Price', color='tab:red', linestyle='-')
    ax.plot(years, csu_arr, linewidth=2.5, label='CSU Price', color='tab:orange', linestyle='--')
    ax.plot(years, csu_emb_arr, linewidth=2.5, label='Embedded CSU Cost', color='tab:green', linestyle=':')
    
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("€/tCO₂", fontsize=14)
    ax.set_title("Cost and Price Trajectories under CTBO", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    if savefig:
        plt.savefig('results/cost_trajectories.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print(f"plot_cost_trajectories output: plotted {len(years)} years")
    
    return len(years)

if __name__ == "__main__":
    plants_clean = pd.read_csv('results/plants_clean.csv')
    transport_hubs = pd.read_csv('data/transport_hubs.csv')
    results = simulate_ctbo(plants_clean, transport_hubs, save_macc=True)
    print(results)