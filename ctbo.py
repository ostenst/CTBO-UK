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
        ktCO2pl = 0
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
            'ktCO2pl': ktCO2pl,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': mCO2_extra * FLH /1000,
            'ktCO2f_ccs': mCO2_total * FLH /1000 * x['capture_rate'],
            'ktCO2cem_ccs': 0,
            'ktCO2pl_ccs': 0,
            'ktCO2b_ccs': 0,
            'ktCO2f_res': mCO2_total * FLH /1000 * (1 - x['capture_rate']),
            'ktCO2cem_res': 0,
            'ktCO2pl_res': 0,
        }
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2pl_ccs'] + ktCO2_dict['ktCO2b_ccs']

        return ktCO2_dict, FLH, mCO2_total, xCO2, OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Biomass':
        ktCO2f = stack['ktCO2'] * (1 - x['fraction_limestone'])
        ktCO2cem = stack['ktCO2'] * x['fraction_limestone']
        ktCO2pl = 0
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
            'ktCO2pl': ktCO2pl,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': 0,
            'ktCO2f_ccs': ktCO2f * x['capture_rate'],
            'ktCO2cem_ccs': ktCO2cem * x['capture_rate'],
            'ktCO2pl_ccs': 0,
            'ktCO2b_ccs': mCO2_extra * FLH /1000 * x['capture_rate'],
            'ktCO2f_res': ktCO2f * (1 - x['capture_rate']),
            'ktCO2cem_res': ktCO2cem * (1 - x['capture_rate']),
            'ktCO2pl_res': 0,
        }
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2pl_ccs'] + ktCO2_dict['ktCO2b_ccs']

        return ktCO2_dict, FLH, mCO2_total, xCO2, OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Class I-HCN' or stack['energy_strategy'] == 'Class I-HRSG':
        ktCO2f = stack['ktCO2']
        ktCO2cem = 0
        ktCO2pl = 0
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
            'ktCO2pl': ktCO2pl,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': 0,
            'ktCO2f_ccs': ktCO2f * x['capture_rate'],
            'ktCO2cem_ccs': 0,
            'ktCO2pl_ccs': 0,
            'ktCO2b_ccs': 0,
            'ktCO2f_res': ktCO2f * (1 - x['capture_rate']),
            'ktCO2cem_res': 0,
            'ktCO2pl_res': 0,
        }
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2pl_ccs'] + ktCO2_dict['ktCO2b_ccs']
        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Drax':
        ktCO2f = 0
        ktCO2cem = 0
        ktCO2pl = 0
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
            'ktCO2pl': 0,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': 0,
            'ktCO2f_ccs': 0,
            'ktCO2cem_ccs': 0,
            'ktCO2pl_ccs': 0,
            'ktCO2b_ccs': ktCO2b * x['capture_rate'],
            'ktCO2f_res': 0,
            'ktCO2cem_res': 0,
            'ktCO2pl_res': 0,
        }
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2pl_ccs'] + ktCO2_dict['ktCO2b_ccs']
        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 
    
    if stack['energy_strategy'] == 'Waste-HCN':
        ktCO2f = 0
        ktCO2cem = 0
        ktCO2pl = stack['ktCO2'] * x['fraction_fossil_waste']
        ktCO2b = stack['ktCO2'] * (1 - x['fraction_fossil_waste'])
        FLH = x['FLH_waste']
        mCO2 = (ktCO2b + ktCO2pl) * 1000 / FLH # [tCO2/h]

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
            'ktCO2pl': ktCO2pl,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': 0,
            'ktCO2f_ccs': 0,
            'ktCO2cem_ccs': 0,
            'ktCO2pl_ccs': ktCO2pl * x['capture_rate'],
            'ktCO2b_ccs': ktCO2b * x['capture_rate'],
            'ktCO2f_res': 0,
            'ktCO2cem_res': 0,
            'ktCO2pl_res': ktCO2pl * (1 - x['capture_rate']),
        }
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2pl_ccs'] + ktCO2_dict['ktCO2b_ccs']
        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 

    if stack['energy_strategy'] == 'LP steam':
        ktCO2f = stack['ktCO2']
        ktCO2cem = 0
        ktCO2pl = 0
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
            'ktCO2pl': ktCO2pl,
            'ktCO2b': ktCO2b,
            'ktCO2f_inc': 0,
            'ktCO2f_ccs': ktCO2f * x['capture_rate'],
            'ktCO2cem_ccs': 0,
            'ktCO2pl_ccs': 0,
            'ktCO2b_ccs': 0,
            'ktCO2f_res': ktCO2f * (1 - x['capture_rate']),
            'ktCO2cem_res': 0,
            'ktCO2pl_res': 0,
        }
        ktCO2_dict['ktCO2tot_ccs'] = ktCO2_dict['ktCO2f_ccs'] + ktCO2_dict['ktCO2cem_ccs'] + ktCO2_dict['ktCO2pl_ccs'] + ktCO2_dict['ktCO2b_ccs']
        return ktCO2_dict, FLH, mCO2, stack['xCO2'], OPEXE, Qgas_boiler, Qbio_boiler 

def approximate_CAPEX(mCO2, xCO2, fixate_CAPEX, CEPCI_2025, CAPEX_m=0.8391, CEPCI_base=798.7, NETL=5.509, debug=False):
    """Estimate CAPEX for CO2 capture using Kim & Leonard (2025) correlation."""
    if debug:
        print(f"approximate_CAPEX inputs: mCO2={mCO2}, xCO2={xCO2}, CAPEX_m={CAPEX_m}")
    
    a, b, c, n = 2.1673, 0.8092, -0.00332, 0.5291 # Note that factor m is replaced by CAPEX_m (a model uncertainty)
    
    nCO2 = mCO2 * 1000 / 44 # [kmolCO2/h]
    n_fluegas = nCO2 / xCO2
    V_fluegas = n_fluegas * 22.4 # [Nm3/h]
    
    n_largest_absorbers = int((V_fluegas/1000) // 1613)
    remaining_V_fluegas = (V_fluegas/1000) % 1613
    
    CAPEX = 0
    for i in range(n_largest_absorbers):
        TEC = a + (b * (xCO2)**n + c) * (1613)**CAPEX_m
        CAPEX += TEC
    
    if remaining_V_fluegas > 0:
        TEC = a + (b * (xCO2)**n + c) * (remaining_V_fluegas)**CAPEX_m
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

def subsidize_outliers(MACC, outliers, debug=False):
    """
    Save initial cost estimates for outlier plants, then set their MAC/CAPEX/OPEX to zero.
    Returns: MACC, cost_initial_outliers
    """
    initial_costs_outliers = {}
    for stack_name in outliers:
        mask = MACC['stack'] == stack_name
        if mask.any():
            initial_costs_outliers[stack_name] = MACC.loc[mask, 'MAC'].values[0]
            MACC.loc[mask, 'MAC'] = 0

    if debug:
        print(f"Initial outlier costs: {initial_costs_outliers}")

    return (
        MACC,
        initial_costs_outliers,
    )

def extend_plant_npv_data(npv_data_all, lifetime_ccs, debug=False):
    """
    Extend each invested plant's annual rows to its individual NPV horizon.
    The horizon is [investment_year, investment_year + lifetime_ccs - 1].
    For years beyond the simulation window, reuse the plant's last available annual values.
    """
    if debug:
        print(
            "extend_plant_npv_data inputs:",
            f"rows={len(npv_data_all)}, lifetime_ccs={lifetime_ccs}",
        )

    extended_rows = []
    for stack, plant_df in npv_data_all.groupby('stack', sort=False):
        plant_df = plant_df.sort_values('year').copy()
        extended_rows.append(plant_df)

        investment_year_series = plant_df['investment_year'].dropna()
        if investment_year_series.empty:
            continue

        investment_year = int(investment_year_series.iloc[0])
        npv_end_year = investment_year + int(lifetime_ccs) - 1
        last_year_available = int(plant_df['year'].max())
        if npv_end_year <= last_year_available:
            continue

        last_row = plant_df.iloc[-1].copy()
        for year in range(last_year_available + 1, npv_end_year + 1):
            row = last_row.copy()
            row['year'] = year
            row['invested'] = True
            row['investment_year'] = investment_year
            extended_rows.append(pd.DataFrame([row]))

    npv_extended = pd.concat(extended_rows, ignore_index=True)
    npv_extended = npv_extended.sort_values(['stack', 'year']).reset_index(drop=True)

    if debug:
        print(
            "extend_plant_npv_data output:",
            f"rows={len(npv_extended)}, max_year={int(npv_extended['year'].max())}",
        )
    return npv_extended

def calculate_plant_npv(npv_data_all, lifetime_ccs, construction_years=3, debug=False):
    """
    Calculate plant-level NPV using plant-specific investment horizons.
    Rules:
    - CAPEX split evenly across construction years from investment year
    - OPEX starts after construction years
    - Revenues = full cash inflow from y and E after construction years
    - All discounted with per-row discount_factor
    """
    if debug:
        print(
            "calculate_plant_npv inputs:",
            f"rows={len(npv_data_all)}, lifetime_ccs={lifetime_ccs}, construction_years={construction_years}",
        )

    records = []
    for stack, plant_df in npv_data_all.groupby('stack', sort=False):
        plant_df = plant_df.sort_values('year').copy()
        sector = plant_df['sector'].iloc[0] if 'sector' in plant_df and len(plant_df) > 0 else None
        investment_year_series = plant_df['investment_year'].dropna()
        if investment_year_series.empty:
            records.append({
                'stack': stack,
                'sector': sector,
                'investment_year': np.nan,
                'npv_end_year': np.nan,
                'NPV_CAPEX': 0.0,
                'NPV_OPEX': 0.0,
                'NPV_REVENUE': 0.0,
                'NPV_total': np.nan,
            })
            continue

        investment_year = int(investment_year_series.iloc[0])
        npv_end_year = investment_year + int(lifetime_ccs) - 1
        period = plant_df[(plant_df['year'] >= investment_year) & (plant_df['year'] <= npv_end_year)].copy()
        if period.empty:
            records.append({
                'stack': stack,
                'sector': sector,
                'investment_year': investment_year,
                'npv_end_year': npv_end_year,
                'NPV_CAPEX': 0.0,
                'NPV_OPEX': 0.0,
                'NPV_REVENUE': 0.0,
                'NPV_total': np.nan,
            })
            continue

        capex_total = float(period['CAPEX'].dropna().iloc[0]) if period['CAPEX'].notna().any() else 0.0
        capex_annual = capex_total / float(construction_years)

        year_values = period['year'].to_numpy()
        discount = period['discount_factor'].to_numpy(dtype=float)
        opex_values = period['OPEX'].to_numpy(dtype=float)
        revenue_values = (period['cash_inflow_y'] + period['cash_inflow_E']).to_numpy(dtype=float)

        construction_end_year = investment_year + int(construction_years) - 1
        in_construction = year_values <= construction_end_year
        in_operation = year_values >= (construction_end_year + 1)

        capex_stream = np.where(in_construction, capex_annual, 0.0)
        opex_stream = np.where(in_operation, opex_values, 0.0)
        revenue_stream = np.where(in_operation, revenue_values, 0.0)

        npv_capex = float(np.sum(capex_stream * discount))
        npv_opex = float(np.sum(opex_stream * discount))
        npv_revenue = float(np.sum(revenue_stream * discount))
        npv_total = npv_revenue - npv_opex - npv_capex

        records.append({
            'stack': stack,
            'sector': sector,
            'investment_year': investment_year,
            'npv_end_year': npv_end_year,
            'NPV_CAPEX': npv_capex,
            'NPV_OPEX': npv_opex,
            'NPV_REVENUE': npv_revenue,
            'NPV_total': npv_total,
        })

    npv_plants = pd.DataFrame(records)
    if debug:
        print(f"calculate_plant_npv output: plants={len(npv_plants)}")
    return npv_plants

def simulate_ctbo(
    # Constants
    plants_clean,
    transport_hubs,
    single_run = False,

    PHASEOUT = False,
    DISCOUNT_RATE = 0.035,
    CTBO_QUADRATIC = 0.4,
    ETS_START = 45, # [£/tCO2]
    ETS_SCENARIO = 'ETS-eq', # ['CTBO-only', 'ETS-eq', '£100-Mix', '£200-Mix', '£300-Mix']),
    DACCS_SCENARIO = '£322', # [£/tCO2] 322, 391, 7th Carbon Budget
    
    START_YEAR = 2025,
    END_YEAR = 2055,
    DIFFUSE_END_YEAR = 2050,
    DIFFUSE_END_FRACTION = 0.30,

    coal_2023 = 17, # [MtCO2] IEA (2025)
    oil_2023 = 139, # [MtCO2] 
    gas_2023 = 127, # [MtCO2] 

    emission_factor_gas = 0.2027, # [tCO2/MWh] DESNZ GHG conversion factors 2025 condensed set
    emission_factor_petrol = 2.339, # [kgCO2e/L] DESNZ
    emission_factor_diesel = 2.661, # [kgCO2e/L] DESNZ
    emission_factor_kerosene = 2.542, # [kgCO2e/L] DESNZ

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

    cgas = 40, # [€/MWh] Mersch et al. (2023)
    celc = 250, # [€/MWh]
    cstraw = 150, # [€/t biostraw]
    cliquefy = 7, # [€/tCO2] @20bar includes CAPEX excludes electricity OPEX
    cHCN = 6, # [€/t steam] Biermann et al. (2022)
    cHRSG = 10, # [€/t steam]
    camine = 4, # [€/tCO2] NOTE: No ref currently!
    CAPEX_gasboiler = (0.04+0.27)/2, # [M€/MW] Danish Energy Agency
    CAPEX_bioboiler = (0.81+1.15)/2, # [M€/MW] Danish Energy Agency
    CONSTRUCTION_YEARS = 3,

    fixate_CAPEX = 0.03, # [-] of CAPEX
    CEPCI_2023 = 898.7,
    CEPCI_2025 = 930,
    NETL_2025 = 5.509,
    CAPEX_m = 0.8391,
    discount_rate_ccs = 0.07,
    lifetime_ccs = 25,
    pounds_to_EUR = 1.15,

    transport_uncertainty = 0.15, # [-] 
    cstorage = 25, # [€/tCO2] CATF (2025)
    results_dir = 'results_baseline',
    figures_dir = 'results_figures',
    save_aux_results = False,

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
    # Calculate the point-source carbon supply. Low-concentration refinery stacks that have NaN as energy_strategy are considered "diffuse" and excluded. Also subtract waste emissions, Drax, and limestone emissions
    plants_clean = plants_clean[plants_clean['energy_strategy'].notna()]
    plants_pre_phaseout = plants_clean.copy()
    total_ktCO2 = plants_clean['ktCO2'].sum()
    waste_ktCO2 = plants_clean[plants_clean['sector'] == 'waste']['ktCO2'].sum()
    drax_ktCO2 = plants_clean[plants_clean['sector'] == 'drax']['ktCO2'].sum()
    cement_ktCO2 = plants_clean[plants_clean['sector'] == 'cement']['ktCO2'].sum() * fraction_limestone
    pointsources_ktCO2f = total_ktCO2 - (waste_ktCO2 + drax_ktCO2 + cement_ktCO2) # [ktCO2f] supplied and emitted in 2023
    
    # Specify whether plants defossilize.
    outliers = ['Padeswood-cement', 'Protos-waste', 'Teeside-ccgt']
    if PHASEOUT:
        plants_clean = plants_clean[~plants_clean['sector'].isin(['steel', 'refinery'])]
        ccgt_plants = plants_clean[plants_clean['sector'] == 'ccgt']
        ccgt_even = ccgt_plants.iloc[::2]
        plants_clean = pd.concat([plants_clean[plants_clean['sector'] != 'ccgt'], ccgt_even])
        # Safeguard: always keep the three designated outlier stacks in the model.
        for outlier_name in outliers:
            if outlier_name not in plants_clean['stack'].values:
                outlier_row = plants_pre_phaseout[plants_pre_phaseout['stack'] == outlier_name]
                if outlier_row.empty:
                    raise ValueError(f"Required outlier stack missing from source data: {outlier_name}")
                plants_clean = pd.concat([plants_clean, outlier_row], ignore_index=True)
        plants_clean = plants_clean.drop_duplicates(subset=['stack'], keep='first')
    if single_run:
        print("Total point-sources of carbon =", plants_clean['ktCO2'].sum(), "ktCO2")

    # ------------------- CONSTRUCT THE MACC ---------------------
    MACC = pd.DataFrame(columns=[
        'sector', 'site', 'stack', 'ktCO2f', 'ktCO2cem', 'ktCO2pl', 'ktCO2b',
        'MAC', 'CAPEX', 'OPEX', 'invested', 'year_invest', 
        'ktCO2f_inc', 'ktCO2f_ccs', 'ktCO2cem_ccs', 'ktCO2pl_ccs', 'ktCO2b_ccs', 'ktCO2tot_ccs', 'ktCO2f_res', 'ktCO2cem_res', 'ktCO2pl_res', 
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
        tCO2_total = mCO2_total * FLH * capture_rate # [tCO2/y captured]
        CAPEX, OPEX_fixed = approximate_CAPEX(
            mCO2_total,
            xCO2,
            fixate_CAPEX,
            CEPCI_2025,
            CAPEX_m=CAPEX_m,
            CEPCI_base=CEPCI_2023,
            NETL=NETL_2025,
            debug=False
        ) # [M€], [M€/yr]
        OPEX_fixed = (OPEX_fixed * 10**6) / tCO2_total 
        OPEX += OPEX_fixed
        CAPEX_boilers, CAPEX_liquefaction = auxiliary_CAPEX(Qgas_boiler, Qbio_boiler, x, liquefy=LIQUEFY) # [M€], [€/tCO2]
        CAPEX += CAPEX_boilers
        CAPEX_levelized = levelize_MEUR(CAPEX, tCO2_total, capture_rate, discount_rate_ccs, lifetime_ccs) # [€/tCO2]
        CAPEX_levelized += CAPEX_liquefaction # NOTE: Later, when separating CAPEX and OPEX in the MACC NPV calculations, we consider liquefaction an OPEX.
        
        # Calculate T&S costs
        onshore_cost = onshore_OPEX(stack, tCO2_total, capture_rate, x)
        shipping_cost = shipping_OPEX(stack, transport_hubs, x)
        offshore_cost = offshore_OPEX(stack, transport_hubs, x)
        OPEX_transtorage = onshore_cost + shipping_cost + offshore_cost
        OPEX += OPEX_transtorage

        # Store results in dictionaries
        MAC = OPEX + CAPEX_levelized
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
            'ktCO2pl': ktCO2_dict['ktCO2pl'], 
            'ktCO2b': ktCO2_dict['ktCO2b'], 
            'MAC': MAC, 
            'CAPEX': CAPEX,
            'OPEX': OPEX, 
            'invested': False,
            'year_invest': None, 
            'ktCO2f_inc': ktCO2_dict['ktCO2f_inc'], 
            'ktCO2f_ccs': ktCO2_dict['ktCO2f_ccs'], 
            'ktCO2cem_ccs': ktCO2_dict['ktCO2cem_ccs'], 
            'ktCO2pl_ccs': ktCO2_dict['ktCO2pl_ccs'], 
            'ktCO2b_ccs': ktCO2_dict['ktCO2b_ccs'], 
            'ktCO2tot_ccs': ktCO2_dict['ktCO2tot_ccs'],
            'ktCO2f_res': ktCO2_dict['ktCO2f_res'], 
            'ktCO2cem_res': ktCO2_dict['ktCO2cem_res'],
            'ktCO2pl_res': ktCO2_dict['ktCO2pl_res'],
        }
    MACC, cost_initial_outliers = subsidize_outliers(MACC, outliers)
    MACC = MACC.sort_values(by='MAC', ascending=True)
  
    if single_run or save_aux_results:
        MACC.to_csv(f'{results_dir}/macc.csv', index=False)
    if single_run:
        plot_macc(MACC, pounds_to_EUR=pounds_to_EUR, figures_dir=figures_dir, savefig=True)

    # ------------------- SIMULATE THE CTBO/ETS policies ---------------------
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

    # Calculate ETS (capped by DACCS costs) and CTBO policy trajectories
    if DACCS_SCENARIO == '£322':
        cost_DACCS = 322 * pounds_to_EUR
    elif DACCS_SCENARIO == '£391':
        cost_DACCS = 391 * pounds_to_EUR
    if ETS_SCENARIO == 'CTBO-only':
        ETS_START = 0
        ETS_END = 0
    elif ETS_SCENARIO == 'ETS-eq': # NOTE: Must calculate ETS price organically!
        ETS_END = 999999
    elif ETS_SCENARIO == '£100-Mix':
        ETS_END = 100
    elif ETS_SCENARIO == '£200-Mix':
        ETS_END = 200 
    elif ETS_SCENARIO == '£300-Mix':
        ETS_END = 300 
    ets_trajectory = np.minimum(
        np.where(
            years <= DIFFUSE_END_YEAR,
            ETS_START + (years - START_YEAR) * ((ETS_END - ETS_START) / (DIFFUSE_END_YEAR - START_YEAR)),
            ETS_END
        ) * pounds_to_EUR,
        cost_DACCS
    )
    # Quadratic CTBO fraction in [0,1] scale; hold constant after DIFFUSE_END_YEAR (default 2050 → 1.0 with default params).
    ctbo_raw = ((years - START_YEAR) * CTBO_QUADRATIC) ** 2 / 100
    ctbo_at_cap = ((DIFFUSE_END_YEAR - START_YEAR) * CTBO_QUADRATIC) ** 2 / 100
    ctbo_trajectory = np.where(years <= DIFFUSE_END_YEAR, ctbo_raw, ctbo_at_cap)

    # Initialize results arrays for carbon (f=fuels, cem=cement, pl=plastic, g=f+cem+pl, b=biomass)
    _supply_ktCO2f = []
    _emitted_ktCO2f = [] 
    _emitted_ktCO2final = []
    _mandate_ktCO2 = []
    _stored_ktCO2g = []
    _stored_ktCO2b = []
    _stored_ktCO2daccs = []

    _cost_marginal = []
    _price_ETS = []
    _price_CSU = []
    _cost_suppliers = [] 
    _cost_emitters = []
    _cost_tax = []
    _cost_consumers = []
    _cost_fuels = []
    _profit_y_policy = []
    _cost_y_policy = []
    _profit_E_policy = []
    _cost_E_policy = []
    _tax_E_policy = []
    _gas_increase_abs = [] # €/MWh
    _gas_increase_pct = [] # % increase
    _petrol_increase_abs = [] # cent/L
    _diesel_increase_abs = [] # cent/L
    _kerosene_increase_abs = [] # cent/L
    _plants_costbenefit = []

    # Simulate the CTBO
    year_DACCS_marginal = None
    ctbo_active = True # Always true in this model version
    stored_ktCO2daccs = 0
    cost_marginal = 0

    for i, year in enumerate(years):

        diffuse_supply = diffuse_trajectory[i]
        ctbo_fraction = ctbo_trajectory[i]
        if year_DACCS_marginal is None:
            ets_price = ets_trajectory[i] 
        else:
            ets_price = ets_trajectory[year_DACCS_marginal - min(years)] # Stabilize ETS prices if DACCS is marginal

        # Plants invest voluntarily if the forced ETS price > MAC (do not account for fossil fuel inc. since CBAM=100%!)
        if ETS_SCENARIO == '£100-Mix' or ETS_SCENARIO == '£200-Mix' or ETS_SCENARIO == '£300-Mix':
            for idx, plant in MACC.iterrows():
                if not plant['invested']:
                    if plant['MAC'] < ets_price:
                        MACC.loc[idx, 'invested'] = True
                        MACC.loc[idx, 'year_invest'] = year
        
        # Base the CTBO mandate on coal, oil, and gas supply (ktCO2f)
        pointsource_supply = MACC['ktCO2f'].sum() + MACC['ktCO2f_inc'].where(MACC['invested'], 0).sum()
        supply_ktCO2f = pointsource_supply + diffuse_supply
        pointsource_emissions = MACC['ktCO2f'].where(~MACC['invested'], 0).sum() + MACC['ktCO2f_res'].where(MACC['invested'], 0).sum()
        emitted_ktCO2f = pointsource_emissions + diffuse_supply

        stored_ktCO2f = MACC['ktCO2f_ccs'].where(MACC['invested'], 0).sum()
        stored_ktCO2cem = MACC['ktCO2cem_ccs'].where(MACC['invested'], 0).sum()
        stored_ktCO2pl = MACC['ktCO2pl_ccs'].where(MACC['invested'], 0).sum()
        stored_ktCO2b = MACC['ktCO2b_ccs'].where(MACC['invested'], 0).sum()

        if ctbo_active:
            # Mandate CTBO compliance for which any CO2 type can be stored
            ctbo_mandate = supply_ktCO2f * ctbo_fraction
            stored_missing = ctbo_mandate - (stored_ktCO2f + stored_ktCO2cem + stored_ktCO2pl + stored_ktCO2b + stored_ktCO2daccs)

            j = 0
            while stored_missing > 0 and j < len(MACC):
                plant = MACC.iloc[j]
                if not plant['invested']:
                    # Consider this plant for the CTBO if it's chepar than DACCS (ignoring extra ETS costs)
                    if plant['MAC'] > cost_DACCS:
                        break
                    MACC.loc[MACC.index[j], 'invested'] = True
                    MACC.loc[MACC.index[j], 'year_invest'] = year
                    stored_ktCO2f += plant['ktCO2f_ccs']
                    stored_ktCO2cem += plant['ktCO2cem_ccs']
                    stored_ktCO2pl += plant['ktCO2pl_ccs']
                    stored_ktCO2b += plant['ktCO2b_ccs']
                    stored_missing = ctbo_mandate - (stored_ktCO2f + stored_ktCO2cem + stored_ktCO2pl + stored_ktCO2b + stored_ktCO2daccs)
                j += 1
            
            # Calculate costs
            plants_invested = MACC[MACC['invested']]
            if len(plants_invested) > 0:
                plant_marginal = plants_invested.loc[plants_invested['MAC'].idxmax()]
                cost_marginal = plant_marginal['MAC']

            if stored_missing > 0 or year_DACCS_marginal is not None:
                if stored_missing > 0:
                    stored_ktCO2daccs += stored_missing
                cost_marginal = cost_DACCS
                if year_DACCS_marginal is None:
                    year_DACCS_marginal = year

            # Determine policy costs from recalculated emissions
            if ETS_SCENARIO == 'CTBO-only':
                E = 0
                y = cost_marginal
            elif ETS_SCENARIO == 'ETS-eq':
                E = max(ETS_START*pounds_to_EUR, cost_marginal)
                y = 0
            else:
                # In mixed policies, ETS covers stored-CO2 support up to marginal cost.
                # E = min(ets_price, max(cost_marginal, ETS_START*pounds_to_EUR)) # In early years, pick ETS_START, in late years, pick marginal cost
                E = ets_price
                y = max(0, cost_marginal - E) # Never negative!

            pointsource_emissions = MACC['ktCO2f'].where(~MACC['invested'], 0).sum() + MACC['ktCO2f_res'].where(MACC['invested'], 0).sum()
            emitted_ktCO2f = pointsource_emissions + diffuse_supply
            emitted_ktCO2_ETS = (
            MACC['ktCO2f'].where(~MACC['invested'], 0).sum() + MACC['ktCO2f_res'].where(MACC['invested'], 0).sum() +
            MACC['ktCO2cem'].where(~MACC['invested'], 0).sum() + MACC['ktCO2cem_res'].where(MACC['invested'], 0).sum() +
            MACC['ktCO2pl'].where(~MACC['invested'], 0).sum() + MACC['ktCO2pl_res'].where(MACC['invested'], 0).sum() )

            # Costs of what CO2 has been abated
            costs_suppliers = y * (stored_ktCO2f + stored_ktCO2cem + stored_ktCO2pl + stored_ktCO2b + stored_ktCO2daccs) # [k€/y] 
            costs_emitters = E * (stored_ktCO2f + stored_ktCO2cem + stored_ktCO2pl + stored_ktCO2b + stored_ktCO2daccs) # [k€/y] 

            # Costs of what CO2 has NOT been abated yet
            costs_tax = E * (emitted_ktCO2_ETS - (stored_ktCO2b + stored_ktCO2daccs)) # [k€/y]

            # Subtract outlier contributions
            for outlier_name in outliers:
                plant = MACC[MACC['stack'] == outlier_name].iloc[0]
                if plant['invested']:
                    costs_suppliers -= y * plant['ktCO2tot_ccs']
                    costs_emitters -= E * plant['ktCO2tot_ccs']
                    costs_tax -= E * (plant['ktCO2f_res'] + plant['ktCO2cem_res'] + plant['ktCO2pl_res'])
                else:
                    costs_tax -= E * (plant['ktCO2f'] + plant['ktCO2cem'] + plant['ktCO2pl'])
            costs_suppliers = max(0, costs_suppliers) #  Ensures no negative costs_suppliers after outlier subtraction
            costs_emitters = max(0, costs_emitters) #  Ensures no negative costs_emitters after outlier subtraction
            costs_tax = max(0, costs_tax) #  Ensures no negative costs_tax after outlier subtraction
            if emitted_ktCO2_ETS <= (stored_ktCO2b + stored_ktCO2daccs):
                if costs_tax != 0:
                    raise ValueError(f"Costs tax is not zero: {costs_tax} k€/y")

            costs_consumers = costs_suppliers + costs_emitters + costs_tax # [k€/y]
            cost_fuels = (costs_suppliers + costs_emitters + costs_tax) / supply_ktCO2f # [€/tCO2] NOTE: No price ceiling implemented beyond 2050!
            
        # Calculate plant and policy costs and profits based on MACC areas
        profit_y_tot = 0 # [k€/y] area A 
        cost_y_tot = 0 # [k€/y] area B
        profit_E_tot = 0 # [k€/y] area C
        cost_E_tot = 0 # [k€/y] area D
        tax_E_tot = 0 # [k€/y] area E

        for idx, plant in MACC.iterrows():
            
            S = plant['MAC'] # Set costs based on MACC costs, and override for outliers
            if plant['stack'] == 'Padeswood-cement':
                S = cost_initial_outliers['Padeswood-cement']
            elif plant['stack'] == 'Protos-waste':
                S = cost_initial_outliers['Protos-waste']
            elif plant['stack'] == 'Teeside-ccgt':
                S = cost_initial_outliers['Teeside-ccgt']

            if not plant['invested']:
                profit_y = 0
                cost_y = 0 # assuming CBAM=100%, so m=1, all embedded costs are passed on.
                profit_E = 0
                cost_E = 0
                tax_E = E * (plant['ktCO2f'] + plant['ktCO2cem'] + plant['ktCO2pl']) # [k€/y]
                cash_inflow_y = 0
                cash_inflow_E = 0
            else:
                profit_y = max(0, cost_marginal - max(E, S)) * plant['ktCO2tot_ccs'] # [k€/y] area A 
                cost_y = max(0, S - E) * plant['ktCO2tot_ccs'] # [k€/y] area B
                profit_E = max(0, E - S) * plant['ktCO2tot_ccs'] # [k€/y] area C
                cost_E = min(E, S) * plant['ktCO2tot_ccs'] # [k€/y] area D
                tax_E = E * (plant['ktCO2f_res'] + plant['ktCO2cem_res'] + plant['ktCO2pl_res']) # [k€/y] area E
                cash_inflow_y = profit_y + cost_y # Full y inflow = area A + area B
                cash_inflow_E = profit_E + cost_E # Full E inflow = area C + area D
            
            # If plant is not an outlier, add its profits and costs to the total
            if plant['stack'] not in outliers: 
                profit_y_tot += profit_y
                cost_y_tot += cost_y
                profit_E_tot += profit_E
                cost_E_tot += cost_E
                tax_E_tot += tax_E

            _plants_costbenefit.append({
                'year': year,
                'stack': plant['stack'],
                'sector': plant['sector'],
                'invested': plant['invested'],
                'investment_year': plant['year_invest'],
                'marginal_plant': cost_marginal == S,
                'CSU_price': y,
                'ETS_price': E,
                'MAC': S,
                'CAPEX': plant['CAPEX'] * 10**3, # [k€]
                'OPEX': plant['OPEX'] * plant['ktCO2tot_ccs'], # [k€/y] 
                'cash_inflow_y': cash_inflow_y, # [k€/y] full cash inflow from y-policy
                'cash_inflow_E': cash_inflow_E, # [k€/y] full cash inflow from E-policy
                # 'relative_profit_y': profit_y, # [k€/y] old relative-profit-only definition
                # 'relative_profit_E': profit_E, # [k€/y] old relative-profit-only definition
                'ktCO2tot_ccs': plant['ktCO2tot_ccs'],
            })

        # Add DACCS and calculate policy costs/profits
        if stored_ktCO2daccs > 0:
            cost_y_tot += y * stored_ktCO2daccs # [k€/y]
            cost_E_tot += E * stored_ktCO2daccs # [k€/y]

        # if single_run:
        #     print("These two methods of calculating policy costs should yield zero")
        #     print(costs_emitters - (cost_E_tot + profit_E_tot))
        #     print(costs_suppliers - (cost_y_tot + profit_y_tot))
        #     print(costs_tax - tax_E_tot)

        # Calculate the fuel price increase and store results
        gas_increase_abs = cost_fuels * emission_factor_gas # [€/MWh]
        petrol_increase_abs = cost_fuels * (emission_factor_petrol/1000) / pounds_to_EUR * 100 # [cent/L]
        diesel_increase_abs = cost_fuels * (emission_factor_diesel/1000) / pounds_to_EUR * 100 # [cent/L]
        kerosene_increase_abs = cost_fuels * (emission_factor_kerosene/1000) / pounds_to_EUR * 100 # [cent/L]

        _supply_ktCO2f.append(supply_ktCO2f)
        _emitted_ktCO2f.append(emitted_ktCO2f)
        _emitted_ktCO2final.append(emitted_ktCO2_ETS + diffuse_supply)
        _mandate_ktCO2.append(ctbo_mandate)
        _stored_ktCO2g.append(stored_ktCO2b+stored_ktCO2daccs+stored_ktCO2cem + stored_ktCO2pl) # Can balance remaining fossil FUEL emissions
        _stored_ktCO2b.append(stored_ktCO2b)
        _stored_ktCO2daccs.append(stored_ktCO2daccs)

        _cost_marginal.append(cost_marginal)
        _price_ETS.append(E)
        _price_CSU.append(y)
        _cost_suppliers.append(costs_suppliers)
        _cost_emitters.append(costs_emitters)
        _cost_tax.append(costs_tax)
        _cost_consumers.append(costs_consumers)
        _cost_fuels.append(cost_fuels)
        _profit_y_policy.append(profit_y_tot)
        _cost_y_policy.append(cost_y_tot)
        _profit_E_policy.append(profit_E_tot)
        _cost_E_policy.append(cost_E_tot)
        _tax_E_policy.append(tax_E_tot)
        _gas_increase_abs.append(gas_increase_abs)
        _petrol_increase_abs.append(petrol_increase_abs)
        _diesel_increase_abs.append(diesel_increase_abs)
        _kerosene_increase_abs.append(kerosene_increase_abs)
    
    # ========== NPV CALCULATIONS ==========
    # Discount factors for annual values (relative to START_YEAR), truncated to 2050
    NPV_END = 2050
    n_npv = NPV_END - START_YEAR + 1
    discount_factors = 1 / (1 + DISCOUNT_RATE) ** (years - START_YEAR)
    df_npv = discount_factors[:n_npv]
    
    # --- Policy-level NPV (up to and including 2050) ---
    NPV_costs_suppliers = (np.array(_cost_suppliers)[:n_npv] * df_npv).sum()   # [k€] equals NPV_profit_y_policy+NPV_cost_y_policy
    NPV_costs_emitters = (np.array(_cost_emitters)[:n_npv] * df_npv).sum()     # [k€] equals NPV_profit_E_policy+NPV_cost_E_policy
    NPV_costs_tax = (np.array(_cost_tax)[:n_npv] * df_npv).sum()               # [k€] equals NPV_tax_E_policy
    NPV_costs_consumers = (np.array(_cost_consumers)[:n_npv] * df_npv).sum()   # [k€]
    NPV_profit_y_policy = (np.array(_profit_y_policy)[:n_npv] * df_npv).sum()  # [k€]
    NPV_cost_y_policy = (np.array(_cost_y_policy)[:n_npv] * df_npv).sum()      # [k€]
    NPV_profit_E_policy = (np.array(_profit_E_policy)[:n_npv] * df_npv).sum()  # [k€]
    NPV_cost_E_policy = (np.array(_cost_E_policy)[:n_npv] * df_npv).sum()      # [k€]
    NPV_tax_E_policy = (np.array(_tax_E_policy)[:n_npv] * df_npv).sum()        # [k€]
    
    # --- Plant-level NPV preprocessing (plant-specific horizons) ---
    NPV_data_all = pd.DataFrame(_plants_costbenefit)
    NPV_data_all = extend_plant_npv_data(NPV_data_all, lifetime_ccs, debug=False)
    investment_year_series = pd.to_numeric(NPV_data_all['investment_year'], errors='coerce')
    years_since_investment = NPV_data_all['year'] - investment_year_series
    NPV_data_all['discount_factor'] = np.where(
        investment_year_series.notna() & (years_since_investment >= 0),
        1 / (1 + discount_rate_ccs) ** years_since_investment,
        np.nan,
    )
    npv_plants = calculate_plant_npv(
        NPV_data_all,
        lifetime_ccs,
        construction_years=CONSTRUCTION_YEARS,
        debug=False
    )
    mac_by_stack = MACC.set_index('stack')['MAC'].to_dict()
    for outlier_name in outliers:
        if outlier_name in cost_initial_outliers:
            mac_by_stack[outlier_name] = cost_initial_outliers[outlier_name]
    npv_plants['MAC'] = npv_plants['stack'].map(mac_by_stack)
    if single_run or save_aux_results:
        NPV_data_all.to_csv(f'{results_dir}/plants_costbenefit_extended.csv', index=False)
        npv_plants.to_csv(f'{results_dir}/plants_npv.csv', index=False)

    results = {}
    results['ctbo_trajectory'] = ctbo_trajectory
    results['ETS_trajectory'] = ets_trajectory

    # Results group 1
    results['supply_ktCO2f'] = _supply_ktCO2f
    results['emitted_ktCO2f'] = _emitted_ktCO2f
    results['emitted_ktCO2final'] = _emitted_ktCO2final
    results['mandate_ktCO2'] = _mandate_ktCO2
    results['stored_ktCO2g'] = _stored_ktCO2g
    results['stored_ktCO2b'] = _stored_ktCO2b
    results['stored_ktCO2daccs'] = _stored_ktCO2daccs

    # Results group 2
    results['cost_marginal'] = _cost_marginal
    results['price_ETS'] = _price_ETS
    results['price_CSU'] = _price_CSU
    results['cost_fuels'] = _cost_fuels

    # Results group 3
    results['costs_suppliers'] = _cost_suppliers
    results['costs_emitters'] = _cost_emitters
    results['costs_tax'] = _cost_tax
    results['costs_consumers'] = _cost_consumers
    results['profit_y_policy'] = _profit_y_policy
    results['cost_y_policy'] = _cost_y_policy
    results['profit_E_policy'] = _profit_E_policy
    results['cost_E_policy'] = _cost_E_policy
    results['tax_E_policy'] = _tax_E_policy
    
    results['NPV_costs_suppliers'] = NPV_costs_suppliers
    results['NPV_costs_emitters'] = NPV_costs_emitters
    results['NPV_costs_tax'] = NPV_costs_tax
    results['NPV_costs_consumers'] = NPV_costs_consumers
    results['NPV_profit_y_policy'] = NPV_profit_y_policy
    results['NPV_cost_y_policy'] = NPV_cost_y_policy
    results['NPV_profit_E_policy'] = NPV_profit_E_policy
    results['NPV_cost_E_policy'] = NPV_cost_E_policy
    results['NPV_tax_E_policy'] = NPV_tax_E_policy

    # Results group 4
    results['gas_increase_abs'] = _gas_increase_abs
    results['petrol_increase_abs'] = _petrol_increase_abs
    results['diesel_increase_abs'] = _diesel_increase_abs
    results['kerosene_increase_abs'] = _kerosene_increase_abs

    # Plant-level NPV results (plant-specific lifetime horizons)
    results['plants_stack'] = npv_plants['stack'].tolist()
    results['plants_sector'] = npv_plants['sector'].tolist()
    # Keep year arrays float so EMA can store NaN consistently across experiments.
    results['plants_investment_year'] = pd.to_numeric(
        npv_plants['investment_year'], errors='coerce'
    ).astype(float).tolist()
    results['plants_npv_end_year'] = pd.to_numeric(
        npv_plants['npv_end_year'], errors='coerce'
    ).astype(float).tolist()
    results['plants_NPV_CAPEX'] = npv_plants['NPV_CAPEX'].tolist()
    results['plants_NPV_OPEX'] = npv_plants['NPV_OPEX'].tolist()
    results['plants_NPV_REVENUE'] = npv_plants['NPV_REVENUE'].tolist()
    results['plants_NPV_total'] = npv_plants['NPV_total'].tolist()
    results['plants_MAC'] = npv_plants['MAC'].tolist()

    if single_run:
        plot_results_groups(years, results, figures_dir=figures_dir, savefig=True)
        plot_plants_npv_scatter(npv_plants, figures_dir=figures_dir, savefig=True)

    return results

def plot_results_groups(years, results, figures_dir='results_figures', savefig=False, debug=False):
    """
    Plot four time-series figures, one for each results group.
    NPVs are annotated in Group 3 instead of plotted as lines.
    """
    if debug:
        print(f"plot_results_groups inputs: years={len(years)}")

    # Group 1: Carbon flows
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(years, results['supply_ktCO2f'], lw=2, label='Supply [ktCO2f]')
    # ax.plot(years, results['emitted_ktCO2f'], lw=2, label='Emitted [ktCO2f]')
    ax.plot(years, results['emitted_ktCO2final'], lw=2, label='Emitted final [ktCO2f,c,p]')
    ax.plot(years, results['emitted_ktCO2f'], lw=2, label='Emitted fuel [ktCO2f]')
    ax.plot(years, results['mandate_ktCO2'], lw=2, label='Mandate [ktCO2]')
    ax.plot(years, results['stored_ktCO2g'], lw=2, label='Stored [ktCO2,b,daccs,cem,pl]')
    ax.plot(years, results['stored_ktCO2b'], lw=2, label='Stored [ktCO2,b]')
    ax.plot(years, results['stored_ktCO2daccs'], lw=2, label='Stored [ktCO2,daccs]')
    ax.set_title('Results Group 1: Carbon flows', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('ktCO2/y', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/group1_timeseries.png', dpi=450, bbox_inches='tight')

    # Group 2: Prices and marginal costs
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(years, results['cost_marginal'], lw=2, label='Marginal cost [€/tCO2]')
    ax.plot(years, results['price_ETS'], lw=2, label='E / ETS price [€/tCO2]')
    ax.plot(years, results['price_CSU'], lw=2, label='gamma / CSU price [€/tCO2]')
    ax.plot(years, results['cost_fuels'], lw=2, label='Fuel cost [€/tCO2]')
    ax.set_title('Results Group 2: Prices and costs', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('€/tCO2', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/group2_timeseries.png', dpi=450, bbox_inches='tight')

    # Group 3: Annual policy costs/profits + NPV annotation
    fig, ax = plt.subplots(figsize=(12, 7))
    scale_billion = 1e-6  # [B€/k€]
    ax.plot(years, np.array(results['costs_suppliers']) * scale_billion, lw=2, label='Costs suppliers [B€/y]')
    ax.plot(years, np.array(results['costs_emitters']) * scale_billion, lw=2, label='Costs emitters [B€/y]')
    ax.plot(years, np.array(results['costs_tax']) * scale_billion, lw=2, label='Costs tax [B€/y]')
    ax.plot(years, np.array(results['costs_consumers']) * scale_billion, lw=2, label='Costs consumers [B€/y]')
    # ax.plot(years, results['profit_y_policy'], lw=2, label='Profit y policy [k€/y]')
    # ax.plot(years, results['cost_y_policy'], lw=2, label='Cost y policy [k€/y]')
    # ax.plot(years, results['profit_E_policy'], lw=2, label='Profit E policy [k€/y]')
    # ax.plot(years, results['cost_E_policy'], lw=2, label='Cost E policy [k€/y]')
    # ax.plot(years, results['tax_E_policy'], lw=2, label='Tax E policy [k€/y]')
    npv_text = (
        f"NPV costs suppliers: {results['NPV_costs_suppliers'] * scale_billion:.3f} B€\n"
        f"NPV costs emitters: {results['NPV_costs_emitters'] * scale_billion:.3f} B€\n"
        f"NPV costs tax: {results['NPV_costs_tax'] * scale_billion:.3f} B€\n"
        f"NPV costs consumers: {results['NPV_costs_consumers'] * scale_billion:.3f} B€\n"
        # f"NPV profit y: {results['NPV_profit_y_policy']:.1f} kEUR\n"
        # f"NPV cost y: {results['NPV_cost_y_policy']:.1f} kEUR\n"
        # f"NPV profit E: {results['NPV_profit_E_policy']:.1f} kEUR\n"
        # f"NPV cost E: {results['NPV_cost_E_policy']:.1f} kEUR\n"
        # f"NPV tax E: {results['NPV_tax_E_policy']:.1f} kEUR"
    )
    ax.text(
        0.02, 0.98, npv_text, transform=ax.transAxes, va='top', ha='left', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray')
    )
    ax.set_title('Results Group 3: Policy annual values', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('B€/y', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/group3_timeseries.png', dpi=450, bbox_inches='tight')

    # Group 4: Fuel price impacts
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(years, results['gas_increase_abs'], lw=2, label='Gas increase [€/MWh]')
    ax.plot(years, results['petrol_increase_abs'], lw=2, label='Petrol increase [cent/L]')
    ax.plot(years, results['diesel_increase_abs'], lw=2, label='Diesel increase [cent/L]')
    ax.plot(years, results['kerosene_increase_abs'], lw=2, label='Kerosene increase [cent/L]')
    ax.set_title('Results Group 4: Fuel price impacts', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Absolute increase', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/group4_timeseries.png', dpi=450, bbox_inches='tight')

    plt.show()

def plot_plants_npv_scatter(npv_plants, figures_dir='results_figures', savefig=False, debug=False):
    """
    Scatter plot of plant NPVs with x=investment year and y=NPV_total.
    Plants in the drax sector are excluded.
    """
    df = pd.DataFrame(npv_plants).copy()
    if debug:
        print(f"plot_plants_npv_scatter inputs: n_plants={len(df)}")

    if 'sector' in df.columns:
        df = df[df['sector'].str.lower() != 'drax']
    df = df.dropna(subset=['investment_year', 'NPV_total', 'sector'])
    if df.empty:
        if debug:
            print("plot_plants_npv_scatter output: no data to plot")
        return 0

    sectors = sorted(df['sector'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(sectors))))
    sector_colors = {sector: color for sector, color in zip(sectors, colors)}

    fig, ax = plt.subplots(figsize=(11, 7))
    for sector in sectors:
        sector_df = df[df['sector'] == sector]
        ax.scatter(
            sector_df['investment_year'],
            sector_df['NPV_total'] / 1000,  # [M€]
            s=55,
            c=[sector_colors[sector]],
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            label=sector,
        )

    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title('Plant NPV by Investment Year (excl. Drax)', fontsize=16)
    ax.set_xlabel('Investment Year', fontsize=14)
    ax.set_ylabel('NPV total [M€]', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title='Sector', fontsize=11, title_fontsize=12, loc='best')

    plt.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/plants_npv_scatter.png', dpi=450, bbox_inches='tight')
    plt.show()

    if debug:
        print(f"plot_plants_npv_scatter output: n_plotted={len(df)}")
    return len(df)

def plot_macc(macc, pounds_to_EUR=1.15, figures_dir='results_figures', savefig=False, debug=False):
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

    # Sort by MAC and compute cumulative (including MAC=0 plants)
    macc_plot = macc_plot.sort_values(by='MAC')
    macc_plot['cumulative_kt'] = macc_plot['ktCO2tot_ccs'].cumsum()
    
    # Filter to only plot plants with MAC != 0 (but cumulative already includes MAC=0 capacity)
    macc_nonzero = macc_plot[macc_plot['MAC'] != 0]

    # Convert MAC to £/tCO₂
    macc_nonzero['MAC'] = macc_nonzero['MAC'] / pounds_to_EUR

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(macc_nonzero['cumulative_kt']/1000, macc_nonzero['MAC'], where='pre', color='black', linewidth=2.5)
    ax.set_xlabel("Cumulative CCS/BECCS capacity [MtCO₂ p.a.]", fontsize=14)
    ax.set_ylabel("Abatement cost of CCS/BECCS [£/tCO₂] ", fontsize=14)
    # ax.set_title("Marginal Abatement Cost Curve", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=12)
    # Get the ylims
    ylims = ax.get_ylim()
    # Set xlims
    ax.set_ylim(0, ylims[1])

    plt.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/macc_curve.png', dpi=450, bbox_inches='tight')

    if debug:
        print("plot_macc output:", macc_plot[['cumulative_kt', 'MAC']].tail(1))

    return len(macc_plot)

if __name__ == "__main__":
    plants_clean = pd.read_csv('results_baseline/plants_clean.csv')
    transport_hubs = pd.read_csv('data/transport_hubs.csv')
    results = simulate_ctbo(plants_clean, transport_hubs, single_run=True, results_dir='results_baseline', figures_dir='results_figures')

    # results['plants_investment_year'] = [str(year) for year in results['plants_investment_year']] #Convert from np.float to string
    # print(f"\nThe stacks that have invested are (alphabetically ordered): {results['plants_stack']}")
    # print(f"\nThe investment years are (alphabetically ordered): {results['plants_investment_year']}")