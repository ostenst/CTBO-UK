import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Prepare data
point_sources = pd.read_csv("data/w2e_plants.csv")
FLH = 8760 * 0.866 # Tolvik report
LHV = 9.52 # [MJ/kg]
emission_factor = 0.98 # [tCO2/twaste]
fossil = 0.465 # fossil content of the CO2 emitted

eta_elec = 605 # [kWh/twaste] *3.6 to get MJ
eta_elec = 2178 # [MJ/twaste]
eta_heat = 110 # [kWh/twaste] *3.6 to get MJ
eta_heat = 396 # [MJ/twaste]
lost_heat = LHV*1000 - eta_elec - eta_heat # [MJ/twaste]

point_sources['CO2'] = point_sources['Capacity 2023 [ktpa]']*1000 * emission_factor # [tCO2/yr]
total_co2 = point_sources['CO2'].sum()

# Calculating costs for CCS at W2E
a = 2.1673
b = 0.8092
c = -0.00332
n = 0.5291
m = 0.8391

results = []
for idx, plant in point_sources.iterrows():
    xCO2 = 0.11 # [-] NOTE: I must recalculate this
    mCO2 = plant['CO2'] #[tCO2/yr]

    mCO2 = mCO2 / FLH # [tCO2/h]
    nCO2 = mCO2*1000 / 44 # [kmolCO2/h]
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
    CAPEX = CAPEX * 5.509 # [MEUR] NETL Methodology
    CEPCI_2025 = 930
    CEPCI_2023 = 798.7
    CAPEX = CAPEX * CEPCI_2025 / CEPCI_2023

    capture_rate = 0.95 # [0-1]
    annualized_CAPEX = CAPEX * 0.07 * (1 + 0.07)**25 / ((1 + 0.07)**25 - 1) *10**6# [EUR/y]
    levelized_CAPEX = annualized_CAPEX / (mCO2*FLH*capture_rate) # [EUR/tCO2]
    levelized_CAPEX_FOAK = 1.7553 * levelized_CAPEX # [EUR/tCO2]

    # Calculate transport cost
    transport_cost = 35 # NOTE: must calculate later using geographic distance
    transport_cost = transport_cost * CEPCI_2025 / CEPCI_2023

    # Estimate energy OPEX (industry)
    Qreb = 3.5 * mCO2*capture_rate*1000/3600 # [MW] NOTE: assuming that waste heat is used
    cost_steam = 4.1 # [EUR/tsteam @130C] [Ali]
    evaporation_enthalpy = 2257 # [MJ/tsteam]
    cost_steam = cost_steam / evaporation_enthalpy # [EUR/MJ]
    Qsteam = Qreb * FLH * 3600 # [MJ/y] assuming all is covered by recovered steam
    OPEX_steam = cost_steam * Qsteam # [EUR/y]        
    OPEX_steam = OPEX_steam / (plant['CO2']*capture_rate) # [EUR/tCO2]
    OPEXE = OPEX_steam
   
    total_cost_foak = levelized_CAPEX_FOAK + transport_cost + OPEXE
    total_cost_noak = levelized_CAPEX + transport_cost + OPEXE
        
    # Store results
    results.append({
        'Plant': plant['Permit name'],
        'CO2_kt_y': plant['CO2']/1000,
        'FLH_h': FLH,
        'CAPEX_FOAK': levelized_CAPEX_FOAK,
        'CAPEX_NOAK': levelized_CAPEX,
        'Transport_EUR_tCO2': transport_cost,
        'OPEX_EUR_tCO2': OPEXE,
        'Total_FOAK': total_cost_foak,
        'Total_NOAK': total_cost_noak
    })

# Drax power station data:
# https://onlinelibrary.wiley.com/doi/full/10.1111/gcbb.12695 NOTE: Has other suggested BECCS projects
Drax_CO2 = 11500000 # [tCO2/yr]
Pinstalled = 2580 # [MW]
eta_P = 33/(1-0.24) # ~44% efficiency before a CCS retrofit, which leads to 33%
Qfuel = Pinstalled / (eta_P/100) # [MWfuel]
emission_factor = 0.3318 # tCO2/MWhfuel [NZIP, 2020]
FLH = Drax_CO2 / (Qfuel * emission_factor) # [h/y] = tCO2/yr / (tCO2/h)

xCO2 = 0.13 # [-] NOTE: I guessed this

mCO2 = Drax_CO2 / FLH # [tCO2/h]
nCO2 = mCO2*1000 / 44 # [kmolCO2/h]
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
CAPEX = CAPEX * 5.509 # [MEUR] NETL Methodology
CEPCI_2025 = 930
CEPCI_2023 = 798.7
CAPEX = CAPEX * CEPCI_2025 / CEPCI_2023

capture_rate = 0.95 # [0-1]
annualized_CAPEX = CAPEX * 0.07 * (1 + 0.07)**25 / ((1 + 0.07)**25 - 1) *10**6# [EUR/y]
levelized_CAPEX = annualized_CAPEX / (Drax_CO2*capture_rate) # [EUR/tCO2]
levelized_CAPEX_FOAK = 1.7553 * levelized_CAPEX # [EUR/tCO2]

# Calculate transport cost
transport_cost = 35 # NOTE: must calculate later using geographic distance
transport_cost = transport_cost * CEPCI_2025 / CEPCI_2023

# Estimate energy OPEX (Drax)
celc = 80 # [EUR/MWh electricity]
profit_baseline = Qfuel * eta_P/100 * FLH * celc # [EUR/y]
profit_BECCS = Qfuel * eta_P/100*(1-0.24) * FLH * celc  # [EUR/y]
difference = profit_baseline - profit_BECCS # [EUR/y]
OPEXE = difference / (Drax_CO2*capture_rate) # [EUR/tCO2]
   
total_cost_foak = levelized_CAPEX_FOAK + transport_cost + OPEXE
total_cost_noak = levelized_CAPEX + transport_cost + OPEXE
        
# Store results
results.append({
    'Plant': 'Drax',
    'CO2_kt_y': Drax_CO2/1000,
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
    print(f"{'Plant Name':<30} | {'CO2':<8} | {'FLH':<6} | {'FOAK':<8} | {'NOAK':<8} | {'Transport':<9} | {'OPEX':<8} | {'TotalFOAK':<9} | {'TotalNOAK':<9}")
    print(f"{'':<30} | {'kt/y':<8} | {'h/y':<6} | {'EUR/tCO2':<8} | {'EUR/tCO2':<8} | {'EUR/tCO2':<9} | {'EUR/tCO2':<8} | {'EUR/tCO2':<9} | {'EUR/tCO2':<9}")
    print("-" * 120)
    
    for result in results:
        print(f"{result['Plant']:<30} | {result['CO2_kt_y']:>8.1f} | {result['FLH_h']:>6.0f} | {result['CAPEX_FOAK']:>8.1f} | {result['CAPEX_NOAK']:>8.1f} | {result['Transport_EUR_tCO2']:>9.1f} | {result['OPEX_EUR_tCO2']:>8.1f} | {result['Total_FOAK']:>9.1f} | {result['Total_NOAK']:>9.1f}")
    
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
    print(f"Number of plant-stacks analyzed: {len(results)}")
    print(f"Total CO2 emissions: {sum([r['CO2_kt_y'] for r in results]):,.1f} ktCO2/y")
    print(f"Average total CCS cost (FOAK): {avg_total_foak:.1f} EUR/tCO2")
    print(f"Average total CCS cost (NOAK): {avg_total_noak:.1f} EUR/tCO2")
    print(f"Cost range (FOAK): {min([r['Total_FOAK'] for r in results]):.1f} - {max([r['Total_FOAK'] for r in results]):.1f} EUR/tCO2")
    print(f"Cost range (NOAK): {min([r['Total_NOAK'] for r in results]):.1f} - {max([r['Total_NOAK'] for r in results]):.1f} EUR/tCO2")