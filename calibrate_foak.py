# A model that predicts TEC costs - note that this is not valid for CCGT if xCO2<5%, or if mCO2>1250 kt/y
a = 2.1673
b = 0.8092
c = -0.00332
n = 0.5291
m = 0.8391

dollar_to_euro = 0.85
# xCO2 [-]
# V_fluegas [10**3 Nm3/h]

# Values taken from Danish Energy Agency: technology_data_for_carbon_capture_transport_and_storage_0, and Sherman FEED study (https://www.osti.gov/servlets/purl/1836563) Front-End Engineering Design (FEED) Study for a Carbon Capture Plant Retrofit to a Natural Gas-Fired Gas Turbine Combined Cycle Power Plant 
dam     = {"name":"dam", "mCO2": 135, "xCO2": 0.13}
nova    = {"name":"nova", "mCO2": 200, "xCO2": 0.13}
shand   = {"name":"shand", "mCO2": 272, "xCO2": 0.13}
klem    = {"name":"klem", "mCO2": 52, "xCO2": 0.11}
norcem  = {"name":"norcem", "mCO2": 55, "xCO2": 0.20}
sherman = {"name":"sherman", "mCO2": 129, "xCO2": 0.06}

for plant in [dam, nova, shand, klem, norcem, sherman]:
    mCO2 = plant["mCO2"]             # [tCO2/h]
    xCO2 = plant["xCO2"]             # [-]

    nCO2 = mCO2*1000 / 44            # [kmolCO2/h]
    n_fluegas = nCO2 / xCO2 # [kmol/h]
    V_fluegas = n_fluegas * 22.4     # [Nm3/h]

    TEC = a + (b * (xCO2)**n + c) * (V_fluegas/1000)**m # [MEUR]
    CAPEX = (a + (b * (xCO2)**n + c) * (V_fluegas/1000)**m) * 5.509 # [MEUR] NETL Methodology
    print(plant["name"], ": ", CAPEX, " MEUR, it's slightly off!")

dam["CAPEX_target"] = 780
nova["CAPEX_target"] = 630
shand["CAPEX_target"] = 810
klem["CAPEX_target"] = 310
norcem["CAPEX_target"] = 390
sherman["CAPEX_target"] = 477 * dollar_to_euro 

# Estimate a calibration constant for the CAPEX modeL using least squares:
# CAPEX_new = calibration * (a + (b * (xCO2)**n + c) * (V_fluegas/1000)**m) * 5.509

import numpy as np

# Calculate calibration constant using least squares
plants = [dam, nova, shand, klem, norcem, sherman]
predicted_values = []
target_values = []

for plant in plants:
    mCO2 = plant["mCO2"]             # [tCO2/h]
    xCO2 = plant["xCO2"]             # [-]

    nCO2 = mCO2*1000 / 44            # [kmolCO2/h]
    n_fluegas = nCO2 / xCO2 # [kmol/h]
    V_fluegas = n_fluegas * 22.4     # [Nm3/h]

    TEC = a + (b * (xCO2)**n + c) * (V_fluegas/1000)**m # [MEUR]
    CAPEX_predicted = TEC * 5.509 # [MEUR] NETL Methodology
    
    predicted_values.append(CAPEX_predicted)
    target_values.append(plant["CAPEX_target"])

# Least squares calibration: minimize sum((target - calibration * predicted)^2)
# Taking derivative and setting to zero: calibration = sum(target * predicted) / sum(predicted^2)
predicted_array = np.array(predicted_values)
target_array = np.array(target_values)

calibration_constant = np.sum(target_array * predicted_array) / np.sum(predicted_array**2)

print(f"\nCalibration constant (least squares): {calibration_constant:.4f}")
print(f"\nCalibrated predictions:")
for i, plant in enumerate(plants):
    calibrated_capex = predicted_values[i] * calibration_constant
    error = abs(calibrated_capex - target_values[i])
    print(f"{plant['name']}: {calibrated_capex:.1f} MEUR (target: {target_values[i]} MEUR, error: {error:.1f} MEUR)")

# Calculate R-squared
ss_res = np.sum((target_array - calibration_constant * predicted_array)**2)
ss_tot = np.sum((target_array - np.mean(target_array))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nR-squared: {r_squared:.4f}")
print(f"Root Mean Square Error: {np.sqrt(ss_res/len(target_array)):.1f} MEUR")