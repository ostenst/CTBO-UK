import numpy as np
from scipy.optimize import least_squares

# --- Original model parameters (used as initial guess) ---
a = 2.1673
b = 0.8092
c = -0.00332
n = 0.5291
m = 0.8391

original_params = np.array([a, b, c, n, m])

dollar_to_euro = 0.85

# --- Plant data with target CAPEX ---
plants = [
    {"name":"dam", "mCO2": 135, "xCO2": 0.13, "CAPEX_target": 780},
    {"name":"nova", "mCO2": 200, "xCO2": 0.13, "CAPEX_target": 630},
    {"name":"shand", "mCO2": 272, "xCO2": 0.13, "CAPEX_target": 810},
    {"name":"klem", "mCO2": 52, "xCO2": 0.11, "CAPEX_target": 310},
    {"name":"norcem", "mCO2": 55, "xCO2": 0.20, "CAPEX_target": 390},
    {"name":"sherman", "mCO2": 129, "xCO2": 0.06, "CAPEX_target": 477 * dollar_to_euro}
]

# --- Model function ---
def capex_model(params, mCO2, xCO2):
    a, b, c, n, m = params
    nCO2 = mCO2 * 1000 / 44           # [kmolCO2/h]
    n_fluegas = nCO2 / xCO2           # [kmol/h]
    V_fluegas = n_fluegas * 22.4      # [Nm3/h]
    TEC = a + (b * (xCO2)**n + c) * (V_fluegas/1000)**m
    CAPEX = TEC * 5.509               # [MEUR]
    return CAPEX

# --- Stage 1: Fit a, b, c only (n and m fixed) ---
def residuals_abc(params_abc):
    a_, b_, c_ = params_abc
    res = []
    for plant in plants:
        CAPEX_pred = capex_model([a_, b_, c_, n, m], plant["mCO2"], plant["xCO2"])
        res.append(CAPEX_pred - plant["CAPEX_target"])
    return np.array(res)

initial_guess_abc = [a, b, c]
result_abc = least_squares(residuals_abc, initial_guess_abc)
a_stage1, b_stage1, c_stage1 = result_abc.x

print("\n--- Stage 1: Fitted a, b, c ---")
print(f"a = {a_stage1:.5f}, b = {b_stage1:.5f}, c = {c_stage1:.5f}")

# --- Stage 2: Fit n and m only (a, b, c fixed) ---
def residuals_nm(params_nm):
    n_, m_ = params_nm
    res = []
    for plant in plants:
        CAPEX_pred = capex_model([a_stage1, b_stage1, c_stage1, n_, m_],
                                 plant["mCO2"], plant["xCO2"])
        res.append(CAPEX_pred - plant["CAPEX_target"])
    return np.array(res)

initial_guess_nm = [n, m]
result_nm = least_squares(residuals_nm, initial_guess_nm)
n_stage2, m_stage2 = result_nm.x

print("\n--- Stage 2: Fitted n, m ---")
print(f"n = {n_stage2:.5f}, m = {m_stage2:.5f}")

# --- Combine final calibrated parameters ---
final_params = [a_stage1, b_stage1, c_stage1, n_stage2, m_stage2]

print("\n--- Final Calibrated Parameters ---")
print(f"a = {final_params[0]:.5f}, b = {final_params[1]:.5f}, c = {final_params[2]:.5f}, "
      f"n = {final_params[3]:.5f}, m = {final_params[4]:.5f}")

# --- Compare predicted vs target CAPEX ---
print("\n--- CAPEX comparison ---")
for plant in plants:
    CAPEX_pred = capex_model(final_params, plant["mCO2"], plant["xCO2"])
    print(f"{plant['name']:8s} | Target: {plant['CAPEX_target']:6.1f} MEUR | Predicted: {CAPEX_pred:6.1f} MEUR")

residuals = []
for plant in plants:
    residuals.append(capex_model(final_params, plant["mCO2"], plant["xCO2"]) - plant["CAPEX_target"])
residuals = np.array(residuals)
print("\nResiduals (Predicted - Target):", residuals)
print("Mean residual:", np.mean(residuals))
print("RMSE:", np.sqrt(np.mean(residuals**2)))

y_true = np.array([p["CAPEX_target"] for p in plants])
y_pred = np.array([capex_model(final_params, p["mCO2"], p["xCO2"]) for p in plants])
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r_squared = 1 - ss_res / ss_tot
print("R²:", r_squared)

# TESTING FOR DRAX
emission_factor = 0.358 # [tCO2/MWh]
capacity = 2580 # [MW]
mCO2 = capacity * emission_factor / 4 # [tCO2/h] there are four biomass boilers @ 645 MWth
xCO2 = 0.14 # [-]
CAPEX_pred = capex_model(final_params, mCO2, xCO2)
print(f"\nDrax Predicted: {CAPEX_pred:6.1f} MEUR at mCO2 [tCO2/h]=", mCO2)

discount_rate = 0.07
lifetime = 25
annual_CO2 = 11 /4 # MtCO2 p.a.
capture_rate = 0.90
annualized_CAPEX = CAPEX_pred * discount_rate * (1 + discount_rate)**lifetime / ((1 + discount_rate)**lifetime - 1)
levelized_CAPEX = annualized_CAPEX / (annual_CO2 * capture_rate)
print(levelized_CAPEX)



# ----------------- PLOTTING -----------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Create a grid of mCO2 and xCO2 values ---
mCO2_vals = np.linspace(30, 300, 50)     # [tCO2/h], adjust as needed
xCO2_vals = np.linspace(0.05, 0.25, 50) # [-], adjust as needed
M, X = np.meshgrid(mCO2_vals, xCO2_vals)

# --- Compute CAPEX predictions on the grid ---
CAPEX_grid = np.zeros_like(M)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        CAPEX_grid[i, j] = capex_model(final_params, M[i, j], X[i, j])

# --- Plot 3D surface ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(M, X, CAPEX_grid, cmap='viridis', alpha=0.35, edgecolor='k')

# --- Scatter the actual plant data for reference ---
plant_mCO2 = np.array([p["mCO2"] for p in plants])
plant_xCO2 = np.array([p["xCO2"] for p in plants])
plant_CAPEX = np.array([p["CAPEX_target"] for p in plants])
ax.scatter(plant_mCO2, plant_xCO2, plant_CAPEX, color='red', s=50, label='FOAK plants')

# --- Labels and title ---
ax.set_xlabel('CO2 Mass Flow [tCO2/h]')
ax.set_ylabel('CO2 Concentration [-]')
ax.set_zlabel('CAPEX [MEUR]')
ax.set_title('CAPEX Prediction Surface (Fitted Parameters)')
ax.view_init(elev=30, azim=-60)  # Adjust viewing angle
ax.legend()
# --- Annotate each FOAK plant ---
for p in plants:
    ax.text(p["mCO2"], p["xCO2"], p["CAPEX_target"] + 10,  # slight offset above point
            p["name"], color='black', fontsize=10, weight='bold')

fig.colorbar(surf, shrink=0.5, aspect=5, label='CAPEX [MEUR]')
plt.show()