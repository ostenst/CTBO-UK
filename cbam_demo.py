"""Load plant CO2 and the pre-learning abatement cost (MAC0) across scenarios."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PREPARE AND PRINT DATA
results_dir = "results_baseline"
plants = pd.read_csv("results_baseline/plants_clean.csv")
macc = pd.read_csv(f"{results_dir}/macc.csv")
plant_ref = pd.read_csv(f"{results_dir}/plant_reference.csv")
mac0 = np.load(f"{results_dir}/outcomes_plants_MAC0.npy")

generated = macc[["stack", "ktCO2f", "ktCO2cem", "ktCO2pl", "ktCO2b"]]
mac0_by_plant = pd.DataFrame({
    "stack": plant_ref["stack"].to_numpy(),
    "MAC0_min": np.nanmin(mac0, axis=0),
    "MAC0_median": np.nanmedian(mac0, axis=0),
    "MAC0_max": np.nanmax(mac0, axis=0),
})
view = plants[["sector", "site", "stack", "ktCO2"]].merge(generated, on="stack", how="left")
view = view.merge(mac0_by_plant, on="stack", how="left")

print(f"plants_clean: {len(plants)} plants")
print(f"macc generated CO2: {len(macc)} plants")
print(f"{results_dir}/outcomes_plants_MAC0.npy: {mac0.shape[0]} scenarios x {mac0.shape[1]} plants")
print(f"plants without MAC0 scenarios: {int(view['MAC0_median'].isna().sum())}")
print("Generated CO2 [ktCO2/y] by type. MAC0 [EUR/tCO2] is the pre-learning cost across scenarios.")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:.1f}")
print(view.to_string(index=False))

# Pnet along the CBAM strength m, for one cement, one CCGT, and one waste plant
E = 200
y = 100
r = 0.90
m = np.linspace(0, 1, 100)
g_cases = [(0.36, "--"), (1.0, "-")]
print(f"E: {E} [EUR/t], y: {y} [EUR/t], r: {r} [-]")

plot_plants = [
    ("Aberthaw-cement", "black"),
    ("Enfield Power Station-ccgt", "gray"),
    ("Runcorn-waste", plt.cm.magma(0.55)),
]
fig, ax = plt.subplots(figsize=(8, 5.5))
for stack, color in plot_plants:
    row = view.loc[view["stack"] == stack].iloc[0]
    ctot = row["ktCO2f"] + row["ktCO2cem"] + row["ktCO2pl"] + row["ktCO2b"]
    cf = row["ktCO2f"]
    ccem = row["ktCO2cem"]
    cpl = row["ktCO2pl"]
    S = row["MAC0_median"]
    print(f"{stack}: ctot={ctot:.1f}, cf={cf:.1f}, ccem={ccem:.1f}, cpl={cpl:.1f} ktCO2/y, S={S:.1f} EUR/tCO2")
    for g, linestyle in g_cases:
        Pnet = E * (ctot * r - (cf + ccem + cpl) * (1 - m)) + y * (ctot * r - cf * g * (1 - m)) - S * ctot * r
        ax.plot(m, Pnet, color=color, lw=2.2, linestyle=linestyle, label=f"{row['sector']}: {row['site']}, g={g}")

ax.axhline(0, color="gray", lw=1)
ax.text(
    0.02, 0.98,
    f"Price scenario\nE = {E:.0f} EUR/tCO₂\ny = {y:.0f} EUR/tCO₂",
    transform=ax.transAxes, va="top", ha="left", fontsize=13, color="black",
)
ax.set_xlabel("CBAM strength m [-]", fontsize=14)
ax.set_ylabel("Pnet [kEUR/y]", fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("results_figures/cbam_pnet.png", dpi=200)
plt.show()