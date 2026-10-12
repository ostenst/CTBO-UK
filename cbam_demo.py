"""Load plant CO2 and the pre-learning abatement cost (MAC0) across scenarios."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

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
E = 0
y = 250
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

# Every stack with S < E+y. Same Pnet(m); one panel per sector.
eligible = view[view["MAC0_median"] < (E + y)]
print(f"Stacks with S < E+y ({E + y:.0f} EUR/tCO2): {len(eligible)}")
sector_order = ["steel", "refinery", "cement", "drax", "waste", "ccgt"]
sectors = [s for s in sector_order if (eligible["sector"] == s).any()]
ncols = 3
nrows = int(np.ceil(len(sectors) / ncols))
solid_lines = []
fig_s, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True)
axes_flat = np.atleast_1d(axes).ravel()
for i, sector in enumerate(sectors):
    ax = axes_flat[i]
    rows = eligible[eligible["sector"] == sector].sort_values("MAC0_median")
    shades = plt.cm.magma(np.linspace(0.15, 0.85, len(rows)))
    print(f"{sector}: {len(rows)} stacks")
    for shade, (_, row) in zip(shades, rows.iterrows()):
        ctot = row["ktCO2f"] + row["ktCO2cem"] + row["ktCO2pl"] + row["ktCO2b"]
        cf, ccem, cpl = row["ktCO2f"], row["ktCO2cem"], row["ktCO2pl"]
        S = row["MAC0_median"]
        for g, linestyle in g_cases:
            Pnet = E * (ctot * r - (cf + ccem + cpl) * (1 - m)) + y * (ctot * r - cf * g * (1 - m)) - S * ctot * r
            line, = ax.plot(m, Pnet, color=shade, lw=1.5, linestyle=linestyle)
            if g == 1.0:
                line.set_visible(False)
                solid_lines.append(line)
    ax.axhline(0, color="gray", lw=1)
    ax.set_title(sector, fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("CBAM strength m [-]", fontsize=13)
    if i % ncols == 0:
        ax.set_ylabel("Pnet [kEUR/y]", fontsize=13)
for ax in axes_flat[len(sectors):]:
    ax.axis("off")
axes_flat[0].text(
    0.02, 0.98,
    f"Price scenario\nE = {E:.0f} EUR/tCO₂\ny = {y:.0f} EUR/tCO₂",
    transform=axes_flat[0].transAxes, va="top", ha="left", fontsize=11, color="black",
)
fig_s.tight_layout(rect=(0, 0.06, 1, 1))
check_ax = fig_s.add_axes([0.01, 0.01, 0.22, 0.04])
g1_toggle = CheckButtons(check_ax, ["show solid g = 1"], [False])
g1_toggle.labels[0].set_fontsize(12)

def _toggle_solid(_label):
    show_solid = g1_toggle.get_status()[0]
    for line in solid_lines:
        line.set_visible(show_solid)
    fig_s.canvas.draw_idle()

g1_toggle.on_clicked(_toggle_solid)
fig_s.savefig("results_figures/cbam_pnet_sectors.png", dpi=200)

# Fraction of plants with Pnet > 0, by policy mix and by CBAM strength m. g = 0.36.
g = 0.36
plants_all = view.dropna(subset=["MAC0_median", "ktCO2f"]).copy()
ctot = plants_all[["ktCO2f", "ktCO2cem", "ktCO2pl", "ktCO2b"]].sum(axis=1).to_numpy()[:, None]
covered = (plants_all["ktCO2f"] + plants_all["ktCO2cem"] + plants_all["ktCO2pl"]).to_numpy()[:, None]
cf = plants_all["ktCO2f"].to_numpy()[:, None]
S = plants_all["MAC0_median"].to_numpy()[:, None]
one_m = (1 - m)[None, :]
sector_of = plants_all["sector"].to_numpy()
policy_mixes = [("EMIT", 250, 0), ("ADD", 150, 100), ("RED", 100, 150), ("CB", 0, 250)]
sector_style = {
    "steel": (plt.cm.magma(0.85), "Steel"),
    "refinery": (plt.cm.magma(0.625), "Refinery"),
    "cement": (plt.cm.magma(0.35), "Cement"),
    "drax": ("#52E7D6", "Drax"),
    "waste": ("#41BCAE", "Waste-to-energy"),
    "ccgt": ("black", "Gas power"),
}
print(f"Policy-mix Pnet: {len(plants_all)} plants, g={g}")
fig_m, axes_m = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, (name, E_mix, y_mix) in zip(axes_m.ravel(), policy_mixes):
    Pnet = E_mix * (ctot * r - covered * one_m) + y_mix * (ctot * r - cf * g * one_m) - S * ctot * r
    for sector, (color, label) in sector_style.items():
        mask = sector_of == sector
        fraction = (Pnet[mask] > 0).mean(axis=0)
        ax.plot(m, fraction, color=color, lw=2.2, label=label)
        print(f"{name} {sector}: m=0 {fraction[0]:.2f}, m=1 {fraction[-1]:.2f} (n={mask.sum()})")
    ax.set_title(f"{name}: E={E_mix:.0f}, y={y_mix:.0f} EUR/tCO₂", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("CBAM strength m [-]", fontsize=13)
    ax.set_ylabel("Fraction of plants with Pnet > 0", fontsize=13)
handles, labels = axes_m[0, 0].get_legend_handles_labels()
fig_m.legend(handles, labels, loc="center right", fontsize=12)
fig_m.tight_layout(rect=(0, 0, 0.84, 1))
fig_m.savefig("results_figures/cbam_pnet_policy_mix.png", dpi=200)
plt.show()