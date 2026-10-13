"""Two UK maps of plant stacks. Bubble color follows the sign of Pnet as m changes."""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, Slider
from shapely.geometry import box

# EMIT and RED. S is the median pre-learning abatement cost (MAC0).
g = 0.36
r = 0.90
SCENARIOS = [("ETS", 250, 0), ("CSU", 0, 250)]
GREEN = "#41BCAE"
RED_COLOR = "#E75263"
BUBBLE_SCALE = 0.8

plants = pd.read_csv("results_baseline/plants_clean.csv")
macc = pd.read_csv("results_baseline/macc.csv")
plant_ref = pd.read_csv("results_baseline/plant_reference.csv")
mac0 = np.load("results_baseline/outcomes_plants_MAC0.npy")
europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")
europe = gpd.clip(europe, box(-9.2, 49.3, 3.2, 59.7))
europe["geometry"] = europe.geometry.simplify(0.02, preserve_topology=True)

s_by_stack = pd.DataFrame({
    "stack": plant_ref["stack"].to_numpy(),
    "S": np.nanmedian(mac0, axis=0),
})
df = plants.merge(
    macc[["stack", "ktCO2f", "ktCO2cem", "ktCO2pl", "ktCO2b"]], on="stack", how="inner"
).merge(s_by_stack, on="stack", how="inner")
df = df.dropna(subset=["S", "latitude", "longitude"])

ctot = df[["ktCO2f", "ktCO2cem", "ktCO2pl", "ktCO2b"]].sum(axis=1).to_numpy()
covered = (df["ktCO2f"] + df["ktCO2cem"] + df["ktCO2pl"]).to_numpy()
cf = df["ktCO2f"].to_numpy()
S = df["S"].to_numpy()
lon = df["longitude"].to_numpy()
lat = df["latitude"].to_numpy()
size = df["ktCO2"].to_numpy() * BUBBLE_SCALE
print(f"Mapped {len(df)} stacks")


def _pnet(E, y, m):
    one = 1 - m
    return E * (ctot * r - covered * one) + y * (ctot * r - cf * g * one) - S * ctot * r


fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.subplots_adjust(bottom=0.18, top=0.90, wspace=0.06)
panels = []
for ax, (name, E, y) in zip(axes, SCENARIOS):
    viable = S <= (E + y)
    europe.plot(ax=ax, color="lightgray", edgecolor="white", alpha=0.45)
    colored = None
    if viable.any():
        colors = np.where(_pnet(E, y, 1)[viable] > 0, GREEN, RED_COLOR)
        colored = ax.scatter(
            lon[viable], lat[viable], s=size[viable], c=colors,
            alpha=0.85, edgecolors="black", linewidth=0.5, zorder=3,
        )
    gray = None
    if (~viable).any():
        gray = ax.scatter(
            lon[~viable], lat[~viable], s=size[~viable], c="gray",
            alpha=0.45, edgecolors="darkgray", linewidth=0.4, zorder=2,
        )
    ax.set_aspect(1.90)
    ax.set_xlim(-9, 3)
    ax.set_ylim(49.5, 59.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"ETS={E:.0f}, CSU={y:.0f} EUR/tCO₂", fontsize=18)
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(plt.cm.magma_r(1.0))
        spine.set_linewidth(8)
        spine.set_alpha(1)
    panels.append((colored, gray, E, y, viable))
    print(f"{name}: {int(viable.sum())} colored, {int((~viable).sum())} gray (S > E+y)")

axes[0].text(0.04, 0.97, "ETS world", transform=axes[0].transAxes, fontsize=20, fontweight="bold", va="top", ha="left", color="black")
axes[1].text(0.04, 0.97, "CTBO world (g=0.36)", transform=axes[1].transAxes, fontsize=20, fontweight="bold", va="top", ha="left", color="black")

axes[0].legend(
    handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor="black", markersize=14, label="Pnet > 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED_COLOR, markeredgecolor="black", markersize=14, label="Pnet ≤ 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray", markeredgecolor="darkgray", markersize=14, label="S > E+y"),
    ],
    loc="lower left", fontsize=15, frameon=True,
)

slider_ax = fig.add_axes([0.22, 0.06, 0.42, 0.04])
m_slider = Slider(slider_ax, "CBAM strength m [%]", 0, 100, valinit=100, valstep=0.05, color=plt.cm.magma_r(1.0), initcolor="none")
m_slider.label.set_fontsize(16)
m_slider.valtext.set_fontsize(16)

check_ax = fig.add_axes([0.68, 0.03, 0.28, 0.09])
gray_toggle = CheckButtons(check_ax, ["show S > E+y"], [True])
gray_toggle.labels[0].set_fontsize(16)


def _on_m(val):
    mm = m_slider.val / 100
    for colored, _gray, E, y, viable in panels:
        if colored is None:
            continue
        colors = np.where(_pnet(E, y, mm)[viable] > 0, GREEN, RED_COLOR)
        colored.set_facecolors(colors)
    shade = plt.cm.magma_r(mm)
    m_slider.poly.set_facecolor(shade)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(shade)
            spine.set_linewidth(1 + 7 * mm)
            spine.set_alpha(mm)
    fig.canvas.draw_idle()


def _on_gray(_label):
    show = gray_toggle.get_status()[0]
    for _colored, gray, _E, _y, _viable in panels:
        if gray is not None:
            gray.set_visible(show)
    fig.canvas.draw_idle()


m_slider.on_changed(_on_m)
gray_toggle.on_clicked(_on_gray)
plt.show()
