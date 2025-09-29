import pandas as pd
import geopandas as gpd

print("This script takes stock on:")
print("1) UK point source emissions (NAEI, 2022)")
print("2) Planned CCS projects (CCSA, 2025)")
print("3) Advanced CCUS projects within Track-1, Track-2 (BEIS, DESNZ)")
print("4) The DACCS potential (Middleton, 202X)")

# Read the pre-filtered CO2 2022 dataset
# https://naei.energysecurity.gov.uk/data/maps/download-gridded-emissions
point_sources = pd.read_csv("data/point_sources_CO2_2022.csv")

# Calculate total CO2 emissions from the pre-filtered dataset
point_sources['CO2'] = point_sources['Emission'] * 3.66
total_co2_emissions = point_sources['CO2'].sum()

print(f"\nTotal CO2 emissions from point sources dataset (2022): {total_co2_emissions:,.2f} tonnes")
print("====> Note that biogenic CO2 seems to be excluded, as Drax has low CO2 emissions.")
print(f"Number of plants in dataset (2022): {len(point_sources)}")

# Print unique sectors
unique_sectors = point_sources['Sector'].unique()
print(f"Unique sectors in point_sources_CO2_2022: {len(unique_sectors)}")
print(f"Sectors: {sorted(unique_sectors)}")

# Calculate emissions from specific sectors
target_sectors = [
    "Waste collection, treatment & disposal",
    "Major power producers", 
    "Minor power producers",
    "Lime",
    "Cement",
    "Iron & steel industries"
]

print(f"\nEmissions by target sectors:")
total_target_emissions = 0

for sector in target_sectors:
    sector_data = point_sources[point_sources['Sector'] == sector]
    if len(sector_data) > 0:
        sector_emissions = sector_data['CO2'].sum()
        total_target_emissions += sector_emissions
        print(f"{sector}: {sector_emissions:,.2f} tonnes CO2 ({len(sector_data)} plants)")
    else:
        print(f"{sector}: No data found")

print(f"\nTotal emissions from target sectors: {total_target_emissions:,.2f} tonnes CO2")
print(f"Percentage of total emissions: {(total_target_emissions/total_co2_emissions)*100:.1f}%")

# Print the 10 largest emitters
print(f"\nTop 10 largest CO2 emitters (2022):")
top_10_emitters = point_sources.nlargest(10, 'CO2')[['PlantID', 'Site', 'Operator', 'Sector', 'CO2']]
for i, (idx, row) in enumerate(top_10_emitters.iterrows(), 1):
    print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tonnes")

europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

# Create a map of UK point sources with CO2 emissions as bubbles
import matplotlib.pyplot as plt

# Convert point sources to GeoDataFrame
point_sources_gdf = gpd.GeoDataFrame(
    point_sources, 
    geometry=gpd.points_from_xy(point_sources['Easting'], point_sources['Northing'], crs="EPSG:27700")
).to_crs("EPSG:4326")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(12*0.80, 15*0.80))
ax.set_aspect(1.90)

# Plot Europe background
europe.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.3)

# Plot point sources as bubbles (size based on CO2 emissions)
scatter = ax.scatter(
    point_sources_gdf.geometry.x, 
    point_sources_gdf.geometry.y,
    s=point_sources_gdf['CO2']/2000,  # Scale down for visibility
    c=point_sources_gdf['CO2'],
    cmap='Reds',
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('CO2 Emissions (tonnes)', fontsize=12)

# Set title and labels
ax.set_title('UK Point Source CO2 Emissions (2022)\nBubble size proportional to emissions', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Set UK-focused view
ax.set_xlim(-9, 3)
ax.set_ylim(49, 62.5)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Set x axis aspect ratio to 1.90