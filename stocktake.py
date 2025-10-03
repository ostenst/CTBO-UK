import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

def map_all(europe, point_sources_gdf, debug=False):
    """
    Create a map showing UK point source CO2 emissions with Europe background.
    
    Args:
        europe: GeoDataFrame containing Europe shapefile data
        point_sources_gdf: GeoDataFrame containing point source emissions data
        debug: If True, print function inputs and outputs
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if debug:
        print(f"map_all inputs: europe shape={europe.shape}, point_sources shape={point_sources_gdf.shape}")
    
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
    
    if debug:
        print(f"map_all output: Created figure with {len(point_sources_gdf)} point sources")
    
    return fig, ax

def map_selected(europe, selected_plants_gdf, debug=False):
    """
    Create a map showing selected CCSA CCS/CCUS projects with Europe background.
    
    Args:
        europe: GeoDataFrame containing Europe shapefile data
        selected_plants_gdf: GeoDataFrame containing selected CCS/CCUS projects data
        debug: If True, print function inputs and outputs
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if debug:
        print(f"map_selected inputs: europe shape={europe.shape}, selected_plants shape={selected_plants_gdf.shape}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12*0.80, 15*0.80))
    ax.set_aspect(1.90)

    # Plot Europe background
    europe.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.3)

    # Plot selected plants as bubbles (size based on CO2 emissions)
    # Convert MtCO2/yr to tonnes for consistency with scaling
    scatter = ax.scatter(
        selected_plants_gdf.geometry.x, 
        selected_plants_gdf.geometry.y,
        s=selected_plants_gdf['CO2']*1000000/2000,  # Convert Mt to tonnes and scale down
        c=selected_plants_gdf['CO2'],
        cmap='Blues',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('CO2 Capture Capacity (MtCO2/yr)', fontsize=12)

    # Set title and labels
    ax.set_title('Selected UK CCS/CCUS Projects (2025)\nBubble size proportional to CO2 capture capacity', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Set UK-focused view
    ax.set_xlim(-9, 3)
    ax.set_ylim(49, 62.5)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if debug:
        print(f"map_selected output: Created figure with {len(selected_plants_gdf)} selected projects")
    
    return fig, ax

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

# Print the 50 largest emitters
print(f"\nTop 50 largest CO2 emitters (2022):")
top_50_emitters = point_sources.nlargest(50, 'CO2')[['PlantID', 'Site', 'Operator', 'Sector', 'CO2']]
for i, (idx, row) in enumerate(top_50_emitters.iterrows(), 1):
    print(f"{i:2d}. {row['Site']:<25} | {row['Operator']:<35} | {row['Sector']:<30} | {row['CO2']:>12,.0f} tonnes")
print(f"Sum of top 50 emitters: {top_50_emitters['CO2'].sum():,.0f} tonnes")

# Remove emitters with less than 100 ktCO2/yr
point_sources = point_sources[point_sources['CO2'] >= 100000]
total_co2_emissions = point_sources['CO2'].sum()
print(f"\nTotal fossil CO2 emissions from point sources dataset (2022) @>100 ktCO2/yr: {total_co2_emissions:,.2f} tonnes")

# Map point sources
europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

point_sources_gdf = gpd.GeoDataFrame(
    point_sources, 
    geometry=gpd.points_from_xy(point_sources['Easting'], point_sources['Northing'], crs="EPSG:27700")
).to_crs("EPSG:4326")

selected_plants = pd.read_csv("data/ccsa_plants_selected.csv")
selected_plants_gdf = gpd.GeoDataFrame(
    selected_plants, 
    geometry=gpd.points_from_xy(selected_plants['Easting'], selected_plants['Northing'], crs="EPSG:27700")
).to_crs("EPSG:4326")

# Create the map
fig1, ax1 = map_all(europe, point_sources_gdf)
fig2, ax2 = map_selected(europe, selected_plants_gdf)
# plt.show()

# Characterise refinery stacks
# https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2022.804163/full
refinery_stacks = pd.read_csv("data/refinery_stacks.csv")

power_emissions = refinery_stacks[refinery_stacks['Unit'] == 'POW']
cracker_emissions = refinery_stacks[refinery_stacks['Unit'] == 'FCC']
distillation_emissions = refinery_stacks[refinery_stacks['Unit'] == 'CDU']
smr_emissions = refinery_stacks[refinery_stacks['Unit'] == 'SMR']
total_emissions = refinery_stacks['CO2_emissions[t/hr]'].sum()
for emissions in [power_emissions, cracker_emissions, distillation_emissions, smr_emissions]:
    print(" ")
    print(emissions)
    print(f"Share of total emissions: {emissions['CO2_emissions[t/hr]'].sum()/total_emissions*100:.1f}%")
remaining_emissions = total_emissions - power_emissions['CO2_emissions[t/hr]'].sum() - cracker_emissions['CO2_emissions[t/hr]'].sum() - distillation_emissions['CO2_emissions[t/hr]'].sum() - smr_emissions['CO2_emissions[t/hr]'].sum()
print(f"\nRemaining emissions: {remaining_emissions:,.2f} t/hr")
print(f"Share of total emissions: {remaining_emissions/total_emissions*100:.1f}%")
print("=> Remaining emissions are approximately at 8% CO2 concentration and could be captured in 1 clustered stack!")

# Plot slices in a pie chart based on: [emissions, color]
power_slice = [power_emissions['CO2_emissions[t/hr]'].sum() , (3*25 + 8*54)/(25+54)]
cracker_slice = [cracker_emissions['CO2_emissions[t/hr]'].sum() , 17]
distillation_slice = [distillation_emissions['CO2_emissions[t/hr]'].sum() , 11]
smr_slice = [smr_emissions['CO2_emissions[t/hr]'].sum() , (8*6 + 24*26)/(6+26)]
remaining_slice = [remaining_emissions , 8]

# Extract emissions and color values
emissions = [power_slice[0], cracker_slice[0], distillation_slice[0], smr_slice[0], remaining_slice[0]]
color_values = [power_slice[1], cracker_slice[1], distillation_slice[1], smr_slice[1], remaining_slice[1]]
labels = ['Power', 'Cracker', 'Distillation', 'SMR', 'Remaining']

# Normalize color values to 0-1 range for magma colormap
# color_values_normalized = np.array(color_values) / max(color_values)
color_values_normalized = np.array(color_values) / 50 # Assuming a colorbar up to 50%

# Create the pie chart with colorbar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})
colors = plt.cm.magma_r(color_values_normalized)

wedges, texts, autotexts = ax1.pie(emissions, labels=labels, colors=colors, autopct='%1.1f%%', 
                                   startangle=90, textprops={'fontsize': 12})

# Enhance the appearance
ax1.set_title('Refinery CO2 Emissions by Unit Type\n(Colored by CO2 Concentration)', 
              fontsize=16, fontweight='bold', pad=20)

# Make percentage text larger
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

# Create colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.magma_r, norm=plt.Normalize(vmin=min(color_values), vmax=max(color_values)))
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax2, orientation='vertical')
cbar.set_label('CO2 Concentration (%)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Remove the second subplot axis
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.tight_layout()
# plt.show()

# Characterise steel stacks (Biermann Excess heat-steel Luleå)
# https://www.sciencedirect.com/science/article/pii/S1750583619303068?casa_token=jTmJHk38ewIAAAAA:U0RaenQ6yhZenVfptMhjr5xnHgEPwkZ0UqtgNI-vgEVVgcxjwp7r2Fe-NWor3A6FhWY4vJsLWA
# [emissions, color]
chp_slice = [59.4, 29.6]
stoves_slice = [22.2, 25.1]
flaring_slice = [10.0, 25.1] # Assumed from Biermann
remaining_slice = [(100-chp_slice[0]-stoves_slice[0]-flaring_slice[0]) , 8] # 8% is just assumed from various combustion processes/kilns

# Create steel emissions pie chart
steel_emissions = [chp_slice[0], stoves_slice[0], flaring_slice[0], remaining_slice[0]]
steel_color_values = [chp_slice[1], stoves_slice[1], flaring_slice[1], remaining_slice[1]]
steel_labels = ['CHP', 'Stoves', 'Flaring', 'Remaining']

# Normalize color values to 0-1 range for magma colormap
steel_color_values_normalized = np.array(steel_color_values) / 50 # Assuming a colorbar up to 50%

# Create the steel pie chart with colorbar
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})
steel_colors = plt.cm.magma_r(steel_color_values_normalized)

wedges, texts, autotexts = ax1.pie(steel_emissions, labels=steel_labels, colors=steel_colors, autopct='%1.1f%%', 
                                   startangle=90, textprops={'fontsize': 12})

# Enhance the appearance
ax1.set_title('Steel Plant CO2 Emissions by Unit Type\n(Colored by CO2 Concentration)', 
              fontsize=16, fontweight='bold', pad=20)

# Make percentage text larger
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

# Create colorbar
sm2 = plt.cm.ScalarMappable(cmap=plt.cm.magma_r, norm=plt.Normalize(vmin=min(steel_color_values), vmax=max(steel_color_values)))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, cax=ax2, orientation='vertical')
cbar2.set_label('CO2 Concentration (%)', fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=12)

# Remove the second subplot axis
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()

