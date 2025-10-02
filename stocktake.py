import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

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
plt.show()

# Estimate CCGT energy balances from NZ Teesside data ... a NEW BUILD!?
# https://www.ten.com/sites/energies/files/2025-05/net-zero-teesside-case-study.pdf
# https://www.gevernova.com/gas-power/resources/case-studies/net-zero-teesside 
# https://iopscience.iop.org/article/10.1088/2515-7620/ad8f99/meta is gas-fired at 840 MW??? Not 740? Or 860 MW capacity?
# https://www.power-technology.com/projects/net-zero-teesside-nzt-project-uk/?cf-view
Pfinal = 742  # MW
mcaptured = 2*10**6 # tCO2/yr which is 95 % of fuel input
mCO2 = mcaptured/0.95 # tCO2/yr
mcarbon = mCO2/44*12 # tC/yr from CH4
cmolar = 12 # kg/kmol
molar_carbon = mcarbon*1000/cmolar # kmol/yr
ch4molar = 16 # kg/kmol
mCH4 = molar_carbon*ch4molar/1000 # tCH4/yr

LHVCH4 = 50 # MJ/kg
QCH4 = mCH4*1000*LHVCH4 # MJ/yr
QCH4 = QCH4/3600 # MWh/yr
# Calculate the FLH from the 1.3Million homes energy delivery???