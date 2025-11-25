import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

def assign_co2_concentration(row, debug=False):
    """
    Assign CO2 concentration based on sector.
    
    Args:
        row: DataFrame row with plant data
        debug: If True, print function inputs and outputs
    
    Returns:
        float: CO2 concentration as fraction (0-1)
    """
    sector = row['Sector']
    site = row['Site']
    
    # Default concentration
    xCO2 = 0.10  # Default 10%
    
    # Power sector
    if 'power producer' in sector.lower():
        xCO2 = 0.05  # CCGT plants
    
    # Refineries
    elif 'petroleum' in sector.lower():
        # Use average of refinery stacks
        xCO2 = 0.15  # Approximate average
    
    # Iron & steel
    elif 'iron' in sector.lower() or 'steel' in sector.lower():
        if 'power' in site.lower():
            xCO2 = 0.296  # CHP
        elif 'blast' in site.lower():
            xCO2 = 0.251  # Stove
        elif 'sinter' in site.lower():
            xCO2 = 0.15  # Sinter
        else:
            xCO2 = 0.23  # Average
    
    # Cement
    elif 'cement' in sector.lower():
        xCO2 = 0.20
    
    # Waste to energy
    elif 'waste' in sector.lower():
        xCO2 = 0.11
    
    if debug:
        print(f"{site}: {sector} -> {xCO2*100:.1f}% CO2")
    
    return xCO2

def create_all_plants_map(europe, plants_gdf, debug=False):
    """
    Create a map showing all plants colored by CO2 concentration.
    
    Args:
        europe: GeoDataFrame containing Europe shapefile data
        plants_gdf: GeoDataFrame containing plant data with locations
        debug: If True, print function inputs and outputs
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if debug:
        print(f"create_all_plants_map inputs: europe shape={europe.shape}, plants shape={plants_gdf.shape}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    ax.set_aspect(1.90)

    # Plot Europe background
    europe.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.45)

    # Plot plants as bubbles (size based on CO2 emissions, color based on concentration)
    scatter = ax.scatter(
        plants_gdf.geometry.x, 
        plants_gdf.geometry.y,
        s=plants_gdf['CO2']/750,  # Scale for visibility (CO2 in tonnes, divide by 1000)
        c=plants_gdf['xCO2']*100,  # Convert to percentage for colorbar
        cmap='magma',
        vmin=0,
        vmax=30,  # Max 30% CO2
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('CO₂ Concentration (%)', fontsize=13)

    # Set title and labels
    ax.set_title('UK Point Source CO₂ Emissions (2022)\nBubble size proportional to annual CO₂ emissions', 
                 fontsize=15, fontweight='bold')

    # Set UK-focused view
    ax.set_xlim(-9, 3)
    ax.set_ylim(49.5, 59.5)

    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend for sectors
    from matplotlib.patches import Patch
    sectors = plants_gdf['Sector'].unique()
    
    # Add statistics box
    total_co2 = plants_gdf['CO2'].sum() / 1e6  # Convert to MtCO2
    n_plants = len(plants_gdf)
    avg_conc = plants_gdf['xCO2'].mean() * 100
    
    stats_text = f'Total plants: {n_plants}\n'
    stats_text += f'Total CO₂: {total_co2:.1f} MtCO₂/yr\n'
    stats_text += f'Avg. concentration: {avg_conc:.1f}%'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
            fontsize=11)

    plt.tight_layout()
    
    if debug:
        print(f"create_all_plants_map output: Created figure with {len(plants_gdf)} plants")
    
    return fig, ax

# Read the point sources data
print("Reading point sources data...")
fossil_plants = pd.read_csv("data/point_sources_CO2_2022.csv")
fossil_plants['CO2'] = fossil_plants['Emission'] * 3.66  # Convert C to CO2
fossil_plants = fossil_plants.nlargest(60, 'CO2')  # Keep only 60 largest plants

print(f"Loaded {len(fossil_plants)} point sources")
print(f"Total CO2: {fossil_plants['CO2'].sum()/1e6:.1f} MtCO2/yr")

# Assign CO2 concentrations to each plant
print("\nAssigning CO2 concentrations based on sector...")
fossil_plants['xCO2'] = fossil_plants.apply(assign_co2_concentration, axis=1, debug=False)

# Display concentration statistics by sector
print("\nCO2 concentration by sector:")
sector_stats = fossil_plants.groupby('Sector').agg({
    'xCO2': 'mean',
    'CO2': ['count', 'sum']
}).round(3)
print(sector_stats)

# Remove plants without valid coordinates
valid_plants = fossil_plants[
    fossil_plants['Easting'].notna() & 
    fossil_plants['Northing'].notna()
].copy()

print(f"\nPlants with valid coordinates: {len(valid_plants)}")

# Load Europe shapefile
print("\nLoading Europe shapefile...")
europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

# Create GeoDataFrame for plants
print("\nCreating GeoDataFrame...")
plants_gdf = gpd.GeoDataFrame(
    valid_plants, 
    geometry=gpd.points_from_xy(valid_plants['Easting'], valid_plants['Northing'], crs="EPSG:27700")
).to_crs("EPSG:4326")

# Create the map
print("\nCreating map...")
fig, ax = create_all_plants_map(europe, plants_gdf, debug=True)

# Save the figure
output_file = 'map_all_plants.png'
plt.savefig(output_file, dpi=400, bbox_inches='tight')
print(f"\nMap saved as '{output_file}'")

# Display top emitters
print("\nTop 20 CO2 emitters:")
top_plants = valid_plants.nlargest(20, 'CO2')[['Site', 'Sector', 'CO2', 'xCO2']]
for i, (idx, plant) in enumerate(top_plants.iterrows(), 1):
    print(f"{i:2d}. {plant['Site']:<35} | {plant['Sector']:<35} | {plant['CO2']/1000:>8.1f} ktCO2/yr | {plant['xCO2']*100:>5.1f}%")

plt.show()

