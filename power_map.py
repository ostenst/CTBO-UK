import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

def create_power_map(europe, power_plants_gdf, cost_column='Total_FOAK', title_suffix='FOAK', vmin=None, vmax=None, median_plant=None, debug=False):
    """
    Create a map showing power plants from cost analysis with Europe background.
    
    Args:
        europe: GeoDataFrame containing Europe shapefile data
        power_plants_gdf: GeoDataFrame containing power plant data with locations
        cost_column: Column name for cost data (Total_FOAK or Total_NOAK)
        title_suffix: Suffix for the title (FOAK or NOAK)
        vmin: Minimum value for colormap normalization
        vmax: Maximum value for colormap normalization
        median_plant: Dictionary with median plant info {'plant_name': str, 'cost': float, 'index': int}
        debug: If True, print function inputs and outputs
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    if debug:
        print(f"create_power_map inputs: europe shape={europe.shape}, power_plants shape={power_plants_gdf.shape}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12*0.80, 15*0.80))
    ax.set_aspect(1.90)

    # Plot Europe background
    europe.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.45)

    # Plot power plants as bubbles (size based on capacity)
    scatter = ax.scatter(
        power_plants_gdf.geometry.x, 
        power_plants_gdf.geometry.y,
        s=power_plants_gdf['Capacity_MW']*2,  # Scale for visibility
        c=power_plants_gdf[cost_column],
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Highlight median plant if provided
    if median_plant is not None:
        median_idx = median_plant['index']
        # Use geometry coordinates (already converted to WGS84 for map display)
        median_x = power_plants_gdf.iloc[median_idx].geometry.x
        median_y = power_plants_gdf.iloc[median_idx].geometry.y
        median_capacity = power_plants_gdf.iloc[median_idx]['Capacity_MW']
        
        # Add a large circle outline around the median plant
        ax.scatter(median_x, median_y, 
                  s=median_capacity*2 + 2000,  # Larger than the plant bubble
                  facecolors='none', 
                  edgecolors='red', 
                  linewidth=3,
                  alpha=0.8)
        
        # Add annotation with median value
        ax.annotate(f'Median: {median_plant["plant_name"]}\n{median_plant["cost"]:.1f} EUR/tCO2', 
                   xy=(median_x, median_y), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='navajowhite', alpha=0.7),
                   fontsize=10,
                   fontweight='bold',
                   ha='left')
        
        if debug:
            print(f"Highlighted median plant: {median_plant['plant_name']} at ({median_x:.2f}, {median_y:.2f})")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(f'Total {title_suffix} Cost (EUR/tCO2)', fontsize=12)

    # Set title and labels
    title_text = f'UK Power Plants - CCS Cost Analysis ({title_suffix})\nBubble size proportional to capacity (MW)'
    if median_plant is not None:
        title_text += f'\nMedian: {median_plant["plant_name"]} ({median_plant["cost"]:.1f} EUR/tCO2)'
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Set UK-focused view
    ax.set_xlim(-9, 3)
    ax.set_ylim(49, 62.5)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if debug:
        print(f"create_power_map output: Created figure with {len(power_plants_gdf)} power plants")
    
    return fig, ax

def match_plants_with_locations(cost_data, point_sources_data, debug=False):
    """
    Match power plants from cost analysis with their locations from point sources data.
    
    Args:
        cost_data: DataFrame with cost analysis results
        point_sources_data: DataFrame with point source locations
        debug: If True, print function inputs and outputs
    
    Returns:
        DataFrame: Merged data with plant locations
    """
    if debug:
        print(f"match_plants_with_locations inputs: cost_data shape={cost_data.shape}, point_sources shape={point_sources_data.shape}")
    
    # Create a copy of cost data
    matched_plants = cost_data.copy()
    
    # Initialize location columns
    matched_plants['Easting'] = np.nan
    matched_plants['Northing'] = np.nan
    matched_plants['CO2_emissions'] = np.nan
    
    # Try to match plants by name
    matched_count = 0
    for idx, plant in matched_plants.iterrows():
        plant_name = plant['Plant']
        
        # Try different matching strategies
        matches = []
        
        # Exact match
        exact_match = point_sources_data[point_sources_data['Site'].str.contains(plant_name, case=False, na=False)]
        if len(exact_match) > 0:
            matches = exact_match
        else:
            # Try partial matching with key words
            key_words = plant_name.split()
            for word in key_words:
                if len(word) > 3:  # Only use words longer than 3 characters
                    partial_match = point_sources_data[point_sources_data['Site'].str.contains(word, case=False, na=False)]
                    if len(partial_match) > 0:
                        matches = partial_match
                        break
        
        if len(matches) > 0:
            # Take the first match (or the one with highest emissions if multiple)
            best_match = matches.loc[matches['Emission'].idxmax()]
            matched_plants.loc[idx, 'Easting'] = best_match['Easting']
            matched_plants.loc[idx, 'Northing'] = best_match['Northing']
            matched_plants.loc[idx, 'CO2_emissions'] = best_match['Emission'] * 3.66  # Convert to CO2
            matched_count += 1
            
            if debug:
                print(f"Matched: {plant_name} -> {best_match['Site']}")
    
    # Remove plants without location data
    matched_plants = matched_plants.dropna(subset=['Easting', 'Northing'])
    
    if debug:
        print(f"match_plants_with_locations output: Matched {matched_count} out of {len(cost_data)} plants")
        print(f"Final matched plants: {len(matched_plants)}")
    
    return matched_plants

# Read the cost analysis results
print("Reading CCS cost analysis results...")
cost_data = pd.read_csv("ccs_cost_analysis_results.csv")

print(f"Loaded {len(cost_data)} power plants from cost analysis")
print(f"Columns: {list(cost_data.columns)}")

# Read the point sources data
print("\nReading point sources data...")
point_sources = pd.read_csv("data/point_sources_CO2_2022.csv")

print(f"Loaded {len(point_sources)} point sources")
print(f"Unique sites: {point_sources['Site'].nunique()}")

# Match plants with locations
print("\nMatching plants with locations...")
matched_plants = match_plants_with_locations(cost_data, point_sources, debug=True)

print(f"\nSuccessfully matched {len(matched_plants)} plants with locations")

# Display matched plants
print("\nMatched plants:")
for idx, plant in matched_plants.iterrows():
    print(f"{plant['Plant']:<30} | {plant['Capacity_MW']:>6.0f} MW | {plant['Total_FOAK']:>8.1f} EUR/tCO2")

# Load Europe shapefile
print("\nLoading Europe shapefile...")
europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

# Create GeoDataFrame for matched plants
power_plants_gdf = gpd.GeoDataFrame(
    matched_plants, 
    geometry=gpd.points_from_xy(matched_plants['Easting'], matched_plants['Northing'], crs="EPSG:27700")
).to_crs("EPSG:4326")

# Calculate unified colormap range from both FOAK and NOAK costs
foak_min, foak_max = power_plants_gdf['Total_FOAK'].min(), power_plants_gdf['Total_FOAK'].max()
noak_min, noak_max = power_plants_gdf['Total_NOAK'].min(), power_plants_gdf['Total_NOAK'].max()

# Use the overall min and max from both datasets
unified_vmin = min(foak_min, noak_min)
unified_vmax = max(foak_max, noak_max)

print(f"\nColormap range:")
print(f"FOAK costs: {foak_min:.1f} - {foak_max:.1f} EUR/tCO2")
print(f"NOAK costs: {noak_min:.1f} - {noak_max:.1f} EUR/tCO2")
print(f"Unified range: {unified_vmin:.1f} - {unified_vmax:.1f} EUR/tCO2")

# Identify median plants for each cost type
print("\nIdentifying median plants...")

# Sort by FOAK costs to find median
foak_sorted = power_plants_gdf.sort_values('Total_FOAK')
foak_median_idx = len(foak_sorted) // 2
foak_median_plant = {
    'plant_name': foak_sorted.iloc[foak_median_idx]['Plant'],
    'cost': foak_sorted.iloc[foak_median_idx]['Total_FOAK'],
    'index': foak_sorted.index[foak_median_idx]
}

# Sort by NOAK costs to find median
noak_sorted = power_plants_gdf.sort_values('Total_NOAK')
noak_median_idx = len(noak_sorted) // 2
noak_median_plant = {
    'plant_name': noak_sorted.iloc[noak_median_idx]['Plant'],
    'cost': noak_sorted.iloc[noak_median_idx]['Total_NOAK'],
    'index': noak_sorted.index[noak_median_idx]
}

print(f"FOAK median plant: {foak_median_plant['plant_name']} ({foak_median_plant['cost']:.1f} EUR/tCO2)")
print(f"NOAK median plant: {noak_median_plant['plant_name']} ({noak_median_plant['cost']:.1f} EUR/tCO2)")

# Create the FOAK cost map
print("\nCreating FOAK cost map...")
fig1, ax1 = create_power_map(europe, power_plants_gdf, cost_column='Total_FOAK', title_suffix='FOAK', 
                             vmin=unified_vmin, vmax=unified_vmax, median_plant=foak_median_plant, debug=True)

# Save the FOAK figure
plt.savefig('power_plants_map_FOAK.png', dpi=400, bbox_inches='tight')
print("FOAK map saved as 'power_plants_map_FOAK.png'")

# Create the NOAK cost map
print("\nCreating NOAK cost map...")
fig2, ax2 = create_power_map(europe, power_plants_gdf, cost_column='Total_NOAK', title_suffix='NOAK', 
                             vmin=unified_vmin, vmax=unified_vmax, median_plant=noak_median_plant, debug=True)

# Save the NOAK figure
plt.savefig('power_plants_map_NOAK.png', dpi=400, bbox_inches='tight')
print("NOAK map saved as 'power_plants_map_NOAK.png'")

plt.show()
