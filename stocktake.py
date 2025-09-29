import pandas as pd
import geopandas as gpd

print("This script takes stock on:")
print("1) UK point source emissions (NAEI, 2022)")
print("2) Planned CCS projects (CCSA, 2025)")
print("3) Advanced CCUS projects within Track-1, Track-2 (BEIS, DESNZ)")
print("4) The DACCS potential (Middleton, 202X)")

point_sources = pd.read_csv("data/NAEIPointsSources_2022_3.csv")

def get_current_emitters(point_sources, debug=False):
    """
    Filter current emitters based on 2022 Carbon Dioxide as Carbon emissions.
    
    Args:
        point_sources: DataFrame containing emission data
        debug: If True, print debug information
    
    Returns:
        Array of unique PlantIDs that emitted CO2 in 2022
    """
    if debug:
        print(f"Input data shape: {point_sources.shape}")
        print(f"Available years: {sorted(point_sources['Year'].unique())}")
        print(f"Available pollutants: {point_sources['Pollutant_Name'].unique()}")
    
    # Filter for 2022 Carbon Dioxide as Carbon emissions
    filtered_data = point_sources[
        (point_sources['Year'] == 2022) & 
        (point_sources['Pollutant_Name'] == 'Carbon Dioxide as Carbon')
    ]
    
    if debug:
        print(f"Filtered data shape: {filtered_data.shape}")
        print(f"Sample of filtered data:\n{filtered_data[['PlantID', 'Site', ' Emission ']].head()}")
    
    current_emitters = filtered_data['PlantID'].unique()
    
    if debug:
        print(f"Unique PlantIDs: {len(current_emitters)}")
        print(f"Sample PlantIDs: {current_emitters[:10]}")
    
    return current_emitters

current_emitters = get_current_emitters(point_sources, debug=False)
print(f"Found {len(current_emitters)} current emitters (plants with CO2 emissions in 2022)")

# Calculate total CO2 emissions from current emitters
current_emitters_data = point_sources[
    (point_sources['Year'] == 2022) & 
    (point_sources['Pollutant_Name'] == 'Carbon Dioxide as Carbon') &
    (point_sources['PlantID'].isin(current_emitters))
]

# Convert emission values to numeric (removing commas and spaces)
emission_values = current_emitters_data[' Emission '].str.replace(',', '').str.replace(' ', '').astype(float)
total_co2_emissions = emission_values.sum()

print(f"Total CO2 emissions from current emitters: {total_co2_emissions:,.2f} tonnes")

# Read the pre-filtered CO2 2022 dataset
point_sources_co2_2022 = pd.read_csv("data/point_sources_CO2_2022.csv")

# Calculate total CO2 emissions from the pre-filtered dataset
total_co2_emissions_2022 = point_sources_co2_2022['Emission'].sum()

print(f"Total CO2 emissions from point_sources_CO2_2022 dataset: {total_co2_emissions_2022:,.2f} tonnes")
print(f"Number of plants in CO2 2022 dataset: {len(point_sources_co2_2022)}")

europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

#Set x axis aspect ratio to 1.90