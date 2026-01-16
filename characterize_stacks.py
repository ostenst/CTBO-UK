import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# Initialize a clean dataframe
stacks_clean = pd.DataFrame(columns=['sector','site','stack','ktCO2','xCO2','energy_strategy','latitude','longitude','hub','km_hub'])

# Read all data into dataframes - NOTE MISSING DRAX
point_sources = pd.read_csv('data/point_sources_NAEI_2022.csv')
power_capacities = pd.read_csv('data/power_capacities_clean.csv')
refinery_stacks = pd.read_csv('data/refinery_stacks.csv')
transport_hubs = pd.read_csv('data/transport_hubs.csv')
waste_incinerators = pd.read_csv('data/waste_incinerators.csv')

# Prepare mapping data and functions
_transformer_osgb_to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
def osgb36_to_wgs84(northing_values, easting_values, debug=False):
    """Convert arrays of OSGB36 northings and eastings to WGS84 lat/lon."""
    longitudes, latitudes = _transformer_osgb_to_wgs84.transform(easting_values, northing_values)
    if debug:
        print("osgb36_to_wgs84 input northings", northing_values[:3])
        print("osgb36_to_wgs84 output latitudes", latitudes[:3])
        print("osgb36_to_wgs84 output longitudes", longitudes[:3])
    return latitudes, longitudes
def assign_hub(latitude, longitude, transport_hubs, debug=False):
    """Return the closest transport hub and its distance for the given coordinates."""
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    hub_lats = np.radians(transport_hubs['latitude'].to_numpy())
    hub_lons = np.radians(transport_hubs['longitude'].to_numpy())
    dlat = hub_lats - lat_rad
    dlon = hub_lons - lon_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(hub_lats) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = 6371.0 * c
    closest_idx = np.argmin(distances)
    closest_hub = transport_hubs.iloc[closest_idx]['hub_name']
    km_distance = distances[closest_idx]
    if debug:
        print(f"assign_transport_hub({latitude}, {longitude}) -> ({closest_hub}, {km_distance:.1f})")
    return closest_hub, km_distance
latitudes, longitudes = osgb36_to_wgs84(point_sources['Northing'].to_numpy(), point_sources['Easting'].to_numpy())
point_sources['Latitude'] = latitudes
point_sources['Longitude'] = longitudes

# Omit decomissioned fossil point sources
point_sources['ktCO2'] = point_sources['Emission'] * 3.66 / 1000
point_sources = point_sources[point_sources['Site'] != "Elgin PUQ"] 
point_sources = point_sources[point_sources['Site'] != "Lindsey Oil Refinery"] 
point_sources = point_sources[point_sources['Site'] != "Ratcliffe on Soar Power Station"]
point_sources = point_sources[~point_sources['Site'].str.startswith('Grangemouth')]
point_sources = point_sources[~point_sources['Site'].str.startswith('Port Talbot')]
fawley_mask = point_sources['Site'] == 'Fawley Refinery'
fawley_total_co2 = point_sources.loc[fawley_mask, 'ktCO2'].sum()
first_fawley_idx = point_sources[fawley_mask].index[0]
point_sources = point_sources[~fawley_mask | (point_sources.index == first_fawley_idx)].copy()
point_sources.loc[first_fawley_idx, 'ktCO2'] = fawley_total_co2
print(point_sources.shape)
print("Total CO2 emissions from point sources: ", point_sources['ktCO2'].sum())

# Clean steel stacks
steel_stack_mask = point_sources['Site'].str.startswith(
    ('Scunthorpe Power Station', 'Scunthorpe Blast Furnaces', 'Scunthorpe Sinter')
)
steel_stacks = point_sources[steel_stack_mask]
for stack_name, stack_data in steel_stacks.groupby('Site'):
    sector = 'steel'
    site = stack_name
    ktCO2 = stack_data['ktCO2'].sum()
    energy_strategy = 'natural_gas'
    latitude = stack_data['Latitude'].iloc[0]
    longitude = stack_data['Longitude'].iloc[0]
    hub, km_hub = assign_hub(latitude, longitude, transport_hubs)
    if stack_name == 'Scunthorpe Power Station':
        stack = stack_name+'power'
        xCO2 = 0.23
    elif stack_name == 'Scunthorpe Blast Furnaces':
        stack = stack_name+'blast'
        xCO2 = 0.21
    elif stack_name == 'Scunthorpe Sinter':
        stack = stack_name+'sinter'
        xCO2 = 0.08
    stacks_clean.loc[len(stacks_clean)] = [sector, site, stack, ktCO2, xCO2, energy_strategy, latitude, longitude, hub, km_hub]

# Clean refinery stacks. Eastham looks unreasonable, so hard-code it to 1/3 of Stanlow. Exclude refineries below 100 ktCO2.
refinery_stacks = point_sources[point_sources['Sector'] == 'Processing & distribution of petroleum products']
stanlow_kt = refinery_stacks.loc[refinery_stacks['Site'] == 'Stanlow Refinery','ktCO2'].iat[0]
refinery_stacks.loc[refinery_stacks['Site'] == 'Eastham', 'ktCO2'] = stanlow_kt / 3
refinery_stacks = refinery_stacks[refinery_stacks['ktCO2'] > 100]
for stack_name, stack_data in refinery_stacks.groupby('Site'):
    print(stack_name, stack_data['ktCO2'])
