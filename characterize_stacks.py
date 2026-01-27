import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# Initialize a clean dataframe
stacks_clean = pd.DataFrame(columns=[
    'sector', 'site', 'stack', 'ktCO2', 'xCO2', 'energy_strategy',
    'latitude', 'longitude', 'hub', 'km_hub', 'land_transport', 'sea_transport'
])

# Read all data into dataframes
point_sources = pd.read_csv('data/point_sources_NAEI_2022.csv')
power_capacities = pd.read_csv('data/power_capacities_clean.csv')
refinery_example = pd.read_csv('data/refinery_emissions.csv')
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
def assign_hub(latitude, longitude, transport_hubs, pipeline_threshold=50, debug=False):
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

    if km_distance < pipeline_threshold:
        land_transport = 'pipeline'
    else:
        land_transport = 'truck'
    if closest_hub in ('Pembroke', 'Bristol'):
        sea_transport = 'ship to Hynet'
    elif closest_hub == 'London':
        sea_transport = 'ship to Bacton'
    else:
        sea_transport = np.nan
    
    if debug:
        print(f"assign_transport_hub({latitude}, {longitude}) -> ({closest_hub}, {km_distance:.1f})")
    return closest_hub, km_distance, land_transport, sea_transport
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

# Clean steel stacks
steel_stack_mask = point_sources['Site'].str.startswith(
    ('Scunthorpe Power Station', 'Scunthorpe Blast Furnaces', 'Scunthorpe Sinter')
)
steel_stacks = point_sources[steel_stack_mask]
for stack_name, stack_data in steel_stacks.groupby('Site'):
    sector = 'steel'
    site = stack_name
    ktCO2 = stack_data['ktCO2'].sum()
    energy_strategy = 'Class III-NG'
    latitude = stack_data['Latitude'].iloc[0]
    longitude = stack_data['Longitude'].iloc[0]
    hub, km_hub, land_transport, sea_transport = assign_hub(latitude, longitude, transport_hubs)
    if stack_name == 'Scunthorpe Power Station':
        stack = stack_name+'-power'
        xCO2 = 0.23
    elif stack_name == 'Scunthorpe Blast Furnaces':
        stack = stack_name+'-blast'
        xCO2 = 0.21
    elif stack_name == 'Scunthorpe Sinter':
        stack = stack_name+'-sinter'
        xCO2 = 0.08
    stacks_clean.loc[len(stacks_clean)] = [sector, site, stack, ktCO2, xCO2, energy_strategy, latitude, longitude, hub, km_hub, land_transport, sea_transport]

# Clean refinery stacks. Calculate illustrative CO2 fractions per stream, then adjust Eastham to 1/3 of Stanlow Refinery. Exclude refineries below 100 ktCO2.
refinery_example_total = refinery_example['CO2_emissions[t/hr]'].sum()
xCO2_dict = {
    "HPU": 0.24,
    "FCC": 0.17,
    "distillation": 0.11,
    "scattered": 0.08,
    "power3%": 0.03,
}
mCO2_dict = {}
for stack_type, xCO2 in xCO2_dict.items():
    mCO2 = refinery_example.loc[refinery_example['yCO2[%vol]'] == xCO2 * 100, 'CO2_emissions[t/hr]'].sum()
    mCO2_dict[stack_type] = mCO2 / refinery_example_total

refinery_sites = point_sources[point_sources['Sector'] == 'Processing & distribution of petroleum products']
stanlow_kt = refinery_sites.loc[refinery_sites['Site'] == 'Stanlow Refinery','ktCO2'].iat[0]
refinery_sites.loc[refinery_sites['Site'] == 'Eastham', 'ktCO2'] = stanlow_kt / 3
refinery_sites = refinery_sites[refinery_sites['ktCO2'] > 100]
for site_name, site_data in refinery_sites.groupby('Site'):
    sector = 'refinery'
    site = site_name
    latitude = site_data['Latitude'].iloc[0]
    longitude = site_data['Longitude'].iloc[0]
    hub, km_hub, land_transport, sea_transport = assign_hub(latitude, longitude, transport_hubs)

    for stack_type, mCO2_fraction in mCO2_dict.items():
        stack = site_name+'-'+stack_type
        ktCO2 = site_data['ktCO2'].sum() * mCO2_fraction
        xCO2 = xCO2_dict[stack_type]
        if stack_type == 'HPU':
            energy_strategy = 'Class I-HCN'
        elif stack_type == 'FCC':
            energy_strategy = 'Class I-HRSG'
        elif stack_type == 'distillation':
            energy_strategy = 'Class III-NG'
        elif stack_type == 'scattered':
            energy_strategy = 'Class III-NG'
        elif stack_type == 'power3%':
            energy_strategy = 'None'
        stacks_clean.loc[len(stacks_clean)] = [sector, site, stack, ktCO2, xCO2, energy_strategy, latitude, longitude, hub, km_hub, land_transport, sea_transport]

# Clean cement stacks above 100 ktCO2
cement_stacks = point_sources[point_sources['Sector'] == 'Cement']
cement_stacks = cement_stacks[cement_stacks['ktCO2'] > 100]
cement_stacks = cement_stacks[cement_stacks['Site'] != 'Cookstown']
for site_name, site_data in cement_stacks.groupby('Site'):
    sector = 'cement'
    site = site_name
    stack = site_name+'-cement'
    ktCO2 = site_data['ktCO2'].sum()
    xCO2 = 0.22
    energy_strategy = 'Biomass'
    latitude = site_data['Latitude'].iloc[0]
    longitude = site_data['Longitude'].iloc[0]
    hub, km_hub, land_transport, sea_transport = assign_hub(latitude, longitude, transport_hubs)
    stacks_clean.loc[len(stacks_clean)] = [sector, site, stack, ktCO2, xCO2, energy_strategy, latitude, longitude, hub, km_hub, land_transport, sea_transport]

# Clean Drax and waste incinerator stacks above 100 ktCO2 (emissions from drax.com, xCO2 from pellet calculation)
latitude = 53.738710
longitude = -0.993030
hub, km_hub, land_transport, sea_transport = assign_hub(latitude, longitude, transport_hubs)
stacks_clean.loc[len(stacks_clean)] = ['drax', 'Drax Power Station', 'Drax-power', 11500, 0.14, 'Drax', latitude, longitude, hub, km_hub, land_transport, sea_transport]

emission_factor_waste = 0.98 # [tCO2/t] Tolvik report
waste_incinerators['ktCO2'] = waste_incinerators['Capacity 2023 [ktpa]'] * emission_factor_waste 
waste_incinerators = waste_incinerators[waste_incinerators['ktCO2'] > 100]
for site_name, site_data in waste_incinerators.groupby('Name'):
    sector = 'waste'
    site = site_name
    stack = site_name+'-waste'
    ktCO2 = site_data['ktCO2'].sum()
    xCO2 = 0.12
    energy_strategy = 'Waste-HCN'
    latitude = site_data['Latitude'].iloc[0]
    longitude = site_data['Longitude'].iloc[0]
    hub, km_hub, land_transport, sea_transport = assign_hub(latitude, longitude, transport_hubs)
    stacks_clean.loc[len(stacks_clean)] = [sector, site, stack, ktCO2, xCO2, energy_strategy, latitude, longitude, hub, km_hub, land_transport, sea_transport]

# Clean gas turbine CCGT stacks above 100 ktCO2. Also add capacity [MW] from power_capacities dataframe.
mask = point_sources['Sector'].isin(['Major power producers', 'Minor power producers'])
ccgt_stacks = point_sources[mask & (point_sources['ktCO2'] > 100)]
ccgt_stacks = ccgt_stacks[ccgt_stacks['Site'] != 'Drax']
ccgt_stacks = ccgt_stacks[ccgt_stacks['Site'] != 'Ferrybridge MF1']
ccgt_stacks = ccgt_stacks[ccgt_stacks['Site'] != 'West Burton']

for site_name, site_data in ccgt_stacks.groupby('Site'):
    sector = 'ccgt'
    site = site_name
    stack = site_name+'-ccgt'
    ktCO2 = site_data['ktCO2'].sum()
    xCO2 = 0.04
    energy_strategy = 'LP steam'
    latitude = site_data['Latitude'].iloc[0]
    longitude = site_data['Longitude'].iloc[0]
    hub, km_hub, land_transport, sea_transport = assign_hub(latitude, longitude, transport_hubs)
    stacks_clean.loc[len(stacks_clean)] = [sector, site, stack, ktCO2, xCO2, energy_strategy, latitude, longitude, hub, km_hub, land_transport, sea_transport]

capacity_map = dict(
    zip(power_capacities['Power plant name'].str.strip(), power_capacities['Capacity [MW]'])
)
def _stack_capacity(row):
    if row['sector'] != 'ccgt':
        return np.nan
    return capacity_map.get(row['site'].strip(), np.nan)
stacks_clean['ccgt_capacity'] = stacks_clean.apply(_stack_capacity, axis=1)

# Print CO2 emissions and plants with km_hub>200km
total_ktCO2 = stacks_clean['ktCO2'].sum()
print("Total CO2 emissions from all cleaned stacks: ", total_ktCO2," ... Below are stacks with emissions above 800 ktCO2:")
print(stacks_clean[stacks_clean['ktCO2'] > 800])
print(stacks_clean[stacks_clean['km_hub'] > 200])
stacks_clean.to_csv('results/plants_clean.csv', index=False)

# Plot a map of the plants. Each plant (stack) is a bubble on the map, colored by its CO2 concentration.
def create_all_plants_map(europe, plants_gdf, bubble_scaling=1.0, remaining_gdf=None, hubs_gdf=None, debug=False):
    """
    Create a map showing all plants colored by CO2 concentration.
    """
    if debug:
        print(f"create_all_plants_map inputs: europe shape={europe.shape}, plants shape={plants_gdf.shape}")
        if remaining_gdf is not None:
            print(f"  remaining plants shape={remaining_gdf.shape}")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    ax.set_aspect(1.90)
    europe.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.45)

    # Plot remaining plants as gray bubbles first
    if remaining_gdf is not None and len(remaining_gdf) > 0:
        ax.scatter(
            remaining_gdf.geometry.x, remaining_gdf.geometry.y,
            s=remaining_gdf['ktCO2'] * bubble_scaling, c='gray', alpha=0.5,
            edgecolors='darkgray', linewidth=0.3,
            label='Other emitters'
        )

    # Plot stacks_clean plants colored by xCO2
    scatter = ax.scatter(
        plants_gdf.geometry.x, plants_gdf.geometry.y,
        s=plants_gdf['ktCO2'] * bubble_scaling, c=plants_gdf['xCO2']*100,
        cmap='magma', vmin=0, vmax=30, alpha=0.7,
        edgecolors='black', linewidth=0.5
        , label='Point source emitters'
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('CO₂ Concentration (%)', fontsize=13)

    ax.set_xlim(-9, 3)
    ax.set_ylim(49.5, 59.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    if hubs_gdf is not None and len(hubs_gdf) > 0:
        if 'km_shipping_clean' in hubs_gdf:
            no_shipping = hubs_gdf[hubs_gdf['km_shipping_clean'].isna()]
            shipping = hubs_gdf[~hubs_gdf['km_shipping_clean'].isna()]
        else:
            no_shipping = hubs_gdf
            shipping = gpd.GeoDataFrame(columns=hubs_gdf.columns)
        if len(no_shipping) > 0:
            ax.scatter(
                no_shipping.geometry.x,
                no_shipping.geometry.y,
                marker='D',
                s=120,
                c='whitesmoke',
                edgecolors='black',
                linewidth=1.0,
                label='Storage hub'
            )
        if len(shipping) > 0:
            ax.scatter(
                shipping.geometry.x,
                shipping.geometry.y,
                marker='D',
                s=120,
                c='#E55064',
                edgecolors='black',
                linewidth=1.0,
                label='Export hub'
            )
    legend = ax.legend(loc='lower left')
    for handle, text in zip(legend.legend_handles, legend.texts):
        if text.get_text() == 'Point source emitters':
            handle.set_sizes([200]) # Custom size for point source emitters
    return fig, ax

def create_sector_map(europe, plants_gdf, bubble_scaling=1.0, remaining_gdf=None, hubs_gdf=None, debug=False):
    """
    Create a map showing all plants colored by sector (using magma colormap).
    """
    if debug:
        print(f"create_sector_map inputs: europe shape={europe.shape}, plants shape={plants_gdf.shape}")
        if remaining_gdf is not None:
            print(f"  remaining plants shape={remaining_gdf.shape}")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    ax.set_aspect(1.90)
    europe.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.45)

    # Plot remaining plants as gray bubbles first
    if remaining_gdf is not None and len(remaining_gdf) > 0:
        ax.scatter(
            remaining_gdf.geometry.x, remaining_gdf.geometry.y,
            s=remaining_gdf['ktCO2'] * bubble_scaling, c='gray', alpha=0.5,
            edgecolors='darkgray', linewidth=0.3,
            label='Other emitters'
        )

    # Create sector colors using magma
    sectors = plants_gdf['sector'].unique()
    sector_labels_dict = {
        'ccgt': 'Gas power',
        'cement': 'Cement',
        'drax': 'Drax',
        'refinery': 'Refinery',
        'steel': 'Steel',
        'waste': 'Waste',
    }
    magma_colors = plt.cm.magma(np.linspace(0.1, 0.9, len(sectors)))
    sector_colors = {s: c for s, c in zip(sorted(sectors), magma_colors)}
    
    if debug:
        print(f"  Sectors: {sorted(sectors)}")

    # Plot plants colored by sector
    legend_handles = []
    for sector in sorted(sectors):
        mask = plants_gdf['sector'] == sector
        sector_data = plants_gdf[mask]
        scatter = ax.scatter(
            sector_data.geometry.x, sector_data.geometry.y,
            s=sector_data['ktCO2'] * bubble_scaling, 
            c=[sector_colors[sector]],
            alpha=0.7,
            edgecolors='black', linewidth=0.5,
            label=sector_labels_dict[sector]
        )
        # Create legend handle with fixed size
        handle = plt.scatter([], [], s=80, c=[sector_colors[sector]], alpha=0.7, 
                            edgecolors='black', linewidth=0.5, label=sector_labels_dict[sector])
        legend_handles.append(handle)

    ax.set_xlim(-9, 3)
    ax.set_ylim(49.5, 59.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    
    # Plot hubs if provided
    if hubs_gdf is not None and len(hubs_gdf) > 0:
        if 'km_shipping_clean' in hubs_gdf:
            no_shipping = hubs_gdf[hubs_gdf['km_shipping_clean'].isna()]
            shipping = hubs_gdf[~hubs_gdf['km_shipping_clean'].isna()]
        else:
            no_shipping = hubs_gdf
            shipping = gpd.GeoDataFrame(columns=hubs_gdf.columns)
        if len(no_shipping) > 0:
            h1 = ax.scatter(
                no_shipping.geometry.x,
                no_shipping.geometry.y,
                marker='D',
                s=120,
                c='whitesmoke',
                edgecolors='black',
                linewidth=1.0,
                label='Storage hub'
            )
            legend_handles.append(h1)
        if len(shipping) > 0:
            h2 = ax.scatter(
                shipping.geometry.x,
                shipping.geometry.y,
                marker='D',
                s=120,
                c='#E55064',
                edgecolors='black',
                linewidth=1.0,
                label='Export hub'
            )
            legend_handles.append(h2)
    
    # Add "Other emitters" to legend if present
    if remaining_gdf is not None and len(remaining_gdf) > 0:
        h_other = plt.scatter([], [], s=80, c='gray', alpha=0.5, 
                              edgecolors='darkgray', linewidth=0.3, label='Other emitters')
        legend_handles.insert(0, h_other)
    
    ax.legend(handles=legend_handles, loc='lower left', fontsize=12)
    return fig, ax

# Identify remaining plants (in point_sources but not in stacks_clean)
clean_names = stacks_clean['site'].unique()
remaining_plants = point_sources[~point_sources['Site'].isin(clean_names)].copy()

print("\nLoading Europe shapefile...")
europe = gpd.read_file("data/shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

plants_gdf = gpd.GeoDataFrame(
    stacks_clean, 
    geometry=gpd.points_from_xy(stacks_clean['longitude'], stacks_clean['latitude'], crs="EPSG:4326")
)
remaining_valid = remaining_plants[
    remaining_plants['Easting'].notna() & remaining_plants['Northing'].notna()
].copy()
remaining_gdf = gpd.GeoDataFrame(
    remaining_valid, 
    geometry=gpd.points_from_xy(remaining_valid['Easting'], remaining_valid['Northing'], crs="EPSG:27700")
).to_crs("EPSG:4326")

transport_hubs['km_shipping_clean'] = pd.to_numeric(transport_hubs['km_shipping'], errors='coerce')
hubs_gdf = gpd.GeoDataFrame(
    transport_hubs,
    geometry=gpd.points_from_xy(transport_hubs['longitude'], transport_hubs['latitude'], crs="EPSG:4326")
)

fig, ax = create_all_plants_map(
    europe,
    plants_gdf,
    bubble_scaling=0.8,
    remaining_gdf=remaining_gdf,
    hubs_gdf=hubs_gdf,
    debug=True
)

output_file = 'results/map_concentrations.png'
plt.savefig(output_file, dpi=400, bbox_inches='tight')

# Create sector map
fig2, ax2 = create_sector_map(
    europe,
    plants_gdf,
    bubble_scaling=0.8,
    remaining_gdf=remaining_gdf,
    hubs_gdf=hubs_gdf,
    debug=True
)
output_file2 = 'results/map_sectors.png'
plt.savefig(output_file2, dpi=400, bbox_inches='tight')

plt.show()