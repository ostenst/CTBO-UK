import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def print_columns_summary(df):
    """
    Print a summary of DataFrame columns, grouping annual columns by range.
    
    Args:
        df: pandas DataFrame to analyze
    """
    # Group columns by their base pattern (without year)
    column_groups = defaultdict(list)

    for col in df.columns:
        # Check if column ends with a 4-digit year
        year_match = re.search(r' (\d{4})$', col)
        if year_match:
            year = year_match.group(1)
            base_name = col[:-5]  # Remove the year and space
            column_groups[base_name].append((year, col))
        else:
            # Non-annual column
            column_groups[col].append(('', col))

    # Print the column names
    print("Column names:")
    col_num = 1

    for base_name, columns in column_groups.items():
        if len(columns) > 1 and all(year for year, _ in columns):
            # This is an annual series
            years = [year for year, _ in columns]
            years.sort()
            start_year = years[0]
            end_year = years[-1]
            if start_year == end_year:
                print(f"{col_num}. {base_name} {start_year}")
            else:
                print(f"{col_num}. {base_name} {start_year}-{end_year}")
            col_num += 1
        else:
            # Non-annual or single-year columns
            for year, col in columns:
                if year:
                    print(f"{col_num}. {col}")
                else:
                    print(f"{col_num}. {col}")
                col_num += 1

    print(f"\nTotal number of columns: {len(df.columns)}")
    print(f"Total number of rows: {len(df)}")

# Read the CSV file with encoding specification
df = pd.read_csv('data/nzip_balanced_scenario_results.csv', encoding='latin-1')

# Print column summary
print_columns_summary(df)

# Get all energy type columns - baseline and total
baseline_gas_cols = [col for col in df.columns if 'Baseline in natural gas use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
total_gas_cols = [col for col in df.columns if 'Total natural gas use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]

baseline_oil_cols = [col for col in df.columns if 'Baseline in petroleum use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
total_oil_cols = [col for col in df.columns if 'Total petroleum use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]

baseline_hydrogen_cols = [col for col in df.columns if 'Baseline in hydrogen use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
total_hydrogen_cols = [col for col in df.columns if 'Total hydrogen use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]

baseline_bioenergy_cols = [col for col in df.columns if 'Baseline in primary bioenergy use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
total_bioenergy_cols = [col for col in df.columns if 'Total  primary bioenergy use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]

baseline_electricity_cols = [col for col in df.columns if 'Baseline electricity use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
total_electricity_cols = [col for col in df.columns if 'Total electricity use (GWh)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]

# Sort all columns by year
all_baseline_cols = [baseline_gas_cols, baseline_oil_cols, baseline_hydrogen_cols, baseline_bioenergy_cols, baseline_electricity_cols]
all_total_cols = [total_gas_cols, total_oil_cols, total_hydrogen_cols, total_bioenergy_cols, total_electricity_cols]

for cols in all_baseline_cols + all_total_cols:
    cols.sort(key=lambda x: int(x.split()[-1]))

# Extract years
years = [int(col.split()[-1]) for col in baseline_gas_cols]

# Calculate totals across all sites for each year
baseline_gas_totals = df[baseline_gas_cols].sum()
total_gas_totals = df[total_gas_cols].sum()
baseline_oil_totals = df[baseline_oil_cols].sum()
total_oil_totals = df[total_oil_cols].sum()
baseline_hydrogen_totals = df[baseline_hydrogen_cols].sum()
total_hydrogen_totals = df[total_hydrogen_cols].sum()
baseline_bioenergy_totals = df[baseline_bioenergy_cols].sum()
total_bioenergy_totals = df[total_bioenergy_cols].sum()
baseline_electricity_totals = df[baseline_electricity_cols].sum()
total_electricity_totals = df[total_electricity_cols].sum()

# Create the plot with distinct colors
plt.figure(figsize=(16, 10))

# Natural Gas - Blue shades
plt.plot(years, baseline_gas_totals.values, label='Baseline Natural Gas', linewidth=2.5, marker='o', color='#1f77b4', markersize=4)
plt.plot(years, total_gas_totals.values, label='Total Natural Gas', linewidth=2.5, marker='s', color='#87ceeb', markersize=4, linestyle='--')

# Petroleum - Red shades
plt.plot(years, baseline_oil_totals.values, label='Baseline Petroleum', linewidth=2.5, marker='^', color='#d62728', markersize=4)
plt.plot(years, total_oil_totals.values, label='Total Petroleum', linewidth=2.5, marker='d', color='#ff7f7f', markersize=4, linestyle='--')

# Hydrogen - Purple shades
plt.plot(years, baseline_hydrogen_totals.values, label='Baseline Hydrogen', linewidth=2.5, marker='v', color='#9467bd', markersize=4)
plt.plot(years, total_hydrogen_totals.values, label='Total Hydrogen', linewidth=2.5, marker='<', color='#c5b0d5', markersize=4, linestyle='--')

# Bioenergy - Green shades
plt.plot(years, baseline_bioenergy_totals.values, label='Baseline Bioenergy', linewidth=2.5, marker='>', color='#2ca02c', markersize=4)
plt.plot(years, total_bioenergy_totals.values, label='Total Bioenergy', linewidth=2.5, marker='p', color='#90ee90', markersize=4, linestyle='--')

# Electricity - Orange shades
plt.plot(years, baseline_electricity_totals.values, label='Baseline Electricity', linewidth=2.5, marker='*', color='#ff7f0e', markersize=5)
plt.plot(years, total_electricity_totals.values, label='Total Electricity', linewidth=2.5, marker='h', color='#ffb347', markersize=4, linestyle='--')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Energy Use (GWh)', fontsize=12)
plt.title('Energy Use: Baseline vs Total Over Time (All Fuel Types)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create a new figure for CO2 emissions (Natural Gas and Oil only)
# Emissions factors (kg CO2 per kWh)
gas_emissions_factor = 0.204  # kgCO2/kWhgas or tCO2/MWhgas or ktCO2/GWhgas
oil_emissions_factor = 0.294 # kgCO2/kWhoil)

# From GWh to ktCO2 to MtCO2
baseline_gas_emissions = baseline_gas_totals * gas_emissions_factor / 1000
total_gas_emissions = total_gas_totals * gas_emissions_factor / 1000
baseline_oil_emissions = baseline_oil_totals * oil_emissions_factor / 1000
total_oil_emissions = total_oil_totals * oil_emissions_factor / 1000

# Get the CO2 emissions columns from the dataset
baseline_emissions_cols = [col for col in df.columns if 'Baseline emissions (MtCO2e)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
post_reee_emissions_cols = [col for col in df.columns if 'Post REEE baseline emissions (MtCO2e)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
direct_abated_cols = [col for col in df.columns if 'Total direct emissions abated (MtCO2e)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
indirect_abated_cols = [col for col in df.columns if 'Total indirect emissions abated (MtCO2e)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]
co2_stored_cols = [col for col in df.columns if 'Total Tonnes of CO2 stored (MtCO2)' in col and col.endswith(tuple([str(year) for year in range(2016, 2071)]))]

# Sort columns by year
baseline_emissions_cols.sort(key=lambda x: int(x.split()[-1]))
post_reee_emissions_cols.sort(key=lambda x: int(x.split()[-1]))
direct_abated_cols.sort(key=lambda x: int(x.split()[-1]))
indirect_abated_cols.sort(key=lambda x: int(x.split()[-1]))
co2_stored_cols.sort(key=lambda x: int(x.split()[-1]))

# Calculate totals across all sites for each year
baseline_emissions_totals = df[baseline_emissions_cols].sum()
post_reee_emissions_totals = df[post_reee_emissions_cols].sum()
direct_abated_totals = df[direct_abated_cols].sum()
indirect_abated_totals = df[indirect_abated_cols].sum()
co2_stored_totals = df[co2_stored_cols].sum()

# Create the CO2 emissions plot
plt.figure(figsize=(16, 10))

# Natural Gas CO2 emissions - Blue shades
plt.plot(years, baseline_gas_emissions.values, label='Baseline Natural Gas CO2', linewidth=2.5, marker='o', color='#1f77b4', markersize=4)
plt.plot(years, total_gas_emissions.values, label='Total Natural Gas CO2', linewidth=2.5, marker='s', color='#87ceeb', markersize=4, linestyle='--')

# Petroleum CO2 emissions - Red shades
plt.plot(years, baseline_oil_emissions.values, label='Baseline Petroleum CO2', linewidth=2.5, marker='^', color='#d62728', markersize=4)
plt.plot(years, total_oil_emissions.values, label='Total Petroleum CO2', linewidth=2.5, marker='d', color='#ff7f7f', markersize=4, linestyle='--')

# Dataset CO2 emissions - Gray shades
plt.plot(years, baseline_emissions_totals.values, label='Baseline Emissions', linewidth=2.5, marker='v', color='#2f2f2f', markersize=4)
plt.plot(years, post_reee_emissions_totals.values, label='Post REEE Baseline Emissions', linewidth=2.5, marker='<', color='#5f5f5f', markersize=4, linestyle='--')
plt.plot(years, direct_abated_totals.values, label='Total Direct Emissions Abated', linewidth=2.5, marker='>', color='#8f8f8f', markersize=4, linestyle=':')
plt.plot(years, indirect_abated_totals.values, label='Total Indirect Emissions Abated', linewidth=2.5, marker='p', color='#bfbfbf', markersize=4, linestyle='-.')

# CO2 stored - Bold black line
plt.plot(years, co2_stored_totals.values, label='Total CO2 Stored', linewidth=4, marker='*', color='#000000', markersize=6, linestyle='-')

plt.xlabel('Year', fontsize=12)
plt.ylabel('CO2 Emissions (MtCO2)', fontsize=12)
plt.title('CO2 Emissions: Energy Carriers vs Dataset Emissions', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()