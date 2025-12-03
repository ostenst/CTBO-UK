import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file with encoding specification
df = pd.read_csv('data/nzip_balanced_scenario_results.csv', encoding='latin-1')
years = list(range(2025, 2061))

# Linear consumption scenarios: decrease from 140 to 70 by 2055, then continue to 2060
# Calculate the value at 2060 to maintain linear trend
# From 2025 to 2055: 140 to 70 (30 years)
# From 2055 to 2060: 70 to ? (5 years)
# Linear trend: (70 - 140) / (2055 - 2025) = -70/30 = -2.33 per year
# At 2060: 70 + (-2.33 * 5) = 70 - 11.67 = 58.33

oil_consumption_totals = np.linspace(140, 58.33, len(years))
gas_consumption_totals = np.linspace(140, 58.33, len(years))

# Calculate industry CO2 storage
co2_stored_cols = [col for col in df.columns if 'Total Tonnes of CO2 stored (MtCO2)' in col and col.endswith(tuple([str(year) for year in range(2025, 2061)]))]
co2_stored_cols.sort(key=lambda x: int(x.split()[-1]))
co2_stored_totals = df[co2_stored_cols].sum()

# Stored fraction: Year 1: 0.1%, Year 2: 0.4% (+0.3%), Year 3: 0.9% (+0.5%), etc.
# Stop incrementing at year 2055 (index 39), remain constant after
stored_fraction = [0.001]  # Start with 0.1% in year 1
increment = 0.003  # Start with 0.3% increment

for year in range(1, len(years)):
    if year <= 33: 
        stored_fraction.append(round(stored_fraction[-1] + increment, 3))
        increment += 0.002  # Increase increment by 0.2% each year
    else:
        # Remain constant at the 2055 value
        stored_fraction.append(stored_fraction[-1])

# Create stackplot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Stackplot of oil and gas consumption
ax1.stackplot(years, oil_consumption_totals, gas_consumption_totals, 
              labels=['Oil Consumption', 'Gas Consumption'],
              colors=['gray', 'lightgray'], alpha=0.6)

# Overlay CO2 stored totals on top
ax1.stackplot(years, co2_stored_totals.values, 
              labels=['Stored CO2 from NZIP industries'],
              colors=['darkgray'], alpha=0.6, hatch='///')

# ax1.set_xlabel('Year', fontsize=16)
ax1.set_ylabel('CO2 (MtCO2)', fontsize=16)
# ax1.set_title('Oil & Gas Consumption with CO2 Storage Overlay', fontsize=18, fontweight='bold')
ax1.legend(loc='upper right', fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
ax1.grid(False)

# Create second y-axis for stored fraction
ax2 = ax1.twinx()
ax2.plot(years, stored_fraction, color='#000000', linewidth=2, label='Stored Fraction')

# Add scatter dot at x=2056, y2=1
ax2.scatter(2055.5, 1, color='#000000', s=70, zorder=5)

# Set y-axis limits and ticks to align with first axis
# First axis goes from 0 to 280 (140+140), second axis should go from 0 to 1
ax2.set_ylim(0, 300/140)
ax1.set_xlim(2025, 2060)

ax2.set_ylabel('Stored Fraction', fontsize=16, color='#000000')
ax2.tick_params(axis='y', labelcolor='#000000', labelsize=14)

# Add legend for the second axis
ax2.legend(loc='upper left', fontsize=14)

plt.tight_layout()
plt.savefig('fig1.png', dpi=600, bbox_inches='tight')
plt.show()
