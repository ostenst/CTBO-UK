import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Set random seed for reproducibility
np.random.seed(42)

# Generate 30 CCS options
n_options = 30

# Generate costs between 50 and 300 £/tCO2
costs = np.linspace(50, 300, n_options)

# Generate volumes between 6 and 0.6 MtCO2/yr
volumes = np.linspace(6, 0.6, n_options)

# Add some random noise to make it more realistic
costs += np.random.normal(0, 5, n_options)  # Add small random variations to costs
volumes += np.random.normal(0, 0.1, n_options)  # Add small random variations to volumes

# Sort by cost to create proper MACC
sorted_indices = np.argsort(costs)
costs = costs[sorted_indices]
volumes = volumes[sorted_indices]

# Calculate cumulative abatement
cumulative_abatement = np.cumsum(volumes)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create the bars with two colors
for i, (cost, volume, cum_abate) in enumerate(zip(costs, volumes, cumulative_abatement)):
    # Calculate the height of each section
    if cost > 70:
        # If cost is above 70, create two sections
        ax.bar(cum_abate - volume/2, 70, width=volume, color='#90EE90', edgecolor='black', linewidth=1)  # Green section
        ax.bar(cum_abate - volume/2, cost - 70, width=volume, bottom=70, color='#D3D3D3', edgecolor='black', linewidth=1)  # Grey section
    else:
        # If cost is below or equal to 70, just create green section
        ax.bar(cum_abate - volume/2, cost, width=volume, color='#90EE90', edgecolor='black', linewidth=1)

# Add horizontal line at 70 £/tCO2
ax.axhline(y=70, color='black', linestyle='--', alpha=0.5, label='70 £/tCO2 ETS price')

# Customize the plot
ax.set_xlabel('Cumulative Abatement (MtCO2/yr)', fontsize=16)
ax.set_ylabel('Cost (£/tCO2)', fontsize=16)
ax.set_title('Marginal Abatement Cost Curve for CCS Options', fontsize=16)

# Set tick label sizes
ax.tick_params(axis='both', which='major', labelsize=16)

# Add grid for better readability
ax.grid(True, alpha=0.3)

# Create custom legend elements
legend_elements = [
    Patch(facecolor='#90EE90', edgecolor='black', label='ETS price incentive'),
    Patch(facecolor='#D3D3D3', edgecolor='black', label='CCS cost difference'),
    Patch(facecolor='none', edgecolor='black', linestyle='--', label='70 £/tCO2 ETS price')
]

# Add legend with larger font size
ax.legend(handles=legend_elements, fontsize=16)

# Adjust layout and save
plt.tight_layout()
plt.savefig('macc.png', dpi=600, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"Total potential abatement: {sum(volumes):.2f} MtCO2/yr")
print(f"Average cost: {np.mean(costs):.2f} £/tCO2")
print(f"Minimum cost: {min(costs):.2f} £/tCO2")
print(f"Maximum cost: {max(costs):.2f} £/tCO2")
print(f"Number of options below 70 £/tCO2: {sum(costs <= 70)}") 