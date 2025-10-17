import matplotlib.pyplot as plt
import numpy as np

def create_macc_curve(debug=False):
    """
    Create a marginal abatement cost curve for 5 CCS plants.
    
    Parameters:
    debug (bool): If True, prints inputs and outputs
    
    Returns:
    tuple: (cumulative_capacity, costs) arrays
    """
    # Plant data: [cumulative_capacity_MtCO2, cost_EUR_per_tCO2]
    plants = [
        [0, 0],      # Starting point
        [2.6, 70],   # Plant 1: 2.6 MtCO2/year, €70/tCO2
        [1.6+2.6, 80],   # Plant 2: 1.6 MtCO2/year, €80/tCO2
        [5.2+2.6+1.6, 120],  # Plant 3: 5.2 MtCO2/year, €120/tCO2
        [1.2+2.6+1.6+5.2, 160],  # Plant 4: 1.2 MtCO2/year, €160/tCO2
        [3+2.6+1.6+5.2+1.2, 300]     # Plant 5: 3.0 MtCO2/year, €300/tCO2
    ]
    
    # Sort plants by cumulative capacity to create proper MACC curve
    plants_sorted = sorted(plants, key=lambda x: x[0])
    
    cumulative_capacity = np.array([plant[0] for plant in plants_sorted])
    costs = np.array([plant[1] for plant in plants_sorted])
    
    if debug:
        print("Input: 5 CCS plants with cumulative capacity and costs")
        print("Cumulative Capacity (MtCO2/year):", cumulative_capacity)
        print("Costs (EUR/tCO2):", costs)
        print("Output: MACC curve data ready for plotting")
    
    return cumulative_capacity, costs

def plot_macc_curve(debug=False):
    """
    Plot the marginal abatement cost curve as horizontal steps (boxes).
    
    Parameters:
    debug (bool): If True, prints inputs and outputs
    """
    cumulative_capacity, costs = create_macc_curve(debug)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create step function data for horizontal lines
    x_steps = []
    y_steps = []
    
    for i in range(len(cumulative_capacity) - 1):
        # Each plant is a horizontal line from current to next cumulative capacity
        x_steps.extend([cumulative_capacity[i], cumulative_capacity[i+1]])
        y_steps.extend([costs[i+1], costs[i+1]])  # Use the cost of the current plant
    
    plt.plot(x_steps, y_steps, 'b-', linewidth=3)
    
    # Formatting
    plt.xlabel('Cumulative CO₂ Capture Capacity (MtCO₂/year)', fontsize=14)
    plt.ylabel('Cost of CCS (EUR/tCO₂)', fontsize=14)
    plt.title('Marginal Abatement Cost Curve - CCS Plants', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    plt.xlim(-0.1, max(cumulative_capacity) + 0.2)
    plt.ylim(-5, max(costs) + 10)
    
    plt.tight_layout()
    plt.show()
    
    if debug:
        print("Plot created successfully with horizontal steps")

if __name__ == "__main__":
    plot_macc_curve(debug=True)
