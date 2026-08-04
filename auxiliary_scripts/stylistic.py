import numpy as np
import matplotlib.pyplot as plt

def plot_stylistic_supply_stored(color='gray', debug=False):
    """
    Stylistic stackplot showing CO2 supply (left y-axis) and CO2 stored (right y-axis).
    """
    if debug:
        print("Creating stylistic supply/stored plot")
    
    # Time axis
    years = np.arange(2025, 2051)
    
    # CO2 supply: linear decrease from 200 to 50 MtCO2/yr
    supply_start = 200  # MtCO2/yr
    supply_end = 50     # MtCO2/yr
    supply = np.linspace(supply_start, supply_end, len(years))
    
    # Setup figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    
    # Left y-axis: CO2 supply (stackplot)
    ax1.stackplot(years, supply, colors=['gray'], alpha=0.45, labels=['CO₂ supply'])
    # ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Carbon supply [MtCO₂/yr]\n'+ r'$\mathbf{diffuse}$' +' emitters', fontsize=14)
    ax1.set_xlim(2025, 2050)
    ax1.set_ylim(0, 300)
    ax1.set_yticks([0, 100, 200, 300])
    ax1.tick_params(labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    # ax1.legend(loc='upper left', fontsize=12)
    
    # Right y-axis: CO2 stored (empty for now, same ticks as left)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Carbon storage [MtCO₂/yr]', fontsize=14, color=color)
    ax2.set_ylim(0, 300)
    ax2.set_yticks([0, 100, 200, 300])
    ax2.tick_params(labelsize=12, colors=color)
    
    plt.tight_layout()
    plt.savefig('results_figures/stylistic_supply_stored.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print("Plot saved to results_figures/stylistic_supply_stored.png")
    
    return fig

def plot_stylistic_supply_vs_stored(color='gray', debug=False):
    """
    Stylistic stackplot showing constant CO2 supply with quadratic CO2 stored overlay.
    """
    if debug:
        print("Creating stylistic supply vs stored plot")
    
    # Time axis
    years = np.arange(2025, 2051)
    ctbo_trajectory = ((years - 2025) * 0.4)**2 / 100
    
    # CO2 supply: constant at 100 MtCO2/yr
    supply = np.full(len(years), 100)
    
    # CO2 stored: quadratic increase from 0 to 150 MtCO2/yr
    stored = 150 * ctbo_trajectory
    
    # Setup figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    
    # Left y-axis: CO2 supply (stackplot)
    ax1.stackplot(years, supply, colors=['gray'], alpha=0.45, labels=['CO₂ supply'])
    # ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Carbon supply [MtCO₂/yr]\n'+ r'$\mathbf{point-source}$' +' emitters', fontsize=14)
    ax1.set_xlim(2025, 2050)
    ax1.set_ylim(0, 300)
    ax1.set_yticks([0, 100, 200, 300])
    ax1.tick_params(labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Right y-axis: CO2 stored (stackplot)
    ax2 = ax1.twinx()
    ax2.stackplot(years, stored, colors=[color], alpha=0.7, labels=['CO₂ stored'])
    ax2.set_ylabel('Carbon storage [MtCO₂/yr]', fontsize=14, color=color)
    ax2.set_ylim(0, 300)
    ax2.set_yticks([0, 100, 200, 300])
    ax2.tick_params(labelsize=12, colors=color)
    
    plt.tight_layout()
    plt.savefig('results_figures/stylistic_supply_vs_stored.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print("Plot saved to results_figures/stylistic_supply_vs_stored.png")
    
    return fig

def plot_stylistic_decreasing_supply_stored(color='gray', debug=False):
    """
    Stylistic stackplot showing decreasing CO2 supply with quadratic CO2 stored overlay.
    """
    if debug:
        print("Creating stylistic decreasing supply vs stored plot")
    
    # Time axis
    years = np.arange(2025, 2051)
    ctbo_trajectory = ((years - 2025) * 0.4)**2 / 100
    
    # CO2 supply: linear decrease from 300 to 150 MtCO2/yr
    supply = np.linspace(300, 150, len(years))
    
    # CO2 stored: quadratic increase from 0 to 150 MtCO2/yr
    stored = 150 * ctbo_trajectory
    
    # Setup figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    
    # Left y-axis: CO2 supply (stackplot)
    ax1.stackplot(years, supply, colors=['gray'], alpha=0.45, labels=['CO₂ supply'])
    # ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Carbon supply [MtCO₂ p.a.]', fontsize=14)
    ax1.set_xlim(2025, 2050)
    ax1.set_ylim(0, 300)
    ax1.set_yticks([0, 100, 200, 300])
    ax1.tick_params(labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Right y-axis: CO2 stored (stackplot)
    ax2 = ax1.twinx()
    ax2.stackplot(years, stored, colors=[color], alpha=0.7, labels=['CO₂ stored'])
    ax2.set_ylabel('Carbon storage [MtCO₂ p.a.]', fontsize=14, color=color)
    ax2.set_ylim(0, 300)
    ax2.set_yticks([0, 100, 200, 300])
    ax2.tick_params(labelsize=12, colors=color)
    
    plt.tight_layout()
    plt.savefig('results_figures/stylistic_decreasing_supply_stored.png', dpi=450, bbox_inches='tight')
    
    if debug:
        print("Plot saved to results_figures/stylistic_decreasing_supply_stored.png")
    
    return fig

if __name__ == "__main__":
    magma = plt.cm.magma
    color = magma(0.2)
    plot_stylistic_supply_stored(color=color, debug=True)
    plot_stylistic_supply_vs_stored(color=color, debug=True)
    plot_stylistic_decreasing_supply_stored(color=color, debug=True)
    plt.show()
