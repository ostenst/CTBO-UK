import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


ETS_SCENARIOS_DEFAULT = ['CTBO-only', '£100-Mix', '£200-Mix', '£300-Mix', 'ETS-eq']


def _scenario_title(ets_scenario, debug=False):
    if debug:
        print(f"_scenario_title input: ets_scenario={ets_scenario}")
    labels = {
        'CTBO-only': 'CTBO only',
        '£100-Mix': 'Mix (£100)',
        '£200-Mix': 'Mix (£200)',
        '£300-Mix': 'Mix (£300)',
        'ETS-eq': 'ETS only',
    }
    title = labels.get(ets_scenario, str(ets_scenario))
    if debug:
        print(f"_scenario_title output: title={title}")
    return title


def _load_experiments(results_dir='results', debug=False):
    if debug:
        print(f"_load_experiments input: results_dir={results_dir}")
    experiments = pd.read_csv(f'{results_dir}/experiments.csv')
    if debug:
        print(f"_load_experiments output: n_rows={len(experiments)}")
    return experiments


def _load_array(results_dir, key, debug=False):
    path = f'{results_dir}/outcomes_{key}.npy'
    arr = np.load(path)
    if debug:
        print(f"_load_array output: key={key}, shape={arr.shape}")
    return arr


def _get_years(results_dir='results', start_year=2025, key='supply_ktCO2f', debug=False):
    arr = _load_array(results_dir, key, debug=debug)
    n_years = arr.shape[1]
    years = np.arange(start_year, start_year + n_years)
    if debug:
        print(f"_get_years output: n_years={n_years}, first={years[0]}, last={years[-1]}")
    return years


def _median_panel(arr, mask):
    if mask.sum() == 0:
        return None
    return np.nanmedian(arr[mask], axis=0)

def _panel_stats(arr, mask):
    if mask.sum() == 0:
        return None, None, None
    panel = arr[mask]
    return np.nanmedian(panel, axis=0), np.nanpercentile(panel, 5, axis=0), np.nanpercentile(panel, 95, axis=0)


def _set_sparse_year_ticks(ax, years, debug=False):
    ticks = [2030, 2040, 2050]
    year_min = int(np.nanmin(years))
    year_max = int(np.nanmax(years))
    ax.set_xlim(year_min, year_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(y) if year_min <= y <= year_max else '' for y in ticks])
    if debug:
        print(f"_set_sparse_year_ticks output: year_min={year_min}, year_max={year_max}, ticks={ticks}")


def _get_sector_colors(sectors, debug=False):
    if debug:
        print(f"_get_sector_colors input: n_sectors={len(sectors)}")
    magma = plt.cm.magma
    base = {
        'cement': magma(0.10),
        'ccgt': magma(0.30),
        'refinery': magma(0.90),
        'steel': magma(0.70),
        'drax': magma(0.50),
        'waste': '#62a7a6',
    }
    fallback = plt.cm.tab10(np.linspace(0, 1, max(1, len(sectors))))
    colors = {}
    for idx, sector in enumerate(sorted(sectors)):
        colors[sector] = base.get(sector, fallback[idx % len(fallback)])
    return colors


def _select_results_dir(debug=False):
    if debug:
        print("_select_results_dir input: awaiting user selection")
    while True:
        choice = input("Use PHASEOUT scenario? Enter 'y' (PHASEOUT=True) or 'n' (PHASEOUT=False): ").strip().lower()
        if choice in ("y", "yes"):
            results_dir = "results_phaseout"
            break
        if choice in ("n", "no"):
            results_dir = "results_baseline"
            break
        print("Please enter 'y' or 'n'.")
    if debug:
        print(f"_select_results_dir output: results_dir={results_dir}")
    return results_dir


def compare_carbon_trajectories(results_dir='results_baseline', figures_dir='results_figures', ets_scenarios=None, start_year=2025, savefig=True, debug=False):
    if ets_scenarios is None:
        ets_scenarios = ETS_SCENARIOS_DEFAULT
    if debug:
        print(f"compare_carbon_trajectories inputs: ets_scenarios={ets_scenarios}")

    experiments = _load_experiments(results_dir, debug=debug)
    years = _get_years(results_dir, start_year=start_year, key='supply_ktCO2f', debug=debug)
    supply = _load_array(results_dir, 'supply_ktCO2f', debug=debug)
    # emitted_f = _load_array(results_dir, 'emitted_ktCO2f', debug=debug)
    stored_total = _load_array(results_dir, 'mandate_ktCO2', debug=debug)
    stored_b = _load_array(results_dir, 'stored_ktCO2b', debug=debug)
    stored_d = _load_array(results_dir, 'stored_ktCO2daccs', debug=debug)
    stored_cdr = stored_b + stored_d

    n = len(ets_scenarios)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 6.2), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    scale = 1000.0
    magma = plt.cm.magma
    for ax, scenario in zip(axes, ets_scenarios):
        mask = experiments['ETS_SCENARIO'] == scenario
        if mask.sum() == 0:
            ax.set_title(f"{_scenario_title(scenario)} (no data)", fontsize=14)
            continue
        med, p5, p95 = _panel_stats(supply, mask)
        ax.plot(years, med / scale, lw=2.3, color=magma(0.0), label='Supply')
        ax.fill_between(years, p5 / scale, p95 / scale, color=magma(0.0), alpha=0.22)
        # Emitted fuel line intentionally hidden for a simpler plot.
        # med, p5, p95 = _panel_stats(emitted_f, mask)
        # ax.plot(years, med / scale, lw=2.3, color=magma(0.25), label='Emitted fuel')
        # ax.fill_between(years, p5 / scale, p95 / scale, color=magma(0.25), alpha=0.22)
        med, p5, p95 = _panel_stats(stored_total, mask)
        ax.plot(years, med / scale, lw=2.3, color=magma(0.45), label='Stored total')
        ax.fill_between(years, p5 / scale, p95 / scale, color=magma(0.45), alpha=0.22)
        med, p5, p95 = _panel_stats(stored_cdr, mask)
        ax.plot(years, med / scale, lw=2.3, color='#62a7a6', label='Stored CDR (BECCS + DACCS)')
        ax.fill_between(years, p5 / scale, p95 / scale, color='#62a7a6', alpha=0.24)
        ax.set_title(_scenario_title(scenario), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Year', fontsize=14)
        _set_sparse_year_ticks(ax, years, debug=debug)

    axes[0].set_ylabel('Carbon [MtCO2/y]', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=11)
    fig.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/compare_carbon_trajectories.png', dpi=450, bbox_inches='tight')
    return fig


def compare_carbon_prices(results_dir='results_baseline', figures_dir='results_figures', ets_scenarios=None, start_year=2025, pounds_to_EUR=1.15, savefig=True, debug=False):
    if ets_scenarios is None:
        ets_scenarios = ETS_SCENARIOS_DEFAULT
    if debug:
        print(f"compare_carbon_prices inputs: ets_scenarios={ets_scenarios}, pounds_to_EUR={pounds_to_EUR}")

    experiments = _load_experiments(results_dir, debug=debug)
    years = _get_years(results_dir, start_year=start_year, key='cost_marginal', debug=debug)
    marginal = _load_array(results_dir, 'cost_marginal', debug=debug)
    ets = _load_array(results_dir, 'price_ETS', debug=debug)
    csu = _load_array(results_dir, 'price_CSU', debug=debug)
    fuel_cost = _load_array(results_dir, 'cost_fuels', debug=debug)
    marginal = marginal / pounds_to_EUR
    ets = ets / pounds_to_EUR
    csu = csu / pounds_to_EUR
    fuel_cost = fuel_cost / pounds_to_EUR

    n = len(ets_scenarios)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 6.2), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    magma = plt.cm.magma
    for ax, scenario in zip(axes, ets_scenarios):
        mask = experiments['ETS_SCENARIO'] == scenario
        if mask.sum() == 0:
            ax.set_title(f"{_scenario_title(scenario)} (no data)", fontsize=14)
            continue
        med, p5, p95 = _panel_stats(marginal, mask)
        ax.plot(years, med, lw=2.3, color=magma(0.05), label='Marginal cost')
        ax.fill_between(years, p5, p95, color=magma(0.05), alpha=0.2)
        med, p5, p95 = _panel_stats(ets, mask)
        ax.plot(years, med, lw=2.3, color=magma(0.35), label='ETS price (E)')
        ax.fill_between(years, p5, p95, color=magma(0.35), alpha=0.2)
        med, p5, p95 = _panel_stats(csu, mask)
        ax.plot(years, med, lw=2.3, color=magma(0.65), label='CSU price (gamma)')
        ax.fill_between(years, p5, p95, color=magma(0.65), alpha=0.2)
        med, p5, p95 = _panel_stats(fuel_cost, mask)
        ax.plot(years, med, lw=2.3, color=magma(0.9), label='Fuel cost')
        ax.fill_between(years, p5, p95, color=magma(0.9), alpha=0.2)
        ax.set_title(_scenario_title(scenario), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Year', fontsize=14)
        _set_sparse_year_ticks(ax, years, debug=debug)

    axes[0].set_ylabel('Price [£/tCO2]', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=11)
    fig.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/compare_carbon_prices.png', dpi=450, bbox_inches='tight')
    return fig


def compare_policy_costs(results_dir='results_baseline', figures_dir='results_figures', ets_scenarios=None, start_year=2025, pounds_to_EUR=1.15, savefig=True, debug=False):
    if ets_scenarios is None:
        ets_scenarios = ETS_SCENARIOS_DEFAULT
    if debug:
        print(f"compare_policy_costs inputs: ets_scenarios={ets_scenarios}, pounds_to_EUR={pounds_to_EUR}")

    experiments = _load_experiments(results_dir, debug=debug)
    years = _get_years(results_dir, start_year=start_year, key='costs_suppliers', debug=debug)
    suppliers = _load_array(results_dir, 'costs_suppliers', debug=debug)
    emitters = _load_array(results_dir, 'costs_emitters', debug=debug)
    tax = _load_array(results_dir, 'costs_tax', debug=debug)
    consumers = _load_array(results_dir, 'costs_consumers', debug=debug)

    n = len(ets_scenarios)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 6.4), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    scale = 1e-6 / pounds_to_EUR  # kEUR -> B£
    magma = plt.cm.magma
    for ax, scenario in zip(axes, ets_scenarios):
        mask = experiments['ETS_SCENARIO'] == scenario
        if mask.sum() == 0:
            ax.set_title(f"{_scenario_title(scenario)} (no data)", fontsize=14)
            continue
        med, p5, p95 = _panel_stats(suppliers, mask)
        ax.plot(years, med * scale, lw=2.3, color=magma(0.1), label='Suppliers')
        ax.fill_between(years, p5 * scale, p95 * scale, color=magma(0.1), alpha=0.2)
        med, p5, p95 = _panel_stats(emitters, mask)
        ax.plot(years, med * scale, lw=2.3, color=magma(0.35), label='Emitters')
        ax.fill_between(years, p5 * scale, p95 * scale, color=magma(0.35), alpha=0.2)
        med, p5, p95 = _panel_stats(tax, mask)
        ax.plot(years, med * scale, lw=2.3, color=magma(0.6), label='Tax')
        ax.fill_between(years, p5 * scale, p95 * scale, color=magma(0.6), alpha=0.2)
        med, p5, p95 = _panel_stats(consumers, mask)
        ax.plot(years, med * scale, lw=2.3, color=magma(0.85), label='Consumers')
        ax.fill_between(years, p5 * scale, p95 * scale, color=magma(0.85), alpha=0.2)
        npv_text = (
            f"NPV suppliers: {experiments.loc[mask, 'NPV_costs_suppliers'].median() * scale:.3f} B£\n"
            f"NPV emitters: {experiments.loc[mask, 'NPV_costs_emitters'].median() * scale:.3f} B£\n"
            f"NPV tax: {experiments.loc[mask, 'NPV_costs_tax'].median() * scale:.3f} B£\n"
            f"NPV consumers: {experiments.loc[mask, 'NPV_costs_consumers'].median() * scale:.3f} B£"
        )
        ax.text(
            0.02, 0.98, npv_text, transform=ax.transAxes, va='top', ha='left', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray')
        )
        ax.set_title(_scenario_title(scenario), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Year', fontsize=14)
        _set_sparse_year_ticks(ax, years, debug=debug)

    axes[0].set_ylabel('Annual policy costs [B£/y]', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=11)
    fig.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/compare_policy_costs.png', dpi=450, bbox_inches='tight')
    return fig


def compare_fuel_inc(
    results_dir='results_baseline',
    figures_dir='results_figures',
    ets_scenarios=None,
    start_year=2025,
    pounds_to_EUR=1.15,
    household_consumption=11200,
    savefig=True,
    debug=False,
):
    if ets_scenarios is None:
        ets_scenarios = ETS_SCENARIOS_DEFAULT
    if debug:
        print(f"compare_fuel_inc inputs: ets_scenarios={ets_scenarios}")

    experiments = _load_experiments(results_dir, debug=debug)
    years = _get_years(results_dir, start_year=start_year, key='gas_increase_abs', debug=debug)
    gas_inc = _load_array(results_dir, 'gas_increase_abs', debug=debug)
    gas_inc_pence_kwh = gas_inc / pounds_to_EUR * (100 / 1000)  # EUR/MWh -> pence/kWh

    n = len(ets_scenarios)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5.8), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    magma = plt.cm.magma
    for ax, scenario in zip(axes, ets_scenarios):
        mask = experiments['ETS_SCENARIO'] == scenario
        if mask.sum() == 0:
            ax.set_title(f"{_scenario_title(scenario)} (no data)", fontsize=14)
            continue
        med, p5, p95 = _panel_stats(gas_inc_pence_kwh, mask)
        ax.plot(years, med, lw=2.4, color=magma(0.7), label='Gas increase')
        ax.fill_between(years, p5, p95, color=magma(0.7), alpha=0.24)
        ax.set_title(_scenario_title(scenario), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Year', fontsize=14)
        _set_sparse_year_ticks(ax, years, debug=debug)
        ax.legend(fontsize=11, loc='best')
        y1_min, y1_max = ax.get_ylim()
        ax2 = ax.twinx()
        ax2.set_ylim(y1_min * household_consumption / 100, y1_max * household_consumption / 100)  # pence -> £/year
        ax2.tick_params(labelsize=11)
        if ax is axes[-1]:
            ax2.set_ylabel('Household bill increase [£/year]\n- assuming 11,200 kWh/year', fontsize=12)

    axes[0].set_ylabel('Gas price increase [pence/kWh]', fontsize=14)
    fig.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/compare_fuel_inc.png', dpi=450, bbox_inches='tight')
    return fig


def compare_plant_NPV(results_dir='results_baseline', figures_dir='results_figures', ets_scenarios=None, pounds_to_EUR=1.15, savefig=True, debug=False):
    if ets_scenarios is None:
        ets_scenarios = ETS_SCENARIOS_DEFAULT
    if debug:
        print(f"compare_plant_NPV inputs: ets_scenarios={ets_scenarios}, pounds_to_EUR={pounds_to_EUR}")

    experiments = _load_experiments(results_dir, debug=debug)
    plant_ref = pd.read_csv(f'{results_dir}/plant_reference.csv')
    npv_total = _load_array(results_dir, 'plants_NPV_total', debug=debug)
    inv_year = _load_array(results_dir, 'plants_investment_year', debug=debug)
    mac = _load_array(results_dir, 'plants_MAC', debug=debug)
    cap_ref = pd.read_csv(f'{results_dir}/plants_costbenefit_extended.csv', usecols=['stack', 'ktCO2tot_ccs'])
    cap_by_stack = cap_ref.groupby('stack', as_index=True)['ktCO2tot_ccs'].median().to_dict()

    sectors = plant_ref['sector'].dropna().unique()
    sector_colors = _get_sector_colors(sectors, debug=debug)

    n = len(ets_scenarios)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 6.3), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, scenario in zip(axes, ets_scenarios):
        mask = experiments['ETS_SCENARIO'] == scenario
        if mask.sum() == 0:
            ax.set_title(f"{_scenario_title(scenario)} (no data)", fontsize=14)
            continue

        median_npv = np.nanmedian(npv_total[mask], axis=0)
        median_inv = np.nanmedian(inv_year[mask], axis=0)
        median_mac = np.nanmedian(np.abs(mac[mask]), axis=0)

        panel = plant_ref.copy()
        panel['NPV_total'] = median_npv
        panel['investment_year'] = median_inv
        panel['MAC'] = median_mac
        panel['ktCO2tot_ccs'] = panel['stack'].map(cap_by_stack)
        panel = panel.dropna(subset=['NPV_total', 'investment_year'])
        panel = panel[panel['sector'].str.lower() != 'drax']

        for sector in sorted(panel['sector'].dropna().unique()):
            sector_df = panel[panel['sector'] == sector]
            sizes = np.clip(sector_df['ktCO2tot_ccs'].to_numpy(dtype=float), 1.0, None) * 0.25
            ax.scatter(
                sector_df['investment_year'].to_numpy(dtype=float),
                sector_df['NPV_total'].to_numpy(dtype=float) / 1000.0 / pounds_to_EUR,  # [M£]
                s=sizes,
                c=[sector_colors.get(sector, 'grey')],
                alpha=0.75,
                edgecolors='black',
                linewidths=0.45,
                label=sector,
            )

        ax.axhline(0, color='grey', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.set_title(_scenario_title(scenario), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Investment year', fontsize=14)

    axes[0].set_ylabel('Plant NPV total [M£]', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        fig.legend(dedup.values(), dedup.keys(), loc='upper right', fontsize=11, title='Sector', title_fontsize=12)
    fig.tight_layout()
    if savefig:
        plt.savefig(f'{figures_dir}/compare_plant_NPV.png', dpi=450, bbox_inches='tight')
    return fig


def plot_mac_by_sector(results_dir='results_baseline', figures_dir='results_figures', pounds_to_EUR=1.15, savefig=True, debug=False):
    """
    Plot MAC distributions by sector as boxplots in one figure.
    Uses plant-level MAC outcomes across all experiments.
    """
    if debug:
        print(f"plot_mac_by_sector input: results_dir={results_dir}, pounds_to_EUR={pounds_to_EUR}")

    plant_ref = pd.read_csv(f'{results_dir}/plant_reference.csv')
    mac = _load_array(results_dir, 'plants_MAC', debug=debug)  # shape: [n_experiments, n_plants]

    if mac.ndim != 2:
        raise ValueError(f"Expected 2D plants_MAC array, got shape {mac.shape}")
    if mac.shape[1] != len(plant_ref):
        raise ValueError(
            f"Mismatch between plants_MAC columns ({mac.shape[1]}) and plant_reference rows ({len(plant_ref)})"
        )

    sectors = sorted(plant_ref['sector'].dropna().unique())
    sector_colors = _get_sector_colors(sectors, debug=debug)

    box_data = []
    labels = []
    colors = []
    for sector in sectors:
        idx = plant_ref.index[plant_ref['sector'] == sector].to_numpy()
        vals = (mac[:, idx] / pounds_to_EUR).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        box_data.append(vals)
        labels.append(sector)
        colors.append(sector_colors.get(sector, 'gray'))

    if len(box_data) == 0:
        raise ValueError("No finite MAC values found to plot.")

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.7),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_title('Plant MAC Distribution by Sector', fontsize=16)
    ax.set_ylabel('MAC [£/tCO2]', fontsize=14)
    ax.set_xlabel('Sector', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.35, axis='y')
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    if savefig:
        plt.savefig(f'{figures_dir}/mac_by_sector_boxplot.png', dpi=450, bbox_inches='tight')
    if debug:
        print(f"plot_mac_by_sector output: sectors_plotted={len(labels)}")
    return fig


def plot_macc_curves(results_dir='results_baseline', figures_dir='results_figures', pounds_to_EUR=1.15, savefig=True, debug=False):
    """
    Plot many MAC curves (one per experiment), with lowest/highest summed cost highlighted.
    """
    if debug:
        print(f"plot_macc_curves input: results_dir={results_dir}, pounds_to_EUR={pounds_to_EUR}")

    mac = _load_array(results_dir, 'plants_MAC', debug=debug)  # [n_experiments, n_plants]
    stacks = _load_array(results_dir, 'plants_stack', debug=debug)  # [n_experiments, n_plants]
    cap_ref = pd.read_csv(f'{results_dir}/plants_costbenefit_extended.csv', usecols=['stack', 'ktCO2tot_ccs'])
    cap_by_stack = cap_ref.groupby('stack', as_index=True)['ktCO2tot_ccs'].median().to_dict()

    n_experiments = mac.shape[0]
    scenario_data = []
    summed_costs = np.full(n_experiments, np.nan)

    for i in range(n_experiments):
        stack_i = np.asarray(stacks[i], dtype=object)
        mac_i = np.asarray(mac[i], dtype=float) / pounds_to_EUR  # [£/tCO2]
        co2_i = np.array([cap_by_stack.get(s, np.nan) for s in stack_i], dtype=float) / 1000.0  # [MtCO2]

        valid = np.isfinite(mac_i) & np.isfinite(co2_i) & (co2_i > 0)
        if not np.any(valid):
            scenario_data.append(None)
            continue

        co2_valid = co2_i[valid]
        mac_valid = mac_i[valid]
        summed_costs[i] = np.sum(co2_valid * mac_valid) /1000

        order = np.argsort(mac_valid)
        co2_sorted = co2_valid[order]
        mac_sorted = mac_valid[order]
        scenario_data.append((np.cumsum(co2_sorted), mac_sorted))

    valid_idx = np.where(np.isfinite(summed_costs))[0]
    if len(valid_idx) == 0:
        raise ValueError("No valid MACC data to plot.")
    idx_min = valid_idx[np.argmin(summed_costs[valid_idx])]
    idx_max = valid_idx[np.argmax(summed_costs[valid_idx])]

    fig, ax = plt.subplots(figsize=(10, 6))
    magma = plt.cm.magma
    cost_min = float(np.nanmin(summed_costs[valid_idx]))
    cost_max = float(np.nanmax(summed_costs[valid_idx]))
    norm = colors.Normalize(vmin=cost_min, vmax=cost_max if cost_max > cost_min else cost_min + 1.0)

    # Draw non-extreme curves first.
    plotted = 0
    for i, data in enumerate(scenario_data):
        if data is None:
            continue
        if i == idx_min or i == idx_max:
            continue
        cum_co2, mac_sorted = data
        plotted += 1
        curve_color = magma(norm(summed_costs[i]))
        ax.step(cum_co2, mac_sorted, where='pre', color=curve_color, alpha=0.45, linewidth=1.1)

    # Draw min/max last so they overlay all other curves.
    for i in [idx_min, idx_max]:
        data = scenario_data[i]
        if data is None:
            continue
        cum_co2, mac_sorted = data
        plotted += 1
        ax.step(cum_co2, mac_sorted, where='pre', color='black', alpha=1.0, linewidth=2.8)
        # Overlay colored line on top of the black highlight.
        ax.step(cum_co2, mac_sorted, where='pre', color=magma(norm(summed_costs[i])), alpha=1.0, linewidth=2.0)

    ax.set_xlabel('Cumulative CCS/BECCS capacity [MtCO2 p.a.]', fontsize=14)
    ax.set_ylabel('Abatement cost of CCS/BECCS [£/tCO2]', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=magma)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Summed cost [B£ p.a.]', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()

    if savefig:
        plt.savefig(f'{figures_dir}/compare_macc_curves.png', dpi=450, bbox_inches='tight')
    if debug:
        print(
            "plot_macc_curves output:",
            f"plotted={plotted}/{n_experiments},",
            f"idx_min={idx_min}, idx_max={idx_max},",
            f"file={figures_dir}/compare_macc_curves.png",
        )
    return fig


if __name__ == "__main__":
    selected_results_dir = _select_results_dir(debug=True)
    compare_carbon_trajectories(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    compare_carbon_prices(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    compare_policy_costs(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    compare_fuel_inc(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    compare_plant_NPV(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_mac_by_sector(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_macc_curves(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plt.show()
