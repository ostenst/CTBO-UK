"""Plot selected experiment outcomes from results_baseline / results_phaseout."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def _select_results_dir(debug=False):
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
        print(f"_select_results_dir: {results_dir}")
    return results_dir


def _load_experiments(results_dir):
    return pd.read_csv(f'{results_dir}/experiments.csv')


def _load_array(results_dir, key):
    return np.load(f'{results_dir}/outcomes_{key}.npy')


def _get_years(results_dir, start_year=2025, key='supply_ktCO2f'):
    n_years = _load_array(results_dir, key).shape[1]
    return np.arange(start_year, start_year + n_years)


def _panel_stats(arr, mask=None):
    panel = arr if mask is None else arr[mask]
    if len(panel) == 0:
        return None, None, None
    return np.nanmedian(panel, axis=0), np.nanpercentile(panel, 5, axis=0), np.nanpercentile(panel, 95, axis=0)


def _set_sparse_year_ticks(ax, years):
    ticks = [2030, 2040, 2050]
    ax.set_xlim(int(np.nanmin(years)), int(np.nanmax(years)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(y) for y in ticks])


def _get_sector_colors(sectors):
    magma = plt.cm.magma
    base = {
        'cement': magma(0.10),
        'ccgt': magma(0.30),
        'refinery': magma(0.85),
        'steel': magma(0.70),
        'drax': magma(0.50),
        'waste': '#62a7a6',
    }
    fallback = plt.cm.tab10(np.linspace(0, 1, max(1, len(sectors))))
    return {
        sector: base.get(sector, fallback[idx % len(fallback)])
        for idx, sector in enumerate(sorted(sectors))
    }


def plot_carbon_trajectories(results_dir='results_baseline', figures_dir='results_figures', start_year=2025, savefig=True, debug=False):
    """One panel: median carbon trajectories across all PRICE_POLICY experiments."""
    years = _get_years(results_dir, start_year=start_year, key='supply_ktCO2f')
    supply = _load_array(results_dir, 'supply_ktCO2f')
    stored_total = _load_array(results_dir, 'mandate_ktCO2')
    stored_b = _load_array(results_dir, 'stored_ktCO2b')
    stored_d = _load_array(results_dir, 'stored_ktCO2daccs')
    stored_cdr = stored_b + stored_d
    scale = 1000.0
    magma = plt.cm.magma

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arr, color, label in [
        (supply, magma(0.0), 'Supply'),
        (stored_total, magma(0.45), 'Stored total'),
        (stored_cdr, '#62a7a6', 'Stored CDR (BECCS + DACCS)'),
    ]:
        med, p5, p95 = _panel_stats(arr)
        ax.plot(years, med / scale, lw=2.3, color=color, label=label)
        ax.fill_between(years, p5 / scale, p95 / scale, color=color, alpha=0.22)

    # Median first year with DACCS > 0; y = median BECCS (ktCO2b) in that year
    first_daccs_idx = np.full(stored_d.shape[0], np.nan)
    for i in range(stored_d.shape[0]):
        hits = np.where(stored_d[i] > 0)[0]
        if len(hits):
            first_daccs_idx[i] = hits[0]
    if np.isfinite(first_daccs_idx).any():
        med_idx = int(np.nanmedian(first_daccs_idx))
        med_year = years[med_idx]
        med_b = np.nanmedian(stored_b[:, med_idx])
        ax.scatter(
            [med_year], [med_b / scale],
            s=70, color='#41BCAE', edgecolors='black', linewidths=1.2, zorder=5,
            label='DACCS start (median)',
        )
        if debug:
            print(f"plot_carbon_trajectories DACCS marker: year={med_year}, ktCO2b={med_b:.1f}")

    ax.set_ylabel('Carbon [MtCO₂/y]', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=12)
    _set_sparse_year_ticks(ax, years)
    ax.legend(fontsize=11, loc='best')
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_carbon_trajectories.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_carbon_trajectories: {out}")
    return fig


def plot_plant_NPV(results_dir='results_baseline', figures_dir='results_figures', pounds_to_EUR=1.15, savefig=True, debug=False):
    """One panel: median plant NPV vs investment year across all PRICE_POLICY experiments."""
    plant_ref = pd.read_csv(f'{results_dir}/plant_reference.csv')
    npv_total = _load_array(results_dir, 'plants_NPV_total')
    inv_year = _load_array(results_dir, 'plants_investment_year')
    mac = _load_array(results_dir, 'plants_MAC')
    cap_ref = pd.read_csv(f'{results_dir}/plants_costbenefit_extended.csv', usecols=['stack', 'ktCO2tot_ccs'])
    cap_by_stack = cap_ref.groupby('stack')['ktCO2tot_ccs'].median().to_dict()
    sector_colors = _get_sector_colors(plant_ref['sector'].dropna().unique())

    panel = plant_ref.copy()
    panel['NPV_total'] = np.nanmedian(npv_total, axis=0)
    panel['investment_year'] = np.nanmedian(inv_year, axis=0)
    panel['MAC'] = np.nanmedian(np.abs(mac), axis=0)
    panel['ktCO2tot_ccs'] = panel['stack'].map(cap_by_stack)
    panel = panel.dropna(subset=['NPV_total', 'investment_year'])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for sector in sorted(panel['sector'].dropna().unique()):
        sector_df = panel[panel['sector'] == sector]
        sizes = np.clip(sector_df['ktCO2tot_ccs'].to_numpy(dtype=float), 1.0, None) * 0.25
        ax.scatter(
            sector_df['investment_year'].to_numpy(dtype=float),
            sector_df['NPV_total'].to_numpy(dtype=float) / 1000.0 / pounds_to_EUR,
            s=sizes,
            c=[sector_colors.get(sector, 'grey')],
            alpha=0.75,
            edgecolors='black',
            linewidths=0.45,
            label=sector,
        )
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Investment year', fontsize=14)
    ax.set_ylabel('Plant NPV total [M£]', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, title='Sector', title_fontsize=12, loc='best')
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_plant_NPV.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_plant_NPV: {out}")
    return fig


def plot_carbon_prices(
    results_dir='results_baseline',
    figures_dir='results_figures',
    PRICE_POLICY='CAP-100£',
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """One panel: carbon price trajectories for a single PRICE_POLICY."""
    experiments = _load_experiments(results_dir)
    mask = experiments['PRICE_POLICY'] == PRICE_POLICY
    if mask.sum() == 0:
        raise ValueError(f"No experiments with PRICE_POLICY={PRICE_POLICY!r} in {results_dir}")

    years = _get_years(results_dir, start_year=start_year, key='cost_marginal')
    magma = plt.cm.magma
    series = [
        (_load_array(results_dir, 'cost_marginal') / pounds_to_EUR, magma(0.05), 'Marginal cost'),
        (_load_array(results_dir, 'price_ETS') / pounds_to_EUR, magma(0.35), 'ETS price (E)'),
        (_load_array(results_dir, 'price_CSU') / pounds_to_EUR, magma(0.65), 'CSU price (y)'),
        (_load_array(results_dir, 'cost_fuels') / pounds_to_EUR, magma(0.9), 'Fuel cost'),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arr, color, label in series:
        med, p5, p95 = _panel_stats(arr, mask)
        ax.plot(years, med, lw=2.3, color=color, label=label)
        ax.fill_between(years, p5, p95, color=color, alpha=0.2)

    ax.set_title(f'Carbon prices — {PRICE_POLICY}', fontsize=16)
    ax.set_ylabel('Price [£/tCO₂]', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=12)
    _set_sparse_year_ticks(ax, years)
    ax.legend(fontsize=11, loc='best')
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_carbon_prices.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_carbon_prices: {out} (n={mask.sum()})")
    return fig


def plot_cfd(
    results_dir='results_baseline',
    figures_dir='results_figures',
    cap_policies=None,
    plot_years=None,
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """Single-panel CfD boxplots: government cost vs taxpayer benefit at selected years."""
    if cap_policies is None:
        cap_policies = ['CAP-50£', 'CAP-100£', 'CAP-200£']
    if plot_years is None:
        plot_years = [2035, 2040, 2045]
    experiments = _load_experiments(results_dir)
    years = _get_years(results_dir, start_year=start_year, key='cost_CfD_government')
    cost_cfd = _load_array(results_dir, 'cost_CfD_government')
    benefit_cfd = _load_array(results_dir, 'benefit_CfD_taxpayer')
    scale = 1e-6 / pounds_to_EUR  # k€ -> B£
    magma = plt.cm.magma
    policy_colors = {
        'CAP-50£': magma(0.25),
        'CAP-100£': magma(0.65),
        'CAP-200£': magma(0.9),
    }

    pol_gap = 0.55
    year_gap = len(cap_policies) * pol_gap + 0.55
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    box_data, positions, facecolors, edgecolors = [], [], [], []
    xtick_pos, xtick_labels = [], []

    for i_year, year in enumerate(plot_years):
        year_idx = int(np.where(years == year)[0][0])
        cluster = i_year * year_gap
        for i_pol, policy in enumerate(cap_policies):
            mask = experiments['PRICE_POLICY'] == policy
            if mask.sum() == 0:
                continue
            color = policy_colors[policy]
            x0 = cluster + i_pol * pol_gap
            # Same x for cost (down) and benefit (up)
            box_data.append(-cost_cfd[mask, year_idx] * scale)
            positions.append(x0)
            facecolors.append((*color[:3], 0.85))
            edgecolors.append('black')
            box_data.append(benefit_cfd[mask, year_idx] * scale)
            positions.append(x0)
            facecolors.append((*color[:3], 0.35))
            edgecolors.append('black')
            xtick_pos.append(x0)
            xtick_labels.append(policy.replace('CAP-', ''))

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.28, patch_artist=True,
        showfliers=False, medianprops=dict(color='black', linewidth=1.5),
    )
    for patch, fc, ec in zip(bp['boxes'], facecolors, edgecolors):
        patch.set_facecolor(fc)
        patch.set_edgecolor(ec)
        patch.set_linewidth(1.0)

    ax.axhline(0, color='black', linewidth=1.0, zorder=1)

    # Year separators / labels under policy ticks
    for i_year, year in enumerate(plot_years):
        cluster = i_year * year_gap
        mid = cluster + (len(cap_policies) - 1) * pol_gap / 2
        ax.text(mid, -0.12, str(year), transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    legend_handles = []
    for p in cap_policies:
        c = policy_colors[p]
        legend_handles.append(Patch(facecolor=(*c[:3], 0.85), edgecolor='black', label=f'{p} gov. cost'))
        legend_handles.append(Patch(facecolor=(*c[:3], 0.35), edgecolor='black', label=f'{p} taxpayer'))

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_ylabel('Annual value [B£/y]\n(NET benefit ↑ / GROSS gov. cost ↓)', fontsize=13)
    # Show absolute magnitudes on both sides of zero
    yticks = ax.get_yticks()
    ax.set_yticklabels([f'{abs(y):.2g}' for y in yticks])
    ax.tick_params(labelsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax.legend(handles=legend_handles, fontsize=9, ncol=2, loc='best')
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_cfd.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_cfd: {out}")
    return fig


def plot_tax_and_gas(
    results_dir='results_baseline',
    figures_dir='results_figures',
    policies=None,
    start_year=2025,
    end_year=2035,
    end_year_long=2050,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """Three slim panels: tax costs, near-term gas rise, and gas rise through 2050."""
    if policies is None:
        policies = ['CAP-50£', 'CAP-100£', 'CAP-200£', 'CTBO', 'ETS']
    experiments = _load_experiments(results_dir)
    years_all = _get_years(results_dir, start_year=start_year, key='costs_tax')
    costs_tax_all = _load_array(results_dir, 'costs_tax')
    gas_all = _load_array(results_dir, 'gas_increase_abs')
    # EUR/MWh -> GBP pence/kWh
    gas_pence = gas_all / pounds_to_EUR * (100.0 / 1000.0)
    tax_scale = 1e-6 / pounds_to_EUR  # k€ -> B£
    magma = plt.cm.magma
    colors = {
        'CAP-50£': magma(0.25),
        'CAP-100£': magma(0.65),
        'CAP-200£': magma(0.85),
        'CTBO': 'black',
        'ETS': 'black',
    }
    linestyles = {
        'CAP-50£': '-',
        'CAP-100£': '-',
        'CAP-200£': '-',
        'CTBO': '--',
        'ETS': '-',
    }

    keep_near = years_all <= end_year
    keep_long = years_all <= end_year_long
    years_near = years_all[keep_near]
    years_long = years_all[keep_long]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 6.5))
    panels = [
        (axes[0], costs_tax_all[:, keep_near], years_near, tax_scale, 'Tax costs [B£/y]', [2030, 2035]),
        (axes[1], gas_pence[:, keep_near], years_near, 1.0, 'Gas price increase [p/kWh]', [2030, 2035]),
        (axes[2], gas_pence[:, keep_long], years_long, 1.0, 'Gas price increase [p/kWh]', [2030, 2040, 2050]),
    ]
    for ax, arr, years, scale, ylabel, ticks in panels:
        for policy in policies:
            mask = experiments['PRICE_POLICY'] == policy
            if mask.sum() == 0:
                continue
            med, p5, p95 = _panel_stats(arr, mask)
            color = colors.get(policy, 'gray')
            ls = linestyles.get(policy, '-')
            ax.plot(years, med * scale, lw=2.3, color=color, ls=ls, label=policy)
            ax.fill_between(years, p5 * scale, p95 * scale, color=color, alpha=0.12)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel('Year', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.set_xlim(int(years.min()), int(years.max()))
        ax.set_xticks(ticks)
        ax.legend(fontsize=9, loc='best')

    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_tax_gas.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_tax_and_gas: {out}")
    return fig


def plot_policy_costs(
    results_dir='results_baseline',
    figures_dir='results_figures',
    policies=None,
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """Three panels of annual policy costs with uncertainty: CAP-100£, CTBO, ETS."""
    if policies is None:
        policies = ['CAP-100£', 'CTBO', 'ETS']
    experiments = _load_experiments(results_dir)
    years = _get_years(results_dir, start_year=start_year, key='costs_suppliers')
    suppliers = _load_array(results_dir, 'costs_suppliers')
    emitters = _load_array(results_dir, 'costs_emitters')
    tax = _load_array(results_dir, 'costs_tax')
    consumers = _load_array(results_dir, 'costs_consumers')
    scale = 1e-6 / pounds_to_EUR  # k€ -> B£
    magma = plt.cm.magma
    series = [
        (suppliers, magma(0.1), 'Suppliers'),
        (emitters, magma(0.35), 'Emitters'),
        (tax, magma(0.6), 'Tax'),
        (consumers, magma(0.85), 'Consumers'),
    ]

    fig, axes = plt.subplots(1, len(policies), figsize=(4.0 * len(policies), 6.4), sharex=True, sharey=True)
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        mask = experiments['PRICE_POLICY'] == policy
        if mask.sum() == 0:
            ax.set_title(f'{policy} (no data)', fontsize=14)
            continue
        for arr, color, label in series:
            med, p5, p95 = _panel_stats(arr, mask)
            ax.plot(years, med * scale, lw=2.3, color=color, label=label)
            ax.fill_between(years, p5 * scale, p95 * scale, color=color, alpha=0.2)
        npv_text = (
            f"NPV suppliers: {experiments.loc[mask, 'NPV_costs_suppliers'].median() * scale:.3f} B£\n"
            f"NPV emitters: {experiments.loc[mask, 'NPV_costs_emitters'].median() * scale:.3f} B£\n"
            f"NPV tax: {experiments.loc[mask, 'NPV_costs_tax'].median() * scale:.3f} B£\n"
            f"NPV consumers: {experiments.loc[mask, 'NPV_costs_consumers'].median() * scale:.3f} B£"
        )
        ax.text(
            0.02, 0.98, npv_text, transform=ax.transAxes, va='top', ha='left', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'),
        )
        ax.set_title(policy, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Year', fontsize=14)
        _set_sparse_year_ticks(ax, years)

    axes[0].set_ylabel('Annual policy costs [B£/y]', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=11)
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_policy_costs.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_policy_costs: {out}")
    return fig


def plot_macc_curves(results_dir='results_baseline', figures_dir='results_figures', pounds_to_EUR=1.15,
                     APPLY_LR=False, savefig=True, debug=False):
    """Many MAC curves (one per experiment); min/max summed cost highlighted.

    APPLY_LR=False → plants_MAC0 (pre-learning).
    APPLY_LR=True  → plants_MAC (learned MAC at FID; final learned if never invested).
    """
    mac_key = 'plants_MAC' if APPLY_LR else 'plants_MAC0'
    mac = _load_array(results_dir, mac_key)
    stacks = _load_array(results_dir, 'plants_stack')
    cap_ref = pd.read_csv(f'{results_dir}/plants_costbenefit_extended.csv', usecols=['stack', 'ktCO2tot_ccs'])
    cap_by_stack = cap_ref.groupby('stack')['ktCO2tot_ccs'].median().to_dict()

    n_experiments = mac.shape[0]
    scenario_data = []
    summed_costs = np.full(n_experiments, np.nan)

    for i in range(n_experiments):
        stack_i = np.asarray(stacks[i], dtype=object)
        mac_i = np.asarray(mac[i], dtype=float) / pounds_to_EUR
        co2_i = np.array([cap_by_stack.get(s, np.nan) for s in stack_i], dtype=float) / 1000.0
        valid = np.isfinite(mac_i) & np.isfinite(co2_i) & (co2_i > 0)
        if not np.any(valid):
            scenario_data.append(None)
            continue
        co2_valid = co2_i[valid]
        mac_valid = mac_i[valid]
        summed_costs[i] = np.sum(co2_valid * mac_valid) / 1000.0
        order = np.argsort(mac_valid)
        scenario_data.append((np.cumsum(co2_valid[order]), mac_valid[order]))

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

    plotted = 0
    for i, data in enumerate(scenario_data):
        if data is None or i in (idx_min, idx_max):
            continue
        cum_co2, mac_sorted = data
        plotted += 1
        ax.step(cum_co2, mac_sorted, where='pre', color=magma(norm(summed_costs[i])), alpha=0.45, linewidth=1.1)

    for i in [idx_min, idx_max]:
        data = scenario_data[i]
        if data is None:
            continue
        cum_co2, mac_sorted = data
        plotted += 1
        ax.step(cum_co2, mac_sorted, where='pre', color='black', alpha=1.0, linewidth=2.8)
        ax.step(cum_co2, mac_sorted, where='pre', color=magma(norm(summed_costs[i])), alpha=1.0, linewidth=2.0)

    ax.set_xlabel('Cumulative CCS/BECCS capacity [MtCO₂ p.a.]', fontsize=14)
    ax.set_ylabel('Abatement cost of CCS/BECCS [£/tCO₂]', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=magma)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Summed cost [B£ p.a.]', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_macc_curves.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_macc_curves: APPLY_LR={APPLY_LR}, key={mac_key}, plotted={plotted}/{n_experiments}, min={idx_min}, max={idx_max}, {out}")
    return fig


if __name__ == "__main__":
    APPLY_LR = False  # False → pre-learning MAC0; True → learned MAC at FID
    selected_results_dir = _select_results_dir(debug=True)
    plot_carbon_trajectories(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_plant_NPV(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_carbon_prices(results_dir=selected_results_dir, figures_dir='results_figures', PRICE_POLICY='CAP-100£', debug=True)
    plot_cfd(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_tax_and_gas(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_policy_costs(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_macc_curves(results_dir=selected_results_dir, figures_dir='results_figures', APPLY_LR=APPLY_LR, debug=True)
    plt.show()
