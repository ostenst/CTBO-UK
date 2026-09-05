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


def _lr_mask(experiments, lr='5%'):
    if lr is None or 'LR' not in experiments.columns:
        return pd.Series(True, index=experiments.index)
    return experiments['LR'].astype(str) == str(lr)


GREEN_CMAP = [
    "#144842",
    "#237067",
    "#33978B",
    "#41BCAE",
    "#52E7D6",
    "#D5FDF7",
]
RED_CMAP = [
    "#F6CDCD",
    "#EE9699",
    "#E75263",
    "#B73A49",
    "#812632",
    "#4B131A",
]
# Low cost (green) → mid (white) → high cost (red)
COST_CMAP = colors.LinearSegmentedColormap.from_list(
    'green_white_red', GREEN_CMAP + ['#FFFFFF'] + RED_CMAP
)

# Friends1: carbon trajectories ↔ plant NPV (matched axes width)
FRIENDS1_FIGSIZE = (5.5, 5.5)
FRIENDS1_RECT = [0.16, 0.12, 0.78, 0.80]  # left, bottom, width, height

# Friends2: carbon prices ↔ macc curves (matched axes height)
FRIENDS2_FIGHEIGHT = 6.5
FRIENDS2_AX_BOTTOM = 0.10
FRIENDS2_AX_HEIGHT = 0.78
FRIENDS2_PRICES_FIGSIZE = (4.0, FRIENDS2_FIGHEIGHT)
FRIENDS2_PRICES_RECT = [0.20, FRIENDS2_AX_BOTTOM, 0.72, FRIENDS2_AX_HEIGHT]
FRIENDS2_MACC_FIGSIZE = (7.2, FRIENDS2_FIGHEIGHT)
FRIENDS2_MACC_RECT = [0.12, FRIENDS2_AX_BOTTOM, 0.68, FRIENDS2_AX_HEIGHT]
FRIENDS2_CBAR_RECT = [0.82, FRIENDS2_AX_BOTTOM, 0.03, FRIENDS2_AX_HEIGHT]


def _friends_axes(figsize, rect):
    """Fixed axes box so paired figures share panel width/height in inches."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    return fig, ax


def _set_sparse_year_ticks(ax, years, end_year=2050):
    ticks = [y for y in (2030, 2040, 2050) if y <= end_year]
    ax.set_xlim(int(np.nanmin(years)), end_year)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(y) for y in ticks])


def _get_sector_colors(sectors):
    magma = plt.cm.magma
    base = {
        'cement': magma(0.325),
        'ccgt': magma(0.10),
        'drax': "#52E7D6",
        'steel': magma(0.925),
        'refinery': magma(0.625),
        'waste': '#41BCAE',
    }
    fallback = plt.cm.tab10(np.linspace(0, 1, max(1, len(sectors))))
    return {
        sector: base.get(sector, fallback[idx % len(fallback)])
        for idx, sector in enumerate(sorted(sectors))
    }


SECTOR_LABELS = {
    'ccgt': 'Gas power',
    'waste': 'Waste-to-energy',
    'cement': 'Cement',
    'drax': 'Drax',
    'steel': 'Steel (stack)',
    'refinery': 'Refinery (stack)',
}


def plot_carbon_trajectories(results_dir='results_baseline', figures_dir='results_figures', start_year=2025, end_year=2050, display_legend=False, savefig=True, debug=False):
    """One panel: median carbon trajectories across all PRICE_POLICY experiments."""
    years_all = _get_years(results_dir, start_year=start_year, key='supply_ktCO2f')
    keep = years_all <= end_year
    years = years_all[keep]
    supply = _load_array(results_dir, 'supply_ktCO2f')[:, keep]
    stored_total = _load_array(results_dir, 'mandate_ktCO2')[:, keep]
    stored_b = _load_array(results_dir, 'stored_ktCO2b')[:, keep]
    stored_d = _load_array(results_dir, 'stored_ktCO2daccs')[:, keep]
    stored_cdr = stored_b + stored_d
    scale = 1000.0
    magma = plt.cm.magma

    fig, ax = _friends_axes(FRIENDS1_FIGSIZE, FRIENDS1_RECT)
    for arr, color, label in [
        (supply, magma(0.0), 'Fossil fuel supply (coal, oil, gas)'),
        (stored_total, magma(0.55), 'Storage capacity (CCS, BECCS, DACCS)'),
        (stored_cdr, '#41BCAE', 'Storage capacity (BECCS, DACCS)'),
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
            label='DACCS median deployment year',
        )
        if debug:
            print(f"plot_carbon_trajectories DACCS marker: year={med_year}, ktCO2b={med_b:.1f}")

    ax.set_ylim(0, 300)
    ax.set_ylabel('Carbon [MtCO₂/y]', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=12)
    _set_sparse_year_ticks(ax, years, end_year=end_year)
    if display_legend:
        ax.legend(fontsize=11, loc='upper left')
    if savefig:
        out = f'{figures_dir}/multiple_carbon_trajectories.png'
        fig.savefig(out, dpi=450)  # no tight crop: keep Friends1 panel width
        if debug:
            print(f"plot_carbon_trajectories: {out}")
    return fig


def plot_plant_NPV(results_dir='results_baseline', figures_dir='results_figures', pounds_to_EUR=1.15, end_year=2050, display_legend=False, savefig=True, debug=False):
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
    panel = panel[panel['investment_year'] <= end_year]

    fig, ax = _friends_axes(FRIENDS1_FIGSIZE, FRIENDS1_RECT)
    y_vals = []
    for sector in sorted(panel['sector'].dropna().unique()):
        sector_df = panel[panel['sector'] == sector]
        sizes = np.clip(sector_df['ktCO2tot_ccs'].to_numpy(dtype=float), 1.0, None) * 0.25
        y = sector_df['NPV_total'].to_numpy(dtype=float) / 1000.0 / pounds_to_EUR
        y_vals.append(y)
        ax.scatter(
            sector_df['investment_year'].to_numpy(dtype=float),
            y,
            s=sizes,
            c=[sector_colors.get(sector, 'grey')],
            alpha=0.75,
            edgecolors='black',
            linewidths=0.45,
            label=SECTOR_LABELS.get(sector, sector),
        )
    ax.set_ylim(0, 10**4)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Investment year', fontsize=14)
    ax.set_ylabel('Plant median NPV [M£]', fontsize=14)
    y_all = np.concatenate(y_vals) if y_vals else np.array([1.0])
    # Log if all positive; else symlog so negatives remain visible
    if np.nanmin(y_all) > 0:
        ax.set_yscale('log')
        max_y = np.nanmax(y_all)
        ax.set_ylim(70, max_y*1.3)
    else:
        ax.set_yscale('symlog', linthresh=max(1.0, float(np.nanpercentile(np.abs(y_all), 10))))

    ax.set_xlim(int(np.nanmin(panel['investment_year'])), end_year)
    ax.set_xticks([2030, 2040, 2050])
    ax.set_xticklabels(['2030', '2040', '2050'])
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=12)
    if display_legend:
        leg = ax.legend(fontsize=12, title_fontsize=12, loc='upper right')
        for handle in leg.legend_handles:
            handle.set_sizes([55])
    if savefig:
        out = f'{figures_dir}/multiple_plant_NPV.png'
        fig.savefig(out, dpi=450)  # no tight crop: keep Friends1 panel width
        if debug:
            print(f"plot_plant_NPV: {out}")
    return fig


def plot_carbon_prices(
    results_dir='results_baseline',
    figures_dir='results_figures',
    PRICE_POLICY='CAP-100£',
    start_year=2025,
    end_year=2050,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """One panel: carbon price trajectories for a single PRICE_POLICY."""
    experiments = _load_experiments(results_dir)
    mask = experiments['PRICE_POLICY'] == PRICE_POLICY
    if mask.sum() == 0:
        raise ValueError(f"No experiments with PRICE_POLICY={PRICE_POLICY!r} in {results_dir}")

    years_all = _get_years(results_dir, start_year=start_year, key='cost_marginal')
    keep = years_all <= end_year
    years = years_all[keep]
    magma = plt.cm.magma
    series = [
        (_load_array(results_dir, 'cost_marginal')[:, keep] / pounds_to_EUR, magma(0.05), 'Marginal cost \nCCS/BECCS/DACCS'),
        (_load_array(results_dir, 'price_ETS')[:, keep] / pounds_to_EUR, magma(0.30), 'ETS price'),
        (_load_array(results_dir, 'price_CSU')[:, keep] / pounds_to_EUR, magma(0.625), 'CSU price'),
        (_load_array(results_dir, 'cost_fuels')[:, keep] / pounds_to_EUR, '#41BCAE', 'Average embedded \npolicy cost (in fuels)'),
    ]

    fig, ax = _friends_axes(FRIENDS2_PRICES_FIGSIZE, FRIENDS2_PRICES_RECT)
    for arr, color, label in series:
        med, p5, p95 = _panel_stats(arr, mask)
        ax.plot(years, med, lw=2.3, color=color, label=label)
        ax.fill_between(years, p5, p95, color=color, alpha=0.2)

    ax.set_title(f'Carbon prices — {PRICE_POLICY}', fontsize=16)
    ax.set_ylabel('Price [£/tCO₂]', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=12)
    _set_sparse_year_ticks(ax, years, end_year=end_year)
    ax.legend(fontsize=12, loc='best')
    if savefig:
        out = f'{figures_dir}/multiple_carbon_prices.png'
        fig.savefig(out, dpi=450)  # no tight crop: keep Friends2 panel height
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
    """Two stacked CfD boxplot panels: taxpayer savings (top), government cost (bottom)."""
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
    box_alpha = 0.95
    policy_colors = {
        'CAP-50£': magma(0.25),
        'CAP-100£': magma(0.65),
        'CAP-200£': colors.to_rgba('#41BCAE'),
    }

    pol_gap = 0.35
    year_gap = len(cap_policies) * pol_gap + 0.40
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    ax_tax, ax_gov = axes

    def _collect_boxes(arr):
        box_data, positions, facecolors = [], [], []
        for i_year, year in enumerate(plot_years):
            year_idx = int(np.where(years == year)[0][0])
            cluster = i_year * year_gap
            for i_pol, policy in enumerate(cap_policies):
                mask = experiments['PRICE_POLICY'] == policy
                if mask.sum() == 0:
                    continue
                color = policy_colors[policy]
                x0 = cluster + i_pol * pol_gap
                box_data.append(arr[mask, year_idx] * scale)
                positions.append(x0)
                facecolors.append((*color[:3], box_alpha))
        return box_data, positions, facecolors

    panels = [
        (ax_tax, benefit_cfd, 'Taxpayer savings [B£/y]'),
        (ax_gov, cost_cfd, 'Government cost [B£/y]'),
    ]
    for ax, arr, ylabel in panels:
        box_data, positions, facecolors = _collect_boxes(arr)
        bp = ax.boxplot(
            box_data, positions=positions, widths=0.28, patch_artist=True,
            showfliers=False, medianprops=dict(color='black', linewidth=1.5),
        )
        for patch, fc in zip(bp['boxes'], facecolors):
            patch.set_facecolor(fc)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
        ax.axhline(0, color='black', linewidth=1.0, zorder=1)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)

    # Year ticks only (CAP policies identified via color legend)
    year_tick_pos = [
        i_year * year_gap + (len(cap_policies) - 1) * pol_gap / 2
        for i_year in range(len(plot_years))
    ]
    ax_gov.set_xticks(year_tick_pos)
    ax_gov.set_xticklabels([str(y) for y in plot_years], fontsize=13)

    from matplotlib.patches import Patch
    label_strings = {
        'CAP-50£': 'CfD replacing CAP-50£',
        'CAP-100£': 'CfD replacing CAP-100£',
        'CAP-200£': 'CfD replacing CAP-200£',
    }
    legend_handles = [
        Patch(facecolor=(*policy_colors[p][:3], box_alpha), edgecolor='black', label=label_strings[p])
        for p in cap_policies
    ]
    ax_tax.legend(handles=legend_handles, fontsize=11, loc='best')
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
    near_start_year=2027,
    near_end_year=2035,
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
        'CAP-200£': '#41BCAE',
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

    keep_near = (years_all >= near_start_year) & (years_all <= near_end_year)
    keep_long = years_all <= end_year_long
    years_near = years_all[keep_near]
    years_long = years_all[keep_long]
    near_ticks = [near_start_year, 2032, near_end_year]

    fig, axes = plt.subplots(1, 3, figsize=(10, 7))
    panels = [
        (axes[0], costs_tax_all[:, keep_near], years_near, tax_scale, 'Tax costs [B£/y]', near_ticks),
        (axes[1], gas_pence[:, keep_near], years_near, 1.0, 'Gas price increase [p/kWh]', near_ticks),
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
        ax.legend(fontsize=12, loc='best')

    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_tax_gas.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_tax_and_gas: {out}")
    return fig


def plot_csu(
    results_dir='results_baseline',
    figures_dir='results_figures',
    cap_policies=None,
    plot_years=None,
    start_year=2025,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """Two stacked CSU boxplot panels split by DIFFUSE_END_FRACTION."""
    if cap_policies is None:
        cap_policies = ['CAP-50£', 'CAP-100£', 'CAP-200£']
    if plot_years is None:
        plot_years = [2035, 2040, 2045]
    experiments = _load_experiments(results_dir)
    years = _get_years(results_dir, start_year=start_year, key='costs_suppliers')
    suppliers = _load_array(results_dir, 'costs_suppliers')
    scale = 1e-6 / pounds_to_EUR  # k€ -> B£
    magma = plt.cm.magma
    box_alpha = 0.95
    policy_colors = {
        'CAP-50£': magma(0.25),
        'CAP-100£': magma(0.65),
        'CAP-200£': colors.to_rgba('#41BCAE'),
    }

    pol_gap = 0.35
    year_gap = len(cap_policies) * pol_gap + 0.40
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True, sharey=True)
    ax_high, ax_low = axes

    def _collect_boxes(row_filter):
        box_data, positions, facecolors = [], [], []
        for i_year, year in enumerate(plot_years):
            year_idx = int(np.where(years == year)[0][0])
            cluster = i_year * year_gap
            for i_pol, policy in enumerate(cap_policies):
                mask = (experiments['PRICE_POLICY'] == policy) & row_filter
                if mask.sum() == 0:
                    continue
                color = policy_colors[policy]
                x0 = cluster + i_pol * pol_gap
                box_data.append(suppliers[mask, year_idx] * scale)
                positions.append(x0)
                facecolors.append((*color[:3], box_alpha))
        return box_data, positions, facecolors

    panels = [
        (ax_high, experiments['DIFFUSE_END_FRACTION'] > 0.25, 'DIFFUSE_END_FRACTION > 0.25'),
        (ax_low, experiments['DIFFUSE_END_FRACTION'] < 0.25, 'DIFFUSE_END_FRACTION < 0.25'),
    ]
    for ax, row_filter, title in panels:
        box_data, positions, facecolors = _collect_boxes(row_filter)
        bp = ax.boxplot(
            box_data, positions=positions, widths=0.28, patch_artist=True,
            showfliers=False, medianprops=dict(color='black', linewidth=1.5),
        )
        for patch, fc in zip(bp['boxes'], facecolors):
            patch.set_facecolor(fc)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
        ax.axhline(0, color='black', linewidth=1.0, zorder=1)
        ax.set_ylabel('CSU demand [B£ p.a.] by fuel suppliers', fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)

    year_tick_pos = [
        i_year * year_gap + (len(cap_policies) - 1) * pol_gap / 2
        for i_year in range(len(plot_years))
    ]
    ax_low.set_xticks(year_tick_pos)
    ax_low.set_xticklabels([str(y) for y in plot_years], fontsize=13)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=(*policy_colors[p][:3], box_alpha), edgecolor='black', label=p)
        for p in cap_policies
    ]
    ax_high.legend(handles=legend_handles, fontsize=11, loc='best')
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_csu.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_csu: {out}")
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
    """Three panels of annual policy costs: CTBO (as 'ETS pretend'), ETS, CAP-100£."""
    if policies is None:
        policies = ['CTBO', 'ETS', 'CAP-100£']
    experiments = _load_experiments(results_dir)
    years = _get_years(results_dir, start_year=start_year, key='costs_suppliers')
    suppliers = _load_array(results_dir, 'costs_suppliers')
    emitters = _load_array(results_dir, 'costs_emitters')
    tax = _load_array(results_dir, 'costs_tax')
    consumers = _load_array(results_dir, 'costs_consumers')
    scale = 1e-6 / pounds_to_EUR  # k€ -> B£
    magma = plt.cm.magma
    emitters_color = magma(0.35)
    series = [
        (suppliers, magma(0.1), 'Supplier costs'),
        (emitters, emitters_color, 'Emitter costs'),
        (tax, magma(0.625), 'Emitter tax'),
        (consumers, magma(0.80), 'Consumer costs (sum)'),
    ]
    panel_titles = {
        'CTBO': 'ETS-without tax',
        'ETS': 'ETS-with tax',
        'CAP-100£': 'CAP-100£',
    }

    fig, axes = plt.subplots(1, len(policies), figsize=(3.333 * len(policies), 7), sharex=True, sharey=True)
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        mask = experiments['PRICE_POLICY'] == policy
        if mask.sum() == 0:
            ax.set_title(f'{panel_titles.get(policy, policy)} (no data)', fontsize=14)
            continue
        pretend = policy == 'CTBO'
        if pretend:
            # CTBO suppliers drawn as Emitters (ETS-like), skip real emitters series
            panel_series = [
                (suppliers, emitters_color, 'Emitters'),
                (tax, magma(0.6), 'Tax'),
                (consumers, magma(0.85), 'Consumers'),
            ]
            npv_text = (
                f"NPV suppliers: {experiments.loc[mask, 'NPV_costs_emitters'].median() * scale:.3f} B£\n"
                f"NPV emitters: {experiments.loc[mask, 'NPV_costs_suppliers'].median() * scale:.3f} B£\n"
                f"NPV tax: {experiments.loc[mask, 'NPV_costs_tax'].median() * scale:.3f} B£\n"
                f"NPV consumers: {experiments.loc[mask, 'NPV_costs_consumers'].median() * scale:.3f} B£"
            )
        else:
            panel_series = series
            npv_text = (
                f"NPV suppliers: {experiments.loc[mask, 'NPV_costs_suppliers'].median() * scale:.3f} B£\n"
                f"NPV emitters: {experiments.loc[mask, 'NPV_costs_emitters'].median() * scale:.3f} B£\n"
                f"NPV tax: {experiments.loc[mask, 'NPV_costs_tax'].median() * scale:.3f} B£\n"
                f"NPV consumers: {experiments.loc[mask, 'NPV_costs_consumers'].median() * scale:.3f} B£"
            )
        for arr, color, label in panel_series:
            med, p5, p95 = _panel_stats(arr, mask)
            ax.plot(years, med * scale, lw=2.3, color=color, label=label)
            ax.fill_between(years, p5 * scale, p95 * scale, color=color, alpha=0.2)
        ax.text(
            0.02, 0.98, npv_text, transform=ax.transAxes, va='top', ha='left', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'),
        )
        ax.set_title(panel_titles.get(policy, policy), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Year', fontsize=14)
        _set_sparse_year_ticks(ax, years)

    axes[0].set_ylabel('Annual policy costs [B£/y]', fontsize=14)
    # Legend inside a non-pretend panel so Suppliers/Emitters both appear
    legend_ax = axes[1] if len(axes) > 1 else axes[0]
    legend_ax.legend(fontsize=12, loc='center left')
    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_policy_costs.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_policy_costs: {out}")
    return fig


def plot_lr_suppliers(
    results_dir='results_baseline',
    figures_dir='results_figures',
    learning_rates=None,
    start_year=2025,
    end_year=2050,
    pounds_to_EUR=1.15,
    savefig=True,
    debug=False,
):
    """CTBO supplier costs by LR, plus paired differences vs LR=0%."""
    if learning_rates is None:
        learning_rates = ['0%', '5%', '10%']
    experiments = _load_experiments(results_dir)
    years_all = _get_years(results_dir, start_year=start_year, key='costs_suppliers')
    keep = years_all <= end_year
    years = years_all[keep]
    suppliers = _load_array(results_dir, 'costs_suppliers')[:, keep]
    scale = 1e-6 / pounds_to_EUR  # k€ -> B£
    magma = plt.cm.magma
    lr_colors = {
        '0%': "black",
        '5%': magma(0.78),
        '10%': magma(0.35),
    }
    diff_specs = [
        ('5%', GREEN_CMAP[0], 'Savings between 0 and 5% LR'),
        ('10%', GREEN_CMAP[3], 'Savings between 0 and 10% LR'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 6.5), sharex=True)
    ax_abs, ax_diff = axes

    for lr in learning_rates:
        mask = (experiments['PRICE_POLICY'] == 'CTBO') & _lr_mask(experiments, lr)
        if mask.sum() == 0:
            continue
        med, p5, p95 = _panel_stats(suppliers, mask)
        color = lr_colors.get(lr, 'gray')
        ax_abs.plot(years, med * scale, lw=2.3, color=color, label=f'LR={lr}')
        ax_abs.fill_between(years, p5 * scale, p95 * scale, color=color, alpha=0.25)
    ax_abs.set_ylabel('GCS costs [B£ p.a.]', fontsize=14)
    ax_abs.set_xlabel('Year', fontsize=14)
    ax_abs.tick_params(labelsize=12)
    ax_abs.grid(True, linestyle='--', alpha=0.35)
    _set_sparse_year_ticks(ax_abs, years, end_year=end_year)
    ax_abs.legend(fontsize=12, loc='best')

    ctbo = experiments['PRICE_POLICY'] == 'CTBO'
    if 'scenario' in experiments.columns:
        for lr_b, color, label in diff_specs:
            mask_0 = ctbo & _lr_mask(experiments, '0%')
            mask_b = ctbo & _lr_mask(experiments, lr_b)
            scen_0 = experiments.loc[mask_0, 'scenario'].to_numpy()
            scen_b = experiments.loc[mask_b, 'scenario'].to_numpy()
            rows_0 = np.flatnonzero(mask_0.to_numpy())
            rows_b = np.flatnonzero(mask_b.to_numpy())
            order_0 = np.argsort(scen_0)
            order_b = np.argsort(scen_b)
            common, i0, ib = np.intersect1d(scen_0[order_0], scen_b[order_b], return_indices=True)
            if len(common) == 0:
                continue
            diff = suppliers[rows_0[order_0][i0]] - suppliers[rows_b[order_b][ib]]
            med, p5, p95 = _panel_stats(diff)
            ax_diff.plot(years, med * scale, lw=2.3, color=color, label=label)
            ax_diff.fill_between(years, p5 * scale, p95 * scale, color=color, alpha=0.18)
            if debug:
                print(f"plot_lr_suppliers paired n={len(common)} for 0% vs {lr_b}")
    ax_diff.axhline(0, color='grey', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_diff.set_ylabel('GCS cost difference [B£ p.a.]', fontsize=14)
    ax_diff.set_xlabel('Year', fontsize=14)
    ax_diff.tick_params(labelsize=12)
    ax_diff.grid(True, linestyle='--', alpha=0.35)
    _set_sparse_year_ticks(ax_diff, years, end_year=end_year)
    ax_diff.legend(fontsize=12, loc='best')

    fig.tight_layout()
    if savefig:
        out = f'{figures_dir}/multiple_lr_suppliers.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_lr_suppliers: {out}")
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

    fig, ax = _friends_axes(FRIENDS2_MACC_FIGSIZE, FRIENDS2_MACC_RECT)
    cost_min = float(np.nanmin(summed_costs[valid_idx]))
    cost_max = float(np.nanmax(summed_costs[valid_idx]))
    norm = colors.Normalize(vmin=cost_min, vmax=cost_max if cost_max > cost_min else cost_min + 1.0)

    plotted = 0
    for i, data in enumerate(scenario_data):
        if data is None or i in (idx_min, idx_max):
            continue
        cum_co2, mac_sorted = data
        plotted += 1
        ax.step(cum_co2, mac_sorted, where='pre', color=COST_CMAP(norm(summed_costs[i])), alpha=0.45, linewidth=1.1)

    for i in [idx_min, idx_max]:
        data = scenario_data[i]
        if data is None:
            continue
        cum_co2, mac_sorted = data
        plotted += 1
        ax.step(cum_co2, mac_sorted, where='pre', color='black', alpha=1.0, linewidth=2.8)
        ax.step(cum_co2, mac_sorted, where='pre', color=COST_CMAP(norm(summed_costs[i])), alpha=1.0, linewidth=2.0)

    ax.set_xlabel('Cumulative CCS/BECCS capacity [MtCO₂ p.a.]', fontsize=14)
    ax.set_ylabel('Abatement cost of CCS/BECCS [£/tCO₂]', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.6)
    cax = fig.add_axes(FRIENDS2_CBAR_RECT)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=COST_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Summed cost [B£ p.a.]', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    if savefig:
        out = f'{figures_dir}/multiple_macc_curves.png'
        fig.savefig(out, dpi=450)  # no tight crop: keep Friends2 panel height
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
    plot_csu(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_policy_costs(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_lr_suppliers(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_macc_curves(results_dir=selected_results_dir, figures_dir='results_figures', APPLY_LR=APPLY_LR, debug=True)
    plt.show()
