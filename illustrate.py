"""Plot selected experiment outcomes from results_baseline / results_phaseout."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        'refinery': magma(0.90),
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
        out = f'{figures_dir}/carbon_trajectories.png'
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
        out = f'{figures_dir}/plant_NPV.png'
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
        safe = str(PRICE_POLICY).replace('£', 'GBP').replace('-', '_')
        out = f'{figures_dir}/carbon_prices_{safe}.png'
        fig.savefig(out, dpi=450, bbox_inches='tight')
        if debug:
            print(f"plot_carbon_prices: {out} (n={mask.sum()})")
    return fig


if __name__ == "__main__":
    selected_results_dir = _select_results_dir(debug=True)
    plot_carbon_trajectories(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_plant_NPV(results_dir=selected_results_dir, figures_dir='results_figures', debug=True)
    plot_carbon_prices(results_dir=selected_results_dir, figures_dir='results_figures', PRICE_POLICY='CAP-100£', debug=True)
    plt.show()
