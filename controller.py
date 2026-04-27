import numpy as np
import pandas as pd
from ctbo import simulate_ctbo
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    ScalarOutcome,
    ArrayOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments,
)

try:
    from ema_workbench import Sample as PolicySample  # EMA 3.x
except ImportError:
    from ema_workbench import Policy as PolicySample  # EMA 2.x


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    PHASEOUT = False
    results_dir = "results_phaseout" if PHASEOUT else "results_baseline"
    figures_dir = "results_figures"
    
    # Load data once
    plants_clean = pd.read_csv("results_baseline/plants_clean.csv")
    transport_hubs = pd.read_csv('data/transport_hubs.csv')

    model = Model("UKCTBO", function=simulate_ctbo)

    # Uncertainties: external factors we cannot control
    model.uncertainties = [
        # Scenario uncertainties
        RealParameter("DIFFUSE_END_FRACTION", 0.05, 0.50),
        CategoricalParameter("DACCS_SCENARIO", ['£322', '£391']),
        # Sector-specific uncertainties
        RealParameter("fraction_limestone", 0.55, 0.65),  # [-] cementite fraction
        RealParameter("fraction_fossil_waste", 0.40, 0.55),  # [-] fossil fraction in waste
        RealParameter("drax_efficiency", 40, 46),  # [%] Drax efficiency
        RealParameter("drax_efficiency_loss", 0.20, 0.28),  # [-] CCS efficiency penalty
        # Technology uncertainties
        RealParameter("capture_rate", 0.90, 0.95),
        RealParameter("qreb", 2.8, 3.8),  # [GJ/tCO2] reboiler duty
        RealParameter("pcomp", 0.20, 0.35),  # [MJ/kgCO2] compression power
        RealParameter("pcomp_liquefy", 0.28, 0.38),  # [MJ/kgCO2] compression & liquefaction power
        RealParameter("FLH_industry", 8000, 8500),
        RealParameter("FLH_waste", 7200, 8000),  # [h/y] waste plant operating hours
        RealParameter("ccgt_efficiency", 0.45, 0.55),
        RealParameter("ccgt_efficiency_loss", 0.10, 0.15),  # [-] CCS efficiency penalty
        # Cost uncertainties
        RealParameter("cgas", 10, 100),  # [€/MWh] 
        RealParameter("celc", 150, 300),  # [€/MWh]
        RealParameter("cstraw", 100, 200),  # [€/t]
        RealParameter("cliquefy", 5, 10),  # [€/tCO2]
        RealParameter("cHCN", 4, 8),  # [€/t steam]
        RealParameter("cHRSG", 8, 15),  # [€/t steam]
        RealParameter("camine", 2, 6),  # [€/tCO2]
        RealParameter("cstorage", 20, 30),  # [€/tCO2]
        RealParameter("transport_uncertainty", 0.0, 0.30),
        RealParameter("CAPEX_gasboiler", 0.10, 0.20),  # [M€/MW]
        RealParameter("CAPEX_bioboiler", 0.80, 1.20),  # [M€/MW]
        # Financial uncertainties
        RealParameter("fixate_CAPEX", 0.02, 0.05),  # [-] fixed OPEX fraction
        RealParameter("CAPEX_m", 0.80, 0.90),  # [-] CAPEX scale exponent
        RealParameter("discount_rate_ccs", 0.06, 0.12),
        IntegerParameter("lifetime_ccs", 20, 30),  # [years]
        RealParameter("CEPCI_2025", 880, 980),
        RealParameter("NETL_2025", 5.0, 6.0),
    ]

    # Levers: policy choices set by each explicit policy below (must stay in sync with Policy/Sample kwargs).
    model.levers = [
        CategoricalParameter("ETS_SCENARIO", ['CTBO-only', 'ETS-eq', '£100-Mix', '£200-Mix', '£300-Mix']),
    ]

    # Outcomes: metrics to track (aligned with simulate_ctbo return keys)
    model.outcomes = [
        # Scalar outcomes (policy NPV)
        ScalarOutcome("NPV_costs_suppliers"),
        ScalarOutcome("NPV_costs_emitters"),
        ScalarOutcome("NPV_costs_tax"),
        ScalarOutcome("NPV_costs_consumers"),
        ScalarOutcome("NPV_profit_y_policy"),
        ScalarOutcome("NPV_cost_y_policy"),
        ScalarOutcome("NPV_profit_E_policy"),
        ScalarOutcome("NPV_cost_E_policy"),
        ScalarOutcome("NPV_tax_E_policy"),
        # Time series outcomes
        ArrayOutcome("supply_ktCO2f"),
        ArrayOutcome("emitted_ktCO2f"),
        ArrayOutcome("emitted_ktCO2final"),
        ArrayOutcome("mandate_ktCO2"),
        ArrayOutcome("stored_ktCO2g"),
        ArrayOutcome("stored_ktCO2b"),
        ArrayOutcome("stored_ktCO2daccs"),
        ArrayOutcome("cost_marginal"),
        ArrayOutcome("price_ETS"),
        ArrayOutcome("price_CSU"),
        ArrayOutcome("cost_fuels"),
        ArrayOutcome("costs_suppliers"),
        ArrayOutcome("costs_emitters"),
        ArrayOutcome("costs_tax"),
        ArrayOutcome("costs_consumers"),
        ArrayOutcome("profit_y_policy"),
        ArrayOutcome("cost_y_policy"),
        ArrayOutcome("profit_E_policy"),
        ArrayOutcome("cost_E_policy"),
        ArrayOutcome("tax_E_policy"),
        ArrayOutcome("gas_increase_abs"),
        ArrayOutcome("petrol_increase_abs"),
        ArrayOutcome("diesel_increase_abs"),
        ArrayOutcome("kerosene_increase_abs"),
        # Plant-level outcomes
        ArrayOutcome("plants_stack"),
        ArrayOutcome("plants_sector"),
        ArrayOutcome("plants_investment_year"),
        ArrayOutcome("plants_npv_end_year"),
        ArrayOutcome("plants_NPV_CAPEX"),
        ArrayOutcome("plants_NPV_OPEX"),
        ArrayOutcome("plants_NPV_REVENUE"),
        ArrayOutcome("plants_NPV_total"),
        ArrayOutcome("plants_MAC"),
    ]

    # Constants: fixed parameters for all runs
    model.constants = [
        Constant("plants_clean", plants_clean),
        Constant("transport_hubs", transport_hubs),
        Constant("single_run", False),
        Constant("PHASEOUT", PHASEOUT),
        Constant("CTBO_QUADRATIC", 0.4),
        Constant("DISCOUNT_RATE", 0.035), # Can try 3.5% (social) or 9% (private)
        Constant("ETS_START", 45),
        Constant("START_YEAR", 2025),
        Constant("END_YEAR", 2055), # But assume all prices stabilize by 2050
        Constant("DIFFUSE_END_YEAR", 2050),
        Constant("pounds_to_EUR", 1.15),
        Constant("CONSTRUCTION_YEARS", 3),
    ]

    # Run experiments
    # Explicit policies: kwargs must match lever names (a dict as 2nd positional arg is wrong — it becomes unique_id).
    policies = [
        PolicySample("CTBO-only", ETS_SCENARIO="CTBO-only"),
        PolicySample("ETS-eq", ETS_SCENARIO="ETS-eq"),
        PolicySample("£100-Mix", ETS_SCENARIO="£100-Mix"),
        PolicySample("£200-Mix", ETS_SCENARIO="£200-Mix"),
        PolicySample("£300-Mix", ETS_SCENARIO="£300-Mix"),
    ]
    n_scenarios = 200
    
    results = perform_experiments(
        model,
        scenarios=n_scenarios,
        policies=policies,
        uncertainty_sampling=Samplers.LHS,
    )
    experiments, outcomes = results

    # Separate scalar and array outcomes
    scalar_outcomes = {k: v for k, v in outcomes.items() if np.asarray(v).ndim == 1}
    array_outcomes = {k: v for k, v in outcomes.items() if np.asarray(v).ndim > 1}
    
    # Save experiments + scalar outcomes as CSV
    scalar_df = pd.DataFrame(scalar_outcomes)
    combined_df = pd.concat([experiments, scalar_df], axis=1)
    combined_df.to_csv(f"{results_dir}/experiments.csv", index=False)

    # Save array outcomes as .npy files
    for name, arr in array_outcomes.items():
        np.save(f"{results_dir}/outcomes_{name}.npy", arr)
    
    # Save plant names reference (alphabetically ordered, same for all runs)
    # Use same constants as model to ensure consistency if PHASEOUT=True
    constants_dict = {c.name: c.value for c in model.constants if c.name not in ['plants_clean', 'transport_hubs']}
    constants_dict["results_dir"] = results_dir
    constants_dict["figures_dir"] = figures_dir
    constants_dict["save_aux_results"] = True
    test_result = simulate_ctbo(plants_clean, transport_hubs, **constants_dict)
    plant_ref = pd.DataFrame({
        'plant_index': range(len(test_result['plants_stack'])),
        'stack': test_result['plants_stack'],
        'sector': test_result['plants_sector']
    })
    plant_ref.to_csv(f"{results_dir}/plant_reference.csv", index=False)

    print("Wrote experiments/outcomes. Run `py regret.py` for regret tables and boxplots.")
    