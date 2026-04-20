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
    perform_experiments
)

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    # Load data once
    plants_clean = pd.read_csv('results/plants_clean.csv')
    transport_hubs = pd.read_csv('data/transport_hubs.csv')

    model = Model("UKCTBO", function=simulate_ctbo)

    # Uncertainties: external factors we cannot control
    model.uncertainties = [
        # Scenario uncertainties
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
        RealParameter("CAPEX_m", 0.75, 0.95),  # [-] CAPEX scale exponent
        RealParameter("discount_rate_ccs", 0.06, 0.12),
        IntegerParameter("lifetime_ccs", 20, 30),  # [years]
        RealParameter("CEPCI_2025", 880, 980),
        RealParameter("NETL_2025", 5.0, 6.0),
    ]

    # Levers: policy choices we can control
    model.levers = [
        CategoricalParameter("ETS_SCENARIO", ['CTBO-only', 'ETS-eq', '£100-Mix', '£200-Mix', '£300-Mix']),
        RealParameter("DIFFUSE_END_FRACTION", 0.05, 0.50),
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
        Constant("PHASEOUT", False),
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
    n_scenarios = 5
    n_policies = 40
    
    results = perform_experiments(
        model, 
        n_scenarios, 
        n_policies, 
        uncertainty_sampling=Samplers.LHS, 
        lever_sampling=Samplers.LHS
    )
    experiments, outcomes = results

    # Separate scalar and array outcomes
    scalar_outcomes = {k: v for k, v in outcomes.items() if np.asarray(v).ndim == 1}
    array_outcomes = {k: v for k, v in outcomes.items() if np.asarray(v).ndim > 1}
    
    # Save experiments + scalar outcomes as CSV
    scalar_df = pd.DataFrame(scalar_outcomes)
    combined_df = pd.concat([experiments, scalar_df], axis=1)
    combined_df.to_csv("results/experiments.csv", index=False)
    
    # Save array outcomes as .npy files
    for name, arr in array_outcomes.items():
        np.save(f"results/outcomes_{name}.npy", arr)
    
    # Save plant names reference (alphabetically ordered, same for all runs)
    # Use same constants as model to ensure consistency if PHASEOUT=True
    constants_dict = {c.name: c.value for c in model.constants if c.name not in ['plants_clean', 'transport_hubs']}
    test_result = simulate_ctbo(plants_clean, transport_hubs, **constants_dict)
    plant_ref = pd.DataFrame({
        'plant_index': range(len(test_result['plants_stack'])),
        'stack': test_result['plants_stack'],
        'sector': test_result['plants_sector']
    })
    plant_ref.to_csv("results/plant_reference.csv", index=False)
    
    print(f"\nCompleted {len(experiments)} experiments")
    print(f"Experiments + scalar outcomes saved to: results/experiments.csv")
    print(f"Array outcomes saved to: results/outcomes_*.npy")
    print(f"Plant reference saved to: results/plant_reference.csv")

    print("\n===> I DON'T THINK THIS MODEL CAN REPRESENT A FULL-ECONOMY ETS SCENARIO <===")
    print("Even though I promised to. This is because I can't make reasonable assumptions on how DACCS, BECCS etc is deployed to abate diffuse emissions")
    print("The distincion between POINT and DIFFUSE allows for a rising CTBO fraction to determine the PACE of storage deployment")
    print("However, I am not modelling a DECLINING CAP for ETS and diffuse emitters, which otherwise would determine the PACE of storage deployment")
    print("Maybe: the model is for GEOLOGICAL NZ, while other models are for economy-wide ATMOSPHERIC NZ?")
    print("Short explanation: the model is not set up to deal with (fully abate) diffuse emissions under a rising ETS price")
    print("So any assumptions I make on how DACCS, BECCS etc. is deployed under the ETS-only scenario did not feel credible")
    print("The model is however set up to deal with (fully abate) diffuse emissions under a rising CTBO fraction")
    print("A possible workaround: just run a £500-Mix scenario, where DACCS constrains the price? NO! This is not representative of the cap-and-trade marginal logic, which has lower costs early on!")


