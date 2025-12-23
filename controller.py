import numpy as np
import pandas as pd
from ctbo_multiple import simulate_ctbo, load_data
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
    
    # Load data once in main process
    data = load_data(debug=False)

    model = Model("UKCTBO", function=simulate_ctbo)

    # Uncertainties: external factors we cannot control
    model.uncertainties = [
        # Economic/market uncertainties
        RealParameter("celc", 200, 300),                    # [EUR/MWh] Electricity cost
        RealParameter("cbio", 80, 160),                     # [EUR/MWh] Biomass fuel cost
        RealParameter("csteam", 3.0, 5.0),                  # [EUR/tsteam] Steam cost
        RealParameter("amine_cost", 30, 60),                # [SEK/tCO2] Amine makeup cost
        RealParameter("extra_transtorage", 20, 40),         # [EUR/tCO2] Transport & storage cost
        # Technology/cost uncertainties
        RealParameter("CEPCI_2025", 880, 980),              # [-] Chemical Engineering Plant Cost Index
        RealParameter("FOAK_MULTIPLIER", 1.5, 2.0),         # [-] First-of-a-kind cost multiplier
        RealParameter("discount_rate_ccs", 0.06, 0.12),     # [-] Discount rate for CCS
        RealParameter("capture_rate", 0.90, 0.95),          # [-] CO2 capture rate
        IntegerParameter("lifetime_ccs", 20, 30),           # [years] CCS project lifetime
        # Technical uncertainties
        RealParameter("qreb", 3.0, 4.0),                    # [MJ/kgCO2] Reboiler heat duty
        RealParameter("pcompr", 0.30, 0.45),                # [MJ/kgCO2] Compression power
        RealParameter("fixed", 0.03, 0.05),                 # [-] Fixed OPEX fraction
        RealParameter("emission_factor_bio", 0.28, 0.38),   # [tCO2/MWhfuel] Biomass emission factor
        # Sector-specific uncertainties
        RealParameter("cement_xCO2", 0.18, 0.22),           # [-] CO2 concentration in cement flue gas
        RealParameter("w2e_xCO2", 0.09, 0.13),              # [-] CO2 concentration in W2E flue gas
        RealParameter("drax_xCO2", 0.11, 0.15),             # [-] CO2 concentration in Drax flue gas
        RealParameter("drax_efficiency_penalty", 0.20, 0.28), # [-] Efficiency penalty from CCS on Drax
        RealParameter("cement_process_fraction", 0.58, 0.68), # [-] Fraction of cement CO2 from process
        RealParameter("FLH_industry", 8400, 8600),
        # Scenario uncertainties
        CategoricalParameter("DACCS_EXPENSIVE", [True, False]),  # [-] DACCS cost scenario
    ]

    # Levers: policy choices we can control
    model.levers = [
        RealParameter("DIFFUSE_END_FRACTION", 0.10, 0.40),  # [-] Diffuse emissions target
        CategoricalParameter("ETS_SCENARIO", ["Low", "Medium", "High"]),
    ]

    # Outcomes: metrics we want to track
    model.outcomes = [
        ScalarOutcome("gas_increase_pct_2040", ScalarOutcome.MINIMIZE),
        ScalarOutcome("gas_increase_pct_2050", ScalarOutcome.MINIMIZE),
        ScalarOutcome("gas_increase_abs_2040", ScalarOutcome.MINIMIZE),
        ScalarOutcome("gas_increase_abs_2050", ScalarOutcome.MINIMIZE),
        ArrayOutcome("gas_increase_pct"),
        ArrayOutcome("fCCS_capacity_vec"),
        ArrayOutcome("BECCS_capacity_vec"),
        ArrayOutcome("DACCS_capacity_vec"),
        ArrayOutcome("CTBO_cost_lev_vec"),
        # Plant-level NPV outcomes (fixed order, NaN for non-invested plants)
        ArrayOutcome("plant_npv_net"),          # [kEUR] Net NPV per plant
        ArrayOutcome("plant_npv_gross"),        # [kEUR] Gross NPV per plant
        ArrayOutcome("plant_investment_year"),  # [year] Investment year per plant
    ]

    # Constants: fixed parameters for all runs
    model.constants = [
        Constant("data", data),
        Constant("CTBO_ENABLED", True),

        Constant("CTBO_growth_factor", 0.4),
        Constant("HALF", False),
        Constant("USE_FOAK", False),

        Constant("VERBOSE", False),
        Constant("debug", False),
        Constant("ETS_LINEAR", True),
        Constant("ets_linear_start", 45),
        Constant("START_YEAR", 2025),
        Constant("END_YEAR", 2055),
        Constant("DIFFUSE_TARGET_YEAR", 2050),
        Constant("DIFFUSE_START_FRACTION", 1.0),
        Constant("DISCOUNT_RATE", 0.035),
        Constant("USE_INVESTMENT_YEAR_AS_BASE", False),
        Constant("NETL", 5.509),
        Constant("CEPCI_2023", 798.7),
        Constant("sek_to_eur", 0.091),
        Constant("pounds_to_EUR", 1.15),
        Constant("evaporation_enthalpy", 2257),
        Constant("qsteam", 0.15),
        Constant("qelc", 0.30),
        Constant("qchp", 0.55),
        Constant("elc_eff", 0.33),
    ]

    n_scenarios = 30
    n_policies = 10

    results = perform_experiments(
        model, 
        n_scenarios, 
        n_policies, 
        uncertainty_sampling=Samplers.LHS, 
        lever_sampling=Samplers.LHS
    )
    experiments, outcomes = results

    # Separate scalar and array outcomes
    scalar_outcomes = {k: v for k, v in outcomes.items() if v.ndim == 1}
    array_outcomes = {k: v for k, v in outcomes.items() if v.ndim > 1}
    
    # Save scalar outcomes as DataFrame
    scalar_df = pd.DataFrame(scalar_outcomes)
    combined_df = pd.concat([experiments, scalar_df], axis=1)
    combined_df.to_csv("experiments.csv", index=False)
    
    # Save array outcomes as .npy files
    for name, arr in array_outcomes.items():
        np.save(f"outcomes_{name}.npy", arr)
    
    # Save plant names reference (same for all runs, run once to get the list)
    # Use same HALF setting as the experiments
    test_result = simulate_ctbo(data, HALF=False, VERBOSE=False, debug=False)
    plant_names = test_result['plant_names']
    pd.DataFrame({'plant_index': range(len(plant_names)), 'plant_name': plant_names}).to_csv(
        "plant_names_reference.csv", index=False
    )
    
    print(f"\nCompleted {len(experiments)} experiments")
    print(f"Experiments + scalar outcomes saved to: experiments.csv")
    print(f"Array outcomes saved to: outcomes_*.npy files")
    print(f"Plant names reference saved to: plant_names_reference.csv")
    print(f"\nRun 'python plot_results.py' to generate plots")