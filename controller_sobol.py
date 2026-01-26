import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from ema_workbench.em_framework import get_SALib_problem
from SALib.analyze import sobol

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
        RealParameter("cgas", 30, 80),  # [€/MWh]
        RealParameter("celc", 150, 300),  # [€/MWh]
        RealParameter("cpellets", 150, 250),  # [€/t]
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
        RealParameter("discount_rate_ccs", 0.05, 0.10),
        IntegerParameter("lifetime_ccs", 20, 30),  # [years]
        RealParameter("CEPCI_2025", 880, 980),
        RealParameter("NETL_2025", 5.0, 6.0),

        # Policy uncertainties
        CategoricalParameter("ETS_SCENARIO", ['£200', '£300', '£400']),
        RealParameter("DIFFUSE_END_FRACTION", 0.05, 0.40),
        
        CategoricalParameter("ASSUME_FOAK", [False, True]),
    ]

    # # Levers: policy choices we can control
    # model.levers = [
    #     CategoricalParameter("ETS_SCENARIO", ['£200', '£300', '£400']),
    #     RealParameter("DIFFUSE_END_FRACTION", 0.05, 0.40),
    # ]

    # Outcomes: metrics to track
    model.outcomes = [
        # Scalar outcomes
        ScalarOutcome("gas_increase_2040"),
        ScalarOutcome("year_DACCS_marginal"),
        ScalarOutcome("NPV_cost_CTBO"),
        ScalarOutcome("NPV_profit_CTBO"),
        ScalarOutcome("benefit2cost_CTBO"),
        ScalarOutcome("NPV_cost_ETS"),
        ScalarOutcome("NPV_profit_ETS"),
        ScalarOutcome("benefit2cost_ETS"),
        # Time series outcomes
        ArrayOutcome("supply_ktCO2f"),
        ArrayOutcome("emitted_ktCO2f"),
        ArrayOutcome("mandate_ktCO2"),
        ArrayOutcome("stored_ktCO2g"),
        ArrayOutcome("stored_ktCO2b"),
        ArrayOutcome("stored_ktCO2daccs"),
        ArrayOutcome("cost_marginal"),
        ArrayOutcome("price_ETS"),
        ArrayOutcome("price_CSU"),
        ArrayOutcome("cost_CTBO_producers"),
        ArrayOutcome("cost_CSU_embedded"),
        ArrayOutcome("cost_CTBO_policy"),
        ArrayOutcome("profit_CTBO_policy"),
        ArrayOutcome("cost_ETS_policy"),
        ArrayOutcome("profit_ETS_policy"),
        ArrayOutcome("gas_increase_abs"),
        ArrayOutcome("gas_increase_pct"),
        # Plant-level outcomes (alphabetically ordered)
        ArrayOutcome("plants_investment_year"),
        ArrayOutcome("plants_NPV_CSU"),
        ArrayOutcome("plants_NPV_total"),
        ArrayOutcome("plants_NPV_ETS"),
        ArrayOutcome("plants_ktCO2tot_ccs"),
    ]

    # Constants: fixed parameters for all runs
    model.constants = [
        Constant("plants_clean", plants_clean),
        Constant("transport_hubs", transport_hubs),
        Constant("single_run", False),
        Constant("CTBO_QUADRATIC", 0.4),
        Constant("DISCOUNT_RATE", 0.035),
        Constant("FOAK_CALIBRATION", 1.6379),
        Constant("ETS_START", 45),
        Constant("START_YEAR", 2025),
        Constant("END_YEAR", 2055),
        Constant("DIFFUSE_END_YEAR", 2050),
        Constant("pounds_to_EUR", 1.15),
        Constant("DEFOSSILIZE", False),
    ]

    # Run experiments
    n_scenarios = 10
    n_policies = 0
    
    results = perform_experiments(
        model, 
        n_scenarios, 
        n_policies, 
        uncertainty_sampling=Samplers.SOBOL, 
        lever_sampling=Samplers.SOBOL
    )
    experiments, outcomes = results

    def analyze(results, ooi):
        """analyze results using SALib sobol, returns a dataframe"""
        _, outcomes = results

        problem = get_SALib_problem(model.uncertainties)
        y = outcomes[ooi]
        sobol_indices = sobol.analyze(problem, y)
        sobol_stats = {key: sobol_indices[key] for key in ["ST", "ST_conf", "S1", "S1_conf"]}
        sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
        sobol_stats.sort_values(by="ST", ascending=False)
        s2 = pd.DataFrame(sobol_indices["S2"], index=problem["names"], columns=problem["names"])
        s2_conf = pd.DataFrame(
            sobol_indices["S2_conf"], index=problem["names"], columns=problem["names"]
        )
        return sobol_stats, s2, s2_conf, problem
    sobol_stats, s2, s2_conf, problem = analyze(results, "gas_increase_2040")

    print(sobol_stats)
    print(s2)
    print(s2_conf)
    sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
    sobol_stats.to_csv("results/sobol_stats.csv")
    sobol_stats_sorted = sobol_stats.sort_values(by="ST", ascending=False)  # Ascending for better readability

    # Create horizontal bar plot
    plt.figure(figsize=(8, 10))  # Adjust figure size for better layout
    sns.barplot(
        y=sobol_stats_sorted.index,  # Parameters on y-axis
        x=sobol_stats_sorted["ST"],  # Sobol indices on x-axis
        xerr=sobol_stats_sorted["ST_conf"],  # Confidence intervals as error bars
        capsize=0.2,
        color="crimson"
    )
    plt.ylabel("Parameter")
    plt.xlabel("Total Sobol Index (ST)")
    plt.title("Total-Order Sobol Indices with Confidence Intervals")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()