import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ctbo import simulate_ctbo
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    ScalarOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments,
)
from ema_workbench.em_framework import get_SALib_problem
from SALib.analyze import sobol


POLICIES = ["CTBO-only", "£100-Mix", "£200-Mix", "£300-Mix", "ETS-eq"]


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


def _build_model(plants_clean, transport_hubs, phaseout=False, policy="CTBO-only", debug=False):
    if debug:
        print(f"_build_model input: phaseout={phaseout}, policy={policy}")
    model = Model("UKCTBOsobol", function=simulate_ctbo)
    model.uncertainties = [
        RealParameter("DIFFUSE_END_FRACTION", 0.05, 0.50),
        CategoricalParameter("DACCS_SCENARIO", ["£322", "£391"]),
        RealParameter("fraction_limestone", 0.55, 0.65),
        RealParameter("fraction_fossil_waste", 0.40, 0.55),
        RealParameter("drax_efficiency", 40, 46),
        RealParameter("drax_efficiency_loss", 0.20, 0.28),
        RealParameter("capture_rate", 0.90, 0.95),
        RealParameter("qreb", 2.8, 3.8),
        RealParameter("pcomp", 0.20, 0.35),
        RealParameter("pcomp_liquefy", 0.28, 0.38),
        RealParameter("FLH_industry", 8000, 8500),
        RealParameter("FLH_waste", 7200, 8000),
        RealParameter("ccgt_efficiency", 0.45, 0.55),
        RealParameter("ccgt_efficiency_loss", 0.10, 0.15),
        RealParameter("cgas", 10, 100),
        RealParameter("celc", 150, 300),
        RealParameter("cstraw", 100, 200),
        RealParameter("cliquefy", 5, 10),
        RealParameter("cHCN", 4, 8),
        RealParameter("cHRSG", 8, 15),
        RealParameter("camine", 2, 6),
        RealParameter("cstorage", 20, 30),
        RealParameter("transport_uncertainty", 0.0, 0.30),
        RealParameter("CAPEX_gasboiler", 0.10, 0.20),
        RealParameter("CAPEX_bioboiler", 0.80, 1.20),
        RealParameter("fixate_CAPEX", 0.02, 0.05),
        RealParameter("CAPEX_m", 0.80, 0.90),
        RealParameter("discount_rate_ccs", 0.06, 0.12),
        IntegerParameter("lifetime_ccs", 20, 30),
        RealParameter("CEPCI_2025", 880, 980),
        RealParameter("NETL_2025", 5.0, 6.0),
    ]
    model.outcomes = [ScalarOutcome("NPV_costs_consumers")]
    model.constants = [
        Constant("plants_clean", plants_clean),
        Constant("transport_hubs", transport_hubs),
        Constant("single_run", False),
        Constant("PHASEOUT", phaseout),
        Constant("ETS_SCENARIO", policy),
        Constant("CTBO_QUADRATIC", 0.4),
        Constant("DISCOUNT_RATE", 0.035),
        Constant("ETS_START", 45),
        Constant("START_YEAR", 2025),
        Constant("END_YEAR", 2055),
        Constant("DIFFUSE_END_YEAR", 2050),
        Constant("pounds_to_EUR", 1.15),
        Constant("CONSTRUCTION_YEARS", 3),
    ]
    return model


def _analyze_sobol(model, results, outcome_name="NPV_costs_consumers", debug=False):
    if debug:
        print(f"_analyze_sobol input: outcome_name={outcome_name}")
    _, outcomes = results
    problem = get_SALib_problem(model.uncertainties)
    y = np.asarray(outcomes[outcome_name], dtype=float)
    indices = sobol.analyze(problem, y, calc_second_order=False)
    stats = pd.DataFrame(
        {
            "S1": indices["S1"],
            "S1_conf": indices["S1_conf"],
            "ST": indices["ST"],
            "ST_conf": indices["ST_conf"],
        },
        index=problem["names"],
    ).sort_values("ST", ascending=False)
    if debug:
        print(f"_analyze_sobol output: n_params={len(stats)}")
    return stats


def run_policy_sobol(
    results_dir="results_baseline",
    figures_dir="results_figures",
    n_scenarios=512,
    debug=False,
):
    if debug:
        print(f"run_policy_sobol input: results_dir={results_dir}, n_scenarios={n_scenarios}")
    phaseout = "phaseout" in results_dir.lower()
    plants_clean = pd.read_csv("results_baseline/plants_clean.csv")
    transport_hubs = pd.read_csv("data/transport_hubs.csv")

    all_rows = []
    fig, axes = plt.subplots(1, len(POLICIES), figsize=(4.1 * len(POLICIES), 7.0), sharey=True)
    if len(POLICIES) == 1:
        axes = [axes]

    for ax, policy in zip(axes, POLICIES):
        model = _build_model(plants_clean, transport_hubs, phaseout=phaseout, policy=policy, debug=debug)
        results = perform_experiments(
            model,
            scenarios=n_scenarios,
            uncertainty_sampling=Samplers.SOBOL,
        )
        stats = _analyze_sobol(model, results, outcome_name="NPV_costs_consumers", debug=debug)
        stats["policy"] = policy
        stats["parameter"] = stats.index
        all_rows.append(stats.reset_index(drop=True))

        top = stats.head(12).iloc[::-1]
        ax.barh(top.index, top["ST"].to_numpy(), color=plt.cm.magma(np.linspace(0.2, 0.85, len(top))))
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(top["parameter"].tolist(), fontsize=10)
        ax.set_title(policy, fontsize=14)
        ax.set_xlabel("Total Sobol index (ST)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(f"{results_dir}/sobol_npv_costs_consumers_by_policy.csv", index=False)

    fig.suptitle("Sobol sensitivity of NPV_costs_consumers by policy", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/sobol_npv_costs_consumers_by_policy.png", dpi=300, bbox_inches="tight")
    if debug:
        print(
            "run_policy_sobol output:",
            f"csv={results_dir}/sobol_npv_costs_consumers_by_policy.csv,",
            f"figure={figures_dir}/sobol_npv_costs_consumers_by_policy.png",
        )
    return summary


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    results_dir = _select_results_dir(debug=True)
    run_policy_sobol(
        results_dir=results_dir,
        figures_dir="results_figures",
        n_scenarios=1,
        debug=True,
    )