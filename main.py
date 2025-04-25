import os
import pandas as pd
from collections import Counter
from data_loader import load_dataset
from moga import nsga2_fixed
from visualize import (
    plot_enhanced_pareto_front,
    plot_pareto_front,
    plot_shift_distribution,
    plot_dataset_summary,
    plot_worker_schedule,
    plot_all_worker_schedules
)
import matplotlib.pyplot as plt

# Configurable parameters
DATA_FOLDER = "data"
OUTPUT_FOLDER = "results"
POPULATION_SIZE = 100
GENERATIONS = 50
PENALTY_WEIGHT = 200
MUTATION_RATE = 0.1
IDLE_TIME_WEIGHT = 0.01

# Ensure results folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def dominates(f1, f2):
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

def get_pareto_front_indices(fitnesses):
    pareto_indices = []
    for i, f1 in enumerate(fitnesses):
        if not any(dominates(f2, f1) for j, f2 in enumerate(fitnesses) if i != j):
            pareto_indices.append(i)
    return pareto_indices

def main():
    dataset_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".dat")]
    metrics_summary = []

    for file in dataset_files:
        dataset_name = file.replace(".dat", "")
        path = os.path.join(DATA_FOLDER, file)
        print(f"\nRunning optimizer on: {file}")

        # Load data
        jobs_df, worker_df, metadata = load_dataset(path)
        num_jobs = metadata['num_jobs']

        # Run optimizer
        final_population, final_fitnesses = nsga2_fixed(
            worker_df=worker_df,
            num_jobs=num_jobs,
            pop_size=POPULATION_SIZE,
            generations=GENERATIONS,
            penalty_weight=PENALTY_WEIGHT,
            mutation_rate=MUTATION_RATE,
            jobs_df=jobs_df,
            idle_time_weight=IDLE_TIME_WEIGHT
        )

        # Save results
        results = pd.DataFrame(final_fitnesses, columns=["ShiftsUsed", "Fairness"])
        results["Chromosome"] = final_population
        results_file = os.path.join(OUTPUT_FOLDER, f"results_{dataset_name}.csv")
        results.to_csv(results_file, index=False)
        print(f"Saved results to: {results_file}")

        # Calculate Metrics
        feasible_count = sum(results["ShiftsUsed"] < 1000)
        feasibility_rate = feasible_count / len(results)
        pareto_indices = get_pareto_front_indices(results[["ShiftsUsed", "Fairness"]].values.tolist())
        fairness_spread = results["Fairness"].max() - results["Fairness"].min()

        metrics_summary.append({
            "Dataset": file,
            "Feasible (%)": round(feasibility_rate * 100, 2),
            "Pareto Front Size": len(pareto_indices),
            "Fairness Spread": round(fairness_spread, 3)
        })

        # Visualize and save plots
        print("Generating visualizations...")

        plot_enhanced_pareto_front(results, save_path=os.path.join(OUTPUT_FOLDER, f"enhanced_pareto_{dataset_name}.png"))
        # plot_pareto_front(results, save_path=os.path.join(OUTPUT_FOLDER, f"pareto_front_{dataset_name}.png"))
        plot_shift_distribution(results.iloc[0]["Chromosome"], num_jobs, save_path=os.path.join(OUTPUT_FOLDER, f"shift_distribution_{dataset_name}.png"))
        plot_dataset_summary(jobs_df, worker_df, save_path=os.path.join(OUTPUT_FOLDER, f"dataset_summary_{dataset_name}.png"))

        # Auto-detect most used worker and plot their schedule
        chromosome = results.iloc[0]["Chromosome"]
        most_used_worker = Counter(chromosome).most_common(1)[0][0]
        plot_worker_schedule(
            chromosome,
            jobs_df,
            worker_id=most_used_worker,
            save_path=os.path.join(OUTPUT_FOLDER, f"worker{most_used_worker}_schedule_{dataset_name}.png")
        )

        # Plot all worker schedules
        plot_all_worker_schedules(
            chromosome,
            jobs_df,
            save_path=os.path.join(OUTPUT_FOLDER, f"all_worker_schedules_{dataset_name}.png")
        )

        print(f"Visualizations saved for dataset: {dataset_name}")

    # Save all metrics at once
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(os.path.join(OUTPUT_FOLDER, "metrics_summary.csv"), index=False)
    print("Saved summary metrics to metrics_summary.csv")


if __name__ == "__main__":
    main()
