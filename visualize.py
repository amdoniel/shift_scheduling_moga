# Contains functions to plot the results of the MOGA optimization.
# Includes Pareto front, fairness analysis, and dataset interpretation visuals.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict

def plot_pareto_front(results_df, save_path=None):
    if results_df.empty or 'ShiftsUsed' not in results_df or 'Fairness' not in results_df:
        print("[plot_pareto_front] Invalid or empty results_df.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(results_df['ShiftsUsed'], results_df['Fairness'], alpha=0.7, color='blue', label="Feasible Solutions")
    plt.title("Pareto Front: Shift Minimization vs Workload Fairness")
    plt.xlabel("Number of Shifts Used")
    plt.ylabel("Workload Imbalance (Std Dev)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_shift_distribution(chromosome, num_jobs, save_path=None):
    if not chromosome or not isinstance(chromosome, list):
        print("[plot_shift_distribution] Invalid chromosome input.")
        return

    assignment = {}
    for job_id, worker_id in enumerate(chromosome):
        if worker_id not in assignment:
            assignment[worker_id] = []
        assignment[worker_id].append(job_id)

    workers = list(assignment.keys())
    task_counts = [len(assignment[w]) for w in workers]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=workers, y=task_counts, palette="viridis")
    plt.title("Task Distribution per Worker")
    plt.xlabel("Worker ID")
    plt.ylabel("Number of Tasks")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_dataset_summary(jobs_df, worker_df, save_path=None):
    if jobs_df.empty or worker_df.empty:
        print("[plot_dataset_summary] jobs_df or worker_df is empty.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in jobs_df.iterrows():
        ax.plot([row['Start'], row['End']], [i, i], color='skyblue')
    ax.set_title("Job Timings (Start-End)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Job Index")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    qual_matrix = np.zeros((len(worker_df), jobs_df.shape[0]))
    for i, row in worker_df.iterrows():
        for job_id in row['QualifiedJobs']:
            if job_id < jobs_df.shape[0]:
                qual_matrix[i, job_id] = 1

    plt.figure(figsize=(12, 6))
    sns.heatmap(qual_matrix, cmap="Blues", cbar=True)
    plt.title("Worker-Job Qualification Matrix")
    plt.xlabel("Job Index")
    plt.ylabel("Worker Index")
    plt.tight_layout()
    if save_path:
        heatmap_path = save_path.replace("dataset_summary", "qualification_heatmap")
        plt.savefig(heatmap_path)
        plt.close()
    else:
        plt.show()

def dominates(f1, f2):
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

def get_pareto_front_indices(fitnesses):
    pareto_indices = []
    for i, f1 in enumerate(fitnesses):
        if not any(dominates(f2, f1) for j, f2 in enumerate(fitnesses) if i != j):
            pareto_indices.append(i)
    return pareto_indices

def plot_enhanced_pareto_front(df, save_path=None):
    df["Feasible"] = df["ShiftsUsed"] < 1000
    pareto_indices = get_pareto_front_indices(df[["ShiftsUsed", "Fairness"]].values.tolist())
    df["Pareto"] = False
    df.loc[pareto_indices, "Pareto"] = True

    plt.figure(figsize=(10, 7))
    feasible = df[df["Feasible"] & ~df["Pareto"]]
    pareto = df[df["Pareto"]]

    plt.scatter(feasible["ShiftsUsed"], feasible["Fairness"], color='skyblue', label="Feasible Points", alpha=0.7)
    plt.scatter(pareto["ShiftsUsed"], pareto["Fairness"], color='green', label="Pareto Front", edgecolor='black', s=80)
    plt.title("Enhanced Pareto Front: Shift Minimization vs Workload Fairness")
    plt.xlabel("Objective 1: Number of Shifts Used")
    plt.ylabel("Objective 2: Workload Imbalance (Std Dev)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_worker_schedule(chromosome, jobs_df, worker_id, save_path=None):
    jobs = [(job_id, jobs_df.loc[job_id, "Start"], jobs_df.loc[job_id, "End"])
            for job_id, w in enumerate(chromosome) if w == worker_id]

    if not jobs:
        print(f"No jobs assigned to worker {worker_id}.")
        return

    jobs = sorted(jobs, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, 4))
    for job_id, start, end in jobs:
        ax.plot([start, end], [1, 1], linewidth=4)
        ax.text((start + end) / 2, 1.05, str(job_id), ha='center', va='bottom', fontsize=8)

    ax.set_title(f"Schedule for Worker {worker_id}")
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_xlim(left=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_all_worker_schedules(chromosome, jobs_df, save_path=None):
    assignments = {}
    for job_id, worker_id in enumerate(chromosome):
        if worker_id not in assignments:
            assignments[worker_id] = []
        assignments[worker_id].append((job_id, jobs_df.loc[job_id, "Start"], jobs_df.loc[job_id, "End"]))

    fig, ax = plt.subplots(figsize=(12, 8))
    yticks, ylabels = [], []

    for idx, (worker_id, jobs) in enumerate(sorted(assignments.items())):
        for job_id, start, end in sorted(jobs, key=lambda x: x[1]):
            ax.plot([start, end], [idx, idx], linewidth=4, label=f"Job {job_id}")
            ax.text((start + end) / 2, idx + 0.1, str(job_id), ha='center', va='bottom', fontsize=7)
        yticks.append(idx)
        ylabels.append(str(worker_id))

    ax.set_title("Schedules for All Workers")
    ax.set_xlabel("Time")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_worker_time_balance(chromosome, jobs_df, save_path=None):
    worker_durations = defaultdict(int)
    worker_gaps = defaultdict(int)
    worker_jobs = defaultdict(list)

    for job_id, worker_id in enumerate(chromosome):
        start = jobs_df.loc[job_id, "Start"]
        end = jobs_df.loc[job_id, "End"]
        worker_durations[worker_id] += (end - start)
        worker_jobs[worker_id].append((start, end))

    for worker_id, jobs in worker_jobs.items():
        jobs.sort()
        for i in range(1, len(jobs)):
            worker_gaps[worker_id] += max(0, jobs[i][0] - jobs[i - 1][1])

    activity_df = pd.DataFrame({
        "Worker": list(worker_durations.keys()),
        "Active Time": list(worker_durations.values()),
        "Idle Time": [worker_gaps.get(w, 0) for w in worker_durations.keys()]
    })

    x = np.arange(len(activity_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, activity_df["Active Time"], width, label="Active Time")
    ax.bar(x + width/2, activity_df["Idle Time"], width, label="Idle Time")

    ax.set_xticks(x)
    ax.set_xticklabels(activity_df["Worker"], rotation=45)
    ax.set_xlabel("Worker ID")
    ax.set_ylabel("Time")
    ax.set_title("Worker Active vs Idle Time")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

