#Defineing the multi-objective fitness evaluation for the MOGA.
# Includes:
#   Objective 1: Minimize number of active shifts (unique workers used)
#   Objective 2: Minimize workload imbalance (std dev of task counts)
#   Constraint penalty
#   Idle time minimization and tracking

import numpy as np

def evaluate_chromosome(chromosome, worker_df, num_jobs):
    worker_to_jobs = {}
    worker_qualified_map = {
        row.WorkerID: set(row.QualifiedJobs) for _, row in worker_df.iterrows()
    }
    num_violations = 0

    for job_id, worker_id in enumerate(chromosome):
        if worker_id not in worker_to_jobs:
            worker_to_jobs[worker_id] = []
        worker_to_jobs[worker_id].append(job_id)
        if worker_id not in worker_qualified_map or job_id not in worker_qualified_map[worker_id]:
            num_violations += 1

    num_shifts_used = len([wid for wid in worker_to_jobs if wid != -1])
    workloads = [len(jobs) for wid, jobs in worker_to_jobs.items() if wid != -1]
    workload_std_dev = np.std(workloads) if workloads else float('inf')

    return num_shifts_used, workload_std_dev, num_violations

def compute_idle_time(chromosome, jobs_df):
    worker_jobs = {}
    for job_id, worker_id in enumerate(chromosome):
        if worker_id not in worker_jobs:
            worker_jobs[worker_id] = []
        start = jobs_df.loc[job_id, "Start"]
        end = jobs_df.loc[job_id, "End"]
        worker_jobs[worker_id].append((start, end))

    idle_time = 0
    for jobs in worker_jobs.values():
        if len(jobs) < 2:
            continue
        jobs = sorted(jobs, key=lambda x: x[0])
        for i in range(1, len(jobs)):
            idle_time += max(0, jobs[i][0] - jobs[i - 1][1])
    return idle_time

def penalized_fitness(chromosome, worker_df, num_jobs, penalty_weight=1000, jobs_df=None, idle_time_weight=0.001):
    f1, f2, violations = evaluate_chromosome(chromosome, worker_df, num_jobs)
    penalty = violations * penalty_weight
    idle_time = compute_idle_time(chromosome, jobs_df) if jobs_df is not None else 0
    f1_penalized = f1 + penalty + idle_time_weight * idle_time
    return f1_penalized, f2 + penalty, idle_time
