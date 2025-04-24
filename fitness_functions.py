# Defining the multi-objective fitness evaluation for the MOGA.
# these includes
#   Objective 1 to Minimize number of active shifts (unique workers used)
#   Objective 2 to Minimize workload imbalance (std dev of task counts)
#   Penalty for constraint violations (unqualified assignments)

import numpy as np

def evaluate_chromosome(chromosome, worker_df, num_jobs):

    worker_to_jobs = {}
    worker_qualified_map = {
        row.WorkerID: set(row.QualifiedJobs) for _, row in worker_df.iterrows()
    }

    num_violations = 0

    for job_id, worker_id in enumerate(chromosome):
        # Track assignments
        if worker_id not in worker_to_jobs:
            worker_to_jobs[worker_id] = []
        worker_to_jobs[worker_id].append(job_id)

        # Check qualification violation
        if worker_id not in worker_qualified_map or job_id not in worker_qualified_map[worker_id]:
            num_violations += 1

    # Objective 1: number of unique shifts (excluding -1 for unassigned)
    num_shifts_used = len([wid for wid in worker_to_jobs if wid != -1])

    # Objective 2: workload std dev (exclude -1)
    workloads = [len(jobs) for wid, jobs in worker_to_jobs.items() if wid != -1]
    workload_std_dev = np.std(workloads) if workloads else float('inf')

    return num_shifts_used, workload_std_dev, num_violations

def penalized_fitness(chromosome, worker_df, num_jobs, penalty_weight=1000):

    f1, f2, violations = evaluate_chromosome(chromosome, worker_df, num_jobs)
    penalty = violations * penalty_weight
    return f1 + penalty, f2 + penalty



