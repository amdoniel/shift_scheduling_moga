# Script that defines the representation of a solution (chromosome) for the shift scheduling problem.
# Each chromosome encodes a full task-to-worker assignment.

import random
import numpy as np

def generate_initial_population(worker_df, num_jobs, population_size):
    """
    Generates an initial population of chromosomes. Each chromosome is a list of worker IDs
    assigned to each job index, constrained to workers who are qualified.

    Parameters:
        worker_df (DataFrame): Contains worker qualifications
        num_jobs (int): Total number of jobs/tasks
        population_size (int): Number of individuals in the population

    Returns:
        List[List[int]]: Population of chromosomes
    """
    # Build a lookup table: job -> list of qualified workers
    job_to_qualified_workers = {job: [] for job in range(num_jobs)}

    for _, row in worker_df.iterrows():
        worker_id = row["WorkerID"]
        for job_id in row["QualifiedJobs"]:
            if job_id in job_to_qualified_workers:
                job_to_qualified_workers[job_id].append(worker_id)

    population = []
    for _ in range(population_size):
        chromosome = []
        for job_id in range(num_jobs):
            qualified_workers = job_to_qualified_workers[job_id]
            if qualified_workers:
                assigned_worker = random.choice(qualified_workers)
            else:
                assigned_worker = -1  # Placeholder for unqualified assignment (infeasible)
            chromosome.append(assigned_worker)
        population.append(chromosome)

    return population

def decode_chromosome(chromosome):
    """
    Converts a chromosome to a readable job assignment.

    Parameters:
        chromosome (List[int]): List of worker IDs

    Returns:
        Dict[int, List[int]]: Worker to assigned jobs mapping
    """
    assignment = {}
    for job_id, worker_id in enumerate(chromosome):
        if worker_id not in assignment:
            assignment[worker_id] = []
        assignment[worker_id].append(job_id)
    return assignment
