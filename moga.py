# NSGA-II optimizer with tunable mutation rate and penalty weight.

import random
import numpy as np
from representation import generate_initial_population
from fitness_functions import penalized_fitness

def dominates(f1, f2):
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

def get_pareto_front_indices(fitnesses):
    pareto_indices = []
    for i, f1 in enumerate(fitnesses):
        if not any(dominates(f2, f1) for j, f2 in enumerate(fitnesses) if i != j):
            pareto_indices.append(i)
    return pareto_indices

def mutation(chromosome, worker_df, mutation_rate=0.05):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            job_id = i
            qualified_workers = [row.WorkerID for _, row in worker_df.iterrows() if job_id in row.QualifiedJobs]
            if qualified_workers:
                chromosome[i] = random.choice(qualified_workers)
    return chromosome

def nsga2_fixed(worker_df, num_jobs, pop_size=100, generations=50, penalty_weight=1000, mutation_rate=0.05, jobs_df=None, idle_time_weight=0.001):
    population = generate_initial_population(worker_df, num_jobs, pop_size)
    fitnesses = []
    for ind in population:
        f1, f2, idle = penalized_fitness(ind, worker_df, num_jobs, penalty_weight, jobs_df, idle_time_weight)
        fitnesses.append((f1, f2, idle))

    for _ in range(generations):
        offspring = []
        while len(offspring) < pop_size:
            if len(population) < 2:
                population = generate_initial_population(worker_df, num_jobs, pop_size)
                fitnesses = []
                for ind in population:
                    f1, f2, idle = penalized_fitness(ind, worker_df, num_jobs, penalty_weight, jobs_df, idle_time_weight)
                    fitnesses.append((f1, f2, idle))
            p1, p2 = random.sample(population, 2)
            point = random.randint(1, num_jobs - 2)
            c1 = p1[:point] + p2[point:]
            c2 = p2[:point] + p1[point:]
            offspring.extend([mutation(c1, worker_df, mutation_rate), mutation(c2, worker_df, mutation_rate)])

        combined = population + offspring
        combined_fitnesses = []
        for ind in combined:
            f1, f2, idle = penalized_fitness(ind, worker_df, num_jobs, penalty_weight, jobs_df, idle_time_weight)
            combined_fitnesses.append((f1, f2, idle))

        front_indices = get_pareto_front_indices([f[:2] for f in combined_fitnesses])

        if len(front_indices) < pop_size:
            selected_indices = front_indices + random.sample(range(len(combined)), pop_size - len(front_indices))
        else:
            selected_indices = front_indices[:pop_size]

        population = [combined[i] for i in selected_indices]
        fitnesses = [combined_fitnesses[i] for i in selected_indices]

    return population, fitnesses
