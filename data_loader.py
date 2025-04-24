import pandas as pd
from collections import defaultdict

def load_dataset(filepath):
 
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Remove comments and blank lines
    clean_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    # Read basic problem metadata
    problem_type = int(clean_lines[0].split('=')[-1].strip())  # Should be 1
    num_jobs = int(clean_lines[1].split('=')[-1].strip())

    # Read job start and end times
    job_lines = clean_lines[2:2 + num_jobs]
    jobs = [list(map(int, line.split())) for line in job_lines]
    jobs_df = pd.DataFrame(jobs, columns=["Start", "End"])

    # Locate and read qualifications section
    qual_line_index = 2 + num_jobs
    num_qualifications = int(clean_lines[qual_line_index].split('=')[-1].strip())

    # Parse worker qualifications
    qualification_lines = clean_lines[qual_line_index + 1:]
    worker_skills = defaultdict(list)

    for line in qualification_lines:
        if ':' in line:
            worker_id, skills_str = line.split(':', 1)
            skills = list(map(int, skills_str.strip().split()))
            worker_skills[int(worker_id.strip())] = skills

    worker_df = pd.DataFrame(worker_skills.items(), columns=["WorkerID", "QualifiedJobs"])

    # Meta-info dictionary
    metadata = {
        'problem_type': problem_type,
        'num_jobs': num_jobs,
        'num_workers': len(worker_skills),
        'num_qualifications': num_qualifications
    }

    return jobs_df, worker_df, metadata
