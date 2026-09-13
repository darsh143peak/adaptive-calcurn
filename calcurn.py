import simpy
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. INFRASTRUCTURE & WORKLOAD DEFINITIONS
# ==========================================
INFRASTRUCTURE = {
    "TELUM_ON_CHIP":      {"speed": 2.5, "vram_limit_gb": 32, "cost_per_sec": 0.01, "interconnect": "DIRECT"},
    "GPU_NODE_1_NVLINK": {"speed": 4.5, "vram_limit_gb": 80, "cost_per_sec": 0.04, "interconnect": "NVLINK"},
    "GPU_NODE_2_PCIE":   {"speed": 3.0, "vram_limit_gb": 40, "cost_per_sec": 0.02, "interconnect": "PCIE"},
    "GPU_NODE_3_PCIE":   {"speed": 2.8, "vram_limit_gb": 40, "cost_per_sec": 0.02, "interconnect": "PCIE"}
}
NODE_KEYS = list(INFRASTRUCTURE.keys())
NUM_NODES = len(NODE_KEYS)

WORKLOAD_TYPES = {
    "LATENCY_CRITICAL": {"sla_target_s": 10.0, "sla_weight": 150.0, "preemption_penalty_s": 0.0},
    "HEAVY_AI":         {"sla_target_s": 25.0, "sla_weight": 30.0,  "preemption_penalty_s": 2.0},
    "BATCH_LOGS":       {"sla_target_s": 50.0, "sla_weight": 5.0,   "preemption_penalty_s": 0.5}
}

# ==========================================
# 2. CONCRETE PSO FITNESS & SOLVER
# ==========================================
def evaluate_schedule(schedule, active_jobs, node_est_free_times, current_time):
    node_timelines = copy.deepcopy(node_est_free_times)
    total_jct, total_wait, total_cost = 0.0, 0.0, 0.0
    sla_penalty_score = 0.0
    
    for job_idx, node_idx in enumerate(schedule):
        job = active_jobs[job_idx]
        node_name = NODE_KEYS[node_idx]
        node_spec = INFRASTRUCTURE[node_name]
        
        interconnect_mult = 1.0 if node_spec["interconnect"] in ["DIRECT", "NVLINK"] else 1.30
        exec_time = (job["base_compute_units"] / node_spec["speed"]) * interconnect_mult + job["preemption_penalty_s"]
        
        start_time = max(current_time, node_timelines[node_name])
        wait_time = start_time - job["arrival_time"]
        completion_time = start_time + exec_time
        node_timelines[node_name] = completion_time
        
        total_jct += (completion_time - job["arrival_time"])
        total_wait += wait_time
        total_cost += exec_time * node_spec["cost_per_sec"]
        
        # Apply SLA-Urgency-Weighted Penalty
        if (completion_time - job["arrival_time"]) > job["sla_target_s"]:
            sla_penalty_score += job["sla_weight"]

    n = max(1, len(active_jobs))
    mean_jct = total_jct / n
    mean_wait = total_wait / n
    
    fitness = mean_jct + (mean_wait * 2.0) + (total_cost * 1.5) + sla_penalty_score
    return fitness

def solve_adaptive_calurn(active_jobs, node_est_free_times, current_time, num_particles=30, max_iter=30):
    if not active_jobs:
        return []
    
    num_jobs = len(active_jobs)
    particles = []
    
    for _ in range(num_particles):
        pos = [random.uniform(0, NUM_NODES - 1) for _ in range(num_jobs)]
        vel = [random.uniform(-0.5, 0.5) for _ in range(num_jobs)]
        sched = [int(round(p)) for p in pos]
        score = evaluate_schedule(sched, active_jobs, node_est_free_times, current_time)
        particles.append({"pos": pos, "vel": vel, "sched": sched, "score": score, "pbest_pos": copy.deepcopy(pos), "pbest_score": score})

    gbest = min(particles, key=lambda x: x["score"])
    gbest_pos = copy.deepcopy(gbest["pos"])

    for iteration in range(max_iter):
        w = 0.9 - (0.5 * (iteration / max_iter))
        for p in particles:
            for j in range(num_jobs):
                r1, r2 = random.random(), random.random()
                cog = 1.5 * r1 * (p["pbest_pos"][j] - p["pos"][j])
                soc = 2.0 * r2 * (gbest_pos[j] - p["pos"][j])
                p["vel"][j] = (w * p["vel"][j]) + cog + soc
                p["pos"][j] = max(0, min(NUM_NODES - 1, p["pos"][j] + p["vel"][j]))
            
            p["sched"] = [int(round(val)) for val in p["pos"]]
            p["score"] = evaluate_schedule(p["sched"], active_jobs, node_est_free_times, current_time)
            
            if p["score"] < p["pbest_score"]:
                p["pbest_pos"] = copy.deepcopy(p["pos"])
                p["pbest_score"] = p["score"]
                
                if p["score"] < gbest["score"]:
                    gbest_pos = copy.deepcopy(p["pos"])
                    gbest["score"] = p["score"]

    return [int(round(val)) for val in gbest_pos]

# ==========================================
# 3. DISCRETE EVENT SIMULATION
# ==========================================
class ClusterSimulator:
    def __init__(self, env, mode="CALURN"):
        self.env = env
        self.mode = mode
        self.job_queue = []
        self.completed_jobs = []
        self.node_resources = {k: simpy.Resource(env, capacity=1) for k in NODE_KEYS}
        self.node_est_free_times = {k: 0.0 for k in NODE_KEYS}

    def run_job(self, job, node_name):
        node_spec = INFRASTRUCTURE[node_name]
        interconnect_mult = 1.0 if node_spec["interconnect"] in ["DIRECT", "NVLINK"] else 1.30
        exec_duration = (job["base_compute_units"] / node_spec["speed"]) * interconnect_mult + job["preemption_penalty_s"]

        with self.node_resources[node_name].request() as req:
            yield req
            start_time = self.env.now
            wait_time = start_time - job["arrival_time"]
            
            yield self.env.timeout(exec_duration)
            
            jct = self.env.now - job["arrival_time"]
            cost = exec_duration * node_spec["cost_per_sec"]
            
            self.completed_jobs.append({
                "id": job["id"],
                "node": node_name,
                "wait_time": wait_time,
                "jct": jct,
                "cost": cost,
                "sla_breached": jct > job["sla_target_s"]
            })

    def job_generator(self, total_jobs=30):
        for i in range(total_jobs):
            category = random.choice(list(WORKLOAD_TYPES.keys()))
            job = {
                "id": f"JOB-{i+1:03d}",
                "category": category,
                "arrival_time": self.env.now,
                "base_compute_units": random.randint(15, 40),
                "vram_req_gb": random.choice([4, 8, 16]),
                "sla_target_s": WORKLOAD_TYPES[category]["sla_target_s"],
                "sla_weight": WORKLOAD_TYPES[category]["sla_weight"],
                "preemption_penalty_s": WORKLOAD_TYPES[category]["preemption_penalty_s"]
            }
            self.job_queue.append(job)
            
            # --- SCHEDULING SELECTION ---
            if self.mode == "ROUND_ROBIN":
                target_node_idx = i % NUM_NODES
            else:  # Adaptive CALURN (PSO)
                schedule = solve_adaptive_calurn(self.job_queue, self.node_est_free_times, self.env.now)
                target_node_idx = schedule[0]
            
            target_node = NODE_KEYS[target_node_idx]
            dispatched_job = self.job_queue.pop(0)
            
            node_spec = INFRASTRUCTURE[target_node]
            interconnect_mult = 1.0 if node_spec["interconnect"] in ["DIRECT", "NVLINK"] else 1.30
            exec_duration = (dispatched_job["base_compute_units"] / node_spec["speed"]) * interconnect_mult + dispatched_job["preemption_penalty_s"]
            
            start_est = max(self.env.now, self.node_est_free_times[target_node])
            self.node_est_free_times[target_node] = start_est + exec_duration
            
            self.env.process(self.run_job(dispatched_job, target_node))
            yield self.env.timeout(random.uniform(0.4, 1.2))

def run_experiment(mode, seed=300, total_jobs=30):
    random.seed(seed)
    env = simpy.Environment()
    cluster = ClusterSimulator(env, mode=mode)
    env.process(cluster.job_generator(total_jobs=total_jobs))
    env.run()
    
    df = pd.DataFrame(cluster.completed_jobs)
    avg_jct = df["jct"].mean()
    avg_wait = df["wait_time"].mean()
    sla_breach_rate = (df["sla_breached"].sum() / len(df)) * 100
    total_cost = df["cost"].sum()
    
    return {
        "Mode": mode,
        "Avg JCT (s)": round(avg_jct, 2),
        "Avg Wait Time (s)": round(avg_wait, 2),
        "SLA Violation (%)": round(sla_breach_rate, 1),
        "Total Cost ($)": round(total_cost, 2)
    }

# ==========================================
# 4. EXECUTE & PLOT
# ==========================================
if __name__ == "__main__":
    print("Running Concrete Benchmark Experiments...\n")
    
    rr_results = run_experiment(mode="ROUND_ROBIN", seed=300, total_jobs=30)
    pso_results = run_experiment(mode="CALURN", seed=300, total_jobs=30)
    
    results_df = pd.DataFrame([rr_results, pso_results])
    print("--- MONDAY BENCHMARK RESULTS ---")
    print(results_df.to_string(index=False))
    
    metrics = ["Avg JCT (s)", "Avg Wait Time (s)", "SLA Violation (%)", "Total Cost ($)"]
    
    plot_df = pd.DataFrame({
        "Round-Robin Baseline": [rr_results[m] for m in metrics],
        "Adaptive CALURN (PSO)": [pso_results[m] for m in metrics]
    }, index=metrics)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", rot=0, ax=ax, color=["#e74c3c", "#2ecc71"])
    plt.title("GPU Cluster Scheduling Performance: Baseline vs. Adaptive CALURN")
    plt.ylabel("Value")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("monday_benchmark_results.png")
    print("\nBenchmark graph saved as 'monday_benchmark_results.png'")