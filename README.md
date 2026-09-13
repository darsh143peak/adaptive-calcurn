# adaptive-calcurn

Clustered, Self-Adapting PSO for Real-Time ML Workload Scheduling IBM Z Datathon 2026 — Real-Time AI for Critical Decisions

Adaptive CALURN is a multi-objective, GPU-cluster scheduling layer for ML training and inference jobs. It sits above Kubernetes / AWS / IBM Z placement and answers the question neither of those layers is built to answer: given many competing jobs with different priorities, deadlines, and costs, what's the best overall scheduling strategy — right now?

The problem

Generic cloud schedulers (Kubernetes bin-packing, priority classes, FIFO) don't understand what makes ML workloads different from ordinary compute tasks:

Dimension	Generic Cloud Task	ML Workload (DL)
Resource scaling	Fixed N CPUs/VMs	Elastic 1–8 GPUs mid-run
Interconnect	Bandwidth-agnostic	Topology-sensitive (NVLink vs. PCIe/Ethernet)
Preemption	Instant pause / save	High-overhead checkpointing (10GB+ models)
Co-location risk	Simple CPU multi-threading	VRAM fragmentation → OOM

Meta-heuristic cloud scheduling (PSO, GA, ACO) has been published for two decades. To be worth building — and publishing — this scheduler has to model what's actually different about deep-learning workloads, not just relabel a generic scheduler.

From CALURN to Adaptive CALURN
CALURN (starting point): Clustered PSO — particles are split into clusters that search semi-independently instead of one global swarm. Tested on the Ackley function (52 particles / 100 iterations), C-PSO consistently approached the global minimum vs. ~20 for standard PSO.
Known limitation: the original paper (Nov 2023) was rejected for insufficient proofs/evidence, and "CALURN + cloud" alone isn't novel — multi-swarm PSO for cloud/fog scheduling is already heavily published.
Our contribution — Adaptive CALURN:
Cluster count and size are no longer fixed — they respond to the workload.
Continuously monitors swarm diversity and convergence.
Dynamically splits, merges, and reorganizes clusters as jobs arrive or finish.
Adapts inertia and acceleration parameters instead of using static coefficients.

Standard PSO converges to a static global best; when new ML jobs arrive or finish, the search space shifts abruptly and the swarm can't react fast enough. Adaptive CALURN is built to react.

Where it sits in the stack

Adaptive CALURN doesn't replace Kubernetes placement or AWS autoscaling — it makes the multi-objective decision that sits above them.

Step	Layer	Role
01 — Predict	ML model	Predicts each job's workload characteristics and risk profile
02 — Decide	Adaptive CALURN	Makes the multi-objective scheduling decision, sub-second
03 — Execute	Kubernetes / AWS / IBM Z	Carries out placement and capacity execution

Kubernetes decides where a job fits (bin-packing, static priority numbers, checkpoint-blind preemption). Adaptive CALURN decides which job should run, when, and how big — factoring in cost, deadline slack, checkpoint overhead, and topology, none of which a K8s priority number can express.

Objectives being balanced

Multi-Objective PSO (MOPSO) can be computationally heavy — if the solver takes 30 seconds per event tick, the scheduling overhead cancels out the efficiency gain. Adaptive CALURN targets sub-second convergence per tick, searching for the Pareto frontier across:

Job completion time — how fast does each ML job finish, end to end?
Hybrid infrastructure cost & energy footprint — what's the cloud spend under a given allocation policy?
SLA fairness — are deadline and fairness guarantees actually honored?

No solution on the frontier can improve one objective without hurting another.

Methodology: laptop-to-cluster pipeline

A trace-driven simulation, grounded by real micro-benchmarks, lets us evaluate cluster-scale behavior (8/16/32+ simulated GPUs, hundreds of jobs) without owning a cluster — then validate a smaller version on 1–2 physical GPUs.

1. Trace-driven simulation Discrete-event simulator (Python simpy or custom engine) fed by public real-world cluster traces:

Microsoft Philly trace
Alibaba GPU cluster trace
Google Borg trace

2. Physical micro-benchmarking Profile real ML workloads (ResNet, LLaMA fine-tuning, BERT, Stable Diffusion) on 1–2 physical GPUs to measure:

Checkpointing time (T_ckpt) during preemption
GPU memory overhead when sharing via NVIDIA MPS
Throughput scaling as batch size / GPU count changes

These empirical profiles ground the simulator in physical reality.

Evaluation

Reviewers expect comparison against established ML cluster schedulers — not just FIFO.

Baselines

Systems / heuristic SOTA: Tiresias (SIGCOMM '19), Gavel (OSDI '20), Pollux (OSDI '21)
Algorithmic SOTA: FIFO & standard priority scheduling, Standard PSO and C-PSO, NSGA-II, Deep RL (DeepRM, Decima)

Metrics

Job completion time
Queue & wait time
Worker / GPU utilization
Resource consumption
Cloud cost
SLA / deadline violations
Why this fits "Real-Time AI for Critical Decisions"
	
Real-time	Sub-second scheduling decisions on every event tick — not a batch job run overnight.
Secure by construction	Runs on IBM Z / LinuxONE's hardware-isolated, highly secure compute — also fits "AI Secured: Innovation Without Exposure."
Critical trade-offs	Every decision balances cost, deadline fairness, and completion time — mistakes are expensive.
Status

Research pitch stage — trace-driven simulation toward a publishable multi-objective scheduler. Open items called out in the deck: strengthening novelty beyond "CALURN + cloud," and comparing against a lightweight cost-aware wrapper on Kubernetes priority classes (not just FIFO/standard PSO) to make the case that full MOPSO is worth its own overhead.

Team / Event

IBM Z Datathon 2026 — Real-Time AI for Critical Decisions track.
