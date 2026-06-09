# Sim-to-Real Transfer in Continuous Control

This repository contains the code and experimental results for the Reinforcement Learning group project, developed for the **Fundamentals of Artificial Intelligence, Machine and Deep Learning (FAIMDL)** course at Politecnico di Torino.

The project investigates the application of Reinforcement Learning (RL) for continuous robotic control, specifically addressing the "sim-to-real" reality gap. It is divided into two main phases:
1. **Part 1:** Establishing foundational RL principles by implementing and analyzing REINFORCE and Actor-Critic architectures on the `Hopper-v4` environment.
2. **Part 2:** Solving a robotic manipulation task (`PandaGym PushTask`) using Soft Actor-Critic (SAC) and a simulated reality gap (mass shift) using Uniform and Automatic Domain Randomization (UDR/ADR).

---

## Repository Structure

The repository is organized into two distinct folders corresponding to the two phases of the project.

### `part1/` - Foundational Policy Gradients
This directory contains the implementations of REINFORCE and Actor-Critic built from scratch to solve the `Hopper-v4` continuous control task.

* **`train.py`**: The main execution script to train the REINFORCE or Actor-Critic agents.
* **`agent.py`**: Contains the core neural network architectures (Policy and Critic MLPs) and the RL agent logic (action sampling, gradient updates, and baseline subtraction).
* **`plots.py`**: A script used to generate reward curves and performance comparisons from the saved `.npz` data arrays.
* **`*.pth` files**: Pre-trained PyTorch model weights for the Actor-Critic (`hopper_policy_AC.pth`) and REINFORCE agents (with and without baselines).
* **`*.npz` files**: Saved training metrics (rewards and execution times) used for evaluation and plotting.
* **`*.png` files**: Generated visual representations of algorithmic reward evolution and time comparisons.

### `part2/` - Sim-to-Real Transfer & Domain Randomization
This directory contains the advanced Deep RL pipeline utilizing Stable-Baselines3 (SB3) to solve the `PandaPush-v3` environment and simulate the reality gap.

* **`train_sb3.py`**: The main training script to launch PPO or SAC training sessions on the source domain.
* **`eval_sb3.py`**: The evaluation script used to test trained policies and quantify the sim-to-real performance drop (or success) across different transfer configurations.
* **`rand_wrapper.py`**: Contains the environment wrappers implementing Uniform Domain Randomization (UDR) and Automatic Domain Randomization (ADR) to alter the cube's mass during training.
* **`plot.py`**: Script for generating the final success rate curves comparing the baseline methods against the domain randomization strategies.
* **`final_sac_models/`**: Directory containing the best-performing `.zip` models trained with SAC across different domain randomization configurations.
* **`tb_logs/`**: Directory containing the raw TensorBoard event files for tracking training metrics.
* **`panda-gym/`**: The local setup or modifications for the PandaGym simulator environment.
* **`*.zip` files**: Pre-trained model checkpoints (e.g., the baseline PPO model).
* **`*.png` files**: The final sim-to-real transfer success rate visualizations.

---

## Reproducing Results

### Part 1: Training Foundational Algorithms (Hopper-v4)
Navigate to the `part1` directory and use the `train.py` script to train the algorithms from scratch.

**Command Template:**
"python train.py --algo <ALGORITHM> --episodes <NUM_EPISODES> --seed <SEED> --baseline <BASELINE_VALUE>"

### Part 2: Training Policies (PandaGym)
Navigate to the part2 directory to train the SB3 agents.
"python train_sb3.py --algo <ALGORITHM> --sampling-strategy <STRATEGY> --env-type <ENV_TYPE> --timesteps <TIMESTEPS>"

### Part 2: Evaluating Sim-to-Real Transfer
To evaluate a trained policy and reproduce the transfer configurations, use the eval_sb3.py script.
"python eval_sb3.py --algo <ALGORITHM> --model-path "<PATH_TO_MODEL>" --episodes <NUM_EPISODES> --env-type <ENV_TYPE>"

Note: for the SAC models, use the "final_sac_models" directory.
