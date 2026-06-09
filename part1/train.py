"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import agent
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from torch.optim.lr_scheduler import LinearLR
import time

# To train a new policy from scratch, use this training script use the following command
# Command Template:
# python train_hopper.py --algo <ALGORITHM (REINFORCE|AC)> --episodes <NUM_EPISODES> --seed <SEED> --baseline (0|20)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE or Actor-Critic on Hopper-v4")
    parser.add_argument("--algo", type=str, default="AC", choices=["REINFORCE", "AC"], 
                        help="The algorithm to train: 'REINFORCE' or 'AC'")
    parser.add_argument("--episodes", type=int, default=250, 
                        help="Number of training episodes. Overrides defaults.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--baseline", type=float, default=0.0, 
                        help="Constant baseline value for REINFORCE (e.g., 0.0 or 20.0)")
    args = parser.parse_args()

    n_episodes = args.episodes 
    seed = args.seed
    
    set_seed(seed)
    env = gym.make('Hopper-v4')

    print(f"Algorithm: {args.algo} | Seed: {seed}")
    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = agent.Policy(obs_dim, act_dim, algorithm=args.algo)
    if args.algo == "REINFORCE":
        rl_agent = agent.Agent(policy, algorithm=args.algo, baseline=args.baseline)
    else:
        rl_agent = agent.Agent(policy, algorithm=args.algo)

    episode_rewards = []

    print(f"Starting {args.algo} training for {n_episodes} episodes...")

    start_time = time.perf_counter()

    for episode in range(n_episodes):
        done = False
        current_seed = seed + episode 
        observation = env.reset(seed=current_seed)[0]
        current_episode_reward = 0 
        
        while not done:
            action, action_log_prob, state_value = rl_agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            done = terminated or truncated

            rl_agent.store_outcome(observation, next_observation, action_log_prob, state_value, reward, done)

            observation = next_observation
    
            current_episode_reward += reward 

        
        rl_agent.update_policy()

        episode_rewards.append(current_episode_reward) 

        if len(episode_rewards) >= 10:
            avg_last_10 = np.mean(episode_rewards[-10:])
            print(f"Finished episode {episode + 1} | Reward: {current_episode_reward:.2f} | 10-Ep Avg: {avg_last_10:.2f}")
            
            if avg_last_10 > 1000:
                print(f"\nAC Target Reached! The average of the last 10 episodes ({avg_last_10:.2f}) is > 1000. Stopping early.")
                break
        else:
            print(f"Finished episode {episode + 1} | Reward: {current_episode_reward:.2f}")
        
        print(f"Finished episode {episode + 1} | Reward: {current_episode_reward:.2f}")
        
    execution_time = time.perf_counter() - start_time

    if args.algo == "REINFORCE":
        model_filename = f"hopper_policy_{args.algo}_{args.baseline}.pth"
        data_filename = f"results_{args.algo}_{args.baseline}.npz"
    else:
        model_filename = f"hopper_policy_{args.algo}.pth"
        data_filename = f"results_{args.algo}.npz"

    torch.save(rl_agent.policy.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

    np.savez(data_filename, rewards=np.array(episode_rewards), time=execution_time)

if __name__ == '__main__':
    main()
