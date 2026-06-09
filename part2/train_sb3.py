import argparse
from collections import deque
import torch
import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import DDPG, PPO, SAC
from rand_wrapper import RandomizationWrapper
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# To train a new policy from scratch, use this training script use the following command
# Command Template:
# python train_sb3.py --algo <ALGORITHM (sac|ppo)> --sampling-strategy <STRATEGY(none|udr|adr)> --env-type <ENV_TYPE (source|target)> --timesteps <TIMESTEPS>

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on PandaPush-v3")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="The RL algorithm to use (ppo or sac)",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    #TODO: add randomization wrapper here
    if args.sampling_strategy == "udr":
        env = RandomizationWrapper(env, mass_range=(0.5, 8.0), mode="udr")
    elif args.sampling_strategy == "adr":
        env = RandomizationWrapper(env, mass_range=(0.5, 12.0), mode="adr")
    
    env = DummyVecEnv([lambda: env])
    #TODO: create model and train it
    if args.algo == "ppo":
        policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]), # pi = Actor, vf = Critic
                activation_fn=torch.nn.ReLU
                )
        print("Initializing PPO model...")
        model = PPO("MultiInputPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,                        
            batch_size=512,                  
            n_epochs=10,                         
            gamma=0.99,                          
            ent_coef=0.03,                 
            policy_kwargs=policy_kwargs,
            verbose=1, tensorboard_log="./tb_logs/"
        )
            
    elif args.algo == "sac":
        eval_env = gym.make("PandaPush-v3", type=args.env_type, reward_type="dense")

        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=f"./best_model_{args.sampling_strategy}_{args.env_type}/",
            log_path="./tb_logs/", 
            eval_freq=10_000,
            deterministic=False, 
            render=False
        )
        print("Initializing UPGRADED SAC model...")

        if args.sampling_strategy == "adr":
            print("Loading pre-trained Source model for ADR Warm Start...")
            
            model = SAC.load("./final_sac_models/sac_push_none_source_500k.zip", env=env, tensorboard_log="./tb_logs/")
        else:
            print("Initializing fresh SAC model...")
            model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
        
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")


    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    # TODO: model.save(save_name)
    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(save_name)
    print(f"Model saved to {save_name}.zip")


if __name__ == "__main__":
    main()