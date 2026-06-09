
import gymnasium as gym
from collections import deque
import numpy as np

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(0.5, 8.0),
        mode="none",
    ):
        super().__init__(env)
        
        
        self.mode = mode
        self.mass_range = mass_range

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

        self.mass_min = self.mass_min_limit
        self.mass_max = self.mass_max_limit
        self.last_sample_type = "fixed"

        # ADR Initialization
        if self.mode == "adr":
            self.base_mass = 1.0  # The source environment mass
            self.mass_min = self.base_mass
            self.mass_max = self.base_mass
            self.adr_step = 0.2   # shkrinking and expending parameter
            
            self.performance_window = deque(maxlen=20) 
            self.high_threshold = 0.70  
            self.low_threshold = 0.30   

    # Mass Sampling
    def sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "fixed"
            return None
            
        elif self.mode == "udr":
            self.last_sample_type = "uniform"
            return np.random.uniform(self.mass_min_limit, self.mass_max_limit)
            
        elif self.mode == "adr":
            self.last_sample_type = "dynamic_uniform"
            return np.random.uniform(self.mass_min, self.mass_max)
            
        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if self.mode == "adr" and done:
            success = float(info.get("is_success", 0.0))
            self.performance_window.append(success)

            # Adjust bounds when the memory buffer is full
            if len(self.performance_window) == self.performance_window.maxlen:
                avg_success = np.mean(self.performance_window)

                # Expand
                if avg_success >= self.high_threshold:
                    self.mass_min = max(self.mass_min_limit, self.mass_min - self.adr_step)
                    self.mass_max = min(self.mass_max_limit, self.mass_max + self.adr_step)
                    self.performance_window.clear() 

                #Shrink
                elif avg_success <= self.low_threshold:
                    self.mass_min = min(self.base_mass, self.mass_min + self.adr_step)
                    self.mass_max = max(self.base_mass, self.mass_max - self.adr_step)
                    self.performance_window.clear() 

        return obs, reward, terminated, truncated, info

    # Reset
    def reset(self, **kwargs):

        new_mass = self.sample_mass() #TODO: sample new mass

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        return super().reset(**kwargs)
