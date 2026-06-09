import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal



def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, algorithm="AC"):
        super().__init__()
        self.algorithm = algorithm
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
        

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic= torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc_critic_value = torch.nn.Linear(self.hidden, 1)


        self.init_weights()
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
         
        torch.nn.init.orthogonal_(self.fc_critic_value.weight, gain=1.0)
        torch.nn.init.zeros_(self.fc_critic_value.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        x_actor = self.tanh(self.fc3_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        sigma = torch.clamp(sigma, min=0.2, max=2.0)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        if self.algorithm == "AC":
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))

            critic_state_value = self.fc_critic_value(x_critic)
        else:
            critic_state_value = None
        
        return normal_dist, critic_state_value


class Agent(object):
    def __init__(self, policy, baseline=20, algorithm="AC", device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.algorithm = algorithm
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.baseline = baseline

        self.gamma = 0.99
        self.entropy_coef = 0.05
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.entropies = []
        self.state_values = []



    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        entropies = torch.stack(self.entropies, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.entropies, self.done = [], [], [], [], [], []

        entropy_coef = self.entropy_coef
        entropy_loss = entropies.mean()
        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #
        if self.algorithm == "REINFORCE":
            # TASK 2:
            #   - compute discounted returns
            #   - compute policy gradient loss function given actions and returns
            #   - compute gradients and step the optimizer

            returns = discount_rewards(rewards, self.gamma)

            baseline = self.baseline
            advantage = returns - baseline
            
            policy_loss = -(action_log_probs * advantage).mean()
            total_loss = policy_loss - (entropy_coef * entropy_loss)

        elif self.algorithm == "AC":
            # TASK 3:
            #   - compute boostrapped discounted return estimates
            #   - compute advantage terms
            #   - compute actor loss and critic loss
            #   - compute gradients and step the optimizer
            state_values = torch.stack(self.state_values, dim=0).to(self.train_device).squeeze(-1)
            self.state_values = []

            _, next_state_values = self.policy(next_states)
            next_state_values = next_state_values.squeeze(-1).detach()
            
            target_values = rewards + self.gamma * next_state_values * (1.0 - done)

            advantages = target_values - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.mse_loss(state_values, target_values)

            total_loss = actor_loss + critic_loss - (entropy_coef * entropy_loss)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=2.0)
        self.optimizer.step()

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:  
            return normal_dist.mean, None, state_value

        else:  
            action = normal_dist.sample()

            action_log_prob = normal_dist.log_prob(action).sum()
            entropy = normal_dist.entropy().sum()
            self.entropies.append(entropy)

            return action, action_log_prob, state_value


    def store_outcome(self, state, next_state, action_log_prob, state_value, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.state_values.append(state_value)
        self.done.append(done)

