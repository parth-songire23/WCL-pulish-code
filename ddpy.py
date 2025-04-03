import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise:
    """
    Ornstein-Uhlenbeck process for adding exploration noise.
    Helps in exploration by generating temporally correlated noise.
    """
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling training experiences.
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.jamming_memory = np.zeros(self.mem_size, dtype=np.float32)  # Store jamming effects

    def store_transition(self, state, action, reward, state_, done, jamming_level):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.jamming_memory[index] = jamming_level  # Store jamming impact
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch],
            self.terminal_memory[batch],
            self.jamming_memory[batch]
        )

class CriticNetwork(nn.Module):
    """
    Critic Network (Q-value function) for estimating state-action value.
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join('./tmp/', name+'_ddpg.pth')

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_value = nn.Linear(n_actions, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        return self.q(state_action_value)

class ActorNetwork(nn.Module):
    """
    Actor Network (Policy function) for generating actions.
    """
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join('./tmp/', name+'_ddpg.pth')

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return T.tanh(self.mu(x))  # Output is bounded between [-1,1]

class Agent:
    """
    Reinforcement Learning Agent using the DDPG algorithm.
    """
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, 
                 n_actions=2, batch_size=64, memory_max_size=1000000, layer1_size=800, layer2_size=600, layer3_size=512, layer4_size=256):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(memory_max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, n_actions, name='Actor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, n_actions, name='Critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, n_actions, name='TargetActor')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, n_actions, name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float32).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float32).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done, jamming = self.memory.sample_buffer(self.batch_size)
        reward, jamming, done = map(lambda x: T.tensor(x, dtype=T.float32).to(self.critic.device), (reward, jamming, done))
        state, action, new_state = map(lambda x: T.tensor(x, dtype=T.float32).to(self.critic.device), (state, action, new_state))

        target_actions = self.target_actor(new_state)
        critic_value_ = self.target_critic(new_state, target_actions)
        critic_value = self.critic(state, action)
        modified_reward = reward - 0.5 * jamming
        target = modified_reward + self.gamma * critic_value_ * done
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        F.mse_loss(target, critic_value).backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        (-self.critic(state, self.actor(state)).mean()).backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
