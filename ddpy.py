import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
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

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class AWGNActionNoise(object):
    def __init__(self, mu = 0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        #self.mu = mu
        #self.sigma = sigma
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, jamming_effect):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions, name,
                 chkpt_dir='C:\\demo\\IRS_DDPG_minimal\\main_foder\\tmp\\ddpg', load_file = ''):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.load_file = 'C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Critic_UAV_ddpg'
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc4_dims)
        f5 = 0.003
        self.q = nn.Linear(self.fc4_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f5, f5)
        T.nn.init.uniform_(self.q.bias.data, -f5, f5)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = self.bn4(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self,load_file = ''):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(load_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions, name,
                 chkpt_dir='C:\\demo\\IRS_DDPG_minimal\\main_foder\\tmp\\ddpg', load_file = ''):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims        
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.load_file = 'C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Actor_UAV_ddpg'
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 0.002
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        #f3 = 0.004
        f5 = 0.003
        self.mu = nn.Linear(self.fc4_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f5, f5)
        T.nn.init.uniform_(self.mu.bias.data, -f5, f5)
        #self.mu.weight.data.uniform_(-f3, f3)
        #self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(load_file))

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, layer3_size=256, layer4_size=128, batch_size=64, noise = 'AWGN', agent_name = 'default', load_file = ''):
        self.load_file = load_file
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.layer4_size = layer4_size
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                  name='Actor_' + agent_name,chkpt_dir=env.data_manager.store_path )
        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                    name='Critic_' + agent_name,chkpt_dir=env.data_manager.store_path)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                         name='TargetActor_' + agent_name,chkpt_dir=env.data_manager.store_path)
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                           name='TargetCritic_' + agent_name,chkpt_dir=env.data_manager.store_path)
        if noise == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(n_actions))
        elif noise == 'AWGN':
            self.noise = AWGNActionNoise(mu = np.zeros(n_actions))
        # tau = 1 means copy parameters to target
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, greedy=0.5, epsilon = 1):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(greedy * self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done, jamming_effect):
        self.memory.store_transition(state, action, reward, new_state, done, jamming_effect)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # the done here is opposite of the done in the env
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        # trun s, a, r, new_s into tensor
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        # trun on evaliation mode of target actor, target critic, critic net
        # fix these three nets
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # argmax_action*(Q(new_state,action*)),i.e. pi(s|pi),i.e. stratage choose the action* maxmium the future Q 
        target_actions = self.target_actor.forward(new_state)
        # caculate the Q(new_state, new_action)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        # caculate the Q(state, action)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        # here update the critic net using mse of (r + gamma * Q_argmax_a*(newstate, a*)) - Q(state, action)
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # here update the actor net by policy gradient
        # first fix the critic net
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self, load_file_actor = '',load_file_critic =''):
        self.actor.load_checkpoint(load_file = load_file_actor)
        self.target_actor.load_checkpoint(load_file = load_file_actor)
        self.critic.load_checkpoint(load_file = load_file_critic)
        self.target_critic.load_checkpoint(load_file = load_file_critic)

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()
