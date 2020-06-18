import numpy as np
import random
from src.replay_buffer import ReplayBuffer, NaivePrioritizedReplayBuffer
from src.config import Config
from src.model import Actor, Critic
from src.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Agent that interacts with and learns from the environment."""

    def __init__(self, id, state_size, action_size, config = Config()):
        """Initialize an Agent object.
        
        Params
        ======
            id (int): id used to identify the agent
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            config (Config): the agents configuration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.id = id

        self.config = config

        random.seed(config.random_seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor & Target Network 
        self.actor_local = Actor(state_size, action_size, config.random_seed, config.actor_hidden_units).to(self.device)
        self.actor_target = Actor(state_size, action_size, config.random_seed, config.actor_hidden_units).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Critic & Target Network
        self.critic_local = Critic(state_size, action_size, config.random_seed, config.critic_hidden_units).to(self.device)
        self.critic_target = Critic(state_size, action_size, config.random_seed, config.critic_hidden_units).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic, weight_decay=config.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, config.random_seed)
        
        # Replay memory
        if config.use_per:
            self.memory = NaivePrioritizedReplayBuffer(action_size, config.buffer_size, config.batch_size, config.random_seed, config.per_alpha,config.per_epsilon)
        else:
            self.memory = ReplayBuffer(action_size, config.buffer_size, config.batch_size, config.random_seed)
    
    def step(self, state, action, reward, next_state, done, beta=None):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.config.buffer_size:
            if self.config.use_per:
                assert(beta != None)
                experiences, weights = self.memory.sample(beta)
                states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
                actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
                rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
                next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
                dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
                weights = torch.from_numpy(np.vstack(weights)).float().to(self.device)

                experiences = (states, actions, rewards, next_states, dones)
                self.learn(experiences, self.config.gamma, weights)
            else:
                experiences = self.memory.sample()

                states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
                actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
                rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
                next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
                dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

                experiences = (states, actions, rewards, next_states, dones)
                self.learn(experiences, self.config.gamma)


    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.config.add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, weights=None):
        """
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            weights (array_like): list of weights for compensation the non-uniform sampling (used only
                                    with prioritized experience replay)
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        if self.config.use_per:
            td_error = Q_expected - Q_targets
            critic_loss = (td_error) ** 2
                
            critic_loss = critic_loss * weights
            critic_loss = critic_loss.mean()

            self.memory.update_priorities(np.hstack(td_error.detach().cpu().numpy()))

        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target networks ------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)                    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def getId(self):
        """ Return the ID of the agent """
        return self.id 

    def summary(self):
        """ Return a brief summary of the agent"""
        s = 'DDPG Agent {}:\n'.format(self.id)
        s += self.config.__str__()
        s += self.actor_local.__str__()
        s += self.critic_local.__str__()
        return s

