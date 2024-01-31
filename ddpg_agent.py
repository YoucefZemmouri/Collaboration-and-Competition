import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_PAUSE = 1 
LEARN_TIMES = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed=21, epsilon=1.0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor 1 Network (w/ Target Network)
        self.actor_local_1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target_1 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer_1 = optim.Adam(self.actor_local_1.parameters(), lr=LR_ACTOR)

        # Actor 1 Network (w/ Target Network)
        self.actor_local_2 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target_2 = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer_2 = optim.Adam(self.actor_local_2.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.epsilon = epsilon

        self.memory_1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memory_2 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        print(device)
    
    def step(self, states, actions, rewards, next_states, dones, current_step, epsilon_decay):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # game_state = states.reshape(-1, )
        # next_game_state = next_states.reshape(-1, )
        self.memory_1.add(states[0], actions[0], rewards[0], next_states[0], dones[0])
        self.memory_2.add(states[1], actions[1], rewards[1], next_states[1], dones[1])
        #self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        #if len(self.memory) > BATCH_SIZE:
        #    experiences = self.memory.sample()
        #    self.learn(experiences, GAMMA)
            
        if current_step % LEARN_PAUSE == 0:
            if len(self.memory_1) > BATCH_SIZE:
                for _ in range(LEARN_TIMES):
                    experiences_1 = self.memory_1.sample()
                    experiences_2 = self.memory_2.sample()
                    self.learn_1(experiences_1, GAMMA, epsilon_decay)
                    self.learn_2(experiences_2, GAMMA, epsilon_decay)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # Agent 1
        state_1 = torch.from_numpy(state[0]).float().to(device)
        self.actor_local_1.eval()
        with torch.no_grad():
            action_1 = self.actor_local_1(state_1).cpu().data.numpy()
        self.actor_local_1.train()
        if add_noise:
            action_1 += self.epsilon * self.noise.sample()
        action_1 = np.clip(action_1, -1, 1)

        # Agent 2
        state_2 = torch.from_numpy(state[1]).float().to(device)
        self.actor_local_2.eval()
        with torch.no_grad():
            action_2 = self.actor_local_2(state_2).cpu().data.numpy()
        self.actor_local_2.train()
        if add_noise:
            action_2 += self.epsilon * self.noise.sample()
        action_2 = np.clip(action_2, -1, 1)

        return np.stack([action_1, action_2]).reshape(2,2)

    def reset(self):
        self.noise.reset()

    def learn_1(self, experiences, gamma, epsilon_decay):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target_1(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()


        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local_1(states)
        actor_loss_1 = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer_1.zero_grad()
        actor_loss_1.backward()
        self.actor_optimizer_1.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local_1, self.actor_target_1, TAU)

        self.epsilon -= epsilon_decay
        self.noise.reset()

    def learn_2(self, experiences, gamma, epsilon_decay):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target_2(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()


        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local_2(states)
        actor_loss_2 = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer_2.zero_grad()
        actor_loss_2.backward()
        self.actor_optimizer_2.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local_2, self.actor_target_2, TAU)

        self.epsilon -= epsilon_decay
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)